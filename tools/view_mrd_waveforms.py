"""
View cardiac waveforms stored in an MRD/H5 file.

This tool opens the selected MRD file read-only, plots ECG, EXT1, and saved PT
waveforms when present, and prints trigger and heart-rate statistics.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from pylottone import mrdhelper
from pylottone.pt import check_waveform_polarity
from pylottone.triggering import calculate_jitter

try:
    import ismrmrd
except ImportError as exc:
    raise ImportError("ismrmrd is required for MRD file I/O. Install with: pip install pylottone[mrd]") from exc


def _get_filepath_from_ui(start_dir: str) -> str:
    try:
        from PySide6.QtWidgets import QApplication, QFileDialog
    except ImportError as exc:
        try:
            from PySide2.QtWidgets import QApplication, QFileDialog  # type: ignore
        except ImportError:
            raise ImportError(
                "UI dependencies are required for the file browser. "
                "Either pass --file or install with: pip install pylottone[ui]"
            ) from exc

    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])

    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    filepath, _ = QFileDialog.getOpenFileName(
        None,
        "Select an MRD image file",
        start_dir,
        "MRD Files (*.mrd *.h5);;All Files (*)",
        options=options,
    )

    if owns_app:
        app.shutdown()

    return filepath


def _read_waveforms_readonly(filepath: str, dataset_name: str) -> tuple[list[Any], float]:
    print(f"Reading waveforms from {filepath}...")
    with ismrmrd.File(filepath, mode="r") as mrd_file:
        dataset = mrd_file[dataset_name]
        first_acq_time_s = 0.0
        if dataset.has_acquisitions():
            first_acq = dataset.acquisitions[:1][0]
            first_acq_time_s = first_acq.acquisition_time_stamp * 2.5e-3
        if not dataset.has_waveforms():
            return [], first_acq_time_s
        waveforms = dataset.waveforms[:]
    print(f"Read {len(waveforms)} waveforms.")
    return waveforms, first_acq_time_s


def _as_1d_float(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).squeeze()
    if values.ndim != 1:
        raise ValueError(f"Expected a one-dimensional waveform, got shape {values.shape}.")
    return values


def _normalize_for_plot(values: np.ndarray) -> np.ndarray:
    values = _as_1d_float(values)
    values = values - np.nanmedian(values)
    scale = np.nanpercentile(np.abs(values), 99)
    if not np.isfinite(scale) or scale == 0:
        scale = np.nanmax(np.abs(values))
    if np.isfinite(scale) and scale > 0:
        values = values / scale
    return values


def _trigger_locs(trigger_waveform: np.ndarray) -> np.ndarray:
    triggers = np.asarray(trigger_waveform).squeeze()
    if triggers.size == 0:
        return np.array([], dtype=int)
    triggers = triggers > 0
    return np.nonzero(triggers)[0]


def _hr_stats(time_s: np.ndarray, trigger_waveform: np.ndarray) -> dict[str, float]:
    locs = _trigger_locs(trigger_waveform)
    if locs.size < 2:
        return {"n_triggers": float(locs.size)}

    rr_intervals_s = np.diff(time_s[locs])
    rr_intervals_s = rr_intervals_s[rr_intervals_s > 0]
    heart_rate_bpm = 60.0 / rr_intervals_s
    return {
        "n_triggers": float(locs.size),
        "mean_bpm": float(np.mean(heart_rate_bpm)),
        "median_bpm": float(np.median(heart_rate_bpm)),
        "std_bpm": float(np.std(heart_rate_bpm)),
    }


def _remove_close_triggers(time_s: np.ndarray, triggers: np.ndarray, min_separation_s: float) -> np.ndarray:
    cleaned_triggers = np.zeros_like(triggers)
    trig_locs = _trigger_locs(triggers)
    if trig_locs.size == 0:
        return cleaned_triggers

    kept_trig_loc = trig_locs[0]
    cleaned_triggers[kept_trig_loc] = 1
    for trig_loc in trig_locs[1:]:
        if time_s[trig_loc] - time_s[kept_trig_loc] < min_separation_s:
            continue
        cleaned_triggers[trig_loc] = 1
        kept_trig_loc = trig_loc

    return cleaned_triggers


def _cleanup_ecg_ext1_triggers(
    waveforms: dict[str, dict[str, np.ndarray]],
    max_hr: float,
) -> list[tuple[str, int, int]]:
    cleanup_counts: list[tuple[str, int, int]] = []
    min_separation_s = 60.0 / max_hr
    for name in ("ECG", "EXT1"):
        if name not in waveforms:
            continue
        entry = waveforms[name]
        raw_triggers = entry["triggers"]
        cleaned_triggers = _remove_close_triggers(entry["time"], raw_triggers, min_separation_s)
        entry["cleaned_triggers"] = cleaned_triggers.astype(np.uint32)
        cleanup_counts.append((name, int(np.count_nonzero(raw_triggers)), int(np.count_nonzero(cleaned_triggers))))

    return cleanup_counts


def _stats_triggers(entry: dict[str, np.ndarray]) -> np.ndarray:
    return entry.get("cleaned_triggers", entry["triggers"])


def _format_stats(label: str, stats: dict[str, float]) -> str:
    lines = [f"{label}:", f"  triggers: {int(stats.get('n_triggers', 0))}"]
    if "mean_bpm" in stats:
        lines.extend(
            [
                f"  HR mean: {stats['mean_bpm']:.2f} bpm",
                f"  HR median: {stats['median_bpm']:.2f} bpm",
                f"  HR std: {stats['std_bpm']:.2f} bpm",
            ]
        )
    else:
        lines.append("  HR: not enough triggers")
    return "\n".join(lines)


def _format_cleanup_stats(cleanup_counts: list[tuple[str, int, int]], max_hr: float) -> str:
    if not cleanup_counts:
        return f"Trigger cleanup ({max_hr:.0f} bpm max):\n  No ECG/EXT1 triggers available."
    lines = [f"Trigger cleanup ({max_hr:.0f} bpm max):"]
    for name, before, after in cleanup_counts:
        lines.append(f"  {name}: {before} beats before, {after} beats after")
    return "\n".join(lines)


def _jitter_stats(
    label: str,
    time_ref: np.ndarray,
    ref_waveform: np.ndarray,
    ref_triggers: np.ndarray,
    time_pt: np.ndarray,
    pt_cardiac: np.ndarray,
    pt_triggers: np.ndarray,
) -> str:
    try:
        peak_diff, missed, extra = calculate_jitter(
            time_pt,
            pt_cardiac,
            time_ref,
            ref_waveform,
            pt_cardiac_trigs=pt_triggers,
            ecg_trigs=ref_triggers,
            skip_time=0.5,
            peak_prominence=0.4,
            max_hr=160,
        )
    except Exception as exc:
        return f"{label} vs PT:\n  jitter: unavailable ({exc})"

    if peak_diff.size == 0:
        return f"{label} vs PT:\n  matched PT triggers: 0"

    diff_ms = peak_diff * 1e3
    return "\n".join(
        [
            f"{label} vs PT:",
            f"  matched PT triggers: {peak_diff.size}",
            f"  missed PT triggers: {missed.size}",
            f"  extra PT triggers: {extra.size}",
            f"  peak diff mean: {np.mean(diff_ms):.2f} ms",
            f"  peak diff median: {np.median(diff_ms):.2f} ms",
            f"  peak diff std: {np.std(diff_ms):.2f} ms",
        ]
    )


def _collect_waveforms(
    wf_list: list[Any],
    first_acq_time_s: float,
) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    waveforms: dict[str, dict[str, np.ndarray]] = {}
    notes: list[str] = []

    wf_dict = mrdhelper.waveforms_asarray2(wf_list)

    if "ecg" in wf_dict:
        time_ecg, ecg = wf_dict["ecg"]
        ecg_waveform = check_waveform_polarity(ecg[:, 0], 0.5, method="width") * ecg[:, 0]
        ecg_triggers = (ecg[:, -1] > 0).astype(np.uint32)
        waveforms["ECG"] = {
            "time": _as_1d_float(time_ecg) - first_acq_time_s,
            "waveform": _as_1d_float(ecg_waveform),
            "triggers": ecg_triggers,
        }
    else:
        notes.append("No ECG waveform found.")

    if "ext1" in wf_dict:
        time_ext1, ext1 = wf_dict["ext1"]
        ext1_triggers = (ext1[:, -1] > 0).astype(np.uint32)
        waveforms["EXT1"] = {
            "time": _as_1d_float(time_ext1) - first_acq_time_s,
            "waveform": _as_1d_float(ext1[:, 0]),
            "triggers": ext1_triggers,
        }
    else:
        notes.append("No EXT1 waveform found.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pt_waveforms = mrdhelper.waveforms_asarray(wf_list)

    if pt_waveforms is not None and "pt_cardiac" in pt_waveforms:
        waveforms["PT"] = {
            "time": _as_1d_float(pt_waveforms["time_pt"]) - first_acq_time_s,
            "waveform": _as_1d_float(pt_waveforms["pt_cardiac"]),
            "triggers": np.asarray(pt_waveforms["pt_cardiac_trigs"], dtype=np.uint32),
        }
    elif "resp" in wf_dict:
        time_resp, resp = wf_dict["resp"]
        waveforms["PT"] = {
            "time": _as_1d_float(time_resp) - first_acq_time_s,
            "waveform": _as_1d_float(resp),
            "triggers": np.zeros(resp.shape, dtype=np.uint32),
        }
        notes.append("Found RESPPT waveform, but no saved PT cardiac triggers.")
    else:
        notes.append("No PT waveform found.")

    return waveforms, notes


def _build_stats_text(
    waveforms: dict[str, dict[str, np.ndarray]],
    notes: list[str],
    cleanup_counts: list[tuple[str, int, int]] | None = None,
    cleanup_max_hr: float = 160.0,
) -> str:
    stats_sections: list[str] = []
    for name in ("ECG", "EXT1", "PT"):
        if name in waveforms:
            entry = waveforms[name]
            stats_sections.append(_format_stats(name, _hr_stats(entry["time"], _stats_triggers(entry))))

    if cleanup_counts is not None:
        stats_sections.append(_format_cleanup_stats(cleanup_counts, cleanup_max_hr))

    if "PT" in waveforms:
        pt = waveforms["PT"]
        for ref_name in ("ECG", "EXT1"):
            if ref_name in waveforms:
                ref = waveforms[ref_name]
                stats_sections.append(
                    _jitter_stats(
                        ref_name,
                        ref["time"],
                        ref["waveform"],
                        _stats_triggers(ref),
                        pt["time"],
                        pt["waveform"],
                        pt["triggers"],
                    )
                )

    return "\n\n".join([*stats_sections, *notes])


def _load_waveform_report(filepath: str, dataset_name: str) -> tuple[dict[str, dict[str, np.ndarray]], list[str], str]:
    filepath = str(Path(filepath).expanduser())
    if not filepath:
        raise SystemExit("No MRD file selected.")
    if not Path(filepath).exists():
        raise FileNotFoundError(filepath)

    wf_list, first_acq_time_s = _read_waveforms_readonly(filepath, dataset_name)
    if not wf_list:
        raise SystemExit("No waveforms found in the selected MRD file.")

    waveforms, notes = _collect_waveforms(wf_list, first_acq_time_s)
    return waveforms, notes, _build_stats_text(waveforms, notes)


def _draw_waveforms(
    fig: plt.Figure,
    filepath: str,
    dataset_name: str,
    waveforms: dict[str, dict[str, np.ndarray]],
    notes: list[str],
    stats_text: str,
) -> None:
    fig.clear()
    plot_names = [name for name in ("ECG", "EXT1", "PT") if name in waveforms]
    if not plot_names:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No ECG, EXT1, or PT waveforms found.", ha="center", va="center")
        ax.set_axis_off()
    else:
        axes = fig.subplots(len(plot_names), 1, sharex=False)
        axes = np.atleast_1d(axes)

        for ax, name in zip(axes, plot_names, strict=True):
            entry = waveforms[name]
            time_s = entry["time"]
            waveform = _normalize_for_plot(entry["waveform"])
            triggers = entry["triggers"]
            locs = _trigger_locs(triggers)
            cleaned_locs = _trigger_locs(entry["cleaned_triggers"]) if "cleaned_triggers" in entry else np.array([], dtype=int)

            ax.plot(time_s, waveform, linewidth=0.9, label=name)
            if locs.size > 0:
                ax.plot(time_s[locs], waveform[locs], "*", markersize=6, label=f"{name} triggers")
            if cleaned_locs.size > 0:
                ax.plot(
                    time_s[cleaned_locs],
                    waveform[cleaned_locs],
                    "o",
                    markersize=4,
                    fillstyle="none",
                    label=f"{name} cleaned triggers",
                )
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Time [s]")

    fig.suptitle(Path(filepath).name)
    fig.text(
        0.01,
        0.04,
        stats_text,
        va="bottom",
        ha="left",
        family="monospace",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    cleanup_ax = fig.add_axes((0.61, 0.03, 0.18, 0.05))
    cleanup_button = Button(cleanup_ax, "Clean ECG/EXT1")

    button_ax = fig.add_axes((0.82, 0.03, 0.16, 0.05))
    load_button = Button(button_ax, "Load another data")

    def cleanup_triggers(_event: object) -> None:
        max_hr = 160.0
        cleanup_counts = _cleanup_ecg_ext1_triggers(waveforms, max_hr=max_hr)
        next_stats_text = _build_stats_text(waveforms, notes, cleanup_counts, cleanup_max_hr=max_hr)
        print(next_stats_text)
        _draw_waveforms(fig, filepath, dataset_name, waveforms, notes, next_stats_text)

    def load_another(_event: object) -> None:
        selected = _get_filepath_from_ui(str(Path(filepath).expanduser().parent))
        if not selected:
            return
        try:
            next_waveforms, next_notes, next_stats_text = _load_waveform_report(selected, dataset_name)
        except Exception as exc:
            print(f"Could not load {selected}: {exc}")
            return
        print(next_stats_text)
        _draw_waveforms(fig, selected, dataset_name, next_waveforms, next_notes, next_stats_text)

    cleanup_button.on_clicked(cleanup_triggers)
    load_button.on_clicked(load_another)
    fig._mrd_buttons = [cleanup_button, load_button]
    fig.tight_layout(rect=(0, 0.21, 1, 0.95))
    fig.canvas.draw_idle()


def _plot_waveforms(
    filepath: str,
    dataset_name: str,
    waveforms: dict[str, dict[str, np.ndarray]],
    notes: list[str],
    stats_text: str,
) -> None:
    fig = plt.figure(figsize=(12, 8))
    _draw_waveforms(fig, filepath, dataset_name, waveforms, notes, stats_text)
    plt.show()


def view_mrd_waveforms(filepath: str, dataset_name: str = "dataset") -> None:
    waveforms, notes, stats_text = _load_waveform_report(filepath, dataset_name)
    print(stats_text)
    _plot_waveforms(filepath, dataset_name, waveforms, notes, stats_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="View ECG, EXT1, and PT waveforms from an MRD/H5 file.")
    parser.add_argument("-f", "--file", help="MRD/H5 file to open. If omitted, a file browser is shown.")
    parser.add_argument("-d", "--dataset", default="dataset", help="MRD dataset name. Default: dataset.")
    args = parser.parse_args()

    filepath = args.file
    if filepath is None:
        filepath = _get_filepath_from_ui(os.path.expanduser("~"))

    view_mrd_waveforms(filepath, dataset_name=args.dataset)


if __name__ == "__main__":
    main()
