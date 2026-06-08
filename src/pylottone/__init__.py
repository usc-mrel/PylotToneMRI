from importlib import import_module

_PT_EXPORTS = (
    "est_dtft",
    "dtft_sum",
    "extract_raw_pt",
    "sniffer_sub",
    "plot_multich_comparison",
    "pickcoilsbycorr",
    "check_waveform_polarity",
    "extract_pilottone_navs",
    "calibrate_pt",
    "apply_pt_calib",
    "process_cplx_pt",
    "pick_cardiac_source",
    "pick_source_bypeak",
)

__all__ = [*_PT_EXPORTS, "main"]


def __getattr__(name: str):
    if name in _PT_EXPORTS:
        module = import_module(".pt", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def main() -> None:
    print("Hello from pylottone!")
