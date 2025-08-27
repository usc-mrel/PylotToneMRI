try:
    from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QListWidget, QPushButton, QFileDialog
except ImportError: # Fallback to PySide2 in case conda is used.
    from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QListWidget, QPushButton, QFileDialog # type: ignore
import sys

class SelectionWindow(QWidget):
    def __init__(self, selection_list):
        super().__init__()

        self.selected_item = None

        # Set up the window
        self.setWindowTitle('Select a group from the dataset.')
        self.setGeometry(100, 100, 300, 200)

        # Set up the layout
        layout = QVBoxLayout()

        # Create the list widget
        self.list_widget = QListWidget()
        self.list_widget.addItems(selection_list)
        self.list_widget.setCurrentRow(len(selection_list)-1)
        layout.addWidget(self.list_widget)

        # Create the button to confirm selection
        self.button = QPushButton('Confirm Selection')
        self.button.clicked.connect(self.confirm_selection)
        layout.addWidget(self.button)

        # Set the layout
        self.setLayout(layout)

    def confirm_selection(self):
        # Get the selected item
        self.selected_item = self.list_widget.currentItem().text()
        # Close the window
        self.close()

def get_selection(selection_list: list[str]):
    '''QT based selection UI.'''
    app = QApplication(sys.argv)
    window = SelectionWindow(selection_list)
    window.show()
    app.exec_()
    app.shutdown()
    return window.selected_item

def get_filepath(dir: str=''):
    '''QT based file selection UI.'''
    app = QApplication(sys.argv)
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    file_path, _ = QFileDialog.getOpenFileName(None, "Select an MRD image file", dir, "MRD Files (*.mrd *.h5);;All Files (*)", options=options)
    app.shutdown()
    return file_path

def get_multiple_filepaths(dir: str=''):
    '''QT based file selection UI for multiple files.'''
    app = QApplication(sys.argv)
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    file_paths, _ = QFileDialog.getOpenFileNames(None, "Select MRD image files", dir, "MRD Files (*.mrd *.h5);;All Files (*)", options=options)
    app.shutdown()
    return file_paths