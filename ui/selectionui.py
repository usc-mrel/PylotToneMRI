from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QListWidget, QPushButton
import sys

def selection_ui(dsetNames: list[str]) -> str:
    ''' Tkinter based selection UI.
    '''
    import tkinter as tk

    root = tk.Tk()
    var = tk.IntVar()
    root.title("Select a group from the dataset.")
    lb = tk.Listbox(root, selectmode=tk.SINGLE, height = len(dsetNames), width = 50) # create Listbox
    for x in dsetNames: lb.insert(tk.END, x)
    lb.pack() # put listbox on window
    lb.select_set(len(dsetNames)-1)
    btn = tk.Button(root,text="Select Group",command=lambda: var.set(1))
    btn.pack()
    # root.mainloop()
    btn.wait_variable(var)
    group = lb.get(lb.curselection()[0])
    root.destroy()
    return group


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
    return window.selected_item