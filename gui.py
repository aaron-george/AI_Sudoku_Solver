from tkinter import *
text=[]

final=[]
def create_gui():
    root=Tk()
    root.title("Sudoku Solver")
    canvas = Canvas(root, height=320, width =350)
    Path_Entry(root)
    Developer_name(root)
    createbuttons(root)
    canvas.pack(side = 'top')
    root.mainloop()

def Path_Entry(top):
    path=Entry(top, width=3, font = 'BOLD')
    path.insert(0,"Enter the path to sudoku Image")
    path.place(x=30, y=10, height=20, width=300)
    text.append(path)

def Developer_name(top):
    status=Label(top,text="Developed By AARON GEORGE",bd=1,relief=SUNKEN,anchor=W)
    status.pack(side=BOTTOM,fill=X)

def message():
    status=Label(text="Path is added, close window to continue")
    status.pack()
    

def PrintPath():
    path=text[0]
    pth=path.get()
    final.append(pth)
    

def createbuttons(root):
    
    button_add_path= Button(root, text="Add path", justify='left',command = lambda: [PrintPath(),message()])
    
    button_add_path.place(x=290, y=10, height=20, width=60)
    

def first_gui():
	create_gui()
	return final[0]





