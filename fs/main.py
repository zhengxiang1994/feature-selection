# the main function
import fs.fileUtil as fUtil
import fs.fcbf as fcbf
import fs.fast as fast
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import ttk
from tkinter import scrolledtext
import numpy as np
from sklearn import cross_validation, svm, neighbors, tree
from sklearn.ensemble import RandomForestClassifier


def select_path():
    path_ = askopenfilename()
    path.set(path_)


def show_result():
    # st1.delete(0., END)
    file_path = entry1.get()
    fs_name = combobox1.get()
    cls_name = combobox2.get()
    data = fUtil.read_data(file_path)
    m, n = data.shape
    st1.insert(END, "-"*5 + fs_name + "-"*5 + "\n")
    best_list = None
    if fs_name == "FULL":
        st1.insert(END, "seleted features: All Features\n")
        best_list = range(n-1)
    elif fs_name == "FAST":
        best_list = fast.alg_fast(data)
        st1.insert(END, "selected features:" + str(best_list) + "\n")
    else:
        best_list = fcbf.alg_fcbf(data)
        st1.insert(END, "selected features:" + str(best_list) + "\n")
    attrs = []
    for f in best_list:
        attrs.append(data[:, f])
    attrs = np.array(attrs).T
    cls = None
    if cls_name == "SVM":
        cls = svm.SVC()
    elif cls_name == "KNN":
        cls = neighbors.KNeighborsClassifier()
    elif cls_name == "C4.5":
        cls = tree.DecisionTreeClassifier()
    elif cls_name == "Random_Forest":
        cls = RandomForestClassifier()
    else:
        pass
    result = cross_validation.cross_val_score(cls, attrs, data[:, -1], cv=5, scoring="accuracy")
    st1.insert(END, "the accuracy using " + cls_name + ":" + str(result) + "\n")
    st1.insert(END, "the average accuracy using " + cls_name + ":" + str(np.average(result)) + "\n")

if __name__ == "__main__":
    root = Tk()
    root.minsize(500, 300)
    path = StringVar()
    label1 = Label(root, text="dataset", width=15)
    label1.grid(row=0, column=0, sticky=W)
    entry1 = Entry(root, textvariable=path, width=30)
    entry1.grid(row=0, column=1, sticky=W)
    button1 = Button(root, text="choose", width=8, height=1, command=select_path)
    button1.grid(row=0, column=2, sticky=W)
    label2 = Label(root, text="FS algorithm", width=15)
    label2.grid(row=1, column=0, sticky=W)
    combobox1 = ttk.Combobox(root, width=28)
    combobox1["values"] = ("FULL", "FAST", "FCBF")
    combobox1.grid(row=1, column=1, sticky=W)
    label3 = Label(root, text="CLF algorithm", width=15)
    label3.grid(row=2, column=0, sticky=W)
    combobox2 = ttk.Combobox(root, width=28)
    combobox2["values"] = ("SVM", "KNN", "C4.5", "Random_Forest")
    combobox2.grid(row=2, column=1, sticky=W)
    button2 = Button(root, text="start", width=8, height=1, command=show_result)
    button2.grid(row=3, column=2, sticky=W)
    st1 = scrolledtext.ScrolledText(root, width=60, height=10, wrap=WORD)
    st1.grid(row=4, columnspan=3)
    root.mainloop()


