import numpy as np

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib import colors
plt.style.use('default')

import model_specific as model

WINDOW_TITLE = 'Mixture models tool'
BG_COLOR = "#ffffff" #"#404040"
BT_COLOR = "#ffffff"
WIDTH = 1280
HEIGHT = 720

def defocus(event):
    event.widget.master.focus_set()

class App:
    def __init__(self):

        self.N = 100
        self.K = 2
        self.AvailColors = [colors.to_rgba('blue'), colors.to_rgba('green'), colors.to_rgba('red'), colors.to_rgba('purple'), colors.to_rgba('orange')]
        self.AvailColorMaps = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges']
        self.KColors = np.array([])
        self.X = np.array([], dtype = np.float32) # <- observable variable
        self.t = None # <- targer variable
        self.classifier = None

        self.initWindow()
        self.initDataGroup()
        self.initModelGroup()
        self.initInfoGroup()
        self.initGraphPanel()

    def initWindow(self):
        self.window = tk.Tk()
        self.window.wm_title(WINDOW_TITLE)
        self.window.geometry('%dx%d+%d+%d' % (WIDTH, HEIGHT, 10, 10))
        self.window.minsize(WIDTH, HEIGHT)
        # self.window.configure(background = BG_COLOR)
        self.window.wm_protocol("WM_DELETE_WINDOW", self.onClose)
        # self.window.iconbitmap('data/img/favicon.ico')

        self.leftFrame = tk.Frame(self.window)
        self.leftFrame.place(relx = 0.03, rely = 0.05, relwidth = 0.25, relheight = 0.9)

        self.rightFrame = tk.Frame(self.window, bg = '#f0f0f0', bd = 1.5)
        self.rightFrame.place(relx = 0.3, rely = 0.05, relwidth = 0.65, relheight = 0.9)

    def initDataGroup(self):
        self.dataGroupLabel = tk.LabelFrame(self.leftFrame, text = '  Data  ')
        self.dataGroupLabel.pack(padx = 5, pady = 5, side = tk.TOP, fill = tk.BOTH)

        NLabelText = tk.StringVar()
        NLabelText.set('N = ')
        NSpinboxVar = tk.IntVar()
        NSpinboxVar.set(100)
        NLabel = tk.Label(self.dataGroupLabel, text = NLabelText, textvariable = NLabelText)
        NLabel.grid(row = 0, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.NSpinbox = tk.Spinbox(self.dataGroupLabel, width = 5, from_ = 1, to = 99999, textvariable = NSpinboxVar)
        self.NSpinbox.grid(row = 0, column = 1, columnspan = 2, sticky = tk.W + tk.E, padx = 5, pady = 5)

        KLabelText = tk.StringVar()
        KLabelText.set('K = ')
        KSpinboxVar = tk.IntVar()
        KSpinboxVar.set(2)
        KLabel = tk.Label(self.dataGroupLabel, text = KLabelText, textvariable = KLabelText)
        KLabel.grid(row = 0, column = 3, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.KSpinbox = tk.Spinbox(self.dataGroupLabel, width = 5, from_ = 2, to = 5, textvariable = KSpinboxVar)
        self.KSpinbox.grid(row = 0, column = 4, columnspan = 2, sticky = tk.W + tk.E, padx = 5, pady = 5)

        FuncLabelText = tk.StringVar()
        FuncLabelText.set('f(x) = ')
        FuncLabel = tk.Label(self.dataGroupLabel, text = FuncLabelText, textvariable = FuncLabelText)
        FuncLabel.grid(row = 2, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        funcs = [ "Gaussian" ]
        self.TrueFuncCBox = ttk.Combobox(self.dataGroupLabel, width = 9, values = funcs, state = "readonly")
        self.TrueFuncCBox.bind("<FocusIn>", defocus) # <- in order to prevent box highlighting
        self.TrueFuncCBox.grid(row = 2, column = 1, columnspan = 2, sticky = tk.W + tk.E, padx = 5, pady = 5)
        self.TrueFuncCBox.current(0)

        self.generateButton = tk.Button(
            self.dataGroupLabel,
            text=' Generate ',
            bg = BT_COLOR,
            command = self.onDataGeneration
        )
        self.generateButton.grid(row = 3, column = 0, columnspan = 3, sticky = tk.W + tk.E, padx = 5, pady = 5)

        separator = ttk.Separator(self.dataGroupLabel, orient = 'horizontal')
        separator.grid(row = 4, column = 0, columnspan = 5, sticky = tk.W + tk.E, padx = 5, pady = (5, 15))

        self.openFileButton = tk.Button(
            self.dataGroupLabel,
            text=' Open .npy file ',
            bg = BT_COLOR,
            command = self.onFileOpening
        )
        self.openFileButton.grid(row = 6, column = 0, columnspan = 3, sticky = tk.W + tk.E, padx = 5, pady = 5)

    def initModelGroup(self):
        self.modelGroupLabel = tk.LabelFrame(self.leftFrame, text = '  Model  ')
        self.modelGroupLabel.pack(padx = 5, pady = 5, side = tk.TOP, fill = tk.BOTH)

        self.modelCButtonVar = tk.IntVar()
        self.modelCButtonVar.set(0)
        self.kmeansRButton = tk.Radiobutton(self.modelGroupLabel, text = 'K-means', variable = self.modelCButtonVar, value = 0)
        self.kmeansRButton.grid(row = 2, column = 0, columnspan = 2, sticky = tk.W + tk.N, padx = (5, 0), pady = 5)
        self.MOGRButton = tk.Radiobutton(self.modelGroupLabel, text = 'MOG', variable = self.modelCButtonVar, value = 1)
        self.MOGRButton.grid(row = 2, column = 2, columnspan = 2, sticky = tk.W + tk.N, padx = (5, 0), pady = 5)
        
        separator = ttk.Separator(self.modelGroupLabel, orient = 'horizontal')
        separator.grid(row = 4, column = 0, columnspan = 6, sticky = tk.W + tk.E, padx = 5, pady = (5, 5))

        EpochsLabelText = tk.StringVar()
        EpochsLabelText.set('Epochs = ')
        EpochsSpinboxVar = tk.IntVar()
        EpochsSpinboxVar.set(5)
        EpochsLabel = tk.Label(self.modelGroupLabel, text = EpochsLabelText, textvariable = EpochsLabelText)
        EpochsLabel.grid(row = 5, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.EpochsSpinbox = tk.Spinbox(self.modelGroupLabel, width = 5, from_ = 1, to = 99999, textvariable = EpochsSpinboxVar)
        self.EpochsSpinbox.grid(row = 5, column = 1, columnspan = 3, sticky = tk.W + tk.E, padx = 5, pady = 5)

        ClustersLabelText = tk.StringVar()
        ClustersLabelText.set('Clusters = ')
        ClustersSpinboxVar = tk.IntVar()
        ClustersSpinboxVar.set(2)
        ClustersLabel = tk.Label(self.modelGroupLabel, text = ClustersLabelText, textvariable = ClustersLabelText)
        ClustersLabel.grid(row = 5, column = 4, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.ClustersSpinbox = tk.Spinbox(self.modelGroupLabel, width = 5, from_ = 1, to = 99999, textvariable = ClustersSpinboxVar)
        self.ClustersSpinbox.grid(row = 5, column = 5, columnspan = 3, sticky = tk.W + tk.E, padx = 5, pady = 5)
        
        separator = ttk.Separator(self.modelGroupLabel, orient = 'horizontal')
        separator.grid(row = 8, column = 0, columnspan = 6, sticky = tk.W + tk.E, padx = 5, pady = (5, 15))

        self.drawDetailsCButtonVar = tk.IntVar()
        self.drawDetailsCButtonVar.set(1)
        self.drawDetailsCButton = tk.Checkbutton(self.modelGroupLabel, text = 'Draw details', onvalue = 1, offvalue = 0, variable = self.drawDetailsCButtonVar)
        self.drawDetailsCButton.grid(row = 11, column = 0, columnspan = 6, sticky = tk.W + tk.N, padx = (5, 0), pady = (5, 5))

        self.drawDesRegsButtonVar = tk.IntVar()
        self.drawDesRegsButtonVar.set(1)
        self.drawDesRegsButton = tk.Checkbutton(self.modelGroupLabel, text = 'Draw desicion regions', onvalue = 1, offvalue = 0, variable = self.drawDesRegsButtonVar)
        self.drawDesRegsButton.grid(row = 12, column = 0, columnspan = 6, sticky = tk.W + tk.N, padx = (5, 0), pady = (5, 5))

        self.fitButton = tk.Button(
            self.modelGroupLabel,
            text=' Fit ',
            bg = BT_COLOR,
            command = lambda: self.onFit()
        )
        self.fitButton.grid(row = 13, column = 0, columnspan = 3, sticky = tk.W + tk.E, padx = 5, pady = 5)

    def initInfoGroup(self):
        self.infoGroupLabel = tk.LabelFrame(self.leftFrame, text = '  Info  ')
        self.infoGroupLabel.pack(padx = 5, pady = 5, side = tk.TOP, fill = tk.BOTH)
        
        AccLabelText = tk.StringVar()
        AccLabelText.set('Accuracy = ')
        AccLabel = tk.Label(self.infoGroupLabel, text = AccLabelText, textvariable = AccLabelText)
        AccLabel.grid(row = 0, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.AccValueLabelText = tk.StringVar()
        self.AccValueLabelText.set('')
        self.AccValueLabel = tk.Label(self.infoGroupLabel, text = self.AccValueLabelText, textvariable = self.AccValueLabelText)
        self.AccValueLabel.grid(row = 0, column = 1, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)

    def initGraphPanel(self):
        self.figure = Figure(figsize = (5, 5), dpi = 100, facecolor = 'gainsboro')
        self.ax = self.figure.add_subplot(111)
        self.ax = self.getAx()

        self.figureCanvas = FigureCanvasTkAgg(self.figure, self.rightFrame)
        self.figureCanvas.draw()
        self.figureCanvas.get_tk_widget().pack(side = tk.LEFT, fill = tk.BOTH,expand = True)

    def start(self):
        print(">>> INFO: Opened")
        self.window.mainloop()
        
    def onClose(self, window = None):
        print(">>> INFO: Closing ...")
        if window is None:
            self.window.quit()
        else:
            window.quit()
        print(">>> INFO: Closed")

    def onFileOpening(self):
        filetypes = (("numpy data","*.npy"),("all files","*.*"))

        filename = filedialog.askopenfilename(
            title = 'Open a file',
            initialdir = './',
            filetypes = filetypes)

        if not filename:
            return

        try:
            with open(filename, 'rb') as f:
                self.X = np.load(f).astype(np.float32)
        except:
            messagebox.showerror(title = 'Error', message = 'Bad data')
            return

        self.t = None
        try:
            with open(filename, 'rb') as f:
                np.load(f)
                self.t = np.load(f).astype(np.int32)
        except:
            pass

        if self.X.size == 0:
            messagebox.showerror(title = 'Error', message = 'Bad X data')
            return
        if self.t is not None:
            if self.t.size:
                messagebox.showerror(title = 'Error', message = 'Bad t data')
                return

        self.X = model.standardizing(self.X)


        self.N = int(self.NSpinbox.get())
        self.K = int(self.KSpinbox.get())

        self.KColors = self.AvailColors[:self.K]

        # Plot data
        ax = self.getAx()
        colors = self.t if self.t is not None else 'b'
        ax.scatter(self.X[:, 0], self.X[:, 1], c = colors, \
            cmap = matplotlib.colors.ListedColormap(self.KColors), label = 'data')
        ax.legend()
        self.figureCanvas.draw()

        if self.K > 2:
            self.modelCButtonVar.set(0)

    def onDataGeneration(self, event = None):
        if not self.NSpinbox.get().isdigit():
            print(">>> ERROR: N must be a number")
            return
        if not self.KSpinbox.get().isdigit():
            print(">>> ERROR: K must be a number")
            return

        self.N = int(self.NSpinbox.get())
        self.K = int(self.KSpinbox.get())
        trueDistr = str(self.TrueFuncCBox.get())

        self.X, self.t = model.generateBlobs(self.N, self.K, trueDistr)
        self.X = model.standardizing(self.X)
        # self.KColors = np.random.uniform(0.3, 0.8, size = (self.K, 3))
        self.KColors = self.AvailColors[:self.K]

        # Plot
        ax = self.getAx()
        ax.scatter(self.X[:, 0], self.X[:, 1], c = self.t, \
            cmap = matplotlib.colors.ListedColormap(self.KColors), label = 'data')
        ax.legend()
        self.figureCanvas.draw()

        if self.K > 2:
            self.modelCButtonVar.set(0)

    def onFit(self):
        X = self.X
        t = self.t
        if X is None:
            print(">>> ERROR: X must be set")
            return
        if X.size == 0:
            print(">>> ERROR: X size must be non zero")
            return

        epochs = int(self.EpochsSpinbox.get())
        clusters = int(self.ClustersSpinbox.get())

        auxParams = \
        {
            'doDrawDetails' : bool(self.drawDetailsCButtonVar.get()),
            'doDrawDesRegs' : bool(self.drawDesRegsButtonVar.get())
        }

        mtdInt = self.modelCButtonVar.get()
        if mtdInt == 0:
            self.classifier = model.KMeansClassificator()
        elif mtdInt == 1:
            self.classifier = model.MOGClassificator()
        else:
            print(">>> ERROR: Undefined classification method")
            return

        Means_est = self.classifier.fit(X, clusters, epochs)
        t_est = self.classifier.predict(X)

        # Plot
        ax = self.getAx()
        
        # Plot data
        colors = self.t if self.t is not None else t_est
        ax.scatter(self.X[:, 0], self.X[:, 1], c = colors, \
            cmap = matplotlib.colors.ListedColormap(self.KColors), label = 'data')

        Npoints = 100
        x1_min, x1_max = ax.get_xlim()
        x2_min, x2_max = ax.get_ylim()
        xx1 = np.linspace(x1_min, x1_max, num = Npoints)
        xx2 = np.linspace(x2_min, x2_max, num = Npoints)

        # Plot decision regions
        if auxParams['doDrawDesRegs']:

            if mtdInt == 0:
                xx, yy = np.meshgrid(xx1, xx2)
                Z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, cmap = matplotlib.colors.ListedColormap(self.KColors), alpha = 0.15)

            elif mtdInt == 1:
                xx, yy = np.meshgrid(xx1, xx2)
                Z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel()])
                Z_probs = self.classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                for k in range(clusters):
                    k_probs = Z_probs[:, k]
                    k_probs = k_probs.reshape(xx.shape)
                    k_probs = np.ma.masked_where(Z != k, k_probs)
                    ax.contourf(xx, yy, k_probs, 20, cmap = self.AvailColorMaps[k], alpha = 0.15)

        # Plot details
        if auxParams['doDrawDetails']:
            
            if mtdInt == 1:
                covars = self.classifier.Covars()
                R = self.classifier.R()
                for k in range(clusters):
                    ids = np.argwhere(np.argmax(R, axis = 1) == k)[:, 0]
                    pts = X[ids, :]
                    pts_probs = R[ids, k]
                    rgba = np.zeros((pts.shape[0], 4))
                    if R.shape[1] > 2:
                        rgba[:, 0] = R[ids, 2]
                    rgba[:, 1] = R[ids, 1]
                    rgba[:, 2] = R[ids, 0]
                    rgba[:, 3] = pts_probs
                    ax.scatter(pts[:, 0], pts[:, 1], color = rgba, label = "cluster {0}".format(k))

                # MoG density
                xx1 = np.linspace(x1_min, x1_max, num = 100)
                xx2 = np.linspace(x2_min, x2_max, num = 100)
                X1_grid, X2_grid = np.meshgrid(xx1, xx2)
                for ki in range(clusters):
                    pdf = np.zeros(X1_grid.shape)
                    for i in range(X1_grid.shape[0]):
                        for j in range(X1_grid.shape[1]):
                            px = np.array([X1_grid[i, j], X2_grid[i, j]])
                            pdf[i, j] = model.multiGaussianPDF(px, Means_est[ki], covars[ki])
                    ax.contour(X1_grid, X2_grid, pdf, cmap = self.AvailColorMaps[ki])

        
        # Plot data once more
        # colors = self.t if self.t is not None else t_est
        # ax.scatter(self.X[:, 0], self.X[:, 1], c = colors, \
        #     cmap = matplotlib.colors.ListedColormap(self.KColors), label = 'data')

        # Plot means
        ax.scatter(Means_est[:, 0], Means_est[:, 1], c = 'w', marker = 'x', s = 140, linewidths = 6)
        ax.scatter(Means_est[:, 0], Means_est[:, 1], c = 'r', marker = 'x', s = 100, linewidths = 2)

        ax.set_xlim([x1_min, x1_max])
        ax.set_ylim([x2_min, x2_max])
        ax.legend()
        self.figureCanvas.draw()

        if t is not None:
            acc = self.classifier.evaluate(t)
            self.AccValueLabelText.set("{:.2f}".format(acc) + '%')
        else:
            self.AccValueLabelText.set("no true labels")

    def getAx(self):
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_title('Mixture models')
        self.ax.set_xlabel('$x1$')
        self.ax.set_ylabel('$x2$', rotation = 0)
        return self.ax


if __name__ == "__main__":
    app = App().start()