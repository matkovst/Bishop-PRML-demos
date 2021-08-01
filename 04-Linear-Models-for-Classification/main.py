import numpy as np
from scipy.stats import multivariate_normal

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

WINDOW_TITLE = 'Linear classification tool'
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
        self.t = np.array([], dtype = np.float32) # <- targer variable
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

        # MLabelText = tk.StringVar()
        # MLabelText.set('Order = ')
        # MSpinboxVar = tk.IntVar()
        # MSpinboxVar.set(6)
        # MLabel = tk.Label(self.modelGroupLabel, text = MLabelText, textvariable = MLabelText)
        # MLabel.grid(row = 0, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        # self.MSpinbox = tk.Spinbox(self.modelGroupLabel, width = 3, from_ = 1, to = 50, textvariable = MSpinboxVar)
        # self.MSpinbox.grid(row = 0, column = 1, columnspan = 2, sticky = tk.W + tk.E, padx = 5, pady = 5)
        
        # LmbdLabelText = tk.StringVar()
        # LmbdLabelText.set('λ = ')
        # LmbdSpinboxVar = tk.DoubleVar()
        # LmbdSpinboxVar.set(0.00001)
        # LmbdLabel = tk.Label(self.modelGroupLabel, text = LmbdLabelText, textvariable = LmbdLabelText)
        # LmbdLabel.grid(row = 0, column = 3, sticky = tk.W + tk.N, padx = 5, pady = 5)
        # self.LmbdSpinbox = tk.Spinbox(self.modelGroupLabel, width = 7, from_ = 0, to = 50, increment = 0.001, format = "%.5f", textvariable = LmbdSpinboxVar)
        # self.LmbdSpinbox.grid(row = 0, column = 4, sticky = tk.W + tk.E, padx = 5, pady = 5)

        self.modelCButtonVar = tk.IntVar()
        self.modelCButtonVar.set(0)
        self.lsRButton = tk.Radiobutton(self.modelGroupLabel, text = 'Least-squares', variable = self.modelCButtonVar, value = 0)
        self.lsRButton.grid(row = 2, column = 0, columnspan = 2, sticky = tk.W + tk.N, padx = (5, 0), pady = 5)
        self.fisherRButton = tk.Radiobutton(self.modelGroupLabel, text = 'Fisher', variable = self.modelCButtonVar, value = 1)
        self.fisherRButton.grid(row = 2, column = 2, columnspan = 2, sticky = tk.W + tk.N, padx = (5, 0), pady = 5)
        self.perceptronRButton = tk.Radiobutton(self.modelGroupLabel, text = 'Perceptron', variable = self.modelCButtonVar, value = 2)
        self.perceptronRButton.grid(row = 2, column = 4, columnspan = 2, sticky = tk.W + tk.N, padx = (5, 0), pady = 5)
        self.genGaussRButton = tk.Radiobutton(self.modelGroupLabel, text = 'Generative Gaussian', variable = self.modelCButtonVar, value = 3)
        self.genGaussRButton.grid(row = 3, column = 0, columnspan = 6, sticky = tk.W + tk.N, padx = (5, 0), pady = 5)
        
        separator = ttk.Separator(self.modelGroupLabel, orient = 'horizontal')
        separator.grid(row = 4, column = 0, columnspan = 6, sticky = tk.W + tk.E, padx = 5, pady = (5, 5))

        EpochsLabelText = tk.StringVar()
        EpochsLabelText.set('Epochs = ')
        EpochsSpinboxVar = tk.IntVar()
        EpochsSpinboxVar.set(10000)
        EpochsLabel = tk.Label(self.modelGroupLabel, text = EpochsLabelText, textvariable = EpochsLabelText)
        EpochsLabel.grid(row = 5, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.EpochsSpinbox = tk.Spinbox(self.modelGroupLabel, width = 5, from_ = 1, to = 99999, textvariable = EpochsSpinboxVar)
        self.EpochsSpinbox.grid(row = 5, column = 1, columnspan = 3, sticky = tk.W + tk.E, padx = 5, pady = 5)

        self.logRegRButton = tk.Radiobutton(self.modelGroupLabel, text = 'Logistic regression', variable = self.modelCButtonVar, value = 4)
        self.logRegRButton.grid(row = 6, column = 0, columnspan = 6, sticky = tk.W + tk.N, padx = (5, 0), pady = 5)
        self.logRegIRLSRButton = tk.Radiobutton(self.modelGroupLabel, text = 'Logistic regression (IRLS)', variable = self.modelCButtonVar, value = 5)
        self.logRegIRLSRButton.grid(row = 7, column = 0, columnspan = 6, sticky = tk.W + tk.N, padx = (5, 0), pady = 5)
        
        separator = ttk.Separator(self.modelGroupLabel, orient = 'horizontal')
        separator.grid(row = 8, column = 0, columnspan = 6, sticky = tk.W + tk.E, padx = 5, pady = (5, 15))

        self.drawDiscrCButtonVar = tk.IntVar()
        self.drawDiscrCButtonVar.set(0)
        self.drawDiscrCButton = tk.Checkbutton(self.modelGroupLabel, text = 'Draw Discriminant', onvalue = 1, offvalue = 0, variable = self.drawDiscrCButtonVar)
        self.drawDiscrCButton.grid(row = 10, column = 0, columnspan = 6, sticky = tk.W + tk.N, padx = (5, 0), pady = (15, 5))

        self.drawDetailsCButtonVar = tk.IntVar()
        self.drawDetailsCButtonVar.set(0)
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

        # separator = ttk.Separator(self.modelGroupLabel, orient = 'horizontal')
        # separator.grid(row = 8, column = 0, columnspan = 5, sticky = tk.W + tk.E, padx = 5, pady = (5, 15))

        # BatchSizeLabelText = tk.StringVar()
        # BatchSizeLabelText.set('Batch = ')
        # BatchSizeSpinboxVar = tk.IntVar()
        # BatchSizeSpinboxVar.set(1)
        # BatchSizeLabel = tk.Label(self.modelGroupLabel, text = BatchSizeLabelText, textvariable = BatchSizeLabelText)
        # BatchSizeLabel.grid(row = 9, column = 0, columnspan = 2, sticky = tk.W + tk.N, padx = 5, pady = 5)
        # self.BatchSizeSpinbox = tk.Spinbox(self.modelGroupLabel, width = 3, from_ = 1, to = 50, textvariable = BatchSizeSpinboxVar)
        # self.BatchSizeSpinbox.grid(row = 9, column = 2, columnspan = 1, sticky = tk.W + tk.E, padx = 5, pady = 5)

        # self.animateCButtonVar = tk.IntVar()
        # self.animateCButtonVar.set(0)
        # self.animateCButton = tk.Checkbutton(self.modelGroupLabel, text = 'Animate', onvalue = 1, offvalue = 0, variable = self.animateCButtonVar)
        # self.animateCButton.grid(row = 9, column = 3, columnspan = 2, sticky = tk.E + tk.N, padx = (5, 0), pady = (5, 5))
        # animateSpinboxVar = tk.IntVar()
        # animateSpinboxVar.set(500)
        # self.animateSpinbox = tk.Spinbox(self.modelGroupLabel, width = 4, from_ = 0, to = 9999, textvariable = animateSpinboxVar)
        # self.animateSpinbox.grid(row = 9, column = 5, columnspan = 1, sticky = tk.W + tk.E, padx = 5, pady = 5)
        # animateLabelText = tk.StringVar()
        # animateLabelText.set('ms')
        # animateLabel = tk.Label(self.modelGroupLabel, text = animateLabelText, textvariable = animateLabelText)
        # animateLabel.grid(row = 9, column = 6, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)

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
        
        # AlphaLabelText = tk.StringVar()
        # AlphaLabelText.set('Estimated α = ')
        # AlphaLabel = tk.Label(self.infoGroupLabel, text = AlphaLabelText, textvariable = AlphaLabelText)
        # AlphaLabel.grid(row = 0, column = 2, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        # self.AlphaValueLabelText = tk.StringVar()
        # self.AlphaValueLabelText.set('')
        # self.AlphaValueLabel = tk.Label(self.infoGroupLabel, text = self.AlphaValueLabelText, textvariable = self.AlphaValueLabelText)
        # self.AlphaValueLabel.grid(row = 0, column = 3, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        
        # MFavourLabelText = tk.StringVar()
        # MFavourLabelText.set('M favour = ')
        # MFavourLabel = tk.Label(self.infoGroupLabel, text = MFavourLabelText, textvariable = MFavourLabelText)
        # MFavourLabel.grid(row = 1, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        # self.MFavourValueLabelText = tk.StringVar()
        # self.MFavourValueLabelText.set('')
        # self.MFavourValueLabel = tk.Label(self.infoGroupLabel, text = self.MFavourValueLabelText, textvariable = self.MFavourValueLabelText)
        # self.MFavourValueLabel.grid(row = 1, column = 1, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        
        # BetaLabelText = tk.StringVar()
        # BetaLabelText.set('Estimated β = ')
        # BetaLabel = tk.Label(self.infoGroupLabel, text = BetaLabelText, textvariable = BetaLabelText)
        # BetaLabel.grid(row = 1, column = 2, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        # self.BetaValueLabelText = tk.StringVar()
        # self.BetaValueLabelText.set('')
        # self.BetaValueLabel = tk.Label(self.infoGroupLabel, text = self.BetaValueLabelText, textvariable = self.BetaValueLabelText)
        # self.BetaValueLabel.grid(row = 1, column = 3, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)

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
                self.X = np.load(f)
                self.t = np.load(f).astype(np.int32)
        except:
            messagebox.showerror(title = 'Error', message = 'Bad data')
            return

        if self.X.size * self.t.size == 0:
            messagebox.showerror(title = 'Error', message = 'Bad data')
            return

        self.N = int(self.NSpinbox.get())
        self.K = int(self.KSpinbox.get())

        self.KColors = self.AvailColors[:self.K]

        # Plot data
        ax = self.getAx()
        ax.scatter(self.X[:, 0], self.X[:, 1], c = self.t, \
            cmap = matplotlib.colors.ListedColormap(self.KColors), label = 'data')
        ax.legend()
        self.figureCanvas.draw()

        if self.K > 2:
            self.modelCButtonVar.set(0)
            self.fisherRButton.configure(state = tk.DISABLED)
            self.perceptronRButton.configure(state = tk.DISABLED)
            self.logRegIRLSRButton.configure(state = tk.DISABLED)
        elif self.K == 2:
            self.fisherRButton.configure(state = tk.NORMAL)
            self.perceptronRButton.configure(state = tk.NORMAL)
            self.logRegIRLSRButton.configure(state = tk.NORMAL)

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
            self.fisherRButton.configure(state = tk.DISABLED)
            self.perceptronRButton.configure(state = tk.DISABLED)
            self.logRegIRLSRButton.configure(state = tk.DISABLED)
        elif self.K == 2:
            self.fisherRButton.configure(state = tk.NORMAL)
            self.perceptronRButton.configure(state = tk.NORMAL)
            self.logRegIRLSRButton.configure(state = tk.NORMAL)

    def onFit(self):
        X = self.X
        t = self.t
        if X is None or t is None:
            print(">>> ERROR: X and t must be set")
            return
        if X.shape[0] != t.size:
            print(">>> ERROR: X and t sizes must be equal")
            return
        if X.size * t.size == 0:
            print(">>> ERROR: X and t size must be non zero")
            return

        epochs = int(self.EpochsSpinbox.get())

        auxParams = \
        {
            'doDrawDiscr' : bool(self.drawDiscrCButtonVar.get()),
            'doDrawDetails' : bool(self.drawDetailsCButtonVar.get()),
            'doDrawDesRegs' : bool(self.drawDesRegsButtonVar.get())
        }

        mtdInt = self.modelCButtonVar.get()
        if mtdInt == 0:
            self.classifier = model.LeastSquaresClassificator(X, t)
        elif mtdInt == 1:
            self.classifier = model.FisherClassificator(X, t)
        elif mtdInt == 2:
            self.classifier = model.PerceptronClassificator(X, t)
        elif mtdInt == 3:
            self.classifier = model.GenerativeGaussianClassificator(X, t)
        elif mtdInt == 4:
            self.classifier = model.LogisticRegression(X, t, epochs = epochs, solver = 'ls', verbose = True)
        elif mtdInt == 5:
            self.classifier = model.LogisticRegression(X, t, epochs = epochs, solver = 'irls', verbose = True)
        else:
            print(">>> ERROR: Undefined classification method")
            return

        W_est = self.classifier.fit()
        if len(W_est.shape) == 1:
            W_est = np.expand_dims(W_est, axis = 1)

        # Plot
        ax = self.getAx()
        
        # Plot data
        ax.scatter(self.X[:, 0], self.X[:, 1], c = self.t, \
            cmap = matplotlib.colors.ListedColormap(self.KColors))

        Npoints = 100
        x1_min, x1_max = ax.get_xlim()
        x2_min, x2_max = ax.get_ylim()
        xx1 = np.linspace(x1_min, x1_max, num = Npoints)
        xx2 = np.linspace(x2_min, x2_max, num = Npoints)

        # Plot decision regions
        if auxParams['doDrawDesRegs']:

            if mtdInt >= 0 and mtdInt <= 2:
                xx, yy = np.meshgrid(xx1, xx2)
                Z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, cmap = matplotlib.colors.ListedColormap(self.KColors), alpha = 0.15)

            elif mtdInt == 3:
                xx, yy = np.meshgrid(xx1, xx2)
                Z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel()])
                Z_probs = self.classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                # ax.contourf(xx, yy, Z, cmap = matplotlib.colors.ListedColormap(self.KColors), alpha = 0.25)
                for k in range(self.K):
                    k_probs = Z_probs[:, k]
                    k_probs = k_probs.reshape(xx.shape)
                    k_probs = np.ma.masked_where(Z != k, k_probs)
                    # norm = plt.cm.colors.Normalize(vmin = np.min(k_probs), vmax = np.max(k_probs))
                    ax.contourf(xx, yy, k_probs, 400, cmap = self.AvailColorMaps[k], alpha = 0.15)

            elif mtdInt == 4 or mtdInt == 5:
                xx, yy = np.meshgrid(xx1, xx2)
                Z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel()])
                Z_probs = self.classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                # ax.contourf(xx, yy, Z, cmap = matplotlib.colors.ListedColormap(self.KColors), alpha = 0.25)
                if self.K == 2:
                    k_probs = 1 - Z_probs
                    k_probs = k_probs.reshape(xx.shape)
                    k_probs = np.ma.masked_where(Z != 0, k_probs)
                    # norm = plt.cm.colors.Normalize(vmin = np.min(k_probs), vmax = np.max(k_probs))
                    ax.contourf(xx, yy, k_probs, 10, cmap = self.AvailColorMaps[0], alpha = 0.15)
                    k_probs = Z_probs
                    k_probs = k_probs.reshape(xx.shape)
                    k_probs = np.ma.masked_where(Z != 1, k_probs)
                    # norm = plt.cm.colors.Normalize(vmin = np.min(k_probs), vmax = np.max(k_probs))
                    ax.contourf(xx, yy, k_probs, 10, cmap = self.AvailColorMaps[1], alpha = 0.15)
                else:
                    for k in range(self.K):
                        k_probs = Z_probs[:, k]
                        k_probs = k_probs.reshape(xx.shape)
                        k_probs = np.ma.masked_where(Z != k, k_probs)
                        # norm = plt.cm.colors.Normalize(vmin = np.min(k_probs), vmax = np.max(k_probs))
                        ax.contourf(xx, yy, k_probs, 10, cmap = self.AvailColorMaps[k], alpha = 0.15)

        # Plot details
        if auxParams['doDrawDetails']:

            if mtdInt == 0: # <- Plot discriminant functions
                for i in range(self.K):
                    x2 = -(W_est[0, i] + W_est[1, i] * xx1) / W_est[2, i]
                    ax.plot(xx1, x2, '--k', label = 'discr func {0}'.format(i + 1))

            elif mtdInt == 3: # <- Plot Gaussian densities
                X1_grid, X2_grid = np.meshgrid(xx1, xx2)
                for ki in range(self.K):
                    distr = multivariate_normal(mean = self.classifier.Means_ml[:, ki], cov = self.classifier.Covars_ml[:, :, ki])
                    pdf = np.zeros(X1_grid.shape)
                    for i in range(X1_grid.shape[0]):
                        for j in range(X1_grid.shape[1]):
                            pdf[i, j] = distr.pdf([X1_grid[i, j], X2_grid[i, j]])
                    ax.contour(X1_grid, X2_grid, pdf, cmap = self.AvailColorMaps[ki])

        # Plot hyperplanes
        if auxParams['doDrawDiscr']:
            if W_est.shape[1] > 1:
                if mtdInt == 0:
                    for i in range(self.K):
                        for j in range(i, self.K):
                            if i == j:
                                continue
                            W_hyperplane = W_est[:, i] - W_est[:, j] #  The decision boundary between classes Ci and Cj occurs where yi(x) = yj(x)
                            x2 = -(W_hyperplane[0] + W_hyperplane[1] * xx1) / W_hyperplane[2]
                            color = (0.5 * self.KColors[i][0] + 0.5 * self.KColors[j][0], \
                                0.5 * self.KColors[i][1] + 0.5 * self.KColors[j][1], \
                                0.5 * self.KColors[i][2] + 0.5 * self.KColors[j][2])
                            ax.plot(xx1, x2, color = color, \
                                linestyle = '-', label = 'hyperplane {0}-{1}'.format(i + 1, j + 1))

                elif mtdInt == 3 or mtdInt == 4:
                    for i in range(self.K):
                        for j in range(i, self.K):
                            if i == j:
                                continue
                            W_hyperplane = W_est[:, i] - W_est[:, j] #  The decision boundary between classes Ci and Cj occurs where p(Ci | x) = p(Cj | x)
                            x2 = -(W_hyperplane[0] + W_hyperplane[1] * xx1) / W_hyperplane[2]
                            color = (0.5 * self.KColors[i][0] + 0.5 * self.KColors[j][0], \
                                0.5 * self.KColors[i][1] + 0.5 * self.KColors[j][1], \
                                0.5 * self.KColors[i][2] + 0.5 * self.KColors[j][2])
                            ax.plot(xx1, x2, color = color, \
                                linestyle = '-', label = 'hyperplane {0}-{1}'.format(i + 1, j + 1))

            else:
                x2 = -(W_est[0] + W_est[1] * xx1) / W_est[2]
                color = (0.5 * self.KColors[0][0] + 0.5 * self.KColors[1][0], \
                    0.5 * self.KColors[0][1] + 0.5 * self.KColors[1][1], \
                    0.5 * self.KColors[0][2] + 0.5 * self.KColors[1][2])
                ax.plot(xx1, x2, color = color, linestyle = '-', label = 'hyperplane')

        
        # Plot data once more
        ax.scatter(self.X[:, 0], self.X[:, 1], c = self.t, \
            cmap = matplotlib.colors.ListedColormap(self.KColors), label = 'data')

        ax.set_xlim([x1_min, x1_max])
        ax.set_ylim([x2_min, x2_max])
        ax.legend()
        self.figureCanvas.draw()

        acc = self.classifier.evaluate()
        self.AccValueLabelText.set("{:.2f}".format(acc) + '%')

    def getAx(self):
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_title('Linear classification')
        self.ax.set_xlabel('$x1$')
        self.ax.set_ylabel('$x2$', rotation = 0)
        return self.ax


if __name__ == "__main__":
    app = App().start()