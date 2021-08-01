import sys
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
from matplotlib import animation
plt.style.use('seaborn-darkgrid')
import model_specific as model

WINDOW_TITLE = 'Linear regression tool'
BG_COLOR = "#ffffff" #"#404040"
BT_COLOR = "#ffffff"
WIDTH = 1280
HEIGHT = 720

def defocus(event):
    event.widget.master.focus_set()

class App:
    def __init__(self):

        self.x = np.array([], dtype = np.float32) # <- observable variable
        self.t = np.array([], dtype = np.float32) # <- targer variable
        self.axMin = 0
        self.axMax = 1
        self.xLinspace = np.array([], dtype = np.float32)
        self.gtLinspace = np.array([], dtype = np.float32)
        self.truePrecision = -1

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

        FromLabelText = tk.StringVar()
        FromLabelText.set('From = ')
        FromSpinboxVar = tk.DoubleVar()
        FromSpinboxVar.set(0)
        FromLabel = tk.Label(self.dataGroupLabel, text = FromLabelText, textvariable = FromLabelText)
        FromLabel.grid(row = 1, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.FromSpinbox = tk.Spinbox(self.dataGroupLabel, width = 5, from_ = -999, to = 999, textvariable = FromSpinboxVar)
        self.FromSpinbox.grid(row = 1, column = 1, columnspan = 2, sticky = tk.W + tk.E, padx = 5, pady = 5)

        ToLabelText = tk.StringVar()
        ToLabelText.set('To = ')
        ToSpinboxVar = tk.DoubleVar()
        ToSpinboxVar.set(1)
        ToLabel = tk.Label(self.dataGroupLabel, text = ToLabelText, textvariable = ToLabelText)
        ToLabel.grid(row = 2, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.ToSpinbox = tk.Spinbox(self.dataGroupLabel, width = 5, from_ = -998, to = 999, textvariable = ToSpinboxVar)
        self.ToSpinbox.grid(row = 2, column = 1, columnspan = 2, sticky = tk.W + tk.E, padx = 5, pady = 5)
        
        BiasLabelText = tk.StringVar()
        BiasLabelText.set('Bias = ')
        BiasSpinboxVar = tk.DoubleVar()
        BiasSpinboxVar.set(0.0)
        BiasLabel = tk.Label(self.dataGroupLabel, text = BiasLabelText, textvariable = BiasLabelText)
        BiasLabel.grid(row = 0, column = 3, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.BiasSpinbox = tk.Spinbox(self.dataGroupLabel, width = 5, from_ = 0, to = 100, increment = 0.1, format = "%.2f", textvariable = BiasSpinboxVar)
        self.BiasSpinbox.grid(row = 0, column = 4, columnspan = 2, sticky = tk.W + tk.E, padx = 5, pady = 5)
        
        NoiseLabelText = tk.StringVar()
        NoiseLabelText.set('Noise = ')
        NoiseSpinboxVar = tk.DoubleVar()
        NoiseSpinboxVar.set(0.3)
        NoiseLabel = tk.Label(self.dataGroupLabel, text = NoiseLabelText, textvariable = NoiseLabelText)
        NoiseLabel.grid(row = 1, column = 3, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.NoiseSpinbox = tk.Spinbox(self.dataGroupLabel, width = 5, from_ = 0.0000001, to = 100, increment = 0.1, format = "%.2f", textvariable = NoiseSpinboxVar)
        self.NoiseSpinbox.grid(row = 1, column = 4, columnspan = 2, sticky = tk.W + tk.E, padx = 5, pady = 5)

        FuncLabelText = tk.StringVar()
        FuncLabelText.set('f(x) = ')
        FuncLabel = tk.Label(self.dataGroupLabel, text = FuncLabelText, textvariable = FuncLabelText)
        FuncLabel.grid(row = 2, column = 3, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        funcs = [ "sin(2pix)", "exp(x)" ]
        self.TrueFuncCBox = ttk.Combobox(self.dataGroupLabel, width = 9, values = funcs, state = "readonly")
        self.TrueFuncCBox.bind("<FocusIn>", defocus) # <- in order to prevent box highlighting
        self.TrueFuncCBox.grid(row = 2, column = 4, columnspan = 2, sticky = tk.W + tk.E, padx = 5, pady = 5)
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

        MLabelText = tk.StringVar()
        MLabelText.set('Order = ')
        MSpinboxVar = tk.IntVar()
        MSpinboxVar.set(6)
        MLabel = tk.Label(self.modelGroupLabel, text = MLabelText, textvariable = MLabelText)
        MLabel.grid(row = 0, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.MSpinbox = tk.Spinbox(self.modelGroupLabel, width = 3, from_ = 1, to = 50, textvariable = MSpinboxVar)
        self.MSpinbox.grid(row = 0, column = 1, columnspan = 2, sticky = tk.W + tk.E, padx = 5, pady = 5)
        
        LmbdLabelText = tk.StringVar()
        LmbdLabelText.set('λ = ')
        LmbdSpinboxVar = tk.DoubleVar()
        LmbdSpinboxVar.set(0.00001)
        LmbdLabel = tk.Label(self.modelGroupLabel, text = LmbdLabelText, textvariable = LmbdLabelText)
        LmbdLabel.grid(row = 0, column = 3, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.LmbdSpinbox = tk.Spinbox(self.modelGroupLabel, width = 7, from_ = 0, to = 50, increment = 0.001, format = "%.5f", textvariable = LmbdSpinboxVar)
        self.LmbdSpinbox.grid(row = 0, column = 4, sticky = tk.W + tk.E, padx = 5, pady = 5)

        self.modelRButtonVar = tk.IntVar()
        self.modelRButtonVar.set(0)
        self.powerRButton = tk.Radiobutton(self.modelGroupLabel, text = 'Power', variable = self.modelRButtonVar, value = 0)
        self.powerRButton.grid(row = 2, column = 0, columnspan = 2, sticky = tk.W + tk.N, padx = (5, 0), pady = 5)
        self.gaussianRButton = tk.Radiobutton(self.modelGroupLabel, text = 'Gaussian', variable = self.modelRButtonVar, value = 1)
        self.gaussianRButton.grid(row = 2, column = 2, columnspan = 2, sticky = tk.W + tk.N, padx = (5, 0), pady = 5)
        self.sigmoidRButton = tk.Radiobutton(self.modelGroupLabel, text = 'Sigmoid', variable = self.modelRButtonVar, value = 2)
        self.sigmoidRButton.grid(row = 2, column = 4, columnspan = 2, sticky = tk.W + tk.N, padx = (5, 0), pady = 5)

        self.sampleWCButtonVar = tk.IntVar()
        self.sampleWCButtonVar.set(0)
        self.sampleWCButton = tk.Checkbutton(self.modelGroupLabel, text = 'Draw samples from p(w|α)', onvalue = 1, offvalue = 0, variable = self.sampleWCButtonVar)
        self.sampleWCButton.grid(row = 5, column = 0, columnspan = 5, sticky = tk.W + tk.N, padx = (5, 0), pady = (15, 5))

        self.drawPredStdevCButtonVar = tk.IntVar()
        self.drawPredStdevCButtonVar.set(1)
        self.drawPredStdevCButton = tk.Checkbutton(self.modelGroupLabel, text = 'Draw uncertainities', onvalue = 1, offvalue = 0, variable = self.drawPredStdevCButtonVar)
        self.drawPredStdevCButton.grid(row = 6, column = 0, columnspan = 5, sticky = tk.W + tk.N, padx = (5, 0), pady = (5, 5))

        self.fitButton = tk.Button(
            self.modelGroupLabel,
            text=' Fit entire dataset ',
            bg = BT_COLOR,
            command = lambda: self.onFit('entire')
        )
        self.fitButton.grid(row = 7, column = 0, columnspan = 3, sticky = tk.W + tk.E, padx = 5, pady = 5)

        separator = ttk.Separator(self.modelGroupLabel, orient = 'horizontal')
        separator.grid(row = 8, column = 0, columnspan = 5, sticky = tk.W + tk.E, padx = 5, pady = (5, 15))

        BatchSizeLabelText = tk.StringVar()
        BatchSizeLabelText.set('Batch = ')
        BatchSizeSpinboxVar = tk.IntVar()
        BatchSizeSpinboxVar.set(1)
        BatchSizeLabel = tk.Label(self.modelGroupLabel, text = BatchSizeLabelText, textvariable = BatchSizeLabelText)
        BatchSizeLabel.grid(row = 9, column = 0, columnspan = 2, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.BatchSizeSpinbox = tk.Spinbox(self.modelGroupLabel, width = 3, from_ = 1, to = 50, textvariable = BatchSizeSpinboxVar)
        self.BatchSizeSpinbox.grid(row = 9, column = 2, columnspan = 1, sticky = tk.W + tk.E, padx = 5, pady = 5)

        self.animateCButtonVar = tk.IntVar()
        self.animateCButtonVar.set(0)
        self.animateCButton = tk.Checkbutton(self.modelGroupLabel, text = 'Animate', onvalue = 1, offvalue = 0, variable = self.animateCButtonVar)
        self.animateCButton.grid(row = 9, column = 3, columnspan = 2, sticky = tk.E + tk.N, padx = (5, 0), pady = (5, 5))
        animateSpinboxVar = tk.IntVar()
        animateSpinboxVar.set(500)
        self.animateSpinbox = tk.Spinbox(self.modelGroupLabel, width = 4, from_ = 0, to = 9999, textvariable = animateSpinboxVar)
        self.animateSpinbox.grid(row = 9, column = 5, columnspan = 1, sticky = tk.W + tk.E, padx = 5, pady = 5)
        animateLabelText = tk.StringVar()
        animateLabelText.set('ms')
        animateLabel = tk.Label(self.modelGroupLabel, text = animateLabelText, textvariable = animateLabelText)
        animateLabel.grid(row = 9, column = 6, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)

        self.fitSequentiallyButton = tk.Button(
            self.modelGroupLabel,
            text=' Fit sequentially ',
            bg = BT_COLOR,
            command = lambda: self.onFit('sequential')
        )
        self.fitSequentiallyButton.grid(row = 10, column = 0, columnspan = 3, sticky = tk.W + tk.E, padx = 5, pady = 5)

    def initInfoGroup(self):
        self.infoGroupLabel = tk.LabelFrame(self.leftFrame, text = '  Info  ')
        self.infoGroupLabel.pack(padx = 5, pady = 5, side = tk.TOP, fill = tk.BOTH)
        
        ErrorLabelText = tk.StringVar()
        ErrorLabelText.set('Error (RMS) = ')
        ErrorLabel = tk.Label(self.infoGroupLabel, text = ErrorLabelText, textvariable = ErrorLabelText)
        ErrorLabel.grid(row = 0, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.ErrorValueLabelText = tk.StringVar()
        self.ErrorValueLabelText.set('')
        self.ErrorValueLabel = tk.Label(self.infoGroupLabel, text = self.ErrorValueLabelText, textvariable = self.ErrorValueLabelText)
        self.ErrorValueLabel.grid(row = 0, column = 1, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        
        AlphaLabelText = tk.StringVar()
        AlphaLabelText.set('Estimated α = ')
        AlphaLabel = tk.Label(self.infoGroupLabel, text = AlphaLabelText, textvariable = AlphaLabelText)
        AlphaLabel.grid(row = 0, column = 2, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.AlphaValueLabelText = tk.StringVar()
        self.AlphaValueLabelText.set('')
        self.AlphaValueLabel = tk.Label(self.infoGroupLabel, text = self.AlphaValueLabelText, textvariable = self.AlphaValueLabelText)
        self.AlphaValueLabel.grid(row = 0, column = 3, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        
        MFavourLabelText = tk.StringVar()
        MFavourLabelText.set('M favour = ')
        MFavourLabel = tk.Label(self.infoGroupLabel, text = MFavourLabelText, textvariable = MFavourLabelText)
        MFavourLabel.grid(row = 1, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.MFavourValueLabelText = tk.StringVar()
        self.MFavourValueLabelText.set('')
        self.MFavourValueLabel = tk.Label(self.infoGroupLabel, text = self.MFavourValueLabelText, textvariable = self.MFavourValueLabelText)
        self.MFavourValueLabel.grid(row = 1, column = 1, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        
        BetaLabelText = tk.StringVar()
        BetaLabelText.set('Estimated β = ')
        BetaLabel = tk.Label(self.infoGroupLabel, text = BetaLabelText, textvariable = BetaLabelText)
        BetaLabel.grid(row = 1, column = 2, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.BetaValueLabelText = tk.StringVar()
        self.BetaValueLabelText.set('')
        self.BetaValueLabel = tk.Label(self.infoGroupLabel, text = self.BetaValueLabelText, textvariable = self.BetaValueLabelText)
        self.BetaValueLabel.grid(row = 1, column = 3, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)

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
                self.x = np.load(f)
                self.t = np.load(f)
        except:
            messagebox.showerror(title = 'Error', message = 'Bad data')
            return

        if self.x.size * self.t.size == 0:
            messagebox.showerror(title = 'Error', message = 'Bad data')
            return

        self.gtLinspace = np.array([])
        self.truePrecision = -1
        self.axMin = np.min(self.x)
        self.axMax = np.max(self.x)

        # Plot data
        ax = self.getAx()
        ax.plot(self.x, self.t, '.', label = 'data')
        ax.legend()
        self.figureCanvas.draw()

    def onDataGeneration(self, event = None):
        if not self.NSpinbox.get().isdigit():
            print(">>> ERROR: N must be number")
            return

        N = int(self.NSpinbox.get())
        self.axMin = float(self.FromSpinbox.get())
        self.axMax = float(self.ToSpinbox.get())
        bias = float(self.BiasSpinbox.get())
        noise_level = float(self.NoiseSpinbox.get())
        TrueFunction = str(self.TrueFuncCBox.get())

        self.x, self.t = model.generateData(N, self.axMin, self.axMax, TrueFunction, bias, noise_level)
        if noise_level > 0:
            self.truePrecision = 1 / noise_level
        else:
            self.truePrecision = 10000000
        
        self.xLinspace = np.linspace(self.axMin, self.axMax, num = 100)
        self.gtLinspace = model.groundTruthProcess(self.xLinspace, bias, TrueFunction)

        # Plot
        ax = self.getAx()
        ax.plot(self.x, self.t, '.', label = 'data')
        ax.plot(self.xLinspace, self.gtLinspace, 'g--', label = 'ground-truth function')
        ax.legend()
        self.figureCanvas.draw()

    def onFit(self, method = 'entire'):
        x = self.x
        t = self.t
        if x is None or t is None:
            print(">>> ERROR: x and t must be set")
            return
        if x.size != t.size:
            print(">>> ERROR: x and t sizes must be equal")
            return
        if x.size * t.size == 0:
            print(">>> ERROR: x and t size must be non zero")
            return
        if not self.MSpinbox.get().isdigit():
            print(">>> ERROR: Order must be number")
            return

        BasisName = ''
        if self.modelRButtonVar.get() == 0:
            BasisName = 'power'
        elif self.modelRButtonVar.get() == 1:
            BasisName = 'gaussian'
        elif self.modelRButtonVar.get() == 2:
            BasisName = 'sigmoid'

        M = int(self.MSpinbox.get())

        bias = float(self.BiasSpinbox.get())

        lmbd = float(self.LmbdSpinbox.get())

        auxParams = \
        {
            'doSampleFromWPosterior' : bool(self.sampleWCButtonVar.get()),
            'doDrawPredicitveStdev' : bool(self.drawPredStdevCButtonVar.get())
        }

        if method == 'entire':
            self.onFitEntireSet(x, t, BasisName, M, bias, lmbd, auxParams)
        elif method == 'sequential':
            self.onFitSequentially(x, t, BasisName, M, bias, lmbd, auxParams)

    def onFitEntireSet(self, x, t, BasisName, M, bias, lmbd, auxParams):

        w_est = np.zeros(M)
        w0_est = 0.0
        b_est_ml = 0.0
        X_train = model.makeFeatureMatrix(x, M, basis = BasisName)
        w_est, w0_est, b_est_ml = model.fitEntireSet(X_train, t, lmbd)
        Precision = b_est_ml
        if self.truePrecision > -1:
            Precision = self.truePrecision

        # Compute evidence function
        a = lmbd * Precision
        NOrders =  10
        Evidences = np.zeros(NOrders, dtype = np.float64)
        for i in range(NOrders):
            Phi = model.makeFeatureMatrix(x, i + 1, basis = BasisName)
            Evidences[i] = model.evaluateEvidence(Phi, t, a, Precision)
        M_est = np.argmax(Evidences) + 1
        # print("Evidences:", Evidences)

        # Plot
        Npoints = 100
        xp = np.linspace(self.axMin, self.axMax, num = Npoints)
        tp = model.polynom(xp, w_est, w0_est, basis = BasisName)
        gtp = model.groundTruthProcess(xp, bias)
        ax = self.getAx()
        ax.plot(x, t, '.', label = 'data')
        if self.gtLinspace.size > 0:
            ax.plot(xp, self.gtLinspace, 'g--', label = 'ground-truth function')

        if auxParams['doSampleFromWPosterior']:
            wMean, wCovar = model.wPosteriorParams(X_train, t, Precision, lmbd)
            NSamples = 10
            wx_sampled = np.random.multivariate_normal(wMean, wCovar, size = NSamples).T
            for i in range(NSamples):
                wPolynom_sampled = model.polynom(xp, wx_sampled[:, i], w0_est, basis = BasisName)
                ax.plot(xp, wPolynom_sampled, 'b--')

        # Draw points from predictive distribution
        wMean, wCovar = model.wPosteriorParams(X_train, t, Precision, lmbd)
        ptp = np.zeros(Npoints, dtype = np.float32)
        ptp_lower = np.zeros(Npoints, dtype = np.float32)
        ptp_upper = np.zeros(Npoints, dtype = np.float32)
        for i in range(Npoints):
            pMean, pSigma = model.predictiveDistrParams(xp[i], Precision, wMean, wCovar, basis = BasisName)
            ptp[i] = pMean
            ptp_lower[i] = pMean - pSigma
            ptp_upper[i] = pMean + pSigma
        ax.plot(xp, ptp, 'r-', label = 'predicted function')
        
        # Draw uncertainity borders from predictive distribution
        if auxParams['doDrawPredicitveStdev']:
            ax.fill_between(xp, ptp_lower, ptp_upper, color = 'r', alpha = 0.06)

        # ax.plot(xp, tp, 'r-', label = 'predicted function')
        # ax.fill_between(xp, (tp - 1/b_est), (tp + 1/b_est), color = 'r', alpha = 0.06)
        ax.legend()
        self.figureCanvas.draw()

        # Estimate α and β
        a0 = 0.001
        b0 = 100
        a_est, b_est = model.estimateHyperparams(X_train, t, a0, b0)

        # Fill info
        self.ErrorValueLabelText.set( "{:.3f}".format(model.RMS(x, w_est, t, basis = BasisName)) )
        self.AlphaValueLabelText.set( "{:.3f}".format(a_est) )
        self.BetaValueLabelText.set( "{:.3f}".format(b_est) )
        self.MFavourValueLabelText.set( M_est )

    def onFitSequentially(self, x, t, BasisName, M, bias, lmbd, auxParams):

        # How about a little animation?
        doAnimate = bool(self.animateCButtonVar.get())
        if doAnimate:
            N = x.shape[0]
            animationDelay = int(self.animateSpinbox.get())
            anim = animation.FuncAnimation(self.figure, self.onFitSequentiallyAnimated, \
                np.arange(1, N), fargs=(x, t, BasisName, M, bias, lmbd, auxParams), \
                interval = animationDelay, repeat = False, blit = False)
            self.figureCanvas.draw()
            return

        batchSize = int(self.BatchSizeSpinbox.get())

        w_est = np.zeros(M)
        w0_est = 0.0
        b_est_ml = 0.0
        X_train = model.makeFeatureMatrix(x, M, basis = BasisName)
        w_est, w0_est, b_est_ml = model.fitSequentially(X_train, t, lmbd, batchSize)
        Precision = b_est_ml
        if self.truePrecision > -1:
            Precision = self.truePrecision

        # Plotting
        Npoints = 100
        xp = np.linspace(self.axMin, self.axMax, num = Npoints)
        ax = self.getAx()

        # Draw ground-truth function
        ax.plot(x, t, '.', label = 'data')
        if self.gtLinspace.size > 0:
            ax.plot(xp, self.gtLinspace, 'g--', label = 'ground-truth function')

        # Draw samples from weight posterior
        if auxParams['doSampleFromWPosterior']:
            wMean, wCovar = model.wPosteriorParams(X_train, t, Precision, lmbd)
            NSamples = 10
            wx_sampled = np.random.multivariate_normal(wMean, wCovar, size = NSamples).T
            for i in range(NSamples):
                wPolynom_sampled = model.polynom(xp, wx_sampled[:, i], w0_est, basis = BasisName)
                ax.plot(xp, wPolynom_sampled, 'b--')

        # Draw points from predictive distribution
        wMean, wCovar = model.wPosteriorParams(X_train, t, Precision, lmbd)
        ptp = np.zeros(Npoints, dtype = np.float32)
        ptp_lower = np.zeros(Npoints, dtype = np.float32)
        ptp_upper = np.zeros(Npoints, dtype = np.float32)
        for i in range(Npoints):
            pMean, pSigma = model.predictiveDistrParams(xp[i], Precision, wMean, wCovar, basis = BasisName)
            ptp[i] = pMean
            ptp_lower[i] = pMean - pSigma
            ptp_upper[i] = pMean + pSigma
        ax.plot(xp, ptp, 'r-', label = 'predicted function')
        
        # Draw uncertainity borders from predictive distribution
        if auxParams['doDrawPredicitveStdev']:
            ax.fill_between(xp, ptp_lower, ptp_upper, color = 'r', alpha = 0.06)

        ax.legend()
        self.figureCanvas.draw()

        # Fill info
        self.ErrorValueLabelText.set( "{:.3f}".format(model.RMS(x, w_est, t, basis = BasisName)) )
        self.AlphaValueLabelText.set( "-" )
        self.BetaValueLabelText.set( "-" )
        self.MFavourValueLabelText.set( "-" )

    def onFitSequentiallyAnimated(self, i, x, t, BasisName, M, bias, lmbd, auxParams):

        x = x[0:i]
        t = t[0:i]

        batchSize = int(self.BatchSizeSpinbox.get())

        w_est = np.zeros(M)
        w0_est = 0.0
        b_est_ml = 0.0
        X_train = model.makeFeatureMatrix(x, M, basis = BasisName)
        w_est, w0_est, b_est_ml = model.fitSequentially(X_train, t, lmbd, batchSize)
        Precision = b_est_ml
        if self.truePrecision > -1:
            Precision = self.truePrecision

        # Plotting
        Npoints = 100
        xp = np.linspace(self.axMin, self.axMax, num = Npoints)
        ax = self.getAx()

        # Draw ground-truth function
        ax.plot(x, t, '.', label = 'data')
        if self.gtLinspace.size > 0:
            ax.plot(xp, self.gtLinspace, 'g--', label = 'ground-truth function')

        # Draw samples from weight posterior
        if auxParams['doSampleFromWPosterior']:
            wMean, wCovar = model.wPosteriorParams(X_train, t, Precision, lmbd)
            NSamples = 10
            wx_sampled = np.random.multivariate_normal(wMean, wCovar, size = NSamples).T
            for i in range(NSamples):
                wPolynom_sampled = model.polynom(xp, wx_sampled[:, i], w0_est, basis = BasisName)
                ax.plot(xp, wPolynom_sampled, 'b--')

        # Draw points from predictive distribution
        wMean, wCovar = model.wPosteriorParams(X_train, t, Precision, lmbd)
        ptp = np.zeros(Npoints, dtype = np.float32)
        ptp_lower = np.zeros(Npoints, dtype = np.float32)
        ptp_upper = np.zeros(Npoints, dtype = np.float32)
        for i in range(Npoints):
            pMean, pSigma = model.predictiveDistrParams(xp[i], Precision, wMean, wCovar, basis = BasisName)
            ptp[i] = pMean
            ptp_lower[i] = pMean - pSigma
            ptp_upper[i] = pMean + pSigma
        ax.plot(xp, ptp, 'r-', label = 'predicted function')
        
        # Draw uncertainity borders from predictive distribution
        if auxParams['doDrawPredicitveStdev']:
            ax.fill_between(xp, ptp_lower, ptp_upper, color = 'r', alpha = 0.06)

        ax.legend()
        self.figureCanvas.draw()

        # Fill info
        self.ErrorValueLabelText.set( "{:.3f}".format(model.RMS(x, w_est, t, basis = BasisName)) )
        self.AlphaValueLabelText.set( "-" )
        self.BetaValueLabelText.set( "-" )
        self.MFavourValueLabelText.set( "-" )

    def getAx(self):
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_title('Linear regression')
        self.ax.set_xlabel('$x$')
        self.ax.set_ylabel('$t$', rotation = 0)
        return self.ax


if __name__ == "__main__":
    app = App().start()