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
import model_specific as model

WINDOW_TITLE = 'Neural network regression tool'
BG_COLOR = "#ffffff" #"#404040"
BG_COLOR_GRAY1 = "#212121"
BG_COLOR_GRAY2 = "#303030"
BG_COLOR_GRAY3 = "#424242"
BT_COLOR = "#424242"
BT_TEXT_COLOR = "#ffffff"
BT_ACTIVE_COLOR = "#424242"
FG_COLOR = "#ffffff"
WIDTH = 1280
HEIGHT = 720

PRIM_COLOR = '#212121'
PRIM_COLOR2 = '#303030'
LABEL_COLOR = (1, 1, 1, 0.9)
CYBERPUNK_COLORS = [
    '#08F7FE',  # teal/cyan
    '#FE53BB',  # pink
    '#F5D300',  # yellow
    '#00ff41', # matrix green
]

# plt.style.use('seaborn-dark')
plt.rcParams["figure.facecolor"] = PRIM_COLOR
plt.rcParams["axes.facecolor"] = PRIM_COLOR
plt.rcParams["savefig.facecolor"] = PRIM_COLOR
plt.rcParams["text.color"] = LABEL_COLOR
plt.rcParams["axes.labelcolor"] = LABEL_COLOR
plt.rcParams["xtick.color"] = LABEL_COLOR
plt.rcParams["ytick.color"] = LABEL_COLOR

def defocus(event):
    event.widget.master.focus_set()

class App:
    def __init__(self):

        self.X = np.array([], dtype = np.float32) # <- observable variable
        self.t = np.array([], dtype = np.float32) # <- targer variable
        self.nnModel = None

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
        self.style = ttk.Style()
        self.style.theme_use('classic')
        self.window.wm_title(WINDOW_TITLE)
        self.window.geometry('%dx%d+%d+%d' % (WIDTH, HEIGHT, 10, 10))
        self.window.minsize(WIDTH, HEIGHT)
        self.window.configure(background = BG_COLOR_GRAY2)
        self.window.wm_protocol("WM_DELETE_WINDOW", self.onClose)
        # self.window.iconbitmap('data/img/favicon.ico')

        self.leftFrame = tk.Frame(self.window, bg = BG_COLOR_GRAY2)
        self.leftFrame.place(relx = 0.03, rely = 0.05, relwidth = 0.25, relheight = 0.9)

        self.rightFrame = tk.Frame(self.window, bg = BG_COLOR_GRAY2, bd = 1.5)
        self.rightFrame.place(relx = 0.3, rely = 0.05, relwidth = 0.65, relheight = 0.9)

    def initDataGroup(self):
        self.dataGroupLabel = tk.LabelFrame(self.leftFrame, text = '  Data  ', fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        self.dataGroupLabel.pack(padx = 5, pady = 5, side = tk.TOP, fill = tk.BOTH)

        NLabelText = tk.StringVar()
        NLabelText.set('N = ')
        NSpinboxVar = tk.IntVar()
        NSpinboxVar.set(100)
        NLabel = tk.Label(self.dataGroupLabel, text = NLabelText, textvariable = NLabelText, fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        NLabel.grid(row = 0, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.NSpinbox = tk.Spinbox(self.dataGroupLabel, width = 5, from_ = 1, to = 99999, textvariable = NSpinboxVar, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2, buttonbackground = BG_COLOR_GRAY2)
        self.NSpinbox.grid(row = 0, column = 1, columnspan = 2, sticky = tk.W + tk.E, padx = 5, pady = 5)

        FromLabelText = tk.StringVar()
        FromLabelText.set('From = ')
        FromSpinboxVar = tk.DoubleVar()
        FromSpinboxVar.set(0)
        FromLabel = tk.Label(self.dataGroupLabel, text = FromLabelText, textvariable = FromLabelText, fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        FromLabel.grid(row = 1, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.FromSpinbox = tk.Spinbox(self.dataGroupLabel, width = 5, from_ = -999, to = 999, textvariable = FromSpinboxVar, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2, buttonbackground = BG_COLOR_GRAY2)
        self.FromSpinbox.grid(row = 1, column = 1, columnspan = 2, sticky = tk.W + tk.E, padx = 5, pady = 5)

        ToLabelText = tk.StringVar()
        ToLabelText.set('To = ')
        ToSpinboxVar = tk.DoubleVar()
        ToSpinboxVar.set(1)
        ToLabel = tk.Label(self.dataGroupLabel, text = ToLabelText, textvariable = ToLabelText, fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        ToLabel.grid(row = 2, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.ToSpinbox = tk.Spinbox(self.dataGroupLabel, width = 5, from_ = -998, to = 999, textvariable = ToSpinboxVar, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2, buttonbackground = BG_COLOR_GRAY2)
        self.ToSpinbox.grid(row = 2, column = 1, columnspan = 2, sticky = tk.W + tk.E, padx = 5, pady = 5)
        
        BiasLabelText = tk.StringVar()
        BiasLabelText.set('Bias = ')
        BiasSpinboxVar = tk.DoubleVar()
        BiasSpinboxVar.set(0.0)
        BiasLabel = tk.Label(self.dataGroupLabel, text = BiasLabelText, textvariable = BiasLabelText, fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        BiasLabel.grid(row = 0, column = 3, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.BiasSpinbox = tk.Spinbox(self.dataGroupLabel, width = 5, from_ = 0, to = 100, increment = 0.1, format = "%.2f", textvariable = BiasSpinboxVar, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2, buttonbackground = BG_COLOR_GRAY2)
        self.BiasSpinbox.grid(row = 0, column = 4, columnspan = 2, sticky = tk.W + tk.E, padx = 5, pady = 5)
        
        NoiseLabelText = tk.StringVar()
        NoiseLabelText.set('Noise = ')
        NoiseSpinboxVar = tk.DoubleVar()
        NoiseSpinboxVar.set(0.3)
        NoiseLabel = tk.Label(self.dataGroupLabel, text = NoiseLabelText, textvariable = NoiseLabelText, fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        NoiseLabel.grid(row = 1, column = 3, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.NoiseSpinbox = tk.Spinbox(self.dataGroupLabel, width = 5, from_ = 0.0000001, to = 100, increment = 0.1, format = "%.2f", textvariable = NoiseSpinboxVar, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2, buttonbackground = BG_COLOR_GRAY2)
        self.NoiseSpinbox.grid(row = 1, column = 4, columnspan = 2, sticky = tk.W + tk.E, padx = 5, pady = 5)

        FuncLabelText = tk.StringVar()
        FuncLabelText.set('f(x) = ')
        FuncLabel = tk.Label(self.dataGroupLabel, text = FuncLabelText, textvariable = FuncLabelText, fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        FuncLabel.grid(row = 2, column = 3, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        funcs = [ "sin(2pix)", "x+0.3sin(2pix)", "exp(x)", "x^2", "|x|", "H(x)" ]
        self.TrueFuncCBox = ttk.Combobox(self.dataGroupLabel, width = 9, values = funcs, state = "readonly", background = BG_COLOR_GRAY2)
        self.TrueFuncCBox.master.option_add( '*TCombobox*Listbox.Background', BG_COLOR_GRAY2)
        self.TrueFuncCBox.master.option_add( '*TCombobox*Listbox.Foreground', BT_TEXT_COLOR)
        self.TrueFuncCBox.master.option_add( '*TCombobox*Listbox.selectBackground', BT_TEXT_COLOR)
        self.TrueFuncCBox.master.option_add( '*TCombobox*Listbox.selectForeground', BG_COLOR_GRAY2)
        self.style.map('TCombobox', fieldbackground=[('readonly', BG_COLOR_GRAY2)])
        self.style.map('TCombobox', selectbackground=[('readonly', BG_COLOR_GRAY2)])
        self.style.map('TCombobox', selectforeground=[('readonly', BT_TEXT_COLOR)])
        self.style.map('TCombobox', background=[('readonly', BG_COLOR_GRAY2)])
        self.style.map('TCombobox', foreground=[('readonly', BT_TEXT_COLOR)])
        self.TrueFuncCBox.bind("<FocusIn>", defocus) # <- in order to prevent box highlighting
        self.TrueFuncCBox.grid(row = 2, column = 4, columnspan = 2, sticky = tk.W + tk.E, padx = 5, pady = 5)
        self.TrueFuncCBox.current(0)

        self.generateButton = tk.Button(
            self.dataGroupLabel,
            text=' Generate ',
            bg = BT_COLOR,
            fg = BT_TEXT_COLOR,
            activebackground = BT_ACTIVE_COLOR,
            activeforeground = BT_TEXT_COLOR,
            command = self.onDataGeneration
        )
        self.generateButton.grid(row = 3, column = 0, columnspan = 3, sticky = tk.W + tk.E, padx = 5, pady = 5)

        # separator = ttk.Separator(self.dataGroupLabel, orient = 'horizontal')
        # separator.grid(row = 4, column = 0, columnspan = 5, sticky = tk.W + tk.E, padx = 5, pady = (5, 15))

        self.openFileButton = tk.Button(
            self.dataGroupLabel,
            text=' Open .npy file ',
            bg = BT_COLOR,
            fg = BT_TEXT_COLOR,
            activebackground = BT_ACTIVE_COLOR,
            activeforeground = BT_TEXT_COLOR,
            command = self.onFileOpening
        )
        self.openFileButton.grid(row = 6, column = 0, columnspan = 3, sticky = tk.W + tk.E, padx = 5, pady = 5)

    def initModelGroup(self):
        self.modelGroupLabel = tk.LabelFrame(self.leftFrame, text = '  Model  ', fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        self.modelGroupLabel.pack(padx = 5, pady = 5, side = tk.TOP, fill = tk.BOTH)

        MLabelText = tk.StringVar()
        MLabelText.set('Hidden units = ')
        MSpinboxVar = tk.IntVar()
        MSpinboxVar.set(3)
        MLabel = tk.Label(self.modelGroupLabel, text = MLabelText, textvariable = MLabelText, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        MLabel.grid(row = 0, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.MSpinbox = tk.Spinbox(self.modelGroupLabel, width = 3, from_ = 1, to = 50, textvariable = MSpinboxVar, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2, buttonbackground = BG_COLOR_GRAY2)
        self.MSpinbox.grid(row = 0, column = 1, columnspan = 2, sticky = tk.W + tk.E, padx = 5, pady = 5)
        
        EpochsLabelText = tk.StringVar()
        EpochsLabelText.set('Epochs = ')
        EpochsSpinboxVar = tk.IntVar()
        EpochsSpinboxVar.set(100)
        EpochsLabel = tk.Label(self.modelGroupLabel, text = EpochsLabelText, textvariable = EpochsLabelText, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        EpochsLabel.grid(row = 1, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.EpochsSpinbox = tk.Spinbox(self.modelGroupLabel, width = 5, from_ = 1, to = 99999, textvariable = EpochsSpinboxVar, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2, buttonbackground = BG_COLOR_GRAY2)
        self.EpochsSpinbox.grid(row = 1, column = 1, columnspan = 3, sticky = tk.W + tk.E, padx = 5, pady = 5)
        
        LRateLabelText = tk.StringVar()
        LRateLabelText.set('Learning rate = ')
        LRateSpinboxVar = tk.DoubleVar()
        LRateSpinboxVar.set(0.05)
        LRateLabel = tk.Label(self.modelGroupLabel, text = LRateLabelText, textvariable = LRateLabelText, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        LRateLabel.grid(row = 2, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.LRateSpinbox = tk.Spinbox(self.modelGroupLabel, width = 4, from_ = 0.001, to = 9999, increment = 0.01, format = "%.3f", textvariable = LRateSpinboxVar, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2, buttonbackground = BG_COLOR_GRAY2)
        self.LRateSpinbox.grid(row = 2, column = 1, columnspan = 3, sticky = tk.W + tk.E, padx = 5, pady = 5)
        
        AlphaLabelText = tk.StringVar()
        AlphaLabelText.set('α = ')
        AlphaSpinboxVar = tk.DoubleVar()
        AlphaSpinboxVar.set(0.00033)
        AlphaLabel = tk.Label(self.modelGroupLabel, text = AlphaLabelText, textvariable = AlphaLabelText, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        AlphaLabel.grid(row = 3, column = 0, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.AlphaSpinbox = tk.Spinbox(self.modelGroupLabel, width = 7, from_ = 0, to = 50, increment = 0.001, format = "%.5f", textvariable = AlphaSpinboxVar, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2, buttonbackground = BG_COLOR_GRAY2)
        self.AlphaSpinbox.grid(row = 3, column = 1, sticky = tk.W + tk.E, padx = 5, pady = 5)
        
        BetaLabelText = tk.StringVar()
        BetaLabelText.set('β = ')
        BetaSpinboxVar = tk.DoubleVar()
        BetaSpinboxVar.set(3.33)
        BetaLabel = tk.Label(self.modelGroupLabel, text = BetaLabelText, textvariable = BetaLabelText, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        BetaLabel.grid(row = 4, column = 0, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.BetaSpinbox = tk.Spinbox(self.modelGroupLabel, width = 7, from_ = 0, to = 50, increment = 0.001, format = "%.5f", textvariable = BetaSpinboxVar, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2, buttonbackground = BG_COLOR_GRAY2)
        self.BetaSpinbox.grid(row = 4, column = 1, sticky = tk.W + tk.E, padx = 5, pady = 5)
        
        LmbdLabelText = tk.StringVar()
        LmbdLabelText.set('λ = ')
        LmbdSpinboxVar = tk.DoubleVar()
        LmbdSpinboxVar.set(0.00001)
        LmbdLabel = tk.Label(self.modelGroupLabel, text = LmbdLabelText, textvariable = LmbdLabelText, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        LmbdLabel.grid(row = 5, column = 0, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.LmbdSpinbox = tk.Spinbox(self.modelGroupLabel, width = 7, from_ = 0, to = 50, increment = 0.001, format = "%.5f", textvariable = LmbdSpinboxVar, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2, buttonbackground = BG_COLOR_GRAY2)
        self.LmbdSpinbox.grid(row = 5, column = 1, sticky = tk.W + tk.E, padx = 5, pady = 5)

        # self.modelRButtonVar = tk.IntVar()
        # self.modelRButtonVar.set(0)
        # self.powerRButton = tk.Radiobutton(self.modelGroupLabel, text = 'Power', variable = self.modelRButtonVar, value = 0)
        # self.powerRButton.grid(row = 2, column = 0, columnspan = 2, sticky = tk.W + tk.N, padx = (5, 0), pady = 5)
        # self.gaussianRButton = tk.Radiobutton(self.modelGroupLabel, text = 'Gaussian', variable = self.modelRButtonVar, value = 1)
        # self.gaussianRButton.grid(row = 2, column = 2, columnspan = 2, sticky = tk.W + tk.N, padx = (5, 0), pady = 5)
        # self.sigmoidRButton = tk.Radiobutton(self.modelGroupLabel, text = 'Sigmoid', variable = self.modelRButtonVar, value = 2)
        # self.sigmoidRButton.grid(row = 2, column = 4, columnspan = 2, sticky = tk.W + tk.N, padx = (5, 0), pady = 5)

        # self.sampleWCButtonVar = tk.IntVar()
        # self.sampleWCButtonVar.set(0)
        # self.sampleWCButton = tk.Checkbutton(self.modelGroupLabel, text = 'Draw samples from p(w|α)', onvalue = 1, offvalue = 0, variable = self.sampleWCButtonVar)
        # self.sampleWCButton.grid(row = 5, column = 0, columnspan = 5, sticky = tk.W + tk.N, padx = (5, 0), pady = (15, 5))

        # self.drawPredStdevCButtonVar = tk.IntVar()
        # self.drawPredStdevCButtonVar.set(1)
        # self.drawPredStdevCButton = tk.Checkbutton(self.modelGroupLabel, text = 'Draw uncertainities', onvalue = 1, offvalue = 0, variable = self.drawPredStdevCButtonVar)
        # self.drawPredStdevCButton.grid(row = 6, column = 0, columnspan = 5, sticky = tk.W + tk.N, padx = (5, 0), pady = (5, 5))

        # self.fitButton = tk.Button(
        #     self.modelGroupLabel,
        #     text=' Fit entire dataset ',
        #     bg = BT_COLOR,
        #     command = lambda: self.onFit('entire')
        # )
        # self.fitButton.grid(row = 7, column = 0, columnspan = 3, sticky = tk.W + tk.E, padx = 5, pady = 5)

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

        self.animateCButtonVar = tk.IntVar()
        self.animateCButtonVar.set(0)
        self.animateCButton = tk.Checkbutton(self.modelGroupLabel, text = 'Animate', onvalue = 1, offvalue = 0, variable = self.animateCButtonVar, \
            fg = FG_COLOR, 
            bg = BG_COLOR_GRAY2, 
            activebackground = BT_ACTIVE_COLOR,
            activeforeground = BT_TEXT_COLOR,
            selectcolor = BG_COLOR_GRAY2)
        self.animateCButton.grid(row = 10, column = 3, columnspan = 2, sticky = tk.E + tk.N, padx = (5, 0), pady = (5, 5))
        animateSpinboxVar = tk.IntVar()
        animateSpinboxVar.set(1)
        self.animateSpinbox = tk.Spinbox(self.modelGroupLabel, width = 4, from_ = 0, to = 9999, textvariable = animateSpinboxVar, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2, buttonbackground = BG_COLOR_GRAY2)
        self.animateSpinbox.grid(row = 10, column = 5, columnspan = 1, sticky = tk.W + tk.E, padx = 5, pady = 5)
        animateLabelText = tk.StringVar()
        animateLabelText.set('ms')
        animateLabel = tk.Label(self.modelGroupLabel, text = animateLabelText, textvariable = animateLabelText, \
            fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        animateLabel.grid(row = 10, column = 6, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)

        self.fitSequentiallyButton = tk.Button(
            self.modelGroupLabel,
            text=' Fit sequentially ',
            bg = BT_COLOR,
            fg = BT_TEXT_COLOR,
            activebackground = BT_ACTIVE_COLOR,
            activeforeground = BT_TEXT_COLOR,
            command = lambda: self.onFit('sequential')
        )
        self.fitSequentiallyButton.grid(row = 10, column = 0, columnspan = 3, sticky = tk.W + tk.E, padx = 5, pady = 5)

    def initInfoGroup(self):
        self.infoGroupLabel = tk.LabelFrame(self.leftFrame, text = '  Info  ', fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        self.infoGroupLabel.pack(padx = 5, pady = 5, side = tk.TOP, fill = tk.BOTH)
        
        ErrorLabelText = tk.StringVar()
        ErrorLabelText.set('Error (RMS) = ')
        ErrorLabel = tk.Label(self.infoGroupLabel, text = ErrorLabelText, textvariable = ErrorLabelText, fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        ErrorLabel.grid(row = 0, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.ErrorValueLabelText = tk.StringVar()
        self.ErrorValueLabelText.set('')
        self.ErrorValueLabel = tk.Label(self.infoGroupLabel, text = self.ErrorValueLabelText, textvariable = self.ErrorValueLabelText, fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        self.ErrorValueLabel.grid(row = 0, column = 1, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        
        AlphaLabelText = tk.StringVar()
        AlphaLabelText.set('Estimated α = ')
        AlphaLabel = tk.Label(self.infoGroupLabel, text = AlphaLabelText, textvariable = AlphaLabelText, fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        AlphaLabel.grid(row = 1, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.AlphaValueLabelText = tk.StringVar()
        self.AlphaValueLabelText.set('')
        self.AlphaValueLabel = tk.Label(self.infoGroupLabel, text = self.AlphaValueLabelText, textvariable = self.AlphaValueLabelText, fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        self.AlphaValueLabel.grid(row = 1, column = 1, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        
        # MFavourLabelText = tk.StringVar()
        # MFavourLabelText.set('M favour = ')
        # MFavourLabel = tk.Label(self.infoGroupLabel, text = MFavourLabelText, textvariable = MFavourLabelText)
        # MFavourLabel.grid(row = 1, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        # self.MFavourValueLabelText = tk.StringVar()
        # self.MFavourValueLabelText.set('')
        # self.MFavourValueLabel = tk.Label(self.infoGroupLabel, text = self.MFavourValueLabelText, textvariable = self.MFavourValueLabelText)
        # self.MFavourValueLabel.grid(row = 1, column = 1, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        
        BetaLabelText = tk.StringVar()
        BetaLabelText.set('Estimated β = ')
        BetaLabel = tk.Label(self.infoGroupLabel, text = BetaLabelText, textvariable = BetaLabelText, fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        BetaLabel.grid(row = 2, column = 0, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)
        self.BetaValueLabelText = tk.StringVar()
        self.BetaValueLabelText.set('')
        self.BetaValueLabel = tk.Label(self.infoGroupLabel, text = self.BetaValueLabelText, textvariable = self.BetaValueLabelText, fg = FG_COLOR, bg = BG_COLOR_GRAY2)
        self.BetaValueLabel.grid(row = 2, column = 1, columnspan = 1, sticky = tk.W + tk.N, padx = 5, pady = 5)

    def initGraphPanel(self):
        self.figure = Figure(figsize = (5, 5), dpi = 100)
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
                self.t = np.load(f)
        except:
            messagebox.showerror(title = 'Error', message = 'Bad data')
            return

        if self.X.size * self.t.size == 0:
            messagebox.showerror(title = 'Error', message = 'Bad data')
            return

        self.gtLinspace = np.array([])
        self.truePrecision = -1
        self.axMin = np.min(self.X)
        self.axMax = np.max(self.X)

        # Plot data
        ax = self.getAx()
        ax.plot(self.X, self.t, '.', label = 'data')
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

        self.X, self.t = model.generateData(N, self.axMin, self.axMax, TrueFunction, bias, noise_level)
        if noise_level > 0:
            self.truePrecision = 1 / noise_level
        else:
            self.truePrecision = 10000000
        
        self.xLinspace = np.linspace(self.axMin, self.axMax, num = 100)
        self.gtLinspace = model.groundTruthProcess(self.xLinspace, bias, TrueFunction)

        # Plot
        ax = self.getAx()
        ax.plot(self.X, self.t, '.', label = 'data', color = CYBERPUNK_COLORS[0])
        ax.plot(self.xLinspace, self.gtLinspace, '--', label = 'ground-truth function', color = CYBERPUNK_COLORS[1])

        # Make glow
        n_lines = 10
        diff_linewidth = 1.05
        alpha_value = 0.03
        for n in range(1, n_lines + 1):
            ax.plot(self.xLinspace, self.gtLinspace, '--',
                    linewidth = 2 + (diff_linewidth * n),
                    alpha = alpha_value,
                    color = CYBERPUNK_COLORS[1])

        ax.legend()
        self.figureCanvas.draw()

    def onFit(self, method = 'entire'):
        X = self.X
        t = self.t
        if X is None or t is None:
            print(">>> ERROR: x and t must be set")
            return
        if X.size != t.size:
            print(">>> ERROR: x and t sizes must be equal")
            return
        if X.size * t.size == 0:
            print(">>> ERROR: x and t size must be non zero")
            return

        epochs = int(self.EpochsSpinbox.get())
        batch_size = 32
        lrate = float(self.LRateSpinbox.get())
        NHiddenUnits = int(self.MSpinbox.get())
        al = float(self.AlphaSpinbox.get())
        bt = float(self.BetaSpinbox.get())
        lmbd = float(self.LmbdSpinbox.get())
        self.nnModel = model.SimpleFeedForwardNetwork(NHiddenUnits)

        # How about a little animation?
        doAnimate = bool(self.animateCButtonVar.get())
        if doAnimate:
            N = X.size
            animationDelay = int(self.animateSpinbox.get())
            anim = animation.FuncAnimation(self.figure, self.onFitSequentiallyAnimated, \
                np.arange(0, 10), fargs = (X, t, epochs // 10, lrate), \
                interval = animationDelay, repeat = False, blit = False)
            self.figureCanvas.draw()
            return

        N = X.size
        X_train = np.ones((N, 2), dtype = np.float32)
        X_train[:, 1:] = np.expand_dims(X, 1)
        self.nnModel.fit(X_train, t, epochs, batch_size, lrate, lmbd, al, bt, verbose = True)

        # Estimate α and β
        a0 = 0.001
        b0 = 100
        a_est, b_est = self.nnModel.hyperparameterOptimization_experimental(X_train, t, a0, b0)

        # Fill info
        # self.ErrorValueLabelText.set( "{:.3f}".format(model.RMS(x, w_est, t, basis = BasisName)) )
        self.AlphaValueLabelText.set( "{:.3f}".format(a_est) )
        self.BetaValueLabelText.set( "{:.3f}".format(b_est) )

        # Plotting
        Npoints = 100
        xp = np.linspace(self.axMin, self.axMax, num = Npoints)
        ax = self.getAx()

        # Plot data
        ax.plot(X, t, '.', label = 'data', color = CYBERPUNK_COLORS[0])

        # Draw ground-truth function
        if self.gtLinspace.size > 0:
            ax.plot(xp, self.gtLinspace, '--', label = 'ground-truth function', color = CYBERPUNK_COLORS[1])

        # Draw points from predictive distribution
        ptp = np.zeros(Npoints, dtype = np.float32)
        ptp_lower = np.zeros(Npoints, dtype = np.float32)
        ptp_upper = np.zeros(Npoints, dtype = np.float32)
        for i in range(Npoints):
            pMean, pSigma = self.nnModel.predictBayesian([1, xp[i]])
            ptp[i] = pMean
            ptp_lower[i] = pMean - pSigma
            ptp_upper[i] = pMean + pSigma
        
        # Draw uncertainity borders from predictive distribution
        if True:
            ax.fill_between(xp, ptp_lower, ptp_upper, color = CYBERPUNK_COLORS[1], alpha = 0.08)

        # Draw predicted function
        ptp = np.zeros(Npoints, dtype = np.float32)
        for i in range(Npoints):
            xi = np.array([1, xp[i]])
            ptp[i] = self.nnModel.forward(xi)[0]
        ax.plot(xp, ptp, '-', label = 'predicted function', color = CYBERPUNK_COLORS[2])

        # Make glow
        n_lines = 10
        diff_linewidth = 1.05
        alpha_value = 0.03
        for n in range(1, n_lines + 1):
            ax.plot(self.xLinspace, self.gtLinspace, '--',
                    linewidth = 2 + (diff_linewidth * n),
                    alpha = alpha_value,
                    color = CYBERPUNK_COLORS[1])
            ax.plot(xp, ptp, '-',
                    linewidth = 2 + (diff_linewidth * n),
                    alpha = alpha_value,
                    color = CYBERPUNK_COLORS[2])

        ax.legend()
        self.figureCanvas.draw()

    def onFitSequentiallyAnimated(self, i, X, t, epochs, lrate):

        N = X.size
        X_train = np.ones((N, 2), dtype = np.float32)
        X_train[:, 1:] = np.expand_dims(X, 1)
        self.nnModel.fit(X_train, t, epochs, lrate)

        # Plotting
        Npoints = 100
        xp = np.linspace(self.axMin, self.axMax, num = Npoints)
        ax = self.getAx()

        # Plot data
        ax.plot(X, t, '.', label = 'data', color = CYBERPUNK_COLORS[0])

        # Draw ground-truth function
        if self.gtLinspace.size > 0:
            ax.plot(xp, self.gtLinspace, '--', label = 'ground-truth function', color = CYBERPUNK_COLORS[1])

        # Draw predicted function
        ptp = np.zeros(Npoints, dtype = np.float32)
        for i in range(Npoints):
            xi = np.array([1, xp[i]])
            ptp[i] = self.nnModel.forward(xi)[0]
        ax.plot(xp, ptp, '-', label = 'predicted function', color = CYBERPUNK_COLORS[2])

        # Make glow
        n_lines = 10
        diff_linewidth = 1.05
        alpha_value = 0.03
        for n in range(1, n_lines + 1):
            ax.plot(self.xLinspace, self.gtLinspace, '--',
                    linewidth = 2 + (diff_linewidth * n),
                    alpha = alpha_value,
                    color = CYBERPUNK_COLORS[1])
            ax.plot(xp, ptp, '-',
                    linewidth = 2 + (diff_linewidth * n),
                    alpha = alpha_value,
                    color = CYBERPUNK_COLORS[2])

        ax.legend()
        self.figureCanvas.draw()

    def getAx(self):
        self.ax.clear()
        self.ax.grid(True, color = PRIM_COLOR2)
        self.ax.set_title('Neural network regression')
        self.ax.set_xlabel('$x$')
        self.ax.set_ylabel('$t$', rotation = 0)
        return self.ax


if __name__ == "__main__":
    app = App().start()