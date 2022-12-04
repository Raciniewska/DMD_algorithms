import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter , AutoMinorLocator,MultipleLocator


def plotVariance():
    df = pd.read_csv ('resultsErrorVar.csv')
    ls = [10,20,40,60,80,100,200,300,400,500]
    df['rząd_macierzy'] =ls
    df.plot(x='rząd_macierzy', y=['Uniform','Gauss','SRFT','SRHT'], kind='line')
    plt.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
    plt.ylabel('wariancja')
    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor', color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    plt.show()

def plotRelativeError():
    df = pd.read_csv ('errorWzgledny.csv')
    ax =df.plot(x='l', y=['Uniform','Gauss','SRFT','SRHT'], kind='bar')
    plt.ylabel('błąd_względny')
    plt.xlabel('rząd_macierzy')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor', color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.minorticks_on()

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()

def plotError():
    df = pd.read_csv ('resultsError.csv')

    ax = df.plot(x='l', y=['Uniform','Gauss','SRFT','SRHT','Deterministic'], kind='bar')

    plt.ylabel('błąd')
    plt.xlabel('rząd_macierzy')


    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor',color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()

def plotTime():
    df = pd.read_csv ('resultsTime.csv')
    ax =df.plot(x='l', y=['Uniform','Gauss','SRFT','SRHT'], kind='bar')
    plt.ylabel('czas')
    plt.xlabel('rząd_macierzy')

    x = np.linspace(0, 500)
    y=[0.314252710342407]*len(x)
    ax.plot(x, y, color='k', label="Deterministic")

    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor',color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()

font = {'family' : 'normal',
        'size'   : 13}

plt.rc('font', **font)
#plotTime()
plotVariance()
#plotRelativeError()
#plotError()