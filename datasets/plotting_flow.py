import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def plotErrorRDMD(df,rank):
    plt.rcParams['font.size'] = '15'
    for labels, dfi in df.groupby("q_val"):
        plt.plot(dfi['p_val'],dfi['error_mean'], label=labels)
    plt.legend(title='liczba iteracji potęgowych',bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title("Rząd docelowy - "+str(rank))
    plt.ylabel('Średni błąd')
    plt.xlabel('Parametr nadpróbkowania')
    plt.yscale('log')
    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor', color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    plt.tight_layout()
    #plt.figure(figsize=(10, 10))
    plt.show()

def plotErrorRDMDbyRank(df,p,q):
    print(df)
    plt.plot(df['target_rank'],df['error_mean'])
    plt.title("Parametr nadpróbkowania - "+str(p) +" \n Liczba iteracji potęgowych - "+str(q))
    plt.ylabel('Średni błąd')
    plt.xlabel('Rząd docelowy')
    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor', color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

def plotTimeRDMDbyRank(df,dfDMD,p,q):
    plt.plot(df['target_rank'], df['time_mean'],label="rDMD")
    plt.plot(dfDMD['target_rank'], dfDMD['time_mean'],label="DMD")
    plt.legend()
    plt.title("Parametr nadpróbkowania - " + str(p) +" \n Liczba iteracji potęgowych - "+str(q))
    plt.ylabel('Średni czas [s]')
    plt.xlabel('Rząd docelowy')
    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor', color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

def plotTimeRDMD(df,rank):
    for labels, dfi in df.groupby("q_val"):
        plt.plot(dfi['p_val'], dfi['time_mean'], label=labels)
    plt.legend(title='liczba iteracji potęgowych',bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title("Rząd docelowy - " + str(rank))
    plt.ylabel('Średni czas [s]')
    plt.xlabel('Parametr nadpróbkowania')
    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor', color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

def ErrorRDMD(path):
    df = pd.read_csv(path)
    cols = df.columns
    df['error_mean'] = 0
    experiments =0
    for colName in cols:
        if colName[0] == "e":
            experiments +=1
            df['error_mean'] +=df[colName]
    df['error_mean']/=experiments

    target_ranks = df['target_rank'].unique()

    for tr in target_ranks:
        df_to_plot = df[ df['target_rank'] == tr]
        plotErrorRDMD(df_to_plot[['p_val','q_val','error_mean']],tr)

def ErrorRDMDforSelectedParams(path,p,q):
    df = pd.read_csv(path)
    cols = df.columns
    df = df[df['p_val'] ==p]
    df = df[df['q_val'] ==q]
    df['error_mean'] = 0
    experiments =0
    for colName in cols:
        if colName[0] == "e":
            experiments +=1
            df['error_mean'] +=df[colName]
    df['error_mean']/=experiments

    target_ranks = df['target_rank'].unique()
    plotErrorRDMDbyRank(df[['target_rank','error_mean']],p,q)

def getTimeDMD(path):
    df = pd.read_csv(path)
    cols = df.columns
    df['time_mean'] = 0
    experiments = 0
    for colName in cols:
        if colName[0:2] == "ti":
            experiments += 1
            df[colName] = pd.to_timedelta(df[colName]).dt.total_seconds()#.astype('timedelta64[s]').astype(float)
            df['time_mean'] += df[colName]
    df['time_mean'] /= experiments
    return df[['target_rank','time_mean']]

def TimeRDMD(path):
    df = pd.read_csv(path)
    cols = df.columns
    df['time_mean'] = 0
    experiments =0
    for colName in cols:
        if colName[0:2] == "ti" :
            experiments +=1
            df[colName]=pd.to_timedelta(df[colName]).dt.total_seconds()
            df['time_mean'] += df[colName]
    df['time_mean']/=experiments

    target_ranks = df['target_rank'].unique()
    dfDMD = getTimeDMD('results_reconstruction/dmd.csv')
    for tr in target_ranks:
        df_to_plot = df[ df['target_rank'] == tr]
        df_to_plot = df_to_plot[['p_val','q_val','time_mean']]
        dmd_time_by_tr = dfDMD[dfDMD['target_rank'] == tr]
        dmd_time = dmd_time_by_tr[['time_mean']].to_numpy()[0][0]
        dmd_rows = {'p_val': df['p_val'].unique(),
                    'q_val': ["Deterministyczny"]*len(df['p_val'].unique()),
                    'time_mean': [dmd_time]*len(df['p_val'].unique())}
        dmd_df = pd.DataFrame.from_dict(dmd_rows)
        df_to_plot = df_to_plot.append(dmd_df)
        plotTimeRDMD(df_to_plot[['p_val','q_val','time_mean']],tr)

def TimeRDMDforSelectedParams(path,p,q):
    df = pd.read_csv(path)
    cols = df.columns
    df = df[df['p_val'] == p]
    df = df[df['q_val'] == q]
    df['time_mean'] = 0
    experiments =0
    for colName in cols:
        if colName[0:2] == "ti" :
            experiments +=1
            df[colName]=pd.to_timedelta(df[colName]).dt.total_seconds()
            df['time_mean'] += df[colName]
    df['time_mean']/=experiments

    dfDMD = getTimeDMD('results_reconstruction/dmd.csv')
    df_to_plot = df[['target_rank','time_mean']]
    dfDMD = dfDMD[['target_rank','time_mean']]
    plotTimeRDMDbyRank(df_to_plot,dfDMD,p,q)

ErrorRDMD('results_reconstruction/rdmd.csv')
#ErrorRDMDforSelectedParams('results_reconstruction/rdmd.csv',10,0)
#TimeRDMD('results_reconstruction/rdmd_v1.csv')
#TimeRDMDforSelectedParams('results_reconstruction/rdmd_v1.csv',10,0)