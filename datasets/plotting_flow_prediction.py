import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def plotPSNRDMD(df, none_rank):
    plt.rcParams['font.size'] = '15'
    print(none_rank.iloc[0])
    print(df['error_mean'])
    plt.plot(df['target_rank'],df['error_mean'], label ="DMD z przyciętym rzędem docelowym")
    plt.axhline(y=none_rank.iloc[0], color='r', linestyle='-', label = "DMD maksymalny rząd docelowy")
    plt.ylabel('Średnia wartość PSNR')
    plt.xlabel('Rząd docelowy')
    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor', color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    plt.legend( bbox_to_anchor=(0.5, -0.17), loc='upper center')
    plt.tight_layout()
    plt.show()

def PSNR_DMD (path):
    df = pd.read_csv(path)
    cols = df.columns
    df['error_mean'] = 0
    experiments = 0
    for colName in cols:
        if colName[0] == "e":
            experiments += 1
            df['error_mean'] += df[colName]
    df['error_mean'] /= experiments
    df_to_plot  = df[df['target_rank'].astype(int) >0]
    none_rank =  df[df['target_rank'].astype(int) == 0]
    plotPSNRDMD(df_to_plot[['error_mean','target_rank']],none_rank['error_mean'] )

def getPSNRDMD(path):
    df = pd.read_csv(path)
    cols = df.columns
    df['error_mean'] = 0
    experiments = 0
    for colName in cols:
        if colName[0] == "e":
            experiments += 1
            df['error_mean'] += df[colName]
    df['error_mean'] /= experiments
    print(df)
    df_to_plot = df[df['target_rank'].astype(int) > 0]
    return df_to_plot[['target_rank','error_mean']]

def getTIMEDMD(path):
    df = pd.read_csv(path)
    cols = df.columns
    df['time_mean'] = 0
    experiments = 0
    for colName in cols:
        if colName[0:2] == "ti":
            experiments += 1
            df['time_mean'] +=  pd.to_timedelta(df[colName]).dt.total_seconds()
    df['time_mean'] /= experiments
    df_to_plot = df[df['target_rank'].astype(int) > 0]
    return df_to_plot[['target_rank','time_mean']]

def plotPSNR_RDMD(df,rank):
    for labels, dfi in df.groupby("q_val"):
        plt.plot(dfi['p_val'], dfi['error_mean'], label=labels)
    #plt.legend(title='liczba iteracji potęgowych',bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title("Rząd docelowy - " + str(rank))
    plt.ylabel('Średnia wartość PSNR')
    plt.xlabel('Parametr nadpróbkowania')
    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor', color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    # plt.yscale('log')
    #plt.ylim(ymin=0)
    plt.tight_layout()
    plt.show()

def plotTIME_RDMD(df,rank):
    for labels, dfi in df.groupby("q_val"):
        plt.plot(dfi['p_val'], dfi['time_mean'], label=labels)
    plt.legend(title='liczba iteracji potęgowych',bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title("Rząd docelowy - " + str(rank))
    plt.ylabel('Średni czas wykonania [s]')
    plt.xlabel('Parametr nadpróbkowania')
    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor', color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    # plt.yscale('log')
    #plt.ylim(ymin=0)
    plt.tight_layout()
    plt.show()

def PSNR_RDMD_by_rank(path):
    df = pd.read_csv(path)
    cols = df.columns
    df['error_mean'] = 0
    experiments = 0
    for colName in cols:
        if colName[0] == "e":
            experiments += 1
            df['error_mean'] += df[colName]
    df['error_mean'] /= experiments

    target_ranks = df['target_rank'].unique()
    dfDMD = getPSNRDMD('results_prediction_noise/dmdv1.csv')

    for tr in target_ranks:
        df_to_plot = df[ df['target_rank'] == tr]
        df_to_plot = df_to_plot[['p_val','q_val','error_mean']]
        dmd_time_by_tr = dfDMD[dfDMD['target_rank'] == tr]
        dmd_time = dmd_time_by_tr[['error_mean']].to_numpy()[0][0]
        print(dmd_time)
        dmd_rows = {'p_val': df['p_val'].unique(),
                    'q_val': ["Deterministyczny"]*len(df['p_val'].unique()),
                    'error_mean': [dmd_time]*len(df['p_val'].unique())}
        dmd_df = pd.DataFrame.from_dict(dmd_rows)
        df_to_plot = df_to_plot.append(dmd_df)

        plotPSNR_RDMD(df_to_plot[['p_val','q_val','error_mean']],tr)

def TIME_RDMD_by_rank(path):
    df = pd.read_csv(path)
    cols = df.columns
    df['time_mean'] = 0
    experiments = 0
    for colName in cols:
        if colName[0:2] == "ti":
            experiments += 1
            df['time_mean'] += pd.to_timedelta(df[colName]).dt.total_seconds()
    df['time_mean'] /= experiments

    target_ranks = df['target_rank'].unique()
    dfDMD = getTIMEDMD('results_prediction/dmd.csv')

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

        plotTIME_RDMD(df_to_plot[['p_val','q_val','time_mean']],tr)

def plotTimeRDMDbyRank(df,dfDMD,p,q):
    plt.plot(df['target_rank'], df['time_mean'],label="rDMD")
    plt.plot(dfDMD['target_rank'], dfDMD['time_mean'],label="DMD")
    plt.legend()
    plt.title("Parametr nadpróbkowania - " + str(p) +" \n Liczba iteracji potęgowych - "+str(q))
    plt.ylabel('Średni czas [s]')
    plt.xlabel('Rząd docelowy')
    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor', color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.ylim(ymin=0)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

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

    dfDMD = getTIMEDMD('results_prediction/dmd.csv')
    df_to_plot = df[['target_rank','time_mean']]
    dfDMD = dfDMD[['target_rank','time_mean']]
    plotTimeRDMDbyRank(df_to_plot,dfDMD,p,q)


def plotErrorRDMDbyRank(df,dfDMD,p,q):
    print(df)
    plt.plot(df['target_rank'], df['error_mean'],label="rDMD")
    plt.plot(dfDMD['target_rank'], dfDMD['error_mean'],label="DMD")
    plt.legend()
    plt.title("Parametr nadpróbkowania - " + str(p) +" \n Liczba iteracji potęgowych - "+str(q))
    plt.ylabel('Średnia wartość PSNR')
    plt.xlabel('Rząd docelowy')
    plt.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    plt.grid(b=True, which='minor', color='#CCCCCC', linestyle='-', alpha=0.2)
    plt.ylim(ymin=0)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

def ErrorRDMDforSelectedParams(path,p,q):
    df = pd.read_csv(path)
    cols = df.columns
    df = df[df['p_val'] == p]
    df = df[df['q_val'] == q]
    df['error_mean'] = 0
    experiments =0
    for colName in cols:
        if colName[0] == "e" :
            experiments +=1
            df['error_mean'] += df[colName]
    df['error_mean']/=experiments

    dfDMD = getPSNRDMD('results_prediction/dmd.csv')
    df_to_plot = df[['target_rank','error_mean']]
    dfDMD = dfDMD[['target_rank','error_mean']]
    print(dfDMD)
    plotErrorRDMDbyRank(df_to_plot,dfDMD,p,q)

def getTimeDiffForParams(path,p,q):
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

    dfDMD = getTIMEDMD('results_prediction/dmd.csv')
    df_to_plot = df[['target_rank','time_mean']]
    dfDMD = dfDMD[['target_rank','time_mean']]
    print(df_to_plot['time_mean'])
    print(dfDMD['time_mean'])

def getPSNRDiffForParams(path,p,q):
    df = pd.read_csv(path)
    cols = df.columns
    df = df[df['p_val'] == p]
    df = df[df['q_val'] == q]
    df['error_mean'] = 0
    experiments =0
    for colName in cols:
        if colName[0] == "e" :
            experiments +=1
            df['error_mean'] += df[colName]
    df['error_mean']/=experiments

    dfDMD = getPSNRDMD('results_prediction/dmd.csv')
    df_to_plot = df[['target_rank','error_mean']]
    dfDMD = dfDMD[['target_rank','error_mean']]
    print(df_to_plot['error_mean'])
    print(dfDMD['error_mean'])





#PSNR_DMD('results_prediction_noise/dmdv1.csv')
PSNR_RDMD_by_rank('results_prediction_noise/rdmdv1.csv')

#TIME_RDMD_by_rank('results_prediction/rdmdv1.csv')
#TimeRDMDforSelectedParams('results_prediction/rdmdv1.csv',0,1)
#ErrorRDMDforSelectedParams('results_prediction_noise/rdmdv1.csv',0,1)

#getTimeDiffForParams('results_prediction/rdmdv1.csv',0,1)
#getPSNRDiffForParams('results_prediction_noise/rdmdv1.csv',0,1)