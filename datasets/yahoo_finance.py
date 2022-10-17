import yfinance as yf
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

def visualizeFinantialData():
    df = pd.read_csv('health_financial.csv')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%S.%f')
    df['month'] = pd.to_datetime(df['date'],utc=True).dt.month
    df['year'] = pd.to_datetime(df['date'],utc=True).dt.year
    df['monthYear'] = df['year'].to_string()+ "-"+df['month'].to_string()
    print("s")
    df['monthYear'] = pd.to_datetime(df['monthYear'], format='%Y-%m')
    print("s")
    df.groupby(['monthYear', 'shortName']).avg('Close')
    print("s")
    plt.figure()
    df.plot(x=["monthYear"], y="Close")
    plt.show()


def saveFinantialData():
    tech_companies =['ABC','ABMD','ABT','ALGN','ALNY','AMGN','BAX','BDX','BIIB','BMRN','BMY',
                     'BSX','CAH','CI','CNC','COO','CVS','DGX','DHR','DXCM','ELV','EW',
                     'GILD','HOLX','IDXX','ILMN','INCY','ISRG','LH','MCK','MDT',
                     'MOH','NBIX','PKI','PODD','REGN','RGEN','RMD','SGEN','STE','TECH',
                     'VRTX','VTRS','WAT','WST','ZBH',]

    toSave =pd.DataFrame(columns =['date','Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
                                   'symbol', 'shortName'])

    start = datetime(2009,12,30)
    end = datetime(2019,12,30)

    for c in tech_companies:
        t = yf.Ticker(c)
        shortName = t.info['shortName']
        h = t.history(start=start, end=end)
        h ['symbol'] = c
        h ['shortName'] = shortName
        h ['date'] = h.index
        toSave =toSave.append(h)

    toSave.to_csv('health_financial.csv', index=False)

#saveFinantialData()
visualizeFinantialData()