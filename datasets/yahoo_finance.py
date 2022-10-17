import pandas as pd
import yfinance as yf

aapl_df = yf.download('AAPL')
print(aapl_df)