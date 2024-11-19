import numpy as np
import pandas as pd
import time
import yfinance as yf
from matplotlib import pyplot as plt
from yahooquery import Ticker


def get_companies():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)
    sp500_df = table[0]
    sp500_symbols = sp500_df['Symbol'].tolist()
    tickers = Ticker(sp500_symbols)
    profiles = tickers.summary_profile
    profiles_df = pd.DataFrame(profiles).T.reset_index()
    profiles_df.rename(columns={'index': 'symbol'}, inplace=True)

    return profiles_df

def filter_all_companies(companies, marketCap_thresh=2_000_000_000, averageVolume_thresh=5_000_000, start_date="2018-11-16", end_date="2022-11-16"):
    """
    Filters companies by market cap and average volume.
    Fetches historical closing prices for each filtered symbol.

    Parameters:
    - companies: DataFrame containing company data, including 'sector' and 'symbol'.
    - marketCap_thresh: Minimum market capitalization (default is 2 billion).
    - averageVolume_thresh: Minimum average volume (default is 5 million).
    - start_date: Start date for historical data.
    - end_date: End date for historical data.
    """
    # Get the list of all company symbols
    all_symbols = companies['symbol'].tolist()

    filtered_symbols = []
    for symbol in all_symbols:
        try:
            # Fetch ticker info with error handling
            ticker_info = yf.Ticker(symbol).info

            # Check if data exists and if marketCap and averageVolume thresholds are met
            market_cap = ticker_info.get("marketCap")
            average_volume = ticker_info.get("averageVolume")
            if market_cap is not None and market_cap > marketCap_thresh and \
                    average_volume is not None and average_volume > averageVolume_thresh:
                filtered_symbols.append(symbol)
        except Exception as e:
            # Print or log error message for the symbol
            print(f"Error fetching data for {symbol}: {e}")
        # Add delay to avoid rate limiting
        time.sleep(0.05)

    # Initialize an empty DataFrame to store the historical closing prices
    filtered_symbols_df = pd.DataFrame()

    # Fetch historical data for each filtered symbol
    for symbol in filtered_symbols:
        try:
            ticker = yf.Ticker(symbol)
            historical_data = ticker.history(start=start_date, end=end_date)
            # Store the 'Close' prices in the DataFrame
            filtered_symbols_df[symbol] = historical_data['Close']
        except Exception as e:
            # Print or log error message for the symbol if historical data cannot be fetched
            print(f"Error fetching historical data for {symbol}: {e}")
        # Add delay to avoid rate limiting
        time.sleep(0.05)

    return filtered_symbols_df
