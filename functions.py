import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import yfinance as yf
from matplotlib import pyplot as plt
from yahooquery import Ticker
from scipy.stats import norm

r= 0.03

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


def filter_all_companies(companies, marketCap_thresh=50_000_000_000, averageVolume_thresh=10_000_000, start_date="2022-11-16", end_date="2024-11-16"):
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


    return filtered_symbols_df


def calculate_time_to_expiration(expiration_date_str: str, reference_date_str: str) -> float:
    """
    Calculate the time to expiration in years from a specific reference date.

    Parameters:
    expiration_date_str (str): Expiration date in the format 'YYYY-MM-DD'
    reference_date_str (str): Reference date in the format 'YYYY-MM-DD'

    Returns:
    float: Time to expiration in years (non-negative)
    """
    # Parse the expiration date and reference date strings to datetime objects
    expiration_date = datetime.strptime(expiration_date_str, "%Y-%m-%d")
    reference_date = datetime.strptime(reference_date_str, "%Y-%m-%d")

    # Calculate the time difference in seconds
    time_diff_seconds = (expiration_date - reference_date).total_seconds()

    # Ensure time difference is non-negative
    if time_diff_seconds <= 0:
        # Option has expired or expires on the reference date; set T to a minimal positive value or filter out
        T = 1 / 365.0  # One day in years
    else:
        # Convert seconds to years
        T = time_diff_seconds / (365.0 * 24 * 60 * 60)  # Total seconds in a year

    return T


def calculate_time_to_expiration_from_today(expiration_date_str: str) -> float:
    """
    Calculate the time to expiration in years from today.

    Parameters:
    expiration_date_str (str): Expiration date in the format 'YYYY-MM-DD'

    Returns:
    float: Time to expiration in years (non-negative)
    """
    # Parse the expiration date string to a datetime object
    expiration_date = datetime.strptime(expiration_date_str, "%Y-%m-%d")

    # Get the current datetime
    current_datetime = datetime.now()

    # Calculate the time difference in seconds
    time_diff_seconds = (expiration_date - current_datetime).total_seconds()

    # Ensure time difference is non-negative
    if time_diff_seconds <= 0:
        # Option has expired or expires today; set T to a minimal positive value or filter out
        T = 1 / 365.0  # One day in years
    else:
        # Convert seconds to years
        T = time_diff_seconds / (365.0 * 24 * 60 * 60)  # Total seconds in a year

    return T



def calculate_time_to_expiration_from_future(df, date_column, days_from_now):
    """
    Calculate the time to expiration in years for a column of dates in a DataFrame,
    considering today + a specified number of days as the reference date.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the date column.
    - date_column (str): The name of the column with expiration dates (format 'YYYY-MM-DD').
    - days_from_now (int): Number of days from today to consider as the reference date.

    Returns:
    - pd.Series: A Series with the time to expiration in years (non-negative).
    """
    # Convert the date column to datetime if it's not already
    df[date_column] = pd.to_datetime(df[date_column])

    # Calculate the future reference date
    future_reference_date = datetime.now() + timedelta(days=days_from_now)

    # Calculate the time to expiration in seconds
    time_diff_seconds = (df[date_column] - future_reference_date).dt.total_seconds()

    # Convert seconds to years and ensure non-negative values
    time_to_expiration = time_diff_seconds.clip(lower=0) / (365.0 * 24 * 60 * 60)

    # Handle cases where the expiration is on or before the reference date
    minimal_value = 1 / 365.0  # One day in years
    time_to_expiration = time_to_expiration.where(time_to_expiration > 0, minimal_value)

    return time_to_expiration

def get_option_chains_spot(ticker_symbol, retries=3, delay=2, start_date=None, end_date=None):
    """
    Fetch option chains and spot price for a given ticker within a specific historical period.

    Parameters:
    ticker_symbol (str): The ticker symbol of the asset.
    retries (int): Number of retry attempts for fetching data.
    delay (int): Delay between retries in seconds.
    start_date (str): Start date for historical spot price in 'YYYY-MM-DD' format. Defaults to None.
    end_date (str): End date for historical spot price in 'YYYY-MM-DD' format. Defaults to None.

    Returns:
    tuple: A tuple containing calls DataFrame, puts DataFrame, and spot price.
    """
    # Fetch the ticker data
    ticker = yf.Ticker(ticker_symbol)
    for attempt in range(retries):
        try:
            # Get the historical spot price
            if start_date and end_date:
                # Fetch spot price for the given date range
                history = ticker.history(start=start_date, end=end_date)
            else:
                # Default to 1-day historical data
                history = ticker.history(period="1d")

            # Check if the history DataFrame is empty
            if history.empty:
                raise ValueError(f"No historical data available for ticker {ticker_symbol}.")

            # Get the spot price from the history DataFrame (use the first available price)
            spot_price = history["Close"].iloc[0]

            # Get expiration dates
            expiration_dates = ticker.options  # Expiration dates

            # Fetch call and put options for each expiration date
            calls_dict = {date: ticker.option_chain(date).calls for date in expiration_dates}
            puts_dict = {date: ticker.option_chain(date).puts for date in expiration_dates}

            # Add expiration column to each DataFrame in calls_dict and puts_dict
            for date, df in calls_dict.items():
                df['expiration'] = date

            for date, df in puts_dict.items():
                df['expiration'] = date

            # Concatenate all DataFrames from calls_dict and puts_dict
            calls_all = pd.concat(calls_dict.values(), ignore_index=True)
            puts_all = pd.concat(puts_dict.values(), ignore_index=True)

            # For calls_all DataFrame
            calls_all = calls_all[["strike", "lastPrice", "impliedVolatility", "expiration"]]
            calls_all["time_to_expiration"] = calls_all["expiration"].apply(
                lambda x: calculate_time_to_expiration(x, history.index[0].strftime('%Y-%m-%d'))
            )
            calls_all = calls_all[calls_all["time_to_expiration"] > 0.0]
            calls_all = calls_all.reset_index(drop=True)

            # For puts_all DataFrame
            puts_all = puts_all[["strike", "lastPrice", "impliedVolatility", "expiration"]]
            puts_all["time_to_expiration"] = puts_all["expiration"].apply(
                lambda x: calculate_time_to_expiration(x, history.index[0].strftime('%Y-%m-%d'))
            )
            puts_all = puts_all[puts_all["time_to_expiration"] > 0.0]
            puts_all = puts_all.reset_index(drop=True)

            # If successful, return the data
            return calls_all, puts_all, spot_price

        except (IndexError, ValueError) as e:
            # Print a warning and retry after a delay
            print(f"Attempt {attempt + 1} failed with error: {e} - Retrying after {delay} seconds...")
            time.sleep(delay)

    # If all retries fail, raise an error
    raise ValueError(f"Failed to get spot price and options data for ticker {ticker_symbol} after {retries} attempts.")

    
    
def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European call option.
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to maturity (in years)
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset
    
    Returns:
    float: Call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European put option.
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to maturity (in years)
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset
    
    Returns:
    float: Put option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def delta_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def delta_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def theta_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    return theta

def theta_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return theta

def rho_call(S, K, T, r, sigma):
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return K * T * np.exp(-r * T) * norm.cdf(d2)

def rho_put(S, K, T, r, sigma):
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)


class OptionsPortfolioOptimizer:
    def __init__(self, options_df, budget, alpha=1.0):
        """
        Initialize the optimizer with the options DataFrame and budget.
        """
        self.options_df = options_df

        # Calculate price difference and expected return
        self.budget = budget
  

    def budget_constraint(self, weights):
        """
        Ensure total cost does not exceed the budget.
        """
        return self.budget - np.sum(weights * self.options_df['market_price'] * 100)

    def delta_neutral_constraint(self, weights):
        """
        Ensure the portfolio is delta neutral.
        """
        return np.sum(weights * self.options_df['delta'])


    def objective_function(self, weights):
        """
        Maximize expected returns.
        """
        return np.sum(weights * self.options_df['expected_return'])
         

def adjust_weights_to_constraints(optimized_portfolio, budget, vega_min, vega_max, theta_min, theta_max, delta_tolerance=1e-5):
    """
    Adjust integer weights in the portfolio to satisfy constraints after rounding.
    """
    # Extract portfolio metrics
    total_cost = (optimized_portfolio['market_price'] * optimized_portfolio['optimized_N'] * 100).sum()
    total_delta = (optimized_portfolio['delta'] * optimized_portfolio['optimized_N'] * 100).sum()
    total_vega = (optimized_portfolio['vega'] * optimized_portfolio['optimized_N'] * 100).sum()
    total_theta = (optimized_portfolio['theta'] * optimized_portfolio['optimized_N'] * 100).sum()

    # Iterate through rows for adjustment
    for index, row in optimized_portfolio.iterrows():
        current_weight = optimized_portfolio.at[index, 'optimized_N']

        if abs(total_delta) > delta_tolerance:
            # Adjust for delta neutrality
            adjustment = -total_delta / (row['delta'] * 100) if row['delta'] != 0 else 0
            adjusted_weight = max(1, current_weight + round(adjustment))  # Keep at least 1 contract
            optimized_portfolio.at[index, 'optimized_N'] = adjusted_weight

        if total_vega < vega_min or total_vega > vega_max:
            # Adjust for vega limits
            adjustment = (vega_min - total_vega) / (row['vega'] * 100) if total_vega < vega_min else (vega_max - total_vega) / (row['vega'] * 100)
            adjusted_weight = max(1, current_weight + round(adjustment))  # Keep at least 1 contract
            optimized_portfolio.at[index, 'optimized_N'] = adjusted_weight

        if total_theta < theta_min or total_theta > theta_max:
            # Adjust for theta limits
            adjustment = (theta_min - total_theta) / (row['theta'] * 100) if total_theta < theta_min else (theta_max - total_theta) / (row['theta'] * 100)
            adjusted_weight = max(1, current_weight + round(adjustment))  # Keep at least 1 contract
            optimized_portfolio.at[index, 'optimized_N'] = adjusted_weight

        # Recalculate metrics after adjustment
        total_cost = (optimized_portfolio['market_price'] * optimized_portfolio['optimized_N'] * 100).sum()
        total_delta = (optimized_portfolio['delta'] * optimized_portfolio['optimized_N'] * 100).sum()
        total_vega = (optimized_portfolio['vega'] * optimized_portfolio['optimized_N'] * 100).sum()
        total_theta = (optimized_portfolio['theta'] * optimized_portfolio['optimized_N'] * 100).sum()

        # Stop adjustments if all constraints are satisfied
        if abs(total_delta) <= delta_tolerance and vega_min <= total_vega <= vega_max and theta_min <= total_theta <= theta_max:
            break

    # Ensure budget constraint is still satisfied
    if total_cost > budget:
        print("Warning: Budget constraint violated after adjustments!")
    
    return optimized_portfolio
                      

def process_portfolio(portfolio_df):
    """
    Processes a portfolio DataFrame and calculates Black-Scholes prices, greeks, and expected returns.

    Parameters:
    - portfolio_df (pd.DataFrame): The DataFrame containing portfolio data. Must include columns:
        ["spot_price", "strike", "time_to_expiration", "impliedVolatility", "market_price", "type"].
    - r (float): The risk-free rate.

    Returns:
    - pd.DataFrame: The updated DataFrame with new columns for calculated values.
    """
    for i in range(len(portfolio_df)):
        # Extracting row values
        spot_price = portfolio_df.loc[i, "spot_price"]
        strike_price = portfolio_df.loc[i, "strike"]
        time_to_expiration = portfolio_df.loc[i, "time_to_expiration"]
        impliedVolatility = portfolio_df.loc[i, "impliedVolatility"]
        market_price = portfolio_df.loc[i, "market_price"]

        # Determine if the option is a call or put and calculate accordingly
        if portfolio_df.loc[i, "type"] == "call":
            bs_price = black_scholes_call(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            portfolio_df.loc[i, "blackScholes_Price"] = bs_price
            portfolio_df.loc[i, "expected_return"] = market_price - bs_price
            portfolio_df.loc[i, "delta"] = delta_call(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            portfolio_df.loc[i, "gamma"] = gamma(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            portfolio_df.loc[i, "vega"] = vega(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            portfolio_df.loc[i, "theta_call"] = theta_call(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            portfolio_df.loc[i, "rho"] = rho_call(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)

        elif portfolio_df.loc[i, "type"] == "put":
            bs_price = black_scholes_put(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            portfolio_df.loc[i, "blackScholes_Price"] = bs_price
            portfolio_df.loc[i, "expected_return"] = market_price - bs_price
            portfolio_df.loc[i, "delta"] = delta_put(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            portfolio_df.loc[i, "gamma"] = gamma(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            portfolio_df.loc[i, "vega"] = vega(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            portfolio_df.loc[i, "theta_call"] = theta_put(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            portfolio_df.loc[i, "rho"] = rho_put(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)

    return portfolio_df


def compute_portfolio_stats(portfolio_df):
    """
    Computes portfolio-level totals for Greeks and expected returns.

    Parameters:
    - portfolio_df (pd.DataFrame): The DataFrame containing portfolio data. Must include columns:
        ["delta", "vega", "gamma", "theta", "rho", "expected_return", "optimized_N"].

    Returns:
    - pd.DataFrame: A single-row DataFrame with aggregated portfolio-level results.
    """
    # List of Greeks to process
    greeks = ["delta", "vega", "gamma", "theta", "rho"]
    
    # Initialize a dictionary to store portfolio totals
    portfolio_data = {}
    
    # Compute the weighted totals for each Greek
    for greek in greeks:
        portfolio_data[f"total_{greek}"] = (portfolio_df[greek] * portfolio_df["optimized_N"]*100).sum()

    # Compute the weighted total expected return
    portfolio_data["total_expected_return"] = (portfolio_df["expected_return"] * portfolio_df["optimized_N"]*100).sum()

    # Convert to DataFrame
    portfolio_summary_df = pd.DataFrame([portfolio_data])
    
    return portfolio_summary_df


def increment_dates(dataframe, date_column, days_increment):
    """
    Increments dates in a specified column by a given number of days.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame.
    - date_column (str): The column name containing dates as strings or datetime objects.
    - days_increment (int): Number of days to increment (can be negative for decrement).

    Returns:
    - pd.DataFrame: The DataFrame with updated dates.
    """
    # Convert the column to datetime if it's not already
    dataframe[date_column] = pd.to_datetime(dataframe[date_column])
    
    # Increment the dates by the specified number of days
    dataframe[date_column] += timedelta(days=days_increment)
    
    return dataframe
