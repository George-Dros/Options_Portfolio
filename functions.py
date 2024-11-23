import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import yfinance as yf
from matplotlib import pyplot as plt
from yahooquery import Ticker
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import plotly.graph_objects as go

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


def update_time_to_expiration(df, expiration_column, time_to_expiration_column, days_from_now):
    """
    Updates the 'time_to_expiration' column in a DataFrame by recalculating it
    from the 'expiration' column based on today + a specified number of days.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - expiration_column (str): The name of the column with expiration dates (format 'YYYY-MM-DD').
    - time_to_expiration_column (str): The name of the column to update with recalculated values.
    - days_from_now (int): Number of days from today to consider as the reference date.

    Returns:
    - pd.DataFrame: The updated DataFrame with modified 'time_to_expiration'.
    """
    # Convert the expiration column to datetime if it's not already
    df[expiration_column] = pd.to_datetime(df[expiration_column])

    # Calculate the future reference date
    future_reference_date = datetime.now() + timedelta(days=int(days_from_now))

    # Calculate time to expiration in seconds
    time_diff_seconds = (df[expiration_column] - future_reference_date).dt.total_seconds()

    # Convert seconds to years and ensure non-negative values
    time_to_expiration = time_diff_seconds.clip(lower=0) / (365.0 * 24 * 60 * 60)

    # Handle cases where the expiration is on or before the reference date
    minimal_value = 1 / 365.0  # One day in years
    df[time_to_expiration_column] = time_to_expiration.where(time_to_expiration > 0, minimal_value)

    return df



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


def analyze_spot_price_impact(portfolio_df, spot_price_multipliers, process_portfolio_func, compute_stats_func):
    """
    Analyzes the impact of spot price changes on portfolio metrics.

    Parameters:
    - portfolio_df (pd.DataFrame): The initial portfolio DataFrame.
    - spot_price_multipliers (list): A list of multipliers to adjust the spot price.
    - process_portfolio_func (function): Function to process the portfolio (e.g., compute Greeks and returns).
    - compute_stats_func (function): Function to compute portfolio-level statistics.

    Returns:
    - pd.DataFrame: A DataFrame containing aggregated portfolio stats for each spot price multiplier.
    """
    # Store the initial version of the portfolio
    initial_portfolio_df = portfolio_df.copy()

    # List to collect results
    results = []

    for multiplier in spot_price_multipliers:
        # Debug: Log current multiplier
        print(f"Analyzing spot price multiplier: {multiplier}")

        # Adjust the spot price
        adjusted_df = initial_portfolio_df.copy()
        adjusted_df.loc[:, "spot_price"] = adjusted_df["spot_price"] * multiplier

        # Debug: Confirm adjusted spot prices
        print(f"Adjusted spot prices:\n{adjusted_df['spot_price']}")

        # Process the portfolio to compute individual option metrics
        processed_df = process_portfolio_func(adjusted_df)

        # Debug: Confirm processed DataFrame
        print(f"Processed DataFrame:\n{processed_df.head()}")

        # Compute portfolio-level statistics
        portfolio_stats = compute_stats_func(processed_df)

        # Add the multiplier to the stats
        portfolio_stats["spot_price_multiplier"] = multiplier

        # Append the stats to the results
        results.append(portfolio_stats)

    # Combine all results into a single DataFrame
    results_df = pd.concat(results, ignore_index=True)

    return results_df






def analyze_implied_volatility_impact(portfolio_df, volatility_multipliers, process_portfolio_func, compute_stats_func):
    """
    Analyzes the impact of implied volatility changes on portfolio metrics by iterating over a set of multipliers.

    Parameters:
    - portfolio_df (pd.DataFrame): The initial portfolio DataFrame.
    - volatility_multipliers (list): A list of multipliers to adjust the implied volatility.
    - process_portfolio_func (function): Function to process the portfolio (e.g., compute Greeks and returns).
    - compute_stats_func (function): Function to compute portfolio-level statistics.

    Returns:
    - pd.DataFrame: A DataFrame containing portfolio stats for each implied volatility multiplier.
    """
    # Store the initial version of the portfolio (safe copy)
    initial_portfolio_df = portfolio_df.copy(deep=True)

    # List to collect results
    results = []

    # Loop through each implied volatility multiplier
    for multiplier in volatility_multipliers:
        # Create a temporary copy of the portfolio for this multiplier
        temp_portfolio_df = initial_portfolio_df.copy()

        # Adjust the implied volatility
        temp_portfolio_df["impliedVolatility"] = temp_portfolio_df["impliedVolatility"] * multiplier

        # Process the portfolio to compute individual option metrics
        processed_df = process_portfolio_func(temp_portfolio_df)

        # Compute portfolio-level statistics
        portfolio_stats = compute_stats_func(processed_df)

        # Add the multiplier to the stats
        portfolio_stats["implied_volatility_multiplier"] = multiplier

        # Append the stats to the results
        results.append(portfolio_stats)

    # Combine all results into a single DataFrame
    results_df = pd.concat(results, ignore_index=True)

    return results_df


def analyze_time_passage_impact(portfolio_df, time_increments, update_time_func, process_portfolio_func, compute_stats_func):
    """
    Analyzes the impact of time passage on portfolio metrics by iterating over a set of time increments.

    Parameters:
    - portfolio_df (pd.DataFrame): The initial portfolio DataFrame.
    - time_increments (list): A list of time increments in days to adjust the time to expiration.
    - update_time_func (function): Function to update time to expiration.
    - process_portfolio_func (function): Function to process the portfolio (e.g., compute Greeks and returns).
    - compute_stats_func (function): Function to compute portfolio-level statistics.

    Returns:
    - pd.DataFrame: A DataFrame containing portfolio stats for each time increment.
    """
    # Store the initial version of the portfolio (safe copy)
    initial_portfolio_df = portfolio_df.copy(deep=True)

    # List to collect results
    results = []

    # Loop through each time increment
    for days in time_increments:
        # Create a temporary copy of the portfolio for this time increment
        temp_portfolio_df = initial_portfolio_df.copy()

        # Update time to expiration
        temp_portfolio_df = update_time_func(
            df=temp_portfolio_df,
            expiration_column="expiration",
            time_to_expiration_column="time_to_expiration",
            days_from_now=days
        )

        # Process the portfolio to compute individual option metrics
        processed_df = process_portfolio_func(temp_portfolio_df)

        # Compute portfolio-level statistics
        portfolio_stats = compute_stats_func(processed_df)

        # Add the time increment to the stats
        portfolio_stats["days_passed"] = days

        # Append the stats to the results
        results.append(portfolio_stats)

    # Combine all results into a single DataFrame
    results_df = pd.concat(results, ignore_index=True)

    return results_df



def analyze_combined_impact(
    portfolio_df,
    spot_price_multipliers,
    volatility_multipliers,
    time_increments,
    update_time_func,
    process_portfolio_func,
    compute_stats_func
):
    """
    Combines the impact analysis of spot price changes, implied volatility changes,
    and time passage on portfolio metrics.

    Parameters:
    - portfolio_df (pd.DataFrame): The initial portfolio DataFrame.
    - spot_price_multipliers (list): A list of multipliers to adjust the spot price.
    - volatility_multipliers (list): A list of multipliers to adjust the implied volatility.
    - time_increments (list): A list of time increments in days to adjust the time to expiration.
    - update_time_func (function): Function to update time to expiration.
    - process_portfolio_func (function): Function to process the portfolio (e.g., compute Greeks and returns).
    - compute_stats_func (function): Function to compute portfolio-level statistics.

    Returns:
    - pd.DataFrame: A DataFrame containing portfolio stats for all combinations of spot price, implied volatility, and time increments.
    """
    # Store the initial version of the portfolio (safe copy)
    initial_portfolio_df = portfolio_df.copy(deep=True)

    # List to collect results
    results = []

    # Loop through each combination of spot price, implied volatility, and time increment
    for spot_multiplier in spot_price_multipliers:
        for vol_multiplier in volatility_multipliers:
            for days in time_increments:
                # Create a temporary copy of the portfolio for this combination
                temp_portfolio_df = initial_portfolio_df.copy()

                # Adjust spot price
                temp_portfolio_df["spot_price"] = temp_portfolio_df["spot_price"] * spot_multiplier

                # Adjust implied volatility
                temp_portfolio_df["impliedVolatility"] = temp_portfolio_df["impliedVolatility"] * vol_multiplier

                # Update time to expiration
                temp_portfolio_df = update_time_func(
                    df=temp_portfolio_df,
                    expiration_column="expiration",
                    time_to_expiration_column="time_to_expiration",
                    days_from_now=days
                )

                # Process the portfolio to compute individual option metrics
                processed_df = process_portfolio_func(temp_portfolio_df)

                # Compute portfolio-level statistics
                portfolio_stats = compute_stats_func(processed_df)

                # Add the combination parameters to the stats
                portfolio_stats["spot_price_multiplier"] = spot_multiplier
                portfolio_stats["implied_volatility_multiplier"] = vol_multiplier
                portfolio_stats["days_passed"] = days

                # Append the stats to the results
                results.append(portfolio_stats)

    # Combine all results into a single DataFrame
    results_df = pd.concat(results, ignore_index=True)

    return results_df


def process_portfolio(portfolio_df):
    """
    Processes a portfolio DataFrame and calculates Black-Scholes prices, greeks, and expected returns.

    Parameters:
    - portfolio_df (pd.DataFrame): The DataFrame containing portfolio data. Must include columns:
        ["spot_price", "strike", "time_to_expiration", "impliedVolatility", "market_price", "type"].

    Returns:
    - pd.DataFrame: A new DataFrame with updated columns for calculated values.
    """
    def calculate_metrics(row):
        # Extract row values
        spot_price = row["spot_price"]
        strike_price = row["strike"]
        time_to_expiration = row["time_to_expiration"]
        impliedVolatility = row["impliedVolatility"]
        market_price = row["market_price"]

        # Initialize results dictionary
        results = {}

        if row["type"] == "call":
            results["blackScholes_Price"] = black_scholes_call(
                S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility
            )
            results["expected_return"] = results["blackScholes_Price"] - market_price
            results["delta"] = delta_call(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            results["gamma"] = gamma(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            results["vega"] = vega(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            results["theta"] = theta_call(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            results["rho"] = rho_call(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)

        elif row["type"] == "put":
            results["blackScholes_Price"] = black_scholes_put(
                S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility
            )
            results["expected_return"] = results["blackScholes_Price"] - market_price
            results["delta"] = delta_put(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            results["gamma"] = gamma(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            results["vega"] = vega(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            results["theta"] = theta_put(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)
            results["rho"] = rho_put(S=spot_price, K=strike_price, T=time_to_expiration, r=r, sigma=impliedVolatility)

        return pd.Series(results)

    # Apply the calculation to each row
    updated_metrics = portfolio_df.apply(calculate_metrics, axis=1)

    # Overwrite existing columns instead of creating duplicates
    for col in updated_metrics.columns:
        portfolio_df[col] = updated_metrics[col]

    return portfolio_df



def compute_portfolio_stats(processed_df):
    """
    Computes portfolio-level totals for Greeks and other key metrics.

    Parameters:
    - processed_df (pd.DataFrame): The processed portfolio DataFrame.

    Returns:
    - pd.DataFrame: A single-row DataFrame with total portfolio-level metrics.
    """
    # Columns to aggregate and their corresponding output keys
    metrics_to_aggregate = {
        "delta": "total_delta",
        "vega": "total_vega",
        "gamma": "total_gamma",
        "theta": "total_theta",
        "rho": "total_rho",
        "expected_return": "total_expected_return",
        "market_price": "total_cost",
    }

    # Initialize stats dictionary
    stats = {}

    for column, total_key in metrics_to_aggregate.items():
        if column in processed_df.columns:
            processed_df[column] = pd.to_numeric(processed_df[column], errors="coerce").fillna(0)
            stats[total_key] = (processed_df[column] * processed_df["optimized_N"] * 100).sum()
        else:
            stats[total_key] = 0  # Default to 0 if the column is missing

    # Convert stats dictionary to a DataFrame
    return pd.DataFrame([stats])




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

def plot_interactive_3d_surface(data, x_col, y_col, z_col):
    """
    Creates an interactive 3D surface plot of portfolio analysis data using Plotly.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the analysis data.
    - x_col (str): The column name to use for the X-axis.
    - y_col (str): The column name to use for the Y-axis.
    - z_col (str): The column name to use for the Z-axis.
    """
    # Extract the data for plotting
    x = data[x_col]
    y = data[y_col]
    z = data[z_col]

    # Create a grid for smoother plotting
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the Z values over the grid
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    # Create the interactive 3D surface plot
    fig = go.Figure(data=[go.Surface(z=zi, x=xi[0], y=yi[:, 0], colorscale='Viridis')])

    # Update layout for better visualization
    fig.update_layout(
        title=f'Interactive 3D Surface Plot of {z_col} vs {x_col} and {y_col}',
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Show the plot
    fig.show()

