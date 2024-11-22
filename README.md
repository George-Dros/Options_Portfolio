# Options Portfolio

## Project Overview

This project implements an Options Portfolio Management and Risk Analysis Tool using Python. It calculates options prices using the Black-Scholes model, computes the Greeks, and performs comprehensive risk analysis. The tool supports both real and simulated market data, visualizes results, and offers extensible features for advanced users.

## Features

- Option Pricing: Computes call and put prices using the Black-Scholes model.
- Greeks Calculation: Computes sensitivities (Delta, Gamma, Vega, Theta, Rho) analytically or numerically.
- Portfolio Management: Supports construction and valuation of options portfolios.
- Risk Analysis: Performs scenario and sensitivity analysis and explores hedging strategies.
- Visualization: Generates interactive graphs and dashboards for insights.
- Backtesting: Validates portfolio performance using historical data.
- Extensibility: Supports implied volatility calculations, American options, Monte Carlo simulations, and machine learning models.

## Installation

1. Clone the repository:

```
bash
git clone https://github.com/your-username/options-portfolio.git
cd options-portfolio
```
2. Install dependencies:

```
bash
pip install -r requirements.txt
```

## Project Structure

```
bash
options-portfolio/
│
├── data/                  # Market data files (if using offline data)
├── functions.py           # Core functions for pricing and Greeks
├── portfolio_config.py    # Configuration for portfolio
├── main.py                # Main script for execution
├── requirements.txt       # Python dependencies
├── visualizations/        # Output graphs and dashboards
├── README.md              # Project documentation
└── testing.py             # Unit tests and validation scripts
```

## Features in Detail

### 1. Option Pricing

- Implements the Black-Scholes model for European call and put options.
- Supports validation of prices with real market data.

### 2. Greeks Calculation

- Computes Delta, Gamma, Vega, Theta, and Rho analytically.
- Visualizes Greeks to show sensitivity to asset prices and time.

### 3. Portfolio Management and Valuation

- Creates an options portfolio with customizable positions.
- Computes total portfolio value and Greeks.

### 4. Risk Analysis

- Scenario analysis to model responses to market changes.
- Implements hedging strategies for Delta-neutral or Gamma-neutral positions.

### 5. Visualization and Reporting

- Generates interactive dashboards using Plotly or Bokeh.
- Summarizes key metrics and performance.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

