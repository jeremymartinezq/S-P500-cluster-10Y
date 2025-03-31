import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
import pandas as pd

# Function to calculate expected annualized return with projections
def calculate_annualized_return_with_projection(ticker, start_date, end_date, projection_end_date):
    # Download historical price data
    data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']

    if data.empty:
        print(f"No data for {ticker}")
        return np.nan

    # Prepare data for Prophet
    df = data.reset_index()
    df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})

    # Fit the Prophet model
    model = Prophet()
    model.fit(df)

    # Create future dataframe with projections
    future = model.make_future_dataframe(periods=(pd.to_datetime(projection_end_date) - pd.to_datetime(end_date)).days)
    forecast = model.predict(future)

    # Extract the projected data
    projected_data = forecast.set_index('ds')['yhat']

    # Combine historical and projected data
    combined_data = pd.concat([data, projected_data[projection_end_date:]], axis=0)

    # Calculate daily returns
    daily_returns = combined_data.pct_change().dropna()

    # Calculate the average daily return
    average_daily_return = daily_returns.mean()

    # Calculate the expected annualized return
    trading_days = 252  # Assuming 252 trading days in a year
    annualized_return = (1 + average_daily_return) ** trading_days - 1

    return annualized_return

# Parameters
start_date = "2020-01-01"  # Start date 5 years ago
end_date = "2024-12-31"  # End date one day before the current date
projection_end_date = "2030-12-31"  # End of projection period

# Fetch S&P 500 tickers and their sectors
def fetch_sp500_tickers_and_sectors():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_table = tables[0]
    tickers = sp500_table[['Symbol', 'Sector']]
    tickers.columns = ['Ticker', 'Sector']
    return tickers

# Fetch tickers and sectors
sp500_data = fetch_sp500_tickers_and_sectors()
sp500_sectors = sp500_data['Sector'].unique()

# Initialize lists to store annualized returns and sector information
all_annualized_returns = []
all_tickers = []
colors = []
sector_labels = []

# Assign colors to sectors
sector_colors = plt.cm.get_cmap('tab20', len(sp500_sectors))

# Calculate and store annualized returns for each sector
for sector in sp500_sectors:
    sector_tickers = sp500_data[sp500_data['Sector'] == sector]['Ticker'].tolist()
    sector_color = sector_colors(len(all_annualized_returns) % len(sp500_sectors))

    for ticker in sector_tickers:
        annualized_return = calculate_annualized_return_with_projection(ticker, start_date, end_date, projection_end_date)
        if not np.isnan(annualized_return):  # Check if the return is valid
            all_annualized_returns.append(annualized_return * 100)  # Convert to percentage
            all_tickers.append(ticker)
            colors.append(sector_color)
            sector_labels.append(sector)

# Plotting the results
x = np.arange(len(all_tickers))  # the label locations

plt.figure(figsize=(24, 12))  # Increased figure width for better visibility

# Plotting each sector with different colors
for sector in sp500_sectors:
    sector_indices = [i for i, label in enumerate(sector_labels) if label == sector]
    sector_returns = [all_annualized_returns[i] for i in sector_indices]
    sector_x = np.array(sector_indices)
    sector_color = sector_colors(sp500_sectors.tolist().index(sector))

    # Scatter plot
    plt.scatter(sector_x, sector_returns, color=sector_color, label=sector, s=100, edgecolor='k', alpha=0.7)

    # Line plot connecting the dots within each sector
    plt.plot(sector_x, sector_returns, color=sector_color, linestyle='-', linewidth=2)

# Adding a red dashed line at y = 20
plt.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Threshold Line (20%)')

# Adding title and labels
plt.title("Expected Annualized Returns for S&P 500 Stocks by Sector (2020-2030)", fontsize=18)
plt.xlabel("Company Ticker", fontsize=16)
plt.ylabel("Annualized Return (%)", fontsize=16)

# Set y-axis limits and ticks
min_value = -25  # Minimum limit for y-axis
max_value = max(all_annualized_returns)
y_padding = 10  # Additional space for better visibility
plt.ylim(min_value, max_value + y_padding)  # Adjust y-axis limits to include negative values

# Set y-axis ticks to go by 10% increments
plt.yticks(np.arange(min_value, max_value + y_padding + 10, 10))

# X-axis labels
plt.xticks(x, all_tickers, rotation=90, fontsize=8)  # Adjust font size for better readability

# Adding sector labels as legend
plt.legend(title="Sectors", bbox_to_anchor=(1.05, 1), loc='upper left')

# Adding grid
plt.grid(True)

# Adjust layout to fit everything properly
plt.subplots_adjust(bottom=0.25, right=0.85)  # Adjust margins for better fitting of labels and legend

# Show plot
plt.show()
