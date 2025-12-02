import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

"""
FNSPID STOCK PRICE PREPROCESSING
run wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip in the Data/FNSPID/Stock_price folder, then extract to get raw data (too big to keep)
"""

price_folder = "/global/cfs/cdirs/m4431/sp2160/FNSPID/Stock_price/full_history"
csv_files = glob(os.path.join(price_folder, "*.csv"))

print("Number of files found:", len(csv_files))

all_data = []
for file in csv_files:
    df = pd.read_csv(file)
    # Standardize column names (strip spaces, lower case)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    # Add ticker (from filename if desired)
    ticker = os.path.basename(file).replace(".csv", "")
    df['ticker'] = ticker
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)
print("Combined DataFrame head:")
display(combined_df.head())

combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
for col in numeric_cols:
    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

combined_df = combined_df.drop_duplicates().dropna(subset=['date', 'ticker'])
print("After cleaning:")
display(combined_df.head())

combined_df = combined_df.sort_values(['ticker', 'date'])
print("Final sorted DataFrame head:")
display(combined_df.head())

combined_df.to_csv("FNSPID/processed_stock_prices.csv", index=False)

"""
FNSPID NEWS HEADLINE PREPROCESSING
run wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv in the Data/FNSPID/Stock_news folder to get raw data (too big to keep)
"""

news_file = "FNSPID/Stock_news/nasdaq_exteral_data.csv"
news_df = pd.read_csv(news_file)
print("Initial headline data:")
display(news_df.head())

news_df['date'] = pd.to_datetime(news_df['Date'], errors='coerce')
print("Converted date column:")
display(news_df[['date', 'Article_title', 'Stock_symbol']].head())

output_df = news_df[['Date', 'Article_title', 'Stock_symbol']]
output_df.to_csv("FNSPID/processed_headlines_subset.csv", index=False)

# Stock Prices: Basic Info
print("Number of tickers:", combined_df['ticker'].nunique())
print("Date range:", combined_df['date'].min(), "to", combined_df['date'].max())
print("Total records:", combined_df.shape[0])
print("\nSample 10 tickers:", combined_df['ticker'].unique()[:10])

# Missing Value Check
print("Missing values per column:")
display(combined_df.isnull().sum())

# News Headlines: Basic Info
print("Number of unique stock symbols in headlines:", output_df['Stock_symbol'].nunique())
print("Headlines date range:", output_df['Date'].min(), "to", output_df['Date'].max())
print("Total headlines:", output_df.shape[0])

# Most frequent headline topics
print("Most common keywords in headlines:")
display(output_df['Article_title'].str.split().explode().value_counts().head(20))

# Price summary by ticker
price_stats = combined_df.groupby('ticker')[['open', 'close', 'adj_close', 'volume']].agg(['mean', 'std', 'min', 'max'])
display(price_stats.head())

# Daily return stats for one ticker
some_ticker = combined_df['ticker'].unique()[0]
ticker_df = combined_df[combined_df['ticker'] == some_ticker].sort_values('date')
ticker_df['daily_return'] = ticker_df['close'].pct_change()
print(f"Stats for {some_ticker}:")
display(ticker_df[['date', 'close', 'daily_return']].head())

# Join subset headlines with prices, for analysis
tmp_headlines = output_df.copy()
# Convert to datetime, then extract only the date part (drops time and timezone)
tmp_headlines['Date'] = pd.to_datetime(tmp_headlines['Date'], errors='coerce').dt.date
tmp_headlines['Stock_symbol'] = tmp_headlines['Stock_symbol'].astype(str)

tmp_prices = combined_df[['date', 'ticker', 'close']].copy()
# Convert to date only (without time or tz)
tmp_prices['date'] = pd.to_datetime(tmp_prices['date'], errors='coerce').dt.date
tmp_prices['ticker'] = tmp_prices['ticker'].astype(str)

# Now you can merge on pure date plus ticker
tmp_joined = pd.merge(
    tmp_headlines,
    tmp_prices,
    left_on=['Date', 'Stock_symbol'],
    right_on=['date', 'ticker'],
    how='left'
)

print("Headlines matched to closing prices (sample):")
display(tmp_joined[['Date', 'Stock_symbol', 'Article_title', 'close']].head(20))

# Number of headlines per year
output_df['year'] = pd.to_datetime(output_df['Date']).dt.year
headline_count_by_year = output_df.groupby('year').size()
print("Headline count per year:")
display(headline_count_by_year)

# Top stocks by headline volume
top_stock_symbols = output_df['Stock_symbol'].value_counts().head(20)
print("Top 20 stocks by headline count:")
display(top_stock_symbols)

# Distribution of headlines per ticker
output_df['Stock_symbol'].value_counts().head(20).plot(kind='bar')
plt.title("Top 20 Stocks by Headlines")
plt.show()

# Stock closing price for sample ticker over time
sample_ticker = combined_df['ticker'].unique()[0]
combined_df[combined_df['ticker'] == sample_ticker].plot(x='date', y='close', title=f"{sample_ticker} Closing Price")
plt.show()

