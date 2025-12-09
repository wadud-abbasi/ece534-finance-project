import pandas as pd
import pyarrow.parquet as pq

headlines = pd.read_parquet("../data/data/merged_lstm_dataset_nickel.parquet")
print(headlines.head())

print(headlines[headlines["sent_pos"] > 0].head())