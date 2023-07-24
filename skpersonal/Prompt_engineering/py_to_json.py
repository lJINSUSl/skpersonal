import pandas as pd

df = pd.read_csv('twcs_processed.csv')
df.to_json('twcs_processed.json', orient='records')