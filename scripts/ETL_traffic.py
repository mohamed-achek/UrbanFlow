import pandas as pd
import os

# Load raw data
file_path = os.path.join("..", "data", "raw", "traffic", "Automated_Traffic_Volume_Counts_20250715.csv")
df = pd.read_csv(file_path)

print(df.head(10))



# Drop rows with missing location
#df = df.dropna(subset=['Latitude', 'Longitude'])