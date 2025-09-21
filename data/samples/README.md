# Sample Datasets

This directory contains sample datasets used for demonstration purposes in the UrbanFlow AI application.

## 📊 Sample Files

| File | Original Size | Sample Size | Description |
|------|---------------|-------------|-------------|
| `merged_dataset_sample.csv` | ~1GB | 0.81MB | Combined bike, weather, and traffic data |
| `citibike_sample.csv` | ~9GB | 1.40MB | Citibike trip data with features |
| `traffic_sample.csv` | ~29MB | 2.06MB | NYC traffic volume data |
| `weather_sample.csv` | ~1.5MB | 0.88MB | Weather data with engineering features |

## 🗂️ Why Sample Data?

The original datasets are too large for GitHub hosting:
- **Original merged dataset**: ~1GB with 8+ million records
- **Original Citibike data**: ~9GB with 40+ million records
- **GitHub file size limit**: 100MB per file

## 📈 Sample Methodology

- **Sample size**: 5,000 randomly selected rows per dataset
- **Random seed**: 42 (for reproducibility)
- **Maintains**: All original columns and data relationships
- **Preserves**: Statistical distributions and patterns
