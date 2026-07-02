import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/wessamsw/Airline_Passenger_Satisfaction/main/airline_passenger_satisfaction.csv"

def load_data():
    """Load the airline passenger satisfaction dataset."""
    return pd.read_csv(DATA_URL)

def check_missing_values(df):
    """Return columns with missing values and their counts."""
    missing = df.isnull().sum()
    return missing[missing > 0]

def check_duplicates(df):
    """Return the number of fully duplicated rows."""
    return df.duplicated().sum()

def check_value_range(df, column, min_val, max_val):
    """Return rows where column value falls outside expected range."""
    return df[(df[column] < min_val) | (df[column] > max_val)]
