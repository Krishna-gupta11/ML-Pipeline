import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):

    df = df.dropna()

    df = df[df['fare_amount'] > 0]
    df = df[df['passenger_count'] > 0]

    return df