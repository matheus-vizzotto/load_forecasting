import pandas as pd
import os

def train_test_split(df: pd.DataFrame, test: int, multivariate=False, y_col: str=None) -> pd.DataFrame:
    train = df.iloc[:-test,:]
    test = df.iloc[-test:,:]
    if multivariate:
        x_cols = [col for col in df.columns if col != y_col] 
        train_x = train[x_cols]
        train_y = train[y_col]
        test_x = test[x_cols]
        test_y = test[y_col]
        return train_x, train_y, test_x, test_y
    else:
        return train, test