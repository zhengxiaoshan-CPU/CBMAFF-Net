# data_processing/utils.py

import pandas as pd

def read_excel(file, sheet_name, column_name):
    df = pd.read_excel(file, sheet_name)
    column_type = df[column_name]
    unique_values = column_type.unique()
    return pd.Series(unique_values)
