import pandas as pd
import os

def load_csv_into_dates(dirname: str,
                        start_date: str = '2014-01-01',
                        end_date: str = '2025-01-06',
                        column_name: str = None):
    """
    Takes one or more CSV files, and load them into the dates that you want.
    """
    df = pd.read_csv(dirname, header=0, index_col=0)
    df.index = pd.to_datetime(df.index)
    if column_name:
        df = df.loc[:, column_name]
    df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')).bfill()
    return df.loc[start_date:end_date]

def load_csv_and_glue_time_series(folder_path: str,
                                  column_name: str):
    """
    Takes all the .csv files from the folder in folder_path, loads and puts them all together.
    """
    first_iteration = True
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            dirname = os.path.join(folder_path, file)
            if first_iteration:
                df = load_csv_into_dates(dirname=dirname, 
                                         column_name=column_name)
                first_iteration = False
                continue
            df_next = load_csv_into_dates(dirname=dirname,
                                        start_date=df.index.max(),
                                        column_name=column_name)
            df = pd.concat([df, df_next])
            df = df[~df.index.duplicated(keep='last')]
    return df