import pandas as pd


def read_landmarks_from_parquet(pq_path: str):
    df = pd.read_parquet(pq_path)
    return df.loc[:, ('x', 'y', 'z')].astype('float32').values.reshape(-1, 543, 3)
