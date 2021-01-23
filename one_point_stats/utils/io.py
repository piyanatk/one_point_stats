import os
import pandas as pd

__all__ = ['check_dir', 'set2df']


def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def set2df(sets, column_names, index=None, sort=True):
    df = pd.DataFrame(list(sets), columns=column_names, index=index)
    if sort:
        df = df.sort(column_names)
        if index:
            df.index = index
        else:
            df.index = range(len(df))
    return df
