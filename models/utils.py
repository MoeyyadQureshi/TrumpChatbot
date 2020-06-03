from pathlib import Path

import pandas as pd


DATA_PATH = Path("../data/realdonaldtrump.csv")


def trump_tweets() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)
