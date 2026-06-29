import pandas as pd
import os

DATA_PATH = "data/"
FILENAME = "HI-Small_Trans.csv"


def get_filepath():
    return os.path.join(DATA_PATH, FILENAME)


def load_transactions(chunksize=500):
    filepath = get_filepath()

    if not os.path.exists(filepath):
        print("Data file not found at: " + filepath)
        return

    for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunksize)):
        chunk.columns = [c.strip().lower().replace(" ", "_") for c in chunk.columns]
        chunk = chunk.rename(
            columns={"account.1": "to_account", "account": "from_account"}
        )
        yield i + 1, chunk


def preview_data():
    filepath = get_filepath()

    if not os.path.exists(filepath):
        print("Data file not found.")
        return

    df = pd.read_csv(filepath, nrows=5)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"account.1": "to_account", "account": "from_account"})

    print("Columns: " + str(list(df.columns)))
    print("Shape preview: 5 rows x " + str(len(df.columns)) + " cols")
    print("")
    for _, row in df.iterrows():
        print(dict(row))
        print("")


if __name__ == "__main__":
    preview_data()
