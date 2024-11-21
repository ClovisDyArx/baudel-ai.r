import pandas as pd


class DatasetPoem:
    def __init__(self, poetry_path, love_poems_path):
        df_poetry = pd.read_csv(poetry_path)
        df_love_poems = pd.read_parquet(love_poems_path)

        self.df = self.preprocess_df(df_poetry, df_love_poems)

    def display_dataset(self):
        print("Poems Dataset : \n")
        print(self.df)

    def get_df(self):
        return self.df

    def preprocess_df(self, df_poetry, df_love_poems):
        df = df_poetry['content']
        # concat the two dataframes
        df = pd.concat([df, df_love_poems['poem']])
        # drop duplicates
        df.drop_duplicates(inplace=True)
        return df


poetry_path = "hf://datasets/merve/poetry/poetry.csv"
love_poems_path = "hf://datasets/asoria/love-poems/data/train-00000-of-00001.parquet"
dataset = DatasetPoem(poetry_path, love_poems_path)

dataset.display_dataset()
