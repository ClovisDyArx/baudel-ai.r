import pandas as pd
import re


def normalize_text(text):
    text = re.sub(r'\r\n|\r|\n', ' ', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def add_special_tokens(text):
    return "<|startoftext|>" + text + "<|endoftext|>"


def preprocess_df(df_poetry, df_love_poems):
    # normalize the content of the poems
    df_poetry['content'] = df_poetry['content'].apply(normalize_text)
    df_love_poems['poem'] = df_love_poems['poem'].apply(normalize_text)

    # concat the two dataframes
    df = pd.concat([df_poetry['content'], df_love_poems['poem']], ignore_index=True)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # add special tokens - GPT2
    df = df.apply(add_special_tokens)

    return df


class DatasetPoem:
    def __init__(self, poetry_path, love_poems_path):
        df_poetry = pd.read_csv(poetry_path)
        df_love_poems = pd.read_parquet(love_poems_path)

        self.df = preprocess_df(df_poetry, df_love_poems)

    def display_dataset(self):
        print("Poems Dataset : \n")
        print(self.df)

    def get_df(self):
        return self.df


if __name__ == "__main__":
    poetry_path = "hf://datasets/merve/poetry/poetry.csv"
    love_poems_path = "hf://datasets/asoria/love-poems/data/train-00000-of-00001.parquet"
    dataset = DatasetPoem(poetry_path, love_poems_path)

    df = dataset.get_df()
    print(df)
