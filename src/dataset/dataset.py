from torch.utils.data import Dataset
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
    df = pd.concat([df_poetry['content'], df_love_poems['poem']], ignore_index=True).to_frame(name='poem')

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # add special tokens - GPT2
    df = df.apply(add_special_tokens)

    return df


# Dataset class : Raw data -> Dataframe
class DatasetPoem:
    def __init__(self, poetry_path="hf://datasets/merve/poetry/poetry.csv", love_poems_path="hf://datasets/asoria/love-poems/data/train-00000-of-00001.parquet"):
        df_poetry = pd.read_csv(poetry_path)
        df_love_poems = pd.read_parquet(love_poems_path)

        self.df = preprocess_df(df_poetry, df_love_poems)

    def display_dataset(self):
        print("Poems Dataset : \n")
        print(self.df)

    def get_df(self):
        return self.df

    def tokenize(self, tokenizer):
        text_data = self.df['poem'].tolist()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenized_data = tokenizer(text_data, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        return tokenized_data


# Dataset class : tokenized data -> PyTorch Dataset
class PoemsDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx],  # For language modeling, labels are the same as input_ids
        }
