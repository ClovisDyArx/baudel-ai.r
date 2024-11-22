from torch.utils.data import Dataset
import pandas as pd
import re
import kagglehub
import os

# dataset paths
poetry_path = "hf://datasets/merve/poetry/poetry.csv"
love_poems_path = "hf://datasets/asoria/love-poems/data/train-00000-of-00001.parquet"
poem_classification_path = "ramjasmaurya/poem-classification-nlp"
poetryfoundationorg_path = "johnhallman/complete-poetryfoundationorg-dataset"
poetry_foundation_path = "tgdivy/poetry-foundation-poems"


def list_files(directory):
    files = []
    for root, dirs, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    return files


def load_kaggle_datasets(dataset_path):
    path = kagglehub.dataset_download(dataset_path)
    files = list_files(path)
    datasets = [pd.read_csv(file) for file in files]
    dataset = pd.concat(datasets, ignore_index=True).copy()
    dataset.dropna(inplace=True)
    dataset.drop_duplicates(inplace=True)
    return dataset


def normalize_text(text):
    text = re.sub(r'\r\n|\r|\n', ' ', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def add_special_tokens(text):
    return "<|startoftext|>" + text + "<|endoftext|>"


def preprocess_df(df_poetry, df_love_poems, df_classification, df_poetryfoundationorg, df_poetry_foundation):

    # normalize the content of the poems
    df_poetry['content'] = df_poetry['content'].apply(normalize_text)
    df_love_poems['poem'] = df_love_poems['poem'].apply(normalize_text)
    df_classification['Poem'] = df_classification['Poem'].apply(normalize_text)
    df_poetryfoundationorg['Content'] = df_poetryfoundationorg['Content'].apply(normalize_text)
    df_poetry_foundation['Poem'] = df_poetry_foundation['Poem'].apply(normalize_text)
    objs = [df_poetry['content'], df_love_poems['poem'], df_classification['Poem'], df_poetryfoundationorg['Content'], df_poetry_foundation['Poem']]

    # concat the two dataframes
    df = pd.concat(objs, ignore_index=True).to_frame(name='poem')

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # add special tokens - GPT2
    df = df.apply(add_special_tokens)

    return df


# Dataset class : Raw data -> Dataframe
class DatasetPoem:
    def __init__(self):
        df_poetry = pd.read_csv(poetry_path)
        df_love_poems = pd.read_parquet(love_poems_path)
        df_classification = load_kaggle_datasets(poem_classification_path)
        df_poetryfoundationorg = load_kaggle_datasets(poetryfoundationorg_path)
        df_poetry_foundation = load_kaggle_datasets(poetry_foundation_path)

        self.df = preprocess_df(df_poetry, df_love_poems, df_classification, df_poetryfoundationorg, df_poetry_foundation)

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
            "labels": self.input_ids[idx],
        }
