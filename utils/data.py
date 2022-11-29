import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import torch
import transformers
from transformers import BertTokenizer


def read_data(data_dir): 
    return pd.read_csv(data_dir, usecols=['sentence', 'label'], encoding = 'utf8', keep_default_na=False)

def clean_str(s): 
    # BioBert or cased-Bert works better with cased letters, so don't use s.lower()
    return s.strip()

def split_train_test_data(data_dir, K, random_state, fold=0):
    df = read_data(data_dir)
    print('- label value counts:')
    print(df.label.value_counts(), '\n')

    df['sentence'] = df.sentence.apply(clean_str)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
    for i, (train_index, test_index) in enumerate(skf.split(df.sentence, df.label)):
        if i == fold:
            break
    train = df.iloc[train_index]
    test = df.iloc[test_index]

    print(f"ALL: {len(df)}   TRAIN: {len(train)}   TEST: {len(test)}")
    label_list = np.unique(train.label)
    return train, test, label_list

def split_X_y(train_data, test_data):
    X_train, X_test = train_data['sentence'], test_data['sentence']
    try: y_train = torch.tensor(train_data['label'], dtype=torch.long)
    except: y_train = torch.tensor(train_data['label'].to_numpy(dtype=np.float64), dtype=torch.long)
    y_test = torch.tensor(test_data['label'].to_numpy(dtype=np.float64), dtype=torch.long)
    return X_train, y_train, X_test, y_test

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_encoded_dataset(args, fold=0):
    train, test, _ = split_train_test_data(args.data_file_path, args.K, args.seed, fold)
    X_train, y_train, X_test, y_test = split_X_y(train, test)

    print(f'Tokenizer from: {args.pretrain_path}')
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_path)
    encodings_train = tokenizer(X_train.tolist(), return_tensors='pt', padding='max_length', max_length=args.max_len, truncation=True)
    encodings_test = tokenizer(X_test.tolist(), return_tensors='pt', padding='max_length', max_length=args.max_len, truncation=True)

    print(encodings_train['input_ids'].shape[1], encodings_test['input_ids'].shape[1])
    print(encodings_train['input_ids'].shape[1] == encodings_test['input_ids'].shape[1])
    print()
    
    train_dataset = MyDataset(encodings_train, y_train)
    test_dataset = MyDataset(encodings_test, y_test)
    return train_dataset, test_dataset

def get_new_data(args):
    columns = ['pmid', 'sentence']
    df = pd.read_csv(args.data_file_path, usecols=columns)
    print(f'all: {len(df):,}    unique sentences: {len(df.sentence.unique()):,}     papers: {len(df.pmid.unique()):,}')

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_path)
    inputs = tokenizer(df['sentence'].tolist(), return_tensors='pt', padding='max_length', max_length=args.max_len, truncation=True)
    return df, inputs