import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
#from torch.utils.data import DataLoader
from transformers import AutoTokenizer #, AutoModelForSequenceClassification
#import torch
import numpy as np
import evaluate

def convert_label(label):
  if label == "neutral":
    return 0
  elif label == "positive":
    return 1
  else:
    return 2

def get_dataloader(data_split: str, batch_size: int = 4, bayes: bool = False):
    """
    Get a pytorch dataloader for a specific data split

    Args:
        data_split (str): the data split
        data_path (str, optional): a data path if the data is not stored at the default path.
            For students using ada, this should always be None. Defaults to None.
        batch_size (int, optional): the desired batch size. Defaults to 4.

    Returns:
        DataLoader: the pytorch dataloader object
    """
    assert data_split in ["train", "dev", "test"]
    data = pd.read_csv("finance-sentiment.csv")

    #Make Labels Integers (0 = neutral, 1 = positive, 2 = negative)
    data["Sentiment"] = data["Sentiment"].apply(convert_label)

    #Split Data (train 40%, dev 20%, test 20%)
    train_sentiment_df, temp_sentiment_df = train_test_split(data, test_size=.4, random_state=457)
    dev_sentiment_df, test_sentiment_df = train_test_split(temp_sentiment_df, test_size=.5, random_state=457)

    if data_split == "train":
        data = train_sentiment_df

    elif data_split == "dev":
        data = dev_sentiment_df
        
    elif data_split == "test":
        data = test_sentiment_df

    if bayes:
       return data

    #Format and Tokenize Data
    dataset = Dataset.from_pandas(data)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = dataset.map(lambda ex: tokenizer(ex["Sentence"], truncation=True, padding="max_length"), batched=True)
    dataset = dataset.with_format("torch")
    # dataloader = DataLoader(dataset, batch_size=batch_size)
    # print(dataset)
    # print(dataset[0]["__index_level_0__"])

    return dataset

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

#get_dataloader("train", 16)