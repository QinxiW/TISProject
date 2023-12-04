import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split

torch.cuda.empty_cache()

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertTokenizer, pipeline
from transformers import TrainingArguments, Trainer
import json
import pandas as pd


def limit_words(word_list, max_words=128):
    return word_list[:max_words]


def main():
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set as {seed}")

    df = pd.read_csv("Data/cleaned_data.csv.gz")
    print(df.head())

    # sentiment_pipeline = pipeline("sentiment-analysis")
    df['description_cleaned_limit'] = df['description_cleaned'].apply(lambda x: limit_words(x, max_words=128))

    # https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads&search=sentiment
    specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

    reviews = df.description_cleaned_tokenized
    tuned_target = np.where(df.points.values > 92, 'pos', np.where(df.points.values < 85, 'neg', 'neu'))

    X, y = reviews, tuned_target

    # Use train_test_split to split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    repo_name = "finetuning-sentiment-model-3000-samples"

    training_args = TrainingArguments(
        output_dir=repo_name,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="epoch",
        push_to_hub=True,
    )

    trainer = Trainer(
        model=specific_model,
        args=training_args,
        train_dataset=X_train,
        eval_dataset=X_test,
        #  tokenizer=tokenizer,
        #  data_collator=data_collator,
        #  compute_metrics=compute_metrics,
    )

    trainer.train()

    df['base_sentiment_label'] = specific_model(list(df.description_cleaned_limit))

    df.to_csv('sentiment_data.csv')


if __name__ == '__main__':
    main()
