import torch
import pandas as pd
import os
import datasets
import dataloader
from sklearn.metrics import f1_score
from tqdm import tqdm
import prompt


def safe_macro_f1(y, y_pred, **kwargs):
    """
    Macro-averaged F1, forcing `sklearn` to report as a multiclass
    problem even when there are just two classes. `y` is the list of
    gold labels and `y_pred` is the list of predicted labels.

    """
    return f1_score(y, y_pred, average='macro', pos_label=None)

def ds_from_sentences_and_sentiments(sentences, sentiments):
    merged = []
    for p, s in zip(sentences, sentiments):
        merged.append(f"{p} {s}")
    return datasets.Dataset.from_dict({'text': merged})


def get_split(dataset_name, split, cache_dir=None, streaming=False, revision=None):
    return datasets.load_dataset(
      dataset_name,
      "dynabench.dynasent.r2.all",
      split=split,
      cache_dir=cache_dir,
      streaming=streaming,
      trust_remote_code=True,
      revision=revision)


def load_sentiment_dataset(cache_dir=None, streaming=False, revision=None):
    dataset = datasets.load_dataset(
      "dynabench/dynasent",
      "dynabench.dynasent.r2.all",
      cache_dir=cache_dir,
      streaming=streaming,
      trust_remote_code=True,
      revision=revision)
    
    train_dataset = ds_from_sentences_and_sentiments(dataset['train']['sentence'], dataset['train']['gold_label'])
    valid_dataset = ds_from_sentences_and_sentiments(dataset['validation']['sentence'], dataset['validation']['gold_label'])

    test_dataset = datasets.Dataset.from_dict(
        {'text': dataset['test']['sentence'], 
         'label': dataset['test']['gold_label']})
    
    return {
        'train': train_dataset,
        'validation': valid_dataset,
        'test': test_dataset
    }

def test_sentiment(model, config, logger, tokenizer):
    ds = load_sentiment_dataset(cache_dir=config.data.cache_dir, streaming=config.data.streaming, revision=config.data.get("train_revision", None))['test']

    masked = []
    for text in ds['text']:
        masked.append(f"{text} <|mask|>5:")

    ps = prompt.PromptDataset(masked, tokenizer, config, device=model.device)
    dl = torch.utils.data.DataLoader(ps, batch_size=config.loader.eval_batch_size, shuffle=False)
  
    labels = []
    for i, (tokenized, masks, lens) in enumerate(dl):
      def projection_fn(x):   
        y = torch.where(masks, tokenized, x)
        return y

      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps, projection_fn=projection_fn)

      text_labels = ps.decode_for_sentiment(samples, lens, tokenizer)
      print(f"text_labels: {text_labels}")
      print(f"text: {ds['text'][i]}, label: {ds['label'][i]}")
      labels.extend(list(text_labels))





