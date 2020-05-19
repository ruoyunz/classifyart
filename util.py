import random
import time
from pathlib import Path
from collections import defaultdict, Counter

# imaging libraries
import cv2
from PIL import Image

# number and data manipulation
import numpy as np
import pandas as pd
import scipy as sp

# loading effects
from tqdm import tqdm

# accuracy metrics
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.model_selection import StratifiedKFold

# data params
N_CLS = 3474
H = 128
W = 128

# Code adapted from PyTorch Tutorials/iMet starter code

def mask(lst, top_n):
    m = np.zeros_like(lst, dtype=np.uint8)
    col_indices = lst[:, -top_n:].reshape(-1)
    row_indices = [i // top_n for i in range(len(col_indices))]
    m[row_indices, col_indices] = 1
    return m

def predictions(probabilities, threshold, min_labels=1, max_labels=10):
    # assert probabilities.shape[1] == N_CLS
    lst = probabilities.argsort(axis=1)
    max_m = mask(lst, max_labels)
    min_m = mask(lst, min_labels)
    prob_m = probabilities > threshold
    return (max_m & prob_m) | min_m

# Split data into parts
def make_folds(df, n_folds, seed):
    cls_counts = Counter(cls for classes in df['attribute_ids'].str.split() for cls in classes)
    fold_cls_counts = defaultdict(int)
    folds = [-1] * len(df)
    for item in df.sample(frac=1, random_state=seed).itertuples():
        cls = min(item.attribute_ids.split(), key=lambda cls: cls_counts[cls])
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts if count == min_count])
        folds[item.Index] = fold
        for cls in item.attribute_ids.split():
            fold_cls_counts[fold, cls] += 1
    df['fold'] = folds
    return df



