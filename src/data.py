import os

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config

_SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}


def download_dataset() -> str:
    return kagglehub.dataset_download("shubhamgoel27/dermnet")


def build_metadata(dataset_path: str) -> pd.DataFrame:
    rows = []
    for root, _, files in os.walk(dataset_path):
        for fname in files:
            if os.path.splitext(fname)[-1].lower() in _SUPPORTED_EXT:
                full_path = os.path.join(root, fname)
                label = os.path.basename(os.path.dirname(full_path))
                rows.append((full_path, label))
    return pd.DataFrame(rows, columns=["image_path", "label"])


def encode_labels(df: pd.DataFrame):
    le = LabelEncoder()
    df = df.copy()
    df["label_enc"] = le.fit_transform(df["label"])
    return df, le


def split_data(df: pd.DataFrame):
    return train_test_split(
        df,
        test_size=1.0 - config.TRAIN_RATIO,
        stratify=df["label_enc"],
        random_state=config.RANDOM_SEED,
    )
