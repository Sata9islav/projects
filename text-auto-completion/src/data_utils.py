import matplotlib.pyplot as plt

import re

from tqdm import tqdm

from typing import List


def read_file(file_path: str) -> List[str]:
    "READ FILE FROM PATH"
    lines: List[str] = []
    with open(file_path, "r") as text_file:
        lines = [line for line in tqdm(text_file)]
    return lines


def preprocessing_text(raw_text: str) -> str:
    """CLEAR TEXT:
    - to lower case
    - remove url, remove emoji, remove mentions
    - replace symbols
    """

    pattern_url: str = r"(https?://[^\s,.;!?]+|www\.[^\s,.;!?]+)"
    pattern_symbols: str = "[^a-z0-9\s]"
    pattern_double_space: str = r"\s+"

    text_lower_without_url: str = re.sub(
        pattern_url,
        "",
        raw_text.lower(),
    )
    text_without_symbols: str = re.sub(
        pattern_symbols,
        "",
        text_lower_without_url,
    )
    text_without_double_space: str = re.sub(
        pattern_double_space,
        " ",
        text_without_symbols,
    ).strip()

    return text_without_double_space


def plot_length_hist(lst_with_lengths: List[int]) -> None:
    plt.hist(lst_with_lengths, bins=50, edgecolor="black")
    plt.title("Text length distribution")
    plt.xlabel("Tweet's legth")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def print_size(data, data_name, type="dataset") -> None:
    if type == "dataset":
        x, y = data[0]
        print(f"{data_name.upper()} DATASET")
        print("X: ", x)
        print("Y: ", y)
        print("X SHAPE: ", x.shape)
        print("Y SHAPE: ", y.shape)
    elif type == "dataloader":
        batch_x, batch_y = next(iter(data))
        print(f"{data_name.upper()} DATALOADER")
        print("BATCH X SHAPE:", batch_x.shape)
        print("BATCH Y SHAPE:", batch_y.shape)
        print("BATCH X OBJECT: ", batch_x[0])
        print("BATCH Y OBJECT: ", batch_y[0])
    else:
        print("UNKNOW DATA TYPE")
