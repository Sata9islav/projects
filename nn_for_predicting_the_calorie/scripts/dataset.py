import albumentations as A

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

import timm


class MultimodalDataset(Dataset):
    def __init__(self, config, transforms, df_type="train"):
        if df_type == "train":
            self.df = pd.read_csv(config.TRAIN_DF_PATH)
        else:
            self.df = pd.read_csv(config.TEST_DF_PATH)

        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "ingredients"]
        if pd.isna(text):
            text = ""
        text = str(text)
        label = float(self.df.loc[idx, "total_calories"])
        mass = float(self.df.loc[idx, "total_mass"])

        img_dir_name = self.df.loc[idx, "dish_id"]

        image = Image.open(f"data/images/{img_dir_name}/rgb.png").convert("RGB")
        image = self.transforms(image=np.array(image))["image"]

        return {"label": label, "image": image, "ingredients": text, "mass": mass}


def collate_fn(batch, tokenizer):
    texts = [item["ingredients"] for item in batch]
    images = torch.stack([item["image"] for item in batch])

    mass = torch.tensor(
        [item["mass"] for item in batch], dtype=torch.float32
    ).unsqueeze(1)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    tokenized_inputs = tokenizer(
        texts, return_tensors="pt", padding="max_length", truncation=True
    )
    return {
        "label": labels,
        "mass": mass,
        "image": images,
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
    }


def get_transforms(config, df_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if df_type == "train":
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0
                ),
                A.RandomCrop(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Affine(
                    scale=(0.8, 1.2),
                    rotate=(-15, 15),
                    translate_percent=(-0.1, 0.1),
                    shear=(-10, 10),
                    fill=0,
                    p=0.4,
                ),
                A.CoarseDropout(
                    num_holes_range=(2, 8),
                    hole_height_range=(
                        int(0.07 * cfg.input_size[1]),
                        int(0.15 * cfg.input_size[1]),
                    ),
                    hole_width_range=(
                        int(0.1 * cfg.input_size[2]),
                        int(0.15 * cfg.input_size[2]),
                    ),
                    fill=0,
                    p=0.2,
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4
                ),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.pytorch.ToTensorV2(p=1.0),
            ],
            seed=42,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0
                ),
                A.CenterCrop(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.pytorch.ToTensorV2(p=1.0),
            ],
            seed=42,
        )

    return transforms
