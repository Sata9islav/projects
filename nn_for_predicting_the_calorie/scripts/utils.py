from functools import partial

import numpy as np

import matplotlib.pyplot as plt

import os

import pandas as pd

from PIL import Image

import random

import timm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchmetrics

from transformers import AutoModel, AutoTokenizer

from typing import Any, Optional, Tuple

from scripts.dataset import collate_fn, get_transforms, MultimodalDataset


def plot_metrics(metrics_history) -> None:
    epochs = range(1, len(metrics_history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, metrics_history["train_loss"], label="Train loss")
    axes[0].plot(epochs, metrics_history["val_loss"], label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train vs Val loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, metrics_history["train_mae"], marker="o", label="Train MAE")
    axes[1].plot(epochs, metrics_history["val_mae"], marker="x", label="Val MAE")
    axes[1].set_title("Train vs Val MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def set_requires_grad(module: nn.Module, unfreeze_pattern: str = "", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
            return

    patterns = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in patterns]):
            param.requires_grad = True
            if verbose:
                print(f"Unfreezed layer: {name}")
        else:
            param.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME, pretrained=True, num_classes=0
        )

        self.text_proj = nn.Linear(
            self.text_model.config.hidden_size, config.HIDDEN_DIM
        )
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)

        self.mass_proj = nn.Sequential(
            nn.Linear(1, config.MASS_DIM),
            nn.LayerNorm(config.MASS_DIM),
            nn.ReLU(),
            nn.Linear(config.MASS_DIM, config.MASS_DIM),
            nn.ReLU(),
        )

        fused_dim = config.HIDDEN_DIM * 2 + config.MASS_DIM

        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(fused_dim // 2, 1),
        )

    def forward(self, input_ids, attention_mask, image, mass):
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[
            :, 0, :
        ]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        mass_emb = self.mass_proj(mass)
        fused_emb = torch.cat([text_emb, image_emb, mass_emb], dim=1)

        preds = self.regressor(fused_emb).squeeze(1)
        return preds


def train(config, calories_scaler, device):
    seed_everything(config.SEED)

    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    set_requires_grad(
        model.text_model, unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True
    )
    set_requires_grad(
        model.image_model, unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True
    )

    weight_decay = getattr(config, "WEIGHT_DECAY", 0.01)

    optimizer = AdamW(
        [
            {"params": model.text_model.parameters(), "lr": config.TEXT_LR},
            {"params": model.image_model.parameters(), "lr": config.IMAGE_LR},
            {"params": model.regressor.parameters(), "lr": config.REGRESSOR_LR},
        ],
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=getattr(config, "LR_FACTOR", 0.5),
        patience=getattr(config, "LR_PATIENCE", 2),
        threshold_mode="abs",
        threshold=getattr(config, "LR_THRESHOLD", 0.5),
        min_lr=getattr(config, "MIN_LR", 1e-6),
    )

    criterion = nn.SmoothL1Loss()

    transforms = get_transforms(config)
    val_transforms = get_transforms(config, df_type="val")

    train_dataset = MultimodalDataset(config, transforms)
    val_dataset = MultimodalDataset(config, val_transforms, df_type="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    best_mae = float("inf")

    metrics_history = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
    }

    labels_mean = float(calories_scaler.mean_[0])
    labels_std = float(calories_scaler.scale_[0])

    print("training started")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0

        mae_train_metric_kcal = torchmetrics.MeanAbsoluteError().to(device)
        mae_val_metric_kcal = torchmetrics.MeanAbsoluteError().to(device)

        for batch in train_loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image": batch["image"].to(device),
                "mass": batch["mass"].to(device),
            }
            labels_scaled = batch["label"].to(device).view(-1)

            optimizer.zero_grad()
            pred_scaled = model(**inputs).view(-1)
            loss = criterion(pred_scaled, labels_scaled)

            loss.backward()
            optimizer.step()

            pred_kcal = pred_scaled.detach() * labels_std + labels_mean
            labels_kcal = labels_scaled * labels_std + labels_mean

            total_loss += loss.item()
            n_batches += 1
            mae_train_metric_kcal.update(pred_kcal, labels_kcal)

        train_loss = total_loss / max(n_batches, 1)
        train_mae_kcal = mae_train_metric_kcal.compute().cpu().item()

        mae_train_metric_kcal.reset()

        val_loss, val_mae_kcal = validate(
            model,
            val_loader,
            device,
            criterion,
            mae_val_metric_kcal,
            labels_mean,
            labels_std,
        )

        mae_val_metric_kcal.reset()

        scheduler.step(val_mae_kcal)

        metrics_history["train_loss"].append(train_loss)
        metrics_history["val_loss"].append(val_loss)
        metrics_history["train_mae"].append(train_mae_kcal)
        metrics_history["val_mae"].append(val_mae_kcal)

        lrs = [pg["lr"] for pg in optimizer.param_groups]
        lr_str = ", ".join([f"{lr:.2e}" for lr in lrs])

        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} | "
            f"Train loss(scaled): {train_loss:.4f} | Val loss(scaled): {val_loss:.4f} | "
            f"Train MAE(kcal): {train_mae_kcal:.2f} | Val MAE(kcal): {val_mae_kcal:.2f} | "
            f"LRs: [{lr_str}]"
        )

        if val_mae_kcal < best_mae:
            print(f"New best model, epoch: {epoch + 1}")
            best_mae = val_mae_kcal
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    return metrics_history


def validate(
    model, val_loader, device, criterion, mae_metric_kcal, labels_mean, labels_std
):
    model.eval()

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image": batch["image"].to(device),
                "mass": batch["mass"].to(device),
            }

            labels_scaled = batch["label"].to(device).view(-1)
            preds_scaled = model(**inputs).view(-1)

            loss = criterion(preds_scaled, labels_scaled)
            total_loss += float(loss.item())
            n_batches += 1

            pred_kcal = preds_scaled * labels_std + labels_mean
            labels_kcal = labels_scaled * labels_std + labels_mean

            mae_metric_kcal.update(pred_kcal, labels_kcal)

        val_loss = total_loss / max(n_batches, 1)
        val_mae_kcal = float(mae_metric_kcal.compute().item())

    return val_loss, val_mae_kcal


def predict_kcal(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    labels_mean: float,
    labels_std: float,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    preds_all = []
    true_all = []

    with torch.no_grad():
        for batch in loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image": batch["image"].to(device),
                "mass": batch["mass"].to(device),
            }

            labels_scaled = batch["label"].to(device).view(-1)
            preds_scaled = model(**inputs).view(-1)

            preds_kcal = preds_scaled * labels_std + labels_mean
            y_kcal = labels_scaled * labels_std + labels_mean

            preds_all.append(preds_kcal.detach().cpu().numpy())
            true_all.append(y_kcal.detach().cpu().numpy())

    return np.concatenate(preds_all), np.concatenate(true_all)


def make_worst_table(
    test_df: pd.DataFrame,
    preds_kcal: np.ndarray,
    true_kcal: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    errors = np.abs(preds_kcal - true_kcal)
    worst_idx = np.argsort(-errors)[:top_k]

    cols = [
        c
        for c in ["dish_id", "ingredients", "total_mass", "total_calories"]
        if c in test_df.columns
    ]
    worst = test_df.loc[worst_idx, cols].copy()

    worst["pred_kcal"] = preds_kcal[worst_idx]
    worst["true_kcal"] = true_kcal[worst_idx]
    worst["abs_error"] = errors[worst_idx]

    worst = worst.sort_values("abs_error", ascending=False).reset_index(drop=False)
    return worst


def show_worst_examples(
    worst_table: pd.DataFrame,
    test_df: pd.DataFrame,
    preds_kcal: np.ndarray,
    true_kcal: np.ndarray,
) -> None:
    for rank, row in enumerate(worst_table.itertuples(index=False), 1):
        orig_idx = int(getattr(row, "index"))
        dish_id = getattr(row, "dish_id", None)

        img_path = f"data/images/{dish_id}/rgb.png"
        img = Image.open(img_path).convert("RGB")

        pred = float(preds_kcal[orig_idx])
        true = float(true_kcal[orig_idx])
        err = abs(pred - true)

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(
            f"Top-{rank} | dish_id={dish_id} | true={true:.1f} | pred={pred:.1f} | abs_err={err:.1f}"
        )
        plt.show()

        if "ingredients" in test_df.columns:
            print("Ingredients:", str(test_df.loc[orig_idx, "ingredients"]))
        print("-" * 90)


def evaluate_on_test(
    config: Any,
    calories_scaler: Any,
    device: Optional[torch.device] = None,
    top_k: int = 5,
    show: bool = True,
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_mean = float(calories_scaler.mean_[0])
    labels_std = float(calories_scaler.scale_[0])

    model = MultimodalModel(config).to(device)
    state = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    test_df = pd.read_csv(config.TEST_DF_PATH).reset_index(drop=True)
    test_tfms = get_transforms(config, df_type="test")
    test_ds = MultimodalDataset(config, test_tfms, df_type="test")

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    preds_kcal, true_kcal = predict_kcal(
        model, test_loader, device, labels_mean, labels_std
    )
    errors_kcal = np.abs(preds_kcal - true_kcal)
    test_mae_kcal = float(errors_kcal.mean())

    worst_table = make_worst_table(test_df, preds_kcal, true_kcal, top_k=top_k)

    if show:
        print(f"TEST MAE (kcal): {test_mae_kcal:.2f}")
        show_worst_examples(worst_table, test_df, preds_kcal, true_kcal)
