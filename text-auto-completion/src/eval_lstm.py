import evaluate

import matplotlib.pyplot as plt

import torch

from tqdm import tqdm


rouge = evaluate.load("rouge")


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

    axes[1].plot(
        epochs,
        metrics_history["val_accuracy"],
        marker="o",
        label="Val accuracy",
    )
    axes[1].set_title("Accuracy vs Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_lstm_model_rouge(test_loader, tokenizer, model, max_length, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    model.eval()

    all_preds = []
    all_refs = []

    all_gen_texts = []
    all_true_texts = []

    for texts_batch in tqdm(test_loader):
        for text in texts_batch:
            token_ids = tokenizer.encode(
                text, add_special_tokens=False, max_length=max_length, truncation=True
            )
            if len(token_ids) < 10:
                continue

            split_idx = int(len(token_ids) * 0.75)
            if split_idx == 0 or split_idx >= len(token_ids):
                continue

            prefix_ids = token_ids[:split_idx]
            target_suffix_ids = token_ids[split_idx:]

            generate_ids = model.generate(
                prefix_ids, max_new_tokens=len(target_suffix_ids), device=device
            )

            gen_suffix_ids = generate_ids[len(prefix_ids) :]

            pred_text = tokenizer.decode(gen_suffix_ids, skip_special_tokens=True)
            ref_text = tokenizer.decode(target_suffix_ids, skip_special_tokens=True)

            all_preds.append(pred_text)
            all_gen_texts.append(pred_text)

            all_refs.append(ref_text)
            all_true_texts.append(ref_text)

    rouge_scores = rouge.compute(predictions=all_preds, references=all_refs)
    return rouge_scores, (all_true_texts, all_gen_texts)


def evaluate_lstm_model_loss(val_loader, model, criterion, device=None):
    if device is None:
        device = next(model.parameters()).device

    model.to(device)

    model.eval()

    correct, total_target_size = 0, 0
    sum_loss = 0

    with torch.no_grad():
        for x_batch, y_batch in tqdm(val_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y_batch).sum().item()
            total_target_size += y_batch.size(0)
            sum_loss += loss.item()

    avg_loss = sum_loss / len(val_loader)
    accuracy = correct / total_target_size
    return avg_loss, accuracy
