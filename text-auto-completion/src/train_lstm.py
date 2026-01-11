from tqdm import tqdm


from src.eval_lstm import evaluate_lstm_model_loss, evaluate_lstm_model_rouge


def train_lstm_model(
    train_loader,
    val_loader,
    test_loader,
    tokenizer,
    model,
    optimizer,
    criterion,
    n_epochs=3,
    device=None,
):
    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    metrics_history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in tqdm(range(n_epochs)):
        model.train()
        train_loss = 0
        for x_batch, y_batch in tqdm(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_loss, val_accuracy = evaluate_lstm_model_loss(
            val_loader, model, criterion, device=device
        )
        val_rouge, (true_texts, gen_texts) = evaluate_lstm_model_rouge(
            test_loader, tokenizer, model, max_length=120, device=device
        )

        metrics_history["train_loss"].append(train_loss)
        metrics_history["val_loss"].append(val_loss)
        metrics_history["val_accuracy"].append(val_accuracy)

        print("--" * 15 + f" EPOOCH: {epoch + 1} " + "--" * 15)
        print(
            f"TRAIN LOSS: {train_loss} | VAL LOSS: {val_loss} | VAL ACCURACY: {val_accuracy}"
        )
        print("VAL ROUGE: " + " ".join(f"{k}: {v:.4f}" for k, v in val_rouge.items()))

        print()

        print("EXAMPLES:")

        for t_txt, g_txt in list(zip(true_texts, gen_texts))[:5]:
            print(f"ORIGINAL|GENERATED: {t_txt}|{g_txt}")

        print("-" * 40)
    return metrics_history
