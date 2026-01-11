import evaluate

from tqdm import tqdm

rouge = evaluate.load("rouge")


def evaluate_transformer_model_rouge(test_texts, generator, max_length) -> None:
    all_preds = []
    all_refs = []

    all_gen_texts = []
    all_true_texts = []

    for text in tqdm(test_texts):
        words = text.split()
        if len(words) < 10:
            continue

        split_idx = int(len(words) * 0.75)

        if split_idx == 0 or split_idx >= len(words):
            continue

        prefix_words = words[:split_idx]
        target_suffix_words = words[split_idx:]

        prompt = " ".join(prefix_words)
        total_max_len = min(
            max_length, len(prefix_words) + len(target_suffix_words) + 5
        )

        out = generator(
            prompt,
            max_new_tokens=len(target_suffix_words),
            # truncation=True,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        )

        generated = out[0]["generated_text"]
        gen_words = generated.split()

        if len(gen_words) <= len(prefix_words):
            continue

        gen_suffix_words = gen_words[
            len(prefix_words) : len(prefix_words) + len(target_suffix_words)
        ]

        pred_text = " ".join(gen_suffix_words)
        refs_text = " ".join(target_suffix_words)

        all_preds.append(pred_text)
        all_refs.append(refs_text)

        all_gen_texts.append(pred_text)
        all_true_texts.append(refs_text)

    rouge_scores = rouge.compute(predictions=all_preds, references=all_refs)

    print("--" * 15 + " RESULT " + "--" * 15)

    print("VAL ROUGE: " + " ".join(f"{k}: {v:.4f}" for k, v in rouge_scores.items()))

    print()

    print("EXAMPLES:")

    for t_txt, g_txt in list(zip(all_true_texts, all_gen_texts))[:5]:
        print(f"ORIGINAL|GENERATED: {t_txt}|{g_txt}")

    print("-" * 40)
