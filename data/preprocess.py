from datasets import load_dataset
import json

# Output paths
ocr_train_path = "mangaocr_train.jsonl"
ocr_valid_path = "mangaocr_valid.jsonl"
ocr_test_path = "mangaocr_test.jsonl"
vqa_path = "mangavqa_train.jsonl"
merged_path = "mangaocr_mangavqa_train.jsonl"

# Helper: clean nulls from message content
def clean_messages(messages):
    cleaned = []
    for message in messages:
        if "content" in message and isinstance(message["content"], list):
            cleaned_content = []
            for item in message["content"]:
                # Remove keys with null values, but keep the item
                new_item = {
                    k: v
                    for k, v in item.items()
                    if not (k in ["image", "text"] and v is None)
                }

                # Keep the item only if it still has at least 'image' or 'text'
                if "image" in new_item or "text" in new_item:
                    cleaned_content.append(new_item)

            message["content"] = cleaned_content
        cleaned.append(message)
    return cleaned


# Define system message (only for mangavqa training set)
system_msg = {
    "role": "system",
    "content": "あなたは日本語の漫画に関する質問に答えるAIです。与えられた画像に基づいて質問に答えてください。",
}

# Helper: save dataset to .jsonl
def save_dataset(dataset, path, add_system=False):
    with open(path, "w", encoding="utf-8") as f:
        for example in dataset:
            if "messages" not in example:
                raise ValueError(
                    f"Missing 'messages' field in example with id {example.get('id')}"
                )

            # Clean null content items
            example["messages"] = clean_messages(example["messages"])

            # Optionally add system message (only for mangavqa training set)
            if add_system:
                example["messages"] = [system_msg] + example["messages"]

            f.write(json.dumps(example, ensure_ascii=False) + "\n")


# 1. Load MangaOCR splits
ocr_train = load_dataset("hal-utokyo/MangaOCR", split="train")
ocr_valid = load_dataset("hal-utokyo/MangaOCR", split="validation")
ocr_test = load_dataset("hal-utokyo/MangaOCR", split="test")

# 2. Load MangaVQA train
vqa_train = load_dataset("hal-utokyo/MangaVQA-train", split="train")

# 3. Save all splits after cleaning
save_dataset(ocr_train, ocr_train_path)
save_dataset(ocr_valid, ocr_valid_path)
save_dataset(ocr_test, ocr_test_path)
save_dataset(vqa_train, vqa_path, add_system=True)

# 4. Merge OCR train + VQA only
with open(merged_path, "w", encoding="utf-8") as f_out:
    for path in [ocr_train_path, vqa_path]:
        with open(path, "r", encoding="utf-8") as f_in:
            f_out.write(f_in.read())
