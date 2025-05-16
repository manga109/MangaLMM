import os
import json
import argparse
import asyncio
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from openai import OpenAI
from datasets import load_dataset


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default=None,
    )
    parser.add_argument(
        "--manga109_path", type=str, default="./data/Manga109_released_2023_12_07",
    )
    return parser.parse_args()


args = parse_args()
model_path = args.model_path
model_name = Path(model_path).name or Path(model_path).parent.name
if args.output_path:
    json_output_path = Path(args.output_path)
else:
    json_output_path = Path(f"outputs/evaluation_results/{model_name}.jsonl")

output_folder = json_output_path.parent
os.makedirs(output_folder, exist_ok=True)
print(model_path, model_name)
print(output_folder, json_output_path)

# model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

dataset_name = "hal-utokyo/MangaVQA"
split = "test"
dataset = load_dataset(dataset_name, split=split)


df = pd.DataFrame(
    {
        "title": dataset["title"],
        "image_number": dataset["image_number"],
        "question": dataset["question"],
        "answer": dataset["answer"],
        "required_info": dataset["required_info"],
        "w5h1": dataset["w5h1"],
        "understanding_type": dataset["understanding_type"],
    }
)

output_text_list = []

PROMPT = "あなたは日本語の漫画に関する質問に答えるAIです。与えられた画像に基づいて質問に答えてください。"
print(PROMPT)

for i in tqdm(range(len(df))):
    img_path = os.path.join(
        args.manga109_path,
        "images",
        df["title"][i],
        f"{df['image_number'][i]:03d}.jpg",
    )

    assert os.path.exists(img_path), f"Image {img_path} not found!"

    # input_text = PROMPT + df["Question"][i]
    input_text = df["question"][i]

    image = Image.open(img_path)

    messages = [
        {"role": "system", "content": PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": input_text},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt",
    ).to("cuda")

    torch.manual_seed(0)
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    output_text_list.append(output_text)

df["generated_answer"] = output_text_list
df.to_json(json_output_path, orient="records", lines=True, force_ascii=False)

df = pd.read_json(json_output_path, lines=True)

system_prompt = """You are an evaluator. Your task is to rate how appropriate a model’s response is to a question about a manga image. For each case, you will be given a question (based on a manga image), a human-written answer, and the model's response. The image is not shown, but the question and answer are based on it. Please evaluate as if the image were available.
Please rate how well the model's response answers the question, considering the intended image context and the human answer as a reference, using a scale from 1 to 10:
1 — Completely inappropriate or unrelated to the question or image context.
2 — Mostly unrelated with major misunderstandings or incorrect information.
3 — Slightly relevant, but largely incorrect or unhelpful.
4 — Somewhat relevant, but contains significant errors or omissions.
5 — Partially correct with noticeable inaccuracies, vagueness, or missing key points.
6 — Generally okay, but missing core points or includes some incorrect interpretations.
7 — Mostly correct and relevant, with only minor issues or small omissions.
8 — Almost entirely accurate with only slight room for improvement.
9 — Very appropriate, accurate, and well-aligned with the question and image context.
10 — Perfectly appropriate, accurate, and fully answers the question as if the image were visible.
Only return a single number (1–10). Do not include any explanations, justifications, or comments."""


def evaluate_with_gpt4o(question, answer, generated_answer):
    evaluation_prompt = f""" Input:
        "question": {question},
        "human-written answer": {answer},
        "model's response": {generated_answer},

    Your score: 
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": evaluation_prompt},
            ],
            max_tokens=100,
            temperature=0,
            seed=0,
        )
        eval_result = response.choices[0].message.content.strip()
        score = int(eval_result)
    except Exception as e:
        eval_log_path = json_output_path.parent / "log_exception_during_evaluation.txt"
        with open(eval_log_path, "a", encoding="utf-8") as f:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            f.write(
                f"[{now}] {model_name}\t[Question]{question}\t[Answer]{answer}\t[Model's response]{generated_answer}\t[GPT_result]{eval_result}\n"
            )
        print(Exception)
        # score = None
        score = 1  # regard exception as lowest score. Exceptions rarely occur. Almost all exceptions occurred when the model's response was empty.

    return eval_result, score


# asyncio wrapper
async def evaluate_async(i):
    question = df["question"][i]
    answer = df["answer"][i]
    generated_answer = df["generated_answer"][i]

    eval_result, score = await asyncio.to_thread(
        evaluate_with_gpt4o, question, answer, generated_answer
    )

    return {
        "title": df["title"][i],
        "image_number": int(df["image_number"][i]),
        "question": question,
        "answer": answer,
        "required_info": df["required_info"][i],
        "w5h1": df["w5h1"][i],
        "understanding_type": df["understanding_type"][i],
        "generated_answer": generated_answer,
        "gpt4o_eval": eval_result,
        "gpt4o_score": score,
    }


# main async routine
async def eval_all():
    tasks = [evaluate_async(i) for i in range(len(df))]
    results = await asyncio.gather(*tasks)
    return results


# run
results = asyncio.run(eval_all())

dir_name = os.path.dirname(json_output_path)
base_name = os.path.basename(json_output_path)
eval_base_name = f"eval_{base_name}"
eval_json_output_path = Path(os.path.join(dir_name, eval_base_name))

df_results = pd.DataFrame(results)
df_results.to_json(
    eval_json_output_path, orient="records", lines=True, force_ascii=False
)

scores = [r["gpt4o_score"] for r in results]
num_score = len(scores)
total_score = sum(scores)
average = total_score / num_score
print(f"Average: {average:.2f} ({total_score}/{num_score})")
print(f"Results saved to {eval_json_output_path}")

log_path = eval_json_output_path.parent / "log_evaluation.txt"
with open(log_path, "a", encoding="utf-8") as f:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    f.write(
        f"[{now}] {model_name}\tAverage: {average:.2f} ({total_score}/{num_score})\n"
    )
