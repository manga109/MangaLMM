import os
import argparse
import json
import math
from pathlib import Path
import re

import torch
from tqdm.auto import tqdm
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    set_seed,
)
from qwen_vl_utils import process_vision_info


DATA_DIR = Path("./data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument(
        "-o", "--output_path", type=Path, default=Path("./outputs/Inference_OCR"),
    )
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-t", "--text_only", action="store_true")
    parser.add_argument("-i", "--split_index", type=int, default=0)
    parser.add_argument("-n", "--split_num", type=int, default=1)
    return parser.parse_args()


def bbox2d_to_quad(bbox_2d):
    xmin, ymin, xmax, ymax = bbox_2d
    return [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]


def parse_text(text: str) -> list[list]:
    if not text.strip():
        return []

    # handle escape
    text = text.replace('\\"', '"')

    # find \n\t{ ... } blocks
    blocks = re.findall(r"\n\t\{.*?\}", text, re.DOTALL)

    ocrs = []
    for block in blocks:
        block = block.strip()  # remove \n\t
        bbox_match = re.search(r'"bbox_2d"\s*:\s*\[([^\]]+)\]', block, flags=re.DOTALL)
        text_match = re.search(
            r'"text_content"\s*:\s*"([^"]*)"', block, flags=re.DOTALL
        )

        if bbox_match and text_match:
            try:
                bbox_list = [int(x.strip()) for x in bbox_match.group(1).split(",")]
                content = text_match.group(1)

                quad = bbox2d_to_quad(bbox_list)
                ocrs.append([content, quad])
            except:
                continue

    return ocrs


def inference(
    data: dict,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
) -> tuple[str, list[str, list[int]], list[str, list[int]]]:
    messages = data["messages"]
    assert messages[-2]["role"] == "user"
    answer_str = messages[-1]["content"][0]["text"]
    messages = messages[:-1]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    for message in messages:
        if isinstance(message["content"], list):
            for content in message["content"]:
                if content["type"] == "image":
                    image_path = (DATA_DIR / content["image"]).resolve()
                    content["image"] = "file://" + str(image_path)
    image_inputs, video_inputs = process_vision_info(
        messages
    )  # image_inputs: list[PIL.Image.Image]
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    # generated_ids = model.generate(**inputs, max_new_tokens=32768)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        do_sample=False,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text, parse_text(output_text), parse_text(answer_str)


def main():
    args = parse_args()

    answer_file = "mangaocr_test.jsonl"
    with open(DATA_DIR / answer_file) as f:
        dataset = [json.loads(line) for line in f]
    split_size = math.ceil(len(dataset) / args.split_num)
    start_index = args.split_index * split_size
    end_index = min((args.split_index + 1) * split_size, len(dataset))
    dataset = dataset[start_index:end_index]
    print(f"Processing {len(dataset)} samples from {start_index} to {end_index}")
    os.makedirs(args.output_path, exist_ok=True)
    result_file = Path(
        args.output_path
        / f"{Path(args.model_path).name}_{args.split_index}-{args.split_num}.jsonl"
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # "sdpa" or "flash_attention_2"
        device_map="auto",
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path)
    set_seed(args.seed)

    for data in tqdm(dataset):
        raw, pred, gt = inference(data, model, processor)
        with result_file.open("a") as f:
            f.write(
                json.dumps(
                    {"id": data["id"], "raw_output": raw, "pred": pred, "gt": gt},
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
