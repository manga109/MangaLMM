from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from qwen_vl_utils import process_vision_info
import safetensors
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    BitsAndBytesConfig,
    HfArgumentParser,
    PreTrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    is_peft_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
)
from transformers.trainer_callback import ExportableState
from trl import SFTConfig, SFTTrainer


local_rank = None


# Constants
IGNORE_INDEX = -100


# Arguments
@dataclass
class Qwen2_5_VLDataArguments:
    data_path: Path = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    image_dir: Path = field(
        default=None, metadata={"help": "Path to the image directory."}
    )
    video_dir: Path = field(
        default=None, metadata={"help": "Path to the video directory."}
    )
    valid_data_path: Path = field(
        default=None, metadata={"help": "Path to the validation data."}
    )
    valid_image_dir: Path = field(
        default=None, metadata={"help": "Path to the validation image directory."}
    )
    valid_video_dir: Path = field(
        default=None, metadata={"help": "Path to the validation video directory."}
    )
    min_pixels: int = field(default=3136)  # 4 (2 x 2) x 28 x 28
    max_pixels: int = field(default=12845056)  # 16384 (128 x 128) x 28 x 28
    fps: float = field(default=2.0)


@dataclass
class Qwen2_5_VLTrainingArguments(SFTConfig):
    base_model: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")

    bits: int = field(default=16)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")  # `fp4` or `nf4`

    optim: str = field(default="adamw_torch")
    warmup_ratio: float = field(default=0.03)

    max_seq_length: int = field(default=32768)
    loss_only_response: bool = field(
        default=True,
        metadata={
            "help": "Whether to calculate the loss only on the response generation."
        },
    )

    freeze_llm: bool = field(default=False)
    learning_rate: float = field(default=1e-5)
    freeze_merger: bool = field(default=False)
    merger_lr: float = field(default=None)
    freeze_visual: bool = field(default=False)
    visual_lr: float = field(default=None)

    lora_enable: bool = field(default=False)
    lora_target_module_patterns: list[str] = field(
        default_factory=lambda: [
            r"visual\.patch_embed",
            r"visual\.blocks\.\d+\.attn\.qkv",
            r"visual\.blocks\.\d+\.attn\.proj",
            r"visual\.blocks\.\d+\.mlp\.gate_proj",
            r"visual\.blocks\.\d+\.mlp\.up_proj",
            r"visual\.blocks\.\d+\.mlp\.down_proj",
            r"visual\.merger\.mlp",
            r"model\.embed_tokens",
            r"model\.layers\.\d+\.self_attn\.q_proj",
            r"model\.layers\.\d+\.self_attn\.k_proj",
            r"model\.layers\.\d+\.self_attn\.v_proj",
            r"model\.layers\.\d+\.self_attn\.o_proj",
            r"model\.layers\.\d+\.mlp\.gate_proj",
            r"model\.layers\.\d+\.mlp\.up_proj",
            r"model\.layers\.\d+\.mlp\.down_proj",
            "lm_head",
        ]
    )
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    use_rslora: bool = field(default=False)
    use_dora: bool = field(default=False)
    init_lora_weights: str = field(default="gaussian")

    use_flash_attn2: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)


# Dataset
def is_relative_filepath(data: str | list[str]) -> bool:
    if isinstance(data, list):
        data = data[0]  # Only check the first element (this is for the video frames)
    if data.startswith("file://"):
        return False  # Absolute file path
    elif data.startswith("http://") or data.startswith("https://"):
        return False  # Remote file
    elif data.startswith("data:image"):
        return False  # Base64 image
    else:
        return True


class Qwen2_5_VLDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        image_dir: Path | None = None,
        video_dir: Path | None = None,
    ):
        super(Qwen2_5_VLDataset, self).__init__()
        self.data_path = data_path
        self.image_dir = image_dir
        if self.image_dir is not None:
            self.image_dir = Path(self.image_dir).resolve()
        self.video_dir = video_dir
        if self.video_dir is not None:
            self.video_dir = Path(self.video_dir).resolve()

        # Load the data
        if self.data_path is None:
            raise ValueError("`data_path` argument is required.")
        try:
            self.data_entries = [
                json.loads(line) for line in open(self.data_path).readlines()
            ]
        except Exception as e:
            raise ValueError(
                f"Error reading data file: {e}. Please check the format of the data file is in JSON Lines format."
            )

        # Check the image/video directory and update paths
        for entry in self.data_entries:
            for message in entry["messages"]:
                if isinstance(message["content"], list):
                    for content in message["content"]:
                        if "image" in content and is_relative_filepath(
                            content["image"]
                        ):
                            if self.image_dir is None:
                                raise ValueError(
                                    "`image_dir` argument is required when using local image files."
                                )
                            content["image"] = "file://" + str(
                                (self.image_dir / content["image"]).resolve()
                            )
                        if "video" in content and is_relative_filepath(
                            content["video"]
                        ):
                            if self.video_dir is None:
                                raise ValueError(
                                    "`video_dir` argument is required when using local video files."
                                )
                            if isinstance(content["video"], list):
                                content["video"] = [
                                    "file://" + str((self.video_dir / frame).resolve())
                                    for frame in content["video"]
                                ]
                            else:
                                content["video"] = "file://" + str(
                                    (self.video_dir / content["video"]).resolve()
                                )

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        return self.data_entries[idx]


class DataCollatorForQwen2_5_VLDataset(object):
    def __init__(
        self, processor: Qwen2_5_VLProcessor, train_args: Qwen2_5_VLTrainingArguments
    ):
        self.processor = processor
        assert isinstance(
            processor, Qwen2_5_VLProcessor
        ), "The processor should be an instance of `Qwen2_5_VLProcessor`."
        self.train_args = train_args
        # Prepare the constants
        self.assistant_start_tokens = processor(text=["<|im_start|>assistant\n"])[
            "input_ids"
        ][0]
        assert self.assistant_start_tokens == [151644, 77091, 198]
        self.assistant_end_tokens = processor(text=["<|im_end|>\n"])["input_ids"][0]
        assert self.assistant_end_tokens == [151645, 198]

    def __call__(self, examples):
        messages = [example["messages"] for example in examples]

        # Prepare images and videos
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare text
        texts = [
            self.processor.apply_chat_template(message, tokenize=False)
            for message in messages
        ]
        batch = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        labels = (
            torch.ones_like(batch["input_ids"], dtype=batch["input_ids"].dtype)
            * IGNORE_INDEX
        )
        if self.train_args.loss_only_response:
            for batch_idx, token_ids in enumerate(batch["input_ids"].tolist()):
                pos = 0
                is_assistant_response = False
                while pos < len(token_ids):
                    if is_assistant_response:
                        if token_ids[pos] == self.assistant_end_tokens[0]:
                            # Check if the assistant response ends
                            is_assistant_end = True
                            for i, assistant_end_token in enumerate(
                                self.assistant_end_tokens[1:], 1
                            ):
                                if token_ids[pos + i] != assistant_end_token:
                                    is_assistant_end = False
                                    break
                            if is_assistant_end:  # End of the assistant response
                                is_assistant_response = False
                                for i in range(
                                    pos, pos + len(self.assistant_end_tokens)
                                ):
                                    # Update the labels (including the end of the assistant response)
                                    labels[batch_idx, i] = token_ids[i]
                                pos += len(self.assistant_end_tokens)
                            else:
                                labels[batch_idx, pos] = token_ids[
                                    pos
                                ]  # Update the labels
                                pos += 1
                        else:
                            labels[batch_idx, pos] = token_ids[pos]  # Update the labels
                            pos += 1
                    else:
                        if token_ids[pos] == self.assistant_start_tokens[0]:
                            is_assistant_start = True
                            for i, assistant_start_token in enumerate(
                                self.assistant_start_tokens[1:], 1
                            ):
                                if token_ids[pos + i] != assistant_start_token:
                                    is_assistant_start = False
                                    break
                            if is_assistant_start:
                                is_assistant_response = True
                                pos += len(self.assistant_start_tokens)
                            else:
                                pos += 1
                        else:
                            pos += 1
        else:
            labels = batch["input_ids"].detach().clone()
            labels[
                labels == self.processor.tokenizer.pad_token_id
            ] = IGNORE_INDEX  # Padding tokens
            labels[
                torch.where((151652 <= labels) & (labels <= 151656))
            ] = IGNORE_INDEX  # Visual tokens
        batch["labels"] = labels
        return batch


# Utils
def rank0_print(*args):
    if local_rank == 0 or local_rank == "0" or local_rank is None:
        print(*args)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


# Trainer
class Qwen2_5_VLTrainer(SFTTrainer):
    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            merger_lr = (
                self.args.merger_lr
                if self.args.merger_lr is not None
                else self.args.learning_rate
            )
            merger_parameters = set(
                [name for name, _ in opt_model.named_parameters() if "merger" in name]
            )
            visual_lr = (
                self.args.visual_lr
                if self.args.visual_lr is not None
                else self.args.learning_rate
            )
            visual_parameters = set(
                [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "visual" in name and "merger" not in name
                ]
            )
            other_parameters = merger_parameters | visual_parameters

            no_decay_patterns = [
                "bias",
                r"visual\.blocks\.\d+\.norm1",
                r"visual\.blocks\.\d+\.norm2",
                r"model\.layers\.\d+\.input_layernorm",
                r"model\.layers\.\d+\.post_attention_layernorm",
                r"model\.norm",
            ]
            no_decay_parameters = set(
                [
                    name
                    for name, _ in opt_model.named_parameters()
                    if any(re.search(pattern, name) for pattern in no_decay_patterns)
                ]
            )

            base_decay_parameters = [
                param
                for name, param in opt_model.named_parameters()
                if (
                    name not in no_decay_parameters
                    and name not in other_parameters
                    and param.requires_grad
                )
            ]
            base_nodecay_parameters = [
                param
                for name, param in opt_model.named_parameters()
                if (
                    name in no_decay_parameters
                    and name not in other_parameters
                    and param.requires_grad
                )
            ]
            optimizer_grouped_parameters = [
                {
                    "params": base_decay_parameters,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": base_nodecay_parameters,
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
            ]

            merger_decay_parameters = [
                param
                for name, param in opt_model.named_parameters()
                if (
                    name not in no_decay_parameters
                    and name in merger_parameters
                    and param.requires_grad
                )
            ]
            merger_nodecay_parameters = [
                param
                for name, param in opt_model.named_parameters()
                if (
                    name in no_decay_parameters
                    and name in merger_parameters
                    and param.requires_grad
                )
            ]
            if len(merger_decay_parameters) > 0:
                optimizer_grouped_parameters.append(
                    {
                        "params": merger_decay_parameters,
                        "weight_decay": self.args.weight_decay,
                        "lr": merger_lr,
                    }
                )
            if len(merger_nodecay_parameters) > 0:
                optimizer_grouped_parameters.append(
                    {
                        "params": merger_nodecay_parameters,
                        "weight_decay": 0.0,
                        "lr": merger_lr,
                    }
                )

            visual_decay_parameters = [
                param
                for name, param in opt_model.named_parameters()
                if (
                    name not in no_decay_parameters
                    and name in visual_parameters
                    and param.requires_grad
                )
            ]
            visual_nodecay_parameters = [
                param
                for name, param in opt_model.named_parameters()
                if (
                    name in no_decay_parameters
                    and name in visual_parameters
                    and param.requires_grad
                )
            ]
            if len(visual_decay_parameters) > 0:
                optimizer_grouped_parameters.append(
                    {
                        "params": visual_decay_parameters,
                        "weight_decay": self.args.weight_decay,
                        "lr": visual_lr,
                    }
                )
            if len(visual_nodecay_parameters) > 0:
                optimizer_grouped_parameters.append(
                    {
                        "params": visual_nodecay_parameters,
                        "weight_decay": 0.0,
                        "lr": visual_lr,
                    }
                )

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                    self.args, opt_model
                )

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes.optim

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp

            self.optimizer = smp.DistributedOptimizer(self.optimizer)
        return self.optimizer

    def _save_checkpoint(self, model, trial):
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)

            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters(), require_grad_only=False
            )
            torch.save(
                non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin")
            )

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
                for cb in [
                    cb
                    for cb in self.callback_handler.callbacks + [self.control]
                    if isinstance(cb, ExportableState)
                ]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(self.state.stateful_callbacks[cb_name], list):
                        self.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        self.state.stateful_callbacks[cb_name] = cb_state
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)
        else:
            super(Qwen2_5_VLTrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: str | None = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (
            (PreTrainedModel,)
            if not is_peft_available()
            else (PreTrainedModel, PeftModel)
        )
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=self.args.save_safetensors,
                )
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
                )
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict,
                        os.path.join(output_dir, SAFE_WEIGHTS_NAME),
                        metadata={"format": "pt"},
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
            )

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def safe_save_model_for_hf_trainer(trainer: Qwen2_5_VLTrainer, output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
        trainer.model.config.save_pretrained(output_dir)


# LoRA
def prepare_lora_target_modules(
    model: Qwen2_5_VLForConditionalGeneration,
    training_args: Qwen2_5_VLTrainingArguments,
) -> list[str]:
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []
    merger_lora_module_names = []
    visual_lora_module_names = []
    for name, module in model.named_modules():
        if not isinstance(module, (linear_cls, embedding_cls)):
            continue
        for pattern in training_args.lora_target_module_patterns:
            if re.match(pattern, name):
                if "merger" in name:
                    merger_lora_module_names.append(name)
                elif "visual" in name:
                    visual_lora_module_names.append(name)
                else:
                    lora_module_names.append(name)
    if len(lora_module_names) > 0 and not training_args.freeze_llm:
        raise ValueError(
            "LoRA is enabled for the LLM parameters but `freeze_llm` is not set to `True`."
        )
    if len(merger_lora_module_names) > 0 and not training_args.freeze_merger:
        raise ValueError(
            "LoRA is enabled for the Merger parameters but `freeze_merger` is not set to `True`."
        )
    if len(visual_lora_module_names) > 0 and not training_args.freeze_visual:
        raise ValueError(
            "LoRA is enabled for the Visual parameters but `freeze_visual` is not set to `True`."
        )
    lora_target_modules = (
        lora_module_names + merger_lora_module_names + visual_lora_module_names
    )
    return lora_target_modules


# Main
def main():
    global local_rank

    parser = HfArgumentParser((Qwen2_5_VLDataArguments, Qwen2_5_VLTrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False  # For processing the `messages` field
    if training_args.dataloader_num_workers > 0:  # For surpassing the warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # Quantization config
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(
            {
                "device_map": {"": training_args.device},
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["visual"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            }
        )

    # Load the base model
    processor = Qwen2_5_VLProcessor.from_pretrained(
        training_args.base_model,
        min_pixels=data_args.min_pixels,
        max_pixels=data_args.max_pixels,
    )
    assert (
        processor.tokenizer.padding_side == "right"
    ), "The padding side should be `right`."
    attn_implementation = (
        "flash_attention_2" if training_args.use_flash_attn2 else "sdpa"
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        training_args.base_model,
        torch_dtype=compute_dtype,
        attn_implementation=attn_implementation,
        **bnb_model_from_pretrained_args,
    )
    model.config.use_cache = False
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side

    # Set `requires_grad`
    visual_params = model.visual.parameters()
    for param in visual_params:
        param.requires_grad = not training_args.freeze_visual
    merger_params = model.visual.merger.parameters()
    for param in merger_params:  # Overwrite `requires_grad` for the merger
        param.requires_grad = not training_args.freeze_merger
    llm_parameters = model.model.parameters()
    for param in llm_parameters:
        param.requires_grad = not training_args.freeze_llm
    lm_head_parameters = model.lm_head.parameters()
    for param in lm_head_parameters:
        param.requires_grad = not training_args.freeze_llm

    # QLoRA
    if training_args.bits in [4, 8]:
        model.config.torch_dtype = compute_dtype
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )
    elif training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # LoRA
    if training_args.lora_enable:
        lora_target_modules = prepare_lora_target_modules(model, training_args)
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            init_lora_weights=training_args.init_lora_weights,
            use_rslora=training_args.use_rslora,
            use_dora=training_args.use_dora,
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

    # Keep some parameters in FP32
    if training_args.bits in [4, 8]:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_token" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16:
                        module = module.to(torch.bfloat16)

    # Dataset
    train_dataset = Qwen2_5_VLDataset(
        data_args.data_path, data_args.image_dir, data_args.video_dir
    )
    eval_dataset = (
        None
        if data_args.valid_data_path is None
        else Qwen2_5_VLDataset(
            data_args.valid_data_path,
            data_args.valid_image_dir,
            data_args.valid_video_dir,
        )
    )
    data_collator = DataCollatorForQwen2_5_VLDataset(processor, training_args)

    # Trainer
    trainer = Qwen2_5_VLTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start training
    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save the model
    processor.save_pretrained(training_args.output_dir)
    # trainer.save_state() # comment out for save disk space. It takes 20GB for qwen 2.5VL 3B
    model.config.use_cache = True
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )
        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_state_dict.bin"),
            )  # in case the model is trained withoout LoRA module
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
