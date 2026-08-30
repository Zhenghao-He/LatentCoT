#!/usr/bin/env python3
"""Evaluate the recovered Qwen3-4B anti-steering intervention.

The intervention is applied to the last token at layers.29 on every valid
prompt-token and generation-token forward:

    hidden -= strength * (relu(selected_sae_pre_acts) @ selected_decoder_rows)

Only the two datasets used by the reproduction scripts are supported.
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from sparsify import Sae
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]

COT_TEMPLATE = """<|im_start|>system
You are a logical reasoning assistant. Solve the user's question by breaking it down into logical steps. Finally, provide the answer in the specified format.<|im_end|>
<|im_start|>user
{question}

Let's think step by step:
1) Analyze the given information.
2) Deduce intermediate conclusions
3) Finalize the answer.

Format your final response as: 'Therefore, the answer is [your answer].'/think<|im_end|>
<|im_start|>assistant
"""


def atomic_json_dump(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def load_gsm8k(limit: int | None) -> list[dict[str, str]]:
    path = (
        REPO_ROOT
        / "data/raw/grade-school-math/grade_school_math/data/test.jsonl"
    )
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows.append(
                {
                    "question": item["question"].strip(),
                    "answer": item["answer"].strip(),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def extract_choice(text: str) -> str:
    if not text:
        return ""
    normalized = text.strip().upper()
    match = re.search(r"\b([ABCDEFGHIJKLMNOPQR])\b", normalized)
    if match:
        return match.group(1)
    match = re.search(r"^\s*([ABCDEFGHIJKLMNOPQR])[.)]", normalized)
    return match.group(1) if match else ""


def load_bbh(limit: int | None) -> list[dict[str, str]]:
    path = (
        REPO_ROOT
        / "data/BIG-Bench-Hard/bbh/logical_deduction_three_objects.json"
    )
    examples = json.loads(path.read_text(encoding="utf-8"))["examples"]
    rows = [
        {"question": item["input"], "answer": extract_choice(item["target"])}
        for item in examples
    ]
    return rows[:limit] if limit is not None else rows


def load_dataset(name: str, limit: int | None) -> list[dict[str, str]]:
    if name == "gsm8k":
        return load_gsm8k(limit)
    if name == "bbh":
        return load_bbh(limit)
    raise ValueError(f"Unsupported dataset: {name}")


def last_meaningful_line(response: str) -> str:
    clean = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    return next((line.strip() for line in reversed(clean.splitlines()) if line.strip()), clean)


def extract_prediction(dataset: str, response: str) -> str:
    final_line = last_meaningful_line(response)
    if dataset == "gsm8k":
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", final_line)
        return numbers[-1].replace(",", "") if numbers else ""
    return extract_choice(final_line)


def normalize_gsm8k_answer(answer: str) -> str:
    final = answer.split("####")[-1]
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", final)
    return numbers[-1].replace(",", "") if numbers else ""


def is_correct(dataset: str, prediction: str, answer: str) -> bool:
    if dataset == "bbh":
        return prediction.strip().upper() == answer.strip().upper()
    try:
        return Decimal(prediction) == Decimal(normalize_gsm8k_answer(answer))
    except (InvalidOperation, ValueError):
        return False


def load_model_and_tokenizer(model_name: str, device: str, dtype: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def terminator_ids(tokenizer) -> list[int]:
    ids = {tokenizer.eos_token_id}
    for token in ("<|endoftext|>", "<|im_end|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None:
            ids.add(token_id)
    return sorted(token_id for token_id in ids if token_id is not None)


class DynamicSuppressionHook:
    """Subtract selected positive SAE decoder contributions from hidden state."""

    def __init__(self, sae, feature_indices: list[int], strength: float):
        self.sae = sae
        self.feature_indices = torch.tensor(
            feature_indices, device=sae.device, dtype=torch.long
        )
        self.strength = strength
        self.calls = 0
        self.processed_rows = 0
        self.active_rows = 0
        self.topk_hit_rows = 0
        self._active_row_mask: torch.Tensor | None = None

    def set_active_rows(self, mask: torch.Tensor) -> None:
        self._active_row_mask = mask

    def __call__(self, module, inputs, output):
        self.calls += 1
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        if self._active_row_mask is None:
            active = torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device)
        else:
            active = self._active_row_mask.to(hidden.device, dtype=torch.bool)
        self._active_row_mask = None
        if not active.any():
            return output

        last_hidden = hidden[active, -1, :]
        self.processed_rows += int(last_hidden.shape[0])
        encoded = self.sae.encode(last_hidden.to(self.sae.dtype))
        selected_pre_acts = encoded.pre_acts.index_select(1, self.feature_indices)
        self.active_rows += int((selected_pre_acts > 0).any(dim=1).sum().item())
        target_in_topk = (
            encoded.top_indices.unsqueeze(-1)
            == self.feature_indices.view(1, 1, -1)
        ).any(dim=(1, 2))
        self.topk_hit_rows += int(target_in_topk.sum().item())

        target_acts = torch.relu(selected_pre_acts)
        decoder_rows = self.sae.W_dec.index_select(0, self.feature_indices)
        delta = -self.strength * (target_acts @ decoder_rows)
        hidden[active, -1, :] = last_hidden + delta.to(last_hidden.dtype)
        return (hidden, *output[1:]) if is_tuple else hidden


def trim_at_eos(row: torch.Tensor, eos_ids: set[int]) -> torch.Tensor:
    for index, token_id in enumerate(row.tolist()):
        if token_id in eos_ids:
            return row[: index + 1]
    return row


@torch.inference_mode()
def generate_with_transformers(
    model, tokenizer, prompts: list[str], max_new_tokens: int, device: str
) -> tuple[list[str], list[int]]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    eos = terminator_ids(tokenizer)
    output = model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        disable_compile=True,
    )
    generated = output[:, encoded.input_ids.shape[1] :]
    rows = [trim_at_eos(row, set(eos)) for row in generated]
    return (
        [tokenizer.decode(row, skip_special_tokens=False) for row in rows],
        [int(row.numel()) for row in rows],
    )


@torch.inference_mode()
def generate_tokenwise(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    device: str,
    suppression: DynamicSuppressionHook | None,
) -> tuple[list[str], list[int]]:
    """Greedy decode with one model forward for every prompt/generated token."""
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    eos = set(terminator_ids(tokenizer))
    pad_id = tokenizer.pad_token_id

    past_key_values = None
    outputs = None
    for token_index in range(input_ids.shape[1]):
        active = attention_mask[:, token_index].bool()
        if suppression is not None:
            suppression.set_active_rows(active)
        outputs = model(
            input_ids=input_ids[:, token_index : token_index + 1],
            attention_mask=attention_mask[:, : token_index + 1],
            position_ids=position_ids[:, token_index : token_index + 1],
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
    if outputs is None:
        raise RuntimeError("Tokenizer returned an empty prompt")

    batch_size = input_ids.shape[0]
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    next_ids = outputs.logits[:, -1, :].argmax(dim=-1)
    for row_index, token_id in enumerate(next_ids.tolist()):
        generated[row_index].append(token_id)
        if token_id in eos:
            finished[row_index] = True

    current_attention = attention_mask
    last_ids = next_ids
    for _ in range(1, max_new_tokens):
        if bool(finished.all()):
            break
        active = ~finished
        next_mask = active.to(current_attention.dtype).unsqueeze(1)
        current_attention = torch.cat([current_attention, next_mask], dim=1)
        positions = (current_attention.sum(dim=1, keepdim=True) - 1).clamp_min_(0)
        feed_ids = torch.where(
            active, last_ids, torch.full_like(last_ids, pad_id)
        ).unsqueeze(1)
        if suppression is not None:
            suppression.set_active_rows(active)
        outputs = model(
            input_ids=feed_ids,
            attention_mask=current_attention,
            position_ids=positions,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        candidates = outputs.logits[:, -1, :].argmax(dim=-1)
        for row_index, token_id in enumerate(candidates.tolist()):
            if not bool(finished[row_index]):
                generated[row_index].append(token_id)
                if token_id in eos:
                    finished[row_index] = True
        last_ids = candidates

    tensors = [torch.tensor(row, dtype=torch.long) for row in generated]
    return (
        [tokenizer.decode(row, skip_special_tokens=False) for row in tensors],
        [len(row) for row in generated],
    )


def build_payload(
    args: argparse.Namespace,
    results: list[dict],
    feature_indices: list[int] | None,
    suppression: DynamicSuppressionHook | None,
    range_end: int,
) -> dict:
    correct = sum(item["correct"] for item in results)
    lengths = [item["num_generated_tokens"] for item in results]
    return {
        "model": args.model,
        "dataset": args.dataset,
        "condition": args.condition,
        "samples": len(results),
        "target_samples": range_end - args.start_index,
        "dataset_range": [args.start_index, range_end],
        "accuracy": correct / len(results) if results else 0.0,
        "correct_count": correct,
        "mean_num_generated_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "layer": args.layer if suppression else None,
        "suppressed_features": feature_indices,
        "suppression_mode": "dynamic" if suppression else None,
        "suppression_strength": args.strength if suppression else None,
        "inference_engine": args.engine,
        "suppression_scope": (
            "all 25 features on every valid prompt and generation token forward"
            if suppression
            else None
        ),
        "hook_calls": suppression.calls if suppression else 0,
        "hook_processed_rows": suppression.processed_rows if suppression else 0,
        "hook_rows_with_any_selected_feature_active": (
            suppression.active_rows if suppression else 0
        ),
        "hook_rows_with_target_in_sae_topk": (
            suppression.topk_hit_rows if suppression else 0
        ),
        "results": results,
    }


def evaluate(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    all_rows = load_dataset(args.dataset, args.samples)
    range_end = args.end_index if args.end_index is not None else len(all_rows)
    if not 0 <= args.start_index <= range_end <= len(all_rows):
        raise ValueError(
            f"Invalid range [{args.start_index}, {range_end}) for {len(all_rows)} rows"
        )
    rows = all_rows[args.start_index:range_end]
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}; use --overwrite")

    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.dtype)
    suppression = None
    feature_indices = None
    hook_handle = None
    if args.condition == "anti-steer":
        if args.engine != "tokenwise":
            raise ValueError("anti-steer requires --engine tokenwise")
        feature_data = json.loads(args.features.read_text(encoding="utf-8"))
        feature_indices = feature_data["feature_indices"]
        if len(feature_indices) != 25:
            raise ValueError(f"Expected 25 features, got {len(feature_indices)}")
        sae = Sae.load_from_disk(args.sae_path, device=args.device)
        sae.eval()
        suppression = DynamicSuppressionHook(sae, feature_indices, args.strength)
        module = model.base_model.get_submodule(args.layer)
        hook_handle = module.register_forward_hook(suppression)

    results: list[dict] = []
    progress = tqdm(
        range(0, len(rows), args.batch_size),
        desc=f"{args.dataset}/{args.condition}",
    )
    for batch_start in progress:
        batch = rows[batch_start : batch_start + args.batch_size]
        prompts = [COT_TEMPLATE.format(question=row["question"]) for row in batch]
        if args.engine == "generate":
            responses, lengths = generate_with_transformers(
                model, tokenizer, prompts, args.max_new_tokens, args.device
            )
        else:
            responses, lengths = generate_tokenwise(
                model,
                tokenizer,
                prompts,
                args.max_new_tokens,
                args.device,
                suppression,
            )
        for offset, (row, response, length) in enumerate(
            zip(batch, responses, lengths)
        ):
            prediction = extract_prediction(args.dataset, response)
            results.append(
                {
                    "question_idx": args.start_index + batch_start + offset,
                    "question": row["question"],
                    "response": response,
                    "num_generated_tokens": length,
                    "predicted_answer": prediction,
                    "ground_truth": row["answer"],
                    "correct": is_correct(args.dataset, prediction, row["answer"]),
                }
            )
        correct = sum(item["correct"] for item in results)
        progress.set_postfix(acc=f"{correct / len(results):.3f}", n=len(results))
        atomic_json_dump(
            build_payload(args, results, feature_indices, suppression, range_end),
            args.output,
        )

    if hook_handle is not None:
        hook_handle.remove()
    payload = build_payload(args, results, feature_indices, suppression, range_end)
    atomic_json_dump(payload, args.output)
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["gsm8k", "bbh"], required=True)
    parser.add_argument("--condition", choices=["baseline", "anti-steer"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--sae-path", type=Path)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--layer", default="layers.29")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--engine", choices=["generate", "tokenwise"], default="tokenwise")
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.condition == "anti-steer" and args.sae_path is None:
        parser.error("--sae-path is required for anti-steer")
    if args.max_new_tokens < 1 or args.batch_size < 1 or args.samples < 1:
        parser.error("samples, batch-size, and max-new-tokens must be positive")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
