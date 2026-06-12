#!/usr/bin/env python3
"""Run R3 generative judge on benchmark pairs (Modal GPU)."""

import json
import os
import random
import time

import modal
from tqdm import tqdm

import _bootstrap  # noqa: F401
from dataset.benchmarks import iter_rewardbench, iter_rmbench, iter_skywork
from utils.config import BENCH, MAX_SAMPLES, MODEL_NAME, OUT_VOLUME_NAME
from utils.inference import R3Judge
from utils.paths import REPO_ROOT
from utils.prompts import build_r3_judge_prompt

app = modal.App("r3-rm-eval")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install(
        "vllm==0.15.1",
        "transformers",
        "datasets",
        "tqdm",
        "pandas",
        "numpy",
    )
    .env({"PYTHONPATH": "/app/src"})
    .add_local_dir(str(REPO_ROOT), remote_path="/app")
)

volume = modal.Volume.from_name(OUT_VOLUME_NAME, create_if_missing=True)


@app.cls(
    gpu="A100-80GB",
    image=vllm_image,
    timeout=3600 * 10,
    scaledown_window=300,
)
class RMJudge:
    @modal.enter()
    def enter(self):
        self.judge = R3Judge()

    @modal.method()
    def judge_all(
        self,
        prompts: list[str],
        chosen_texts: list[str] | None = None,
        rejected_texts: list[str] | None = None,
    ) -> tuple[list[str], list[int], list[int], list[int] | None, list[int] | None]:
        return self.judge.judge_all(prompts, chosen_texts, rejected_texts)


@app.function(
    image=vllm_image,
    volumes={"/outputs": volume},
    timeout=3600 * 10,
)
def generate_responses():
    random.seed(42)
    os.makedirs("/outputs", exist_ok=True)
    json_path = f"/outputs/{BENCH}__R3_Qwen3_14B_14k__responses.json"
    if os.path.exists(json_path):
        os.remove(json_path)

    if BENCH == "skywork":
        iterator = iter_skywork
    elif BENCH == "rmbench":
        iterator = iter_rmbench
    else:
        iterator = iter_rewardbench

    judge = RMJudge()
    print(f"Generating responses for {BENCH.upper()} (MAX_SAMPLES={MAX_SAMPLES}, model={MODEL_NAME})")

    all_ex, all_prompts = [], []
    for ex in tqdm(iterator(MAX_SAMPLES), desc="Building prompts"):
        swap = random.random() < 0.5
        if swap:
            resp1, resp2 = ex["rejected"], ex["chosen"]
            chosen_response = "Response 2"
        else:
            resp1, resp2 = ex["chosen"], ex["rejected"]
            chosen_response = "Response 1"
        all_ex.append((ex, chosen_response))
        all_prompts.append(build_r3_judge_prompt(ex["prompt"], resp1, resp2))

    chosen_list = [ex["chosen"] for (ex, _) in all_ex]
    rejected_list = [ex["rejected"] for (ex, _) in all_ex]

    t0 = time.time()
    texts, tokens_in_list, tokens_out_list, chosen_tokens_list, rejected_tokens_list = judge.judge_all.remote(
        all_prompts, chosen_list, rejected_list
    )
    total_dt = time.time() - t0
    print(f"Inference finished in {total_dt:.1f}s ({len(all_prompts) / total_dt:.1f} samples/sec)")

    results = []
    for (ex, chosen_response), out_text, tokens_in, tokens_out, chosen_tokens, rejected_tokens in zip(
        all_ex, texts, tokens_in_list, tokens_out_list, chosen_tokens_list, rejected_tokens_list
    ):
        blob = {
            "prompt": ex["prompt"],
            "response": out_text,
            "chosen_response": ex["chosen"],
            "rejected_response": ex["rejected"],
            "judge_tokens": tokens_in + tokens_out,
            "chosen_response_tokens": chosen_tokens,
            "rejected_response_tokens": rejected_tokens,
            "answer": chosen_response,
        }
        for key in ("id", "chosen_style", "rejected_style"):
            if key in ex:
                blob[key] = ex[key]
        results.append(blob)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} responses to {json_path}")
    return results


if __name__ == "__main__":
    fn = modal.Function.from_name("r3-rm-eval", "generate_responses")
    print(f"Launching {BENCH.upper()} job in background mode...")
    future = fn.spawn()
    print(f"Job started. Object ID: {future.object_id}")
    print(f"  modal volume get {OUT_VOLUME_NAME} /outputs/{BENCH}__R3_Qwen3_14B_14k__responses.json .")
