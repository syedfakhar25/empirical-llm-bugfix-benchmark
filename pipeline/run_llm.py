# ============================================================
# run_llm.py
# - CPU optimized
# - Deterministic runs
# - Proper energy + power logging
# - Thread controlled for reproducibility
# - Supports HF and Ollama backends
# ============================================================

import json
import os
import re
import time
import tracemalloc
from urllib import error, request

import pyRAPL
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_block import extract_buggy_block

# ------------------------------
# Deterministic + Controlled CPU
# ------------------------------
torch.manual_seed(42)
torch.set_num_threads(4)  # Control CPU usage (adjust if needed)

MODEL_NAME = os.environ["MODEL_NAME"]
BUGGY_FILE = os.environ["BUGGY_FILE"]
PATCH_FILE = os.environ["PATCH_FILE"]
LLM_BACKEND = os.environ.get("LLM_BACKEND", "hf").strip().lower()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "0")
WARMUP_RUNS = int(os.environ.get("WARMUP_RUNS", "0"))
OLLAMA_REQUEST_TIMEOUT = int(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "900"))


def resolve_ollama_model_name(model_name):
    """
    Accepts either Ollama tags (e.g. qwen2.5-coder:1.5b)
    or some HF-style ids and maps common ones.
    """
    lowered = model_name.lower().strip()

    # already looks like an ollama tag
    if ":" in lowered and "/" not in lowered:
        return lowered

    mapping = {
        "qwen/qwen2.5-coder-1.5b": "qwen2.5-coder:1.5b",
        "qwen/qwen2.5-coder-3b": "qwen2.5-coder:3b",
        "qwen/qwen2.5-coder-7b": "qwen2.5-coder:7b",
        "codellama:7b-instruct": "codellama:7b-instruct",
        "codellama/codellama-7b-instruct-hf": "codellama:7b-instruct",
        "deepseek-ai/deepseek-coder-1.3b-base": "deepseek-coder:1.3b",
    }
    return mapping.get(lowered, model_name)


# ------------------------------
# Model Loading
# ------------------------------
def load_hf_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,  # Stable on CPU
        low_cpu_mem_usage=True,
    )

    model.eval()
    return tok, model


def generate_with_ollama(prompt):
    resolved_model = resolve_ollama_model_name(MODEL_NAME)
    payload = {
        "model": resolved_model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "num_predict": 240,
            "seed": 42,
        },
    }

    req = request.Request(
        f"{OLLAMA_URL.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=OLLAMA_REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "")
    except error.URLError as exc:
        raise RuntimeError(
            f"Failed to call Ollama at {OLLAMA_URL}. "
            f"Check service is running and model is pulled. Details: {exc}"
        )


def generate_with_hf(prompt, tokenizer, model):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=200,
            temperature=0.0,
            do_sample=False,
        )

    gen_only = output_ids[0][encoded["input_ids"].shape[1] :]
    return tokenizer.decode(gen_only, skip_special_tokens=True)


# ------------------------------
# Clean Generated Code
# ------------------------------
def extract_clean_code(text):
    text = text.replace("```python", "").replace("```", "").strip()

    m = re.search(r"(def|class)\s+[A-Za-z0-9_]+", text)
    if not m:
        return ""

    return text[m.start() :].strip()


def enforce_block_name(generated_code, block_type, expected_name):
    """
    Ensure the generated top-level def/class keeps the same target name.
    """
    if not generated_code.strip():
        return generated_code

    pattern = r"^(\s*)(def|class)\s+([A-Za-z0-9_]+)"
    m = re.search(pattern, generated_code, flags=re.MULTILINE)
    if not m:
        return generated_code

    actual_kind = m.group(2)
    if actual_kind != block_type:
        return generated_code

    if m.group(3) == expected_name:
        return generated_code

    start, end = m.span(3)
    return generated_code[:start] + expected_name + generated_code[end:]


# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":

    with open(BUGGY_FILE, "r") as f:
        source = f.read()

    block_info = extract_buggy_block(source, PATCH_FILE)
    if not block_info:
        open("llm_patch_block.py", "w").write("# ERROR: no block detected")
        print("Patch generated → llm_patch_block.py (empty)")
        exit(0)

    block_type, block_name, buggy_block = block_info

    prompt = f"""
Fix the following {block_type} '{block_name}'.
Return ONLY the corrected {block_type} code.
Keep the exact same {block_type} name: {block_name}

### CODE ###
{buggy_block}
"""

    tokenizer, model = (None, None)
    if LLM_BACKEND != "ollama":
        tokenizer, model = load_hf_model()

    # Optional warm-up runs before measured run
    for _ in range(max(WARMUP_RUNS, 0)):
        if LLM_BACKEND == "ollama":
            _ = generate_with_ollama(prompt)
        else:
            _ = generate_with_hf(prompt, tokenizer, model)

    # ------------------------------
    # Energy Measurement Start
    # ------------------------------
    pyRAPL.setup()
    meter = pyRAPL.Measurement("llm_inference")
    meter.begin()

    tracemalloc.start()
    start = time.time()

    if LLM_BACKEND == "ollama":
        raw = generate_with_ollama(prompt)
    else:
        raw = generate_with_hf(prompt, tokenizer, model)

    # ------------------------------
    # Energy Measurement End
    # ------------------------------
    meter.end()

    duration = time.time() - start

    # Energy in microjoules → joules
    energy_uj = meter.result.pkg[0] if meter.result.pkg else 0.0
    energy_joules = energy_uj / 1e6

    # Compute Power (W = J / s)
    power_watts = energy_joules / duration if duration > 0 else 0.0

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ------------------------------
    # Clean Output
    # ------------------------------
    cleaned = extract_clean_code(raw)
    if not cleaned.strip():
        cleaned = "# ERROR: LLM returned no valid Python code"
    else:
        cleaned = enforce_block_name(
            cleaned,
            "def" if block_type == "function" else "class",
            block_name,
        )

    open("llm_patch_block.py", "w").write(cleaned)

    # ------------------------------
    # Final Logging
    # ------------------------------
    print("Patch generated → llm_patch_block.py")
    print(
        f"LLM_RESULT "
        f"duration={duration:.4f}s "
        f"mem_peak={peak_mem}bytes "
        f"energy={energy_joules:.6f}J "
        f"power={power_watts:.4f}W "
        f"backend={LLM_BACKEND} "
        f"warmup_runs={WARMUP_RUNS} "
        f"ollama_keep_alive={OLLAMA_KEEP_ALIVE} "
        f"ollama_request_timeout={OLLAMA_REQUEST_TIMEOUT}"
    )
