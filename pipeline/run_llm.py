# ============================================================
# run_llm.py
# - Removed device_map (no accelerate needed)
# - Deterministic runs
# - Proper energy logging
# ============================================================

import os
import time
import tracemalloc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
#from codecarbon import EmissionsTracker
import pyRAPL
from extract_block import extract_buggy_block
import re

torch.manual_seed(42)

MODEL_NAME = os.environ["MODEL_NAME"]
BUGGY_FILE = os.environ["BUGGY_FILE"]
PATCH_FILE = os.environ["PATCH_FILE"]

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32   # Explicit CPU dtype
    )
    model.eval()  # Important for inference stability
    return tok, model

def extract_clean_code(text):
    text = text.replace("```python", "").replace("```", "").strip()

    m = re.search(r"(def|class)\s+[A-Za-z0-9_]+", text)
    if not m:
        return ""

    return text[m.start():].strip()

if __name__ == "__main__":

    tokenizer, model = load_model()

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

### CODE ###
{buggy_block}
"""
    pyRAPL.setup()
    meter = pyRAPL.Measurement("llm_inference")
    meter.begin()

    tracemalloc.start()
    start = time.time()

    encoded = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():  # important for inference
        output_ids = model.generate(
            **encoded,
            max_new_tokens=300,
            temperature=0.0,
            do_sample=False
        )

    gen_only = output_ids[0][encoded["input_ids"].shape[1]:]
    raw = tokenizer.decode(gen_only, skip_special_tokens=True)

    meter.end()

    energy_uj = meter.result.pkg[0] if meter.result.pkg else 0.0
    energy_joules = energy_uj / 1e6
    emissions = 0.0

    duration = time.time() - start

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()


    cleaned = extract_clean_code(raw)
    if not cleaned.strip():
        cleaned = "# ERROR: LLM returned no valid Python code"

    open("llm_patch_block.py", "w").write(cleaned)

    print("Patch generated → llm_patch_block.py")
    print(f"LLM_RESULT duration={duration} mem={peak_mem} energy={energy_joules} emissions={emissions}")
