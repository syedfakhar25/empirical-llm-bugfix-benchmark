#!/usr/bin/env python3

import argparse
import json
import time
import traceback
import sys
import threading

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# NVML GPU ENERGY SUPPORT
# =========================

try:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage,
        nvmlDeviceGetName,
        nvmlDeviceGetMemoryInfo
    )
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


# =========================
# GPU ENERGY MONITOR
# =========================

class GPUEnergyMonitor:

    def __init__(self, device_index=0, sample_interval=0.1):
        self.device_index = device_index
        self.sample_interval = sample_interval
        self.energy_joules = 0.0
        self.running = False
        self.thread = None

    def _sample_loop(self):
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception:
            self.running = False
            return

        last_time = time.time()

        while self.running:
            try:
                power_mw = nvmlDeviceGetPowerUsage(handle)
                power_w = power_mw / 1000.0
            except Exception:
                power_w = 0.0

            now = time.time()
            delta = now - last_time
            last_time = now

            self.energy_joules += power_w * delta
            time.sleep(self.sample_interval)

        try:
            nvmlShutdown()
        except Exception:
            pass

    def start(self):
        if not NVML_AVAILABLE:
            return
        self.running = True
        self.thread = threading.Thread(target=self._sample_loop)
        self.thread.start()

    def stop(self):
        if not NVML_AVAILABLE:
            return
        self.running = False
        if self.thread:
            self.thread.join()


# =========================
# GPU INFORMATION
# =========================

def get_gpu_info():
    gpu_name = "CPU"
    gpu_memory_used = 0
    gpu_memory_total = 0

    if not NVML_AVAILABLE or not torch.cuda.is_available():
        return gpu_name, gpu_memory_used, gpu_memory_total

    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)

        gpu_name = nvmlDeviceGetName(handle).decode()

        mem = nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_used = mem.used
        gpu_memory_total = mem.total

        nvmlShutdown()
    except Exception:
        pass

    return gpu_name, gpu_memory_used, gpu_memory_total


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt_file")
    parser.add_argument("--max_new_tokens", type=int, default=800)
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    prompt = None
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8", errors="ignore") as f:
            prompt = f.read()
    elif args.prompt:
        prompt = args.prompt

    if not prompt:
        raise ValueError("Prompt is empty")

    try:
        # -------------------------
        # DEVICE DETECTION
        # -------------------------
        if args.device == "cpu":
            device = "cpu"
        elif args.device == "cuda":
            device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        device_type = "gpu" if device == "cuda" else "cpu"

        print(f"[INFO] Using device: {device}", file=sys.stderr)

        # -------------------------
        # LOAD TOKENIZER
        # -------------------------
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        # -------------------------
        # LOAD MODEL
        # -------------------------
        if device_type == "gpu":
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
                device_map=None
            )
            model.to("cuda")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float32,
                device_map=None
            )
            model.to("cpu")

        model.eval()
        print("[INFO] Model loaded", file=sys.stderr)

        # -------------------------
        # TOKENIZE
        # -------------------------
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_tokens = inputs["input_ids"].shape[-1]

        # -------------------------
        # ENERGY MONITOR
        # -------------------------
        monitor = GPUEnergyMonitor()
        monitor.start()

        if device_type == "gpu":
            torch.cuda.reset_peak_memory_stats()

        # -------------------------
        # GENERATION
        # -------------------------
        print("[INFO] Starting generation...", file=sys.stderr)
        gen_start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_end = time.time()
        print("[INFO] Generation finished", file=sys.stderr)

        monitor.stop()

        duration = gen_end - gen_start
        energy_joules = monitor.energy_joules if NVML_AVAILABLE else -1

        gpu_name, gpu_memory_used, gpu_memory_total = get_gpu_info()

        peak_gpu_memory = 0
        if device_type == "gpu":
            peak_gpu_memory = torch.cuda.max_memory_allocated()

        total_tokens = outputs.shape[-1]
        tokens_generated = total_tokens - input_tokens

        generated_text = tokenizer.decode(
            outputs[0][input_tokens:],
            skip_special_tokens=True
        )

        result = {
            "status": "OK",
            "device": device_type,
            "duration_seconds": duration,
            "energy_joules": energy_joules,
            "gpu_name": gpu_name,
            "gpu_memory_used": gpu_memory_used,
            "gpu_memory_total": gpu_memory_total,
            "gpu_peak_memory": peak_gpu_memory,
            "input_tokens": input_tokens,
            "tokens_generated": tokens_generated,
            "output": generated_text
        }

        print(json.dumps(result))
        sys.exit(0)

    except Exception as e:
        print("[ERROR]", str(e), file=sys.stderr)

        error = {
            "status": "LLM_CRASH",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

        print(json.dumps(error))
        sys.exit(1)


if __name__ == "__main__":
    main()