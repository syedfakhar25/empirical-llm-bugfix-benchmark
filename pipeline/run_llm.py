#!/usr/bin/env python3

import argparse
import json
import time
import traceback
import sys
import threading

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ============================================================
# GPU ENERGY MEASUREMENT (NVML)
# ============================================================

try:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage,
    )
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


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

            current_time = time.time()
            delta = current_time - last_time
            last_time = current_time

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


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    start_total = time.time()

    try:
        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

        # ----------------------------------------------------
        # Start GPU energy monitor
        # ----------------------------------------------------
        monitor = GPUEnergyMonitor()
        monitor.start()

        gen_start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False
            )

        gen_end = time.time()
        monitor.stop()

        duration = gen_end - gen_start
        energy_joules = monitor.energy_joules if NVML_AVAILABLE else -1

        generated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        result = {
            "status": "OK",
            "duration_seconds": duration,
            "energy_joules": energy_joules,
            "output": generated_text,
        }

        print(json.dumps(result))
        sys.exit(0)

    except Exception as e:
        error = {
            "status": "LLM_CRASH",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(error))
        sys.exit(1)


if __name__ == "__main__":
    main()