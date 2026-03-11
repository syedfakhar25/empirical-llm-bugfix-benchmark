#!/usr/bin/env python3

import argparse
import json
import time
import traceback
import sys
import threading

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# NVML GPU ENERGY SUPPORT

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


# GPU ENERGY MONITOR

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


# GPU INFORMATION

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


# MAIN

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    try:

        # DEVICE SELECTION
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        device_type = "gpu" if device.type == "cuda" else "cpu"

        # LOAD MODEL
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if device_type == "gpu" else torch.float32,
        )

        model.to(device)

        # PREPARE INPUT

        inputs = tokenizer(
            args.prompt,
            return_tensors="pt"
        ).to(device)

        input_tokens = inputs["input_ids"].shape[-1]

        # GPU ENERGY MONITOR

        monitor = GPUEnergyMonitor()

        monitor.start()

        if device_type == "gpu":
            torch.cuda.reset_peak_memory_stats()

        # GENERATION

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

        # GPU MEMORY METRICS

        gpu_name, gpu_memory_used, gpu_memory_total = get_gpu_info()

        peak_gpu_memory = 0

        if device_type == "gpu":

            peak_gpu_memory = torch.cuda.max_memory_allocated()

        # OUTPUT PROCESSING

        generated_tokens = outputs.shape[-1]

        generated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # RESULT JSON

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

            "tokens_generated": generated_tokens,

            "output": generated_text
        }

        print(json.dumps(result))

        sys.exit(0)

    except Exception as e:

        error = {

            "status": "LLM_CRASH",

            "error": str(e),

            "traceback": traceback.format_exc()
        }

        print(json.dumps(error))

        sys.exit(1)


if __name__ == "__main__":
    main()