# ============================================================
#For ME: ALWAYS ->  venv/bin/activate
# run_experiment.py
# - Repetition support (run_id)
# - Robust failure logging
# - AST-safe patch replacement
# - Scientific CSV logging
# ============================================================

import argparse
import ast
import csv
import os
import re
import shutil
import subprocess
import time
from datetime import datetime

from extract_block import extract_buggy_block

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, "pipeline")
PROJECTS = os.path.join(ROOT, "..", "BugsInPy", "projects")
RESULTS_FILE = os.path.join(ROOT, "results/results.csv")
RUN_LLM = os.path.join(PIPE, "run_llm.py")


def run_cmd(cmd, cwd=None, timeout=None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def replace_block_in_file(file_path, block_name, new_block):
    with open(file_path, "r") as f:
        source = f.read()

    tree = ast.parse(source)
    target_node = None

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == block_name:
            target_node = node
            break

        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == block_name:
                    target_node = node
                    break

    if not target_node:
        print("ERROR: AST block not found.")
        return False

    lines = source.splitlines()
    start = target_node.lineno - 1
    end = target_node.end_lineno

    updated_lines = lines[:start] + new_block.splitlines() + lines[end:]

    with open(file_path, "w") as f:
        f.write("\n".join(updated_lines))

    return True


def run_single(
    domain,
    project,
    bug_id,
    model,
    run_id,
    backend,
    ollama_url,
    ollama_keep_alive,
    warmup_runs,
    llm_timeout_seconds,
):

    print(f"\n=== {project} | Bug {bug_id} | {model} | Run {run_id} | Backend {backend} ===")

    eval_dir = os.path.join(PIPE, f"eval_{project}_{bug_id}_{run_id}")
    if os.path.exists(eval_dir):
        shutil.rmtree(eval_dir)
    os.makedirs(eval_dir)

    # Clone repository
    project_info = os.path.join(PROJECTS, project, "project.info")
    url = None
    for line in open(project_info):
        if line.startswith("github_url"):
            url = line.split("=")[1].strip().strip('"')

    run_cmd(f"git clone {url} {eval_dir}/{project}")

    # Checkout buggy commit
    bug_info = os.path.join(PROJECTS, project, "bugs", str(bug_id), "bug.info")
    commit = None
    for line in open(bug_info):
        if "buggy_commit_id" in line:
            commit = line.split("=")[1].strip().strip('"')

    repo = os.path.join(eval_dir, project)
    run_cmd(f"git checkout {commit}", cwd=repo)

    # Locate buggy file
    patch_file = os.path.join(PROJECTS, project, "bugs", str(bug_id), "bug_patch.txt")
    diff_line = [l for l in open(patch_file) if l.startswith("diff --git")][0]
    buggy_rel = re.search(r"a/(.+?) ", diff_line).group(1)
    buggy_path = os.path.join(repo, buggy_rel)

    # Run LLM
    os.environ["MODEL_NAME"] = model
    os.environ["BUGGY_FILE"] = buggy_path
    os.environ["PATCH_FILE"] = patch_file
    os.environ["LLM_BACKEND"] = backend
    os.environ["OLLAMA_URL"] = ollama_url
    os.environ["OLLAMA_KEEP_ALIVE"] = ollama_keep_alive
    os.environ["WARMUP_RUNS"] = str(warmup_runs)

    llm_crashed = 0
    llm_timeout = 0
    llm_error_message = ""

    llm_start = time.time()
    try:
        llm_output = run_cmd(f"python3 {RUN_LLM}", cwd=eval_dir, timeout=llm_timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        llm_crashed = 1
        llm_timeout = 1
        llm_error_message = f"LLM timeout after {llm_timeout_seconds}s"
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

        class TimeoutResult:
            returncode = 124

        llm_output = TimeoutResult()
        llm_output.stdout = stdout
        llm_output.stderr = stderr + f"\n{llm_error_message}\n"

    llm_wall_time = time.time() - llm_start

    if llm_output.returncode != 0:
        llm_crashed = 1
        if not llm_error_message:
            llm_error_message = f"run_llm.py failed with exit code {llm_output.returncode}"
        print("LLM crashed!")
        print("STDERR:", llm_output.stderr)

    out_text = llm_output.stdout

    duration_match = re.search(r"duration=([0-9\.]+)", out_text)
    energy_match = re.search(r"energy=([0-9\.]+)", out_text)
    mem_match = re.search(r"mem_peak=([0-9]+)", out_text)

    # If run_llm crashed before metric print, fall back to process wall time
    duration = float(duration_match.group(1)) if duration_match else llm_wall_time
    energy_joules = float(energy_match.group(1)) if energy_match else 0.0
    avg_power_watts = energy_joules / duration if duration > 0 else 0.0
    memory_bytes = int(mem_match.group(1)) if mem_match else 0

    passed = 0

    patch_file_path = os.path.join(eval_dir, "llm_patch_block.py")

    if os.path.exists(patch_file_path) and llm_crashed == 0:
        fixed_block = open(patch_file_path).read().strip()

        generated_match = re.search(r"(class|def)\s+([A-Za-z0-9_]+)", fixed_block)
        with open(buggy_path, "r") as bf:
            source_code = bf.read()
        target_info = extract_buggy_block(source_code, patch_file)

        if target_info and generated_match:
            target_kind, block_name, _ = target_info

            # If LLM changed the name, rewrite it back to target
            kind = generated_match.group(1)
            gen_name = generated_match.group(2)
            if gen_name != block_name:
                fixed_block = re.sub(
                    rf"^({kind}\s+){re.escape(gen_name)}",
                    rf"\1{block_name}",
                    fixed_block,
                    count=1,
                    flags=re.MULTILINE,
                )

            expected_kind = "def" if target_kind == "function" else "class"
            if kind == expected_kind:
                replaced = replace_block_in_file(buggy_path, block_name, fixed_block)

                if replaced:
                    result = run_cmd("bash run_test.sh", cwd=repo)
                    passed = 1 if result.returncode == 0 else 0

    # Always log result
    with open(RESULTS_FILE, "a", newline="") as f:
        csv.writer(f).writerow(
            [
                datetime.now().isoformat(),
                domain,
                project,
                bug_id,
                model,
                backend,
                run_id,
                passed,
                duration,
                memory_bytes,
                energy_joules,
                avg_power_watts,
                warmup_runs,
                ollama_keep_alive,
                llm_crashed,
                llm_timeout,
                llm_wall_time,
                llm_error_message,
            ]
        )

    print("→ Done.")


def init_results():
    results_dir = os.path.dirname(RESULTS_FILE)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not os.path.exists(RESULTS_FILE) or os.path.getsize(RESULTS_FILE) == 0:
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "domain",
                    "project",
                    "bug_id",
                    "model",
                    "backend",
                    "run_id",
                    "pass_at_1",
                    "duration_seconds",
                    "memory_bytes",
                    "energy_joules",
                    "avg_power_watts",
                    "warmup_runs",
                    "ollama_keep_alive",
                    "llm_crashed",
                    "llm_timeout",
                    "llm_wall_time_seconds",
                    "llm_error_message",
                ]
            )
        print("[INIT] results.csv created with headers")


if __name__ == "__main__":

    init_results()

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--bug", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "ollama"])
    parser.add_argument("--ollama_url", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--ollama_keep_alive", type=str, default="0")
    parser.add_argument("--warmup_runs", type=int, default=0)
    parser.add_argument("--llm_timeout_seconds", type=int, default=1800)

    args = parser.parse_args()

    run_single(
        domain="manual",
        project=args.project,
        bug_id=args.bug,
        model=args.model,
        run_id=args.run_id,
        backend=args.backend,
        ollama_url=args.ollama_url,
        ollama_keep_alive=args.ollama_keep_alive,
        warmup_runs=args.warmup_runs,
        llm_timeout_seconds=args.llm_timeout_seconds,
    )
