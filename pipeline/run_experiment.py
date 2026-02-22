# ============================================================
# For ME: ALWAYS ->  venv/bin/activate
# run_experiment.py
# - Repetition support (run_id)
# - Robust failure logging
# - AST-safe patch replacement
# - Scientific CSV logging
# - Local/HPC portable paths + metadata
# ============================================================

import argparse
import ast
import csv
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import time
from datetime import datetime

from extract_block import extract_buggy_block

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, "pipeline")
DEFAULT_PROJECTS = os.path.join(ROOT, "..", "BugsInPy", "projects")
RESULTS_FILE = os.path.join(ROOT, "results/results.csv")
RUN_LLM = os.path.join(PIPE, "run_llm.py")
SELECTED_BUGS_FILE = os.path.join(ROOT, "selected_bugs.json")


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


def detect_job_scheduler():
    # Human note: tiny helper to tag where this run happened.
    if os.environ.get("SLURM_JOB_ID"):
        return "slurm", os.environ.get("SLURM_JOB_ID")
    if os.environ.get("PBS_JOBID"):
        return "pbs", os.environ.get("PBS_JOBID")
    if os.environ.get("LSB_JOBID"):
        return "lsf", os.environ.get("LSB_JOBID")
    return "local", ""


def read_selected_bugs(path):
    with open(path, "r") as f:
        data = json.load(f)

    pairs = []
    for _, projects in data.items():
        for project, bugs in projects.items():
            for bug_id in bugs:
                pairs.append((project, int(bug_id)))
    return pairs


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


def ensure_repo_copy(repo_cache_root, url, project, eval_dir):
    cache_repo = os.path.join(repo_cache_root, project)
    target_repo = os.path.join(eval_dir, project)

    if not os.path.exists(cache_repo):
        os.makedirs(repo_cache_root, exist_ok=True)
        clone = run_cmd(f"git clone {url} {cache_repo}")
        if clone.returncode != 0:
            raise RuntimeError(f"Failed to clone cache repo: {clone.stderr}")

    if os.path.exists(target_repo):
        shutil.rmtree(target_repo)

    # Human note: copy from cached checkout -> faster on HPC than recloning each run.
    shutil.copytree(cache_repo, target_repo)
    return target_repo


def run_single(args, domain, project, bug_id, run_id):
    print(f"\n=== {project} | Bug {bug_id} | {args.model} | Run {run_id} | Backend {args.backend} ===")

    scheduler, scheduler_job_id = detect_job_scheduler()
    hostname = socket.gethostname()
    cpu_count = os.cpu_count() or 0

    eval_dir = os.path.join(args.eval_root, f"eval_{project}_{bug_id}_{run_id}")
    if os.path.exists(eval_dir):
        shutil.rmtree(eval_dir)
    os.makedirs(eval_dir, exist_ok=True)

    project_info = os.path.join(args.bugsinpy_projects_dir, project, "project.info")
    url = None
    for line in open(project_info):
        if line.startswith("github_url"):
            url = line.split("=")[1].strip().strip('"')

    if not url:
        raise RuntimeError(f"Missing github_url in {project_info}")

    repo = ensure_repo_copy(args.repo_cache_root, url, project, eval_dir)

    bug_info = os.path.join(args.bugsinpy_projects_dir, project, "bugs", str(bug_id), "bug.info")
    commit = None
    for line in open(bug_info):
        if "buggy_commit_id" in line:
            commit = line.split("=")[1].strip().strip('"')

    checkout = run_cmd(f"git checkout {commit}", cwd=repo)
    if checkout.returncode != 0:
        raise RuntimeError(f"git checkout failed: {checkout.stderr}")

    patch_file = os.path.join(args.bugsinpy_projects_dir, project, "bugs", str(bug_id), "bug_patch.txt")
    diff_line = [l for l in open(patch_file) if l.startswith("diff --git")][0]
    buggy_rel = re.search(r"a/(.+?) ", diff_line).group(1)
    buggy_path = os.path.join(repo, buggy_rel)

    os.environ["MODEL_NAME"] = args.model
    os.environ["BUGGY_FILE"] = buggy_path
    os.environ["PATCH_FILE"] = patch_file
    os.environ["LLM_BACKEND"] = args.backend
    os.environ["OLLAMA_URL"] = args.ollama_url
    os.environ["OLLAMA_KEEP_ALIVE"] = args.ollama_keep_alive
    os.environ["WARMUP_RUNS"] = str(args.warmup_runs)
    os.environ["OLLAMA_REQUEST_TIMEOUT"] = str(args.ollama_request_timeout)

    llm_crashed = 0
    llm_timeout = 0
    llm_error_message = ""

    llm_start = time.time()
    try:
        llm_output = run_cmd(f"python3 {RUN_LLM}", cwd=eval_dir, timeout=args.llm_timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        llm_crashed = 1
        llm_timeout = 1
        llm_error_message = f"LLM timeout after {args.llm_timeout_seconds}s"

        class TimeoutResult:
            returncode = 124
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\n{llm_error_message}\n"

        llm_output = TimeoutResult()

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
    rapl_match = re.search(r"rapl_available=([0-1])", out_text)

    duration = float(duration_match.group(1)) if duration_match else llm_wall_time
    energy_joules = float(energy_match.group(1)) if energy_match else 0.0
    avg_power_watts = energy_joules / duration if duration > 0 else 0.0
    memory_bytes = int(mem_match.group(1)) if mem_match else 0
    rapl_available = int(rapl_match.group(1)) if rapl_match else 0

    passed = 0
    test_returncode = -1

    patch_file_path = os.path.join(eval_dir, "llm_patch_block.py")

    if os.path.exists(patch_file_path) and llm_crashed == 0:
        fixed_block = open(patch_file_path).read().strip()

        generated_match = re.search(r"(class|def)\s+([A-Za-z0-9_]+)", fixed_block)
        with open(buggy_path, "r") as bf:
            source_code = bf.read()
        target_info = extract_buggy_block(source_code, patch_file)

        if target_info and generated_match:
            target_kind, block_name, _ = target_info

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
                    test_returncode = result.returncode
                    passed = 1 if result.returncode == 0 else 0

    with open(args.results_file, "a", newline="") as f:
        csv.writer(f).writerow(
            [
                datetime.now().isoformat(),
                domain,
                project,
                bug_id,
                args.model,
                args.backend,
                run_id,
                passed,
                duration,
                memory_bytes,
                energy_joules,
                avg_power_watts,
                args.warmup_runs,
                args.ollama_keep_alive,
                llm_crashed,
                llm_timeout,
                llm_wall_time,
                llm_error_message,
                rapl_available,
                scheduler,
                scheduler_job_id,
                hostname,
                cpu_count,
                platform.platform(),
                args.eval_root,
                args.bugsinpy_projects_dir,
                test_returncode,
            ]
        )

    print("→ Done.")


def init_results(results_file):
    results_dir = os.path.dirname(results_file)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not os.path.exists(results_file) or os.path.getsize(results_file) == 0:
        with open(results_file, "w", newline="") as f:
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
                    "rapl_available",
                    "scheduler",
                    "scheduler_job_id",
                    "hostname",
                    "cpu_count",
                    "platform",
                    "eval_root",
                    "bugsinpy_projects_dir",
                    "test_returncode",
                ]
            )
        print("[INIT] results.csv created with headers")


def build_job_list(args):
    # Human note: one script can run single bug or whole selected set.
    if args.all_selected:
        return read_selected_bugs(args.selected_bugs_file)

    if args.project and args.bug is not None:
        return [(args.project, args.bug)]

    raise ValueError("Use --project/--bug for one run, or --all_selected for batch mode.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", type=str)
    parser.add_argument("--bug", type=int)
    parser.add_argument("--all_selected", action="store_true")
    parser.add_argument("--selected_bugs_file", type=str, default=SELECTED_BUGS_FILE)

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "ollama"])

    parser.add_argument("--bugsinpy_projects_dir", type=str, default=DEFAULT_PROJECTS)
    parser.add_argument("--eval_root", type=str, default=PIPE)
    parser.add_argument("--repo_cache_root", type=str, default=os.path.join(PIPE, "repo_cache"))
    parser.add_argument("--results_file", type=str, default=RESULTS_FILE)

    parser.add_argument("--ollama_url", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--ollama_keep_alive", type=str, default="0")
    parser.add_argument("--warmup_runs", type=int, default=0)
    parser.add_argument("--ollama_request_timeout", type=int, default=900)
    parser.add_argument("--llm_timeout_seconds", type=int, default=1800)

    args = parser.parse_args()

    init_results(args.results_file)
    jobs = build_job_list(args)

    for project, bug_id in jobs:
        run_single(
            args=args,
            domain="manual",
            project=project,
            bug_id=bug_id,
            run_id=args.run_id,
        )
