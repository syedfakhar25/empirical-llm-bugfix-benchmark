# ============================================================
# For ME: ALWAYS ->  venv/bin/activate
# ============================================================
# run_experiment.py (Per-bug isolated venv version)
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


def create_virtualenv(venv_path):
    run_cmd(f"python3 -m venv {venv_path}")
    pip_path = os.path.join(venv_path, "bin", "pip")
    run_cmd(f"{pip_path} install --upgrade pip")


def install_requirements(venv_path, requirements_file):
    pip_path = os.path.join(venv_path, "bin", "pip")
    if os.path.exists(requirements_file):
        run_cmd(f"{pip_path} install -r {requirements_file}")

def install_common_test_tools(venv_path):
    pip_path = os.path.join(venv_path, "bin", "pip")
    run_cmd(f"{pip_path} install pytest")
    run_cmd(f"{pip_path} install pytest-xdist")
    run_cmd(f"{pip_path} install pytest-httpbin")

def detect_job_scheduler():
    if os.environ.get("SLURM_JOB_ID"):
        return "slurm", os.environ.get("SLURM_JOB_ID")
    return "local", ""


def ensure_repo_copy(repo_cache_root, url, project, eval_dir):
    cache_repo = os.path.join(repo_cache_root, project)
    target_repo = os.path.join(eval_dir, project)

    if not os.path.exists(cache_repo):
        os.makedirs(repo_cache_root, exist_ok=True)
        clone = run_cmd(f"git clone {url} {cache_repo}")
        if clone.returncode != 0:
            raise RuntimeError(f"Clone failed: {clone.stderr}")

    if os.path.exists(target_repo):
        shutil.rmtree(target_repo)

    shutil.copytree(cache_repo, target_repo)
    return target_repo


def run_single(args, project, bug_id, run_id):

    model_safe = args.model.replace("/", "_")
    eval_dir = os.path.join(
        args.eval_root,
        f"eval_{project}_{bug_id}_{model_safe}_{run_id}",
    )

    if os.path.exists(eval_dir):
        shutil.rmtree(eval_dir)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"\n=== {project} Bug {bug_id} | {args.model} | Run {run_id} ===")

    scheduler, scheduler_job_id = detect_job_scheduler()
    hostname = socket.gethostname()
    cpu_count = os.cpu_count() or 0

    project_info = os.path.join(args.bugsinpy_projects_dir, project, "project.info")
    url = None
    for line in open(project_info):
        if line.startswith("github_url"):
            url = line.split("=")[1].strip().strip('"')

    repo = ensure_repo_copy(args.repo_cache_root, url, project, eval_dir)

    bug_info = os.path.join(
        args.bugsinpy_projects_dir, project, "bugs", str(bug_id), "bug.info"
    )
    commit = None
    for line in open(bug_info):
        if "buggy_commit_id" in line:
            commit = line.split("=")[1].strip().strip('"')

    run_cmd(f"git checkout {commit}", cwd=repo)

    # -------------------------
    # CREATE PER-BUG VENV
    # -------------------------
    venv_path = os.path.join(eval_dir, "venv")
    create_virtualenv(venv_path)

    requirements_file = os.path.join(
        args.bugsinpy_projects_dir,
        project,
        "bugs",
        str(bug_id),
        "requirements.txt",
    )

    install_requirements(venv_path, requirements_file)
    install_common_test_tools(venv_path)

    python_bin = os.path.join(venv_path, "bin", "python")

    # -------------------------
    # RUN LLM
    # -------------------------
    patch_file = os.path.join(
        args.bugsinpy_projects_dir,
        project,
        "bugs",
        str(bug_id),
        "bug_patch.txt",
    )

    diff_line = [l for l in open(patch_file) if l.startswith("diff --git")][0]
    buggy_rel = re.search(r"a/(.+?) ", diff_line).group(1)
    buggy_path = os.path.join(repo, buggy_rel)

    os.environ["MODEL_NAME"] = args.model
    os.environ["BUGGY_FILE"] = buggy_path
    os.environ["PATCH_FILE"] = patch_file
    os.environ["LLM_BACKEND"] = args.backend

    llm_start = time.time()
    llm_output = run_cmd(f"python3 {RUN_LLM}", cwd=eval_dir)
    llm_wall_time = time.time() - llm_start

    duration_match = re.search(r"duration=([0-9\.]+)", llm_output.stdout)
    energy_match = re.search(r"energy=([0-9\.]+)", llm_output.stdout)

    duration = float(duration_match.group(1)) if duration_match else llm_wall_time
    energy_joules = float(energy_match.group(1)) if energy_match else 0.0

    # -------------------------
    # APPLY PATCH + RUN TEST
    # -------------------------
    passed = 0
    test_returncode = -1

    patch_file_path = os.path.join(eval_dir, "llm_patch_block.py")

    if os.path.exists(patch_file_path):
        fixed_block = open(patch_file_path).read()

        target_info = extract_buggy_block(open(buggy_path).read(), patch_file)

        if target_info:
            block_type, block_name, _ = target_info

            tree = ast.parse(open(buggy_path).read())
            lines = open(buggy_path).read().splitlines()

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == block_name:
                    start = node.lineno - 1
                    end = node.end_lineno
                    lines = lines[:start] + fixed_block.splitlines() + lines[end:]
                    break

            with open(buggy_path, "w") as f:
                f.write("\n".join(lines))

            test_cmd = f"{python_bin} -m pytest"
            result = run_cmd(test_cmd, cwd=repo)
            test_returncode = result.returncode
            passed = 1 if result.returncode == 0 else 0

    # -------------------------
    # LOG RESULTS
    # -------------------------
    with open(args.results_file, "a", newline="") as f:
        csv.writer(f).writerow(
            [
                datetime.now().isoformat(),
                project,
                bug_id,
                args.model,
                run_id,
                passed,
                duration,
                energy_joules,
                test_returncode,
                scheduler,
                scheduler_job_id,
                hostname,
                cpu_count,
                platform.platform(),
            ]
        )

    print("→ Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--bug", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--backend", default="hf")
    parser.add_argument("--bugsinpy_projects_dir", default=DEFAULT_PROJECTS)
    parser.add_argument("--eval_root", default=PIPE)
    parser.add_argument("--repo_cache_root", default=os.path.join(PIPE, "repo_cache"))
    parser.add_argument("--results_file", default=RESULTS_FILE)

    args = parser.parse_args()

    run_single(args, args.project, args.bug, args.run_id)