# ============================================================
# FINAL STABLE THESIS RUNNER
# - Separate LLM runtime
# - Separate bug runtime
# - Dependency isolation
# - Always logs energy + duration
# ============================================================

import argparse
import ast
import csv
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime

from extract_block import extract_buggy_block

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, "pipeline")
DEFAULT_PROJECTS = os.path.join(ROOT, "..", "BugsInPy", "projects")
RESULTS_FILE = os.path.join(ROOT, "results/results.csv")
RUN_LLM = os.path.join(PIPE, "run_llm.py")


def run_cmd(cmd, cwd=None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def create_virtualenv(venv_path, python_exec):
    result = run_cmd(f"{python_exec} -m venv {venv_path}")
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    pip = os.path.join(venv_path, "bin", "pip")
    run_cmd(f"{pip} install --upgrade pip setuptools wheel")


def install_requirements(venv_path, requirements_file):
    pip = os.path.join(venv_path, "bin", "pip")

    if os.path.exists(requirements_file):
        r = run_cmd(f"{pip} install -r {requirements_file}")
        if r.returncode != 0:
            return False

    run_cmd(f"{pip} install pytest")
    return True


def run_single(args):

    project = args.project
    bug_id = args.bug
    run_id = args.run_id
    model = args.model

    model_safe = model.replace("/", "_")
    eval_dir = os.path.join(
        args.eval_root,
        f"eval_{project}_{bug_id}_{model_safe}_{run_id}",
    )

    if os.path.exists(eval_dir):
        shutil.rmtree(eval_dir)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"\n=== {project} Bug {bug_id} | {model} | Run {run_id} ===")

    # -------------------- Clone repo --------------------

    project_info = os.path.join(args.bugsinpy_projects_dir, project, "project.info")
    url = None
    for line in open(project_info):
        if line.startswith("github_url"):
            url = line.split("=")[1].strip().strip('"')

    repo = os.path.join(eval_dir, project)
    if not os.path.exists(repo):
        run_cmd(f"git clone {url} {repo}")

    bug_info = os.path.join(
        args.bugsinpy_projects_dir, project, "bugs", str(bug_id), "bug.info"
    )

    for line in open(bug_info):
        if "buggy_commit_id" in line:
            commit = line.split("=")[1].strip().strip('"')

    run_cmd(f"git checkout {commit}", cwd=repo)

    # -------------------- Bug runtime --------------------

    venv_path = os.path.join(eval_dir, "venv")
    create_virtualenv(venv_path, args.bug_python)

    requirements_file = os.path.join(
        args.bugsinpy_projects_dir,
        project,
        "bugs",
        str(bug_id),
        "requirements.txt",
    )

    deps_ok = install_requirements(venv_path, requirements_file)
    python_bin = os.path.join(venv_path, "bin", "python")

    # -------------------- LLM runtime --------------------

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

    os.environ["MODEL_NAME"] = model
    os.environ["BUGGY_FILE"] = buggy_path
    os.environ["PATCH_FILE"] = patch_file
    os.environ["LLM_BACKEND"] = args.backend

    start = time.time()
    llm_out = run_cmd(f"{args.llm_python} {RUN_LLM}", cwd=eval_dir)
    duration = time.time() - start

    energy_match = re.search(r"energy=([0-9\.]+)", llm_out.stdout)
    energy_joules = float(energy_match.group(1)) if energy_match else 0.0

    status = "LLM_DONE"
    passed = 0
    test_returncode = -1

    if not deps_ok:
        status = "DEPENDENCY_INSTALL_FAIL"

    else:
        patch_file_path = os.path.join(eval_dir, "llm_patch_block.py")

        if not os.path.exists(patch_file_path):
            status = "LLM_NO_VALID_CODE"
        else:
            fixed_block = open(patch_file_path).read()
            target_info = extract_buggy_block(open(buggy_path).read(), patch_file)

            if target_info:
                _, block_name, _ = target_info

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

                result = run_cmd(
                    f"PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 {python_bin} -m pytest",
                    cwd=repo,
                )

                test_returncode = result.returncode
                passed = 1 if result.returncode == 0 else 0

                status = "LLM_OK_TEST_PASS" if passed else "LLM_OK_TEST_FAIL"

    # -------------------- Log --------------------

    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)

    with open(args.results_file, "a", newline="") as f:
        csv.writer(f).writerow(
            [
                datetime.now().isoformat(),
                project,
                bug_id,
                model,
                run_id,
                status,
                passed,
                duration,
                energy_joules,
                test_returncode,
                socket.gethostname(),
                os.cpu_count(),
                platform.platform(),
            ]
        )

    print("→ Done:", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--bug", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--backend", default="hf")
    parser.add_argument("--bug-python", required=True)
    parser.add_argument("--llm-python", default=sys.executable)
    parser.add_argument("--bugsinpy_projects_dir", default=DEFAULT_PROJECTS)
    parser.add_argument("--eval_root", default=PIPE)
    parser.add_argument("--results_file", default=RESULTS_FILE)

    args = parser.parse_args()
    run_single(args)