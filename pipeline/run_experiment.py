#!/usr/bin/env python3

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
import sys
import time
from datetime import datetime

from extract_block import extract_buggy_block


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, "pipeline")
DEFAULT_PROJECTS = os.path.join(ROOT, "..", "BugsInPy", "projects")
RESULTS_FILE = os.path.join(ROOT, "results/results.csv")
RUN_LLM = os.path.join(PIPE, "run_llm.py")


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def run_cmd(cmd, cwd=None):

    result = subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print(result.stdout, flush=True)
    print(result.stderr, flush=True)

    return result


# ------------------------------------------------------------
# Virtual Environment
# ------------------------------------------------------------

def create_virtualenv(venv_path, python_exec):

    base_env = os.path.join(ROOT, "base_env")

    # Create base environment once
    if not os.path.exists(base_env):

        subprocess.run(f"{python_exec} -m venv {base_env}", shell=True)

        pip = os.path.join(base_env, "bin", "pip")

        subprocess.run(
            f"{pip} install --upgrade pip",
            shell=True
        )

        subprocess.run(
            f"{pip} install 'setuptools<70' 'cython<3' wheel pytest nose pytest-xdist",
            shell=True
        )

    # Copy base env for this bug
    if os.path.exists(venv_path):
        shutil.rmtree(venv_path)

    shutil.copytree(base_env, venv_path)

# ------------------------------------------------------------
# Dependency Installation
# ------------------------------------------------------------

def install_dependencies(repo, pip, project, bug_id, bugsinpy_projects_dir, eval_dir):

    logs = []

    def safe_install(req_file, cwd=None):

        with open(req_file) as f:
            lines = f.readlines()

        cleaned = [
            l for l in lines
            if "pkg-resources==0.0.0" not in l
        ]

        temp_req = os.path.join(eval_dir, "temp_requirements.txt")

        with open(temp_req, "w") as f:
            f.writelines(cleaned)

        r = run_cmd(
                    f"{pip} install --prefer-binary --no-input -r {temp_req} || true",
                    cwd=cwd
                )

        logs.append(r.stdout + r.stderr)

    bug_req = os.path.join(
        bugsinpy_projects_dir,
        project,
        "bugs",
        str(bug_id),
        "requirements.txt",
    )

    if os.path.exists(bug_req):

        print("Installing bug specific requirements")

        safe_install(bug_req)

    req_files = [
        "requirements.txt",
        "requirements-dev.txt",
        "dev-requirements.txt",
        "test-requirements.txt",
    ]

    for rf in req_files:

        path = os.path.join(repo, rf)

        if os.path.exists(path):

            print(f"Installing {rf}")

            safe_install(path, cwd=repo)

    if os.path.exists(os.path.join(repo, "setup.py")):

        print("Installing project (setup.py)")

        r = run_cmd(f"{pip} install --no-deps -e . || true", cwd=repo)

        logs.append(r.stdout + r.stderr)

    with open(os.path.join(eval_dir, "dependency_install_log.txt"), "w") as f:

        for l in logs:
            f.write(l)


# ------------------------------------------------------------
# LLM Output Cleaning
# ------------------------------------------------------------

def clean_llm_output(raw_output, expected_name):

    if not raw_output:
        return "", "EMPTY_OUTPUT"

    text = raw_output.replace("```python", "").replace("```", "")

    pattern = rf"(def|class)\s+{expected_name}\b"

    match = re.search(pattern, text)

    if not match:
        return "", "NO_MATCHING_BLOCK"

    candidate = text[match.start():]

    lines = candidate.splitlines()

    for i in range(len(lines), 0, -1):

        snippet = "\n".join(lines[:i])

        try:

            tree = ast.parse(snippet)

            if len(tree.body) > 0:

                node = tree.body[0]

                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):

                    return "\n".join(
                        snippet.splitlines()[node.lineno - 1: node.end_lineno]
                    ), None

        except Exception:
            continue

    return "", "AST_PARSE_FAIL"


# ------------------------------------------------------------
# Safe JSON extraction
# ------------------------------------------------------------

def parse_llm_json(output):

    try:
        return json.loads(output)
    except:

        match = re.search(r"\{.*\}", output, re.DOTALL)

        if match:
            return json.loads(match.group(0))

    return None


# ------------------------------------------------------------
# CSV HEADER
# ------------------------------------------------------------

def ensure_csv_header(path):

    if os.path.exists(path):
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:

        csv.writer(f).writerow([
            "timestamp",
            "project",
            "bug_id",
            "model",
            "run_id",
            "status",
            "passed",
            "duration_seconds",
            "energy_joules",
            "device",
            "gpu_name",
            "gpu_memory_used",
            "gpu_memory_total",
            "gpu_peak_memory",
            "input_tokens",
            "tokens_generated",
            "test_returncode",
            "hostname",
            "cpu_count",
            "platform"
        ])


# ------------------------------------------------------------
# Run Single Bug
# ------------------------------------------------------------

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

    project_info = os.path.join(args.bugsinpy_projects_dir, project, "project.info")

    url = None

    for line in open(project_info):

        if line.startswith("github_url"):
            url = line.split("=")[1].strip().strip('"')

    repo = os.path.join(eval_dir, project)

    result = run_cmd(f"git clone {url} {repo}")
    if result.returncode != 0:
        raise RuntimeError(f"GIT CLONE FAILED")

    bug_info = os.path.join(
        args.bugsinpy_projects_dir, project, "bugs", str(bug_id), "bug.info"
    )

    commit = None

    for line in open(bug_info):

        if "buggy_commit_id" in line:
            commit = line.split("=")[1].strip().strip('"')

    result = run_cmd(f"git checkout {commit}", cwd=repo)
    if result.returncode != 0:
        raise RuntimeError(f"GIT CHECKOUT FAILED")

    venv_path = os.path.join(eval_dir, "venv")

    create_virtualenv(venv_path, args.bug_python)

    pip = os.path.join(venv_path, "bin", "pip")

    install_dependencies(
        repo,
        pip,
        project,
        bug_id,
        args.bugsinpy_projects_dir,
        eval_dir,
    )

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

    with open(buggy_path) as f:
        source = f.read()

    block_info = extract_buggy_block(source, patch_file)

    if not block_info:

        status = "BLOCK_DETECTION_FAIL"
        passed = 0
        duration = 0
        energy_joules = 0
        test_returncode = -1

    else:

        block_type, block_name, buggy_block = block_info

        prompt = f"""Fix the following {block_type} '{block_name}'.

Return ONLY corrected code.

{buggy_block}
"""

        prompt = prompt.replace('"', '\\"')

        start = time.time()

        cmd = (
            f"{args.llm_python} {RUN_LLM} "
            f"--model \"{model}\" "
            f"--prompt \"{prompt}\""
        )

        llm_out = run_cmd(cmd, cwd=eval_dir)

        duration = time.time() - start

        with open(os.path.join(eval_dir, "llm_stdout.txt"), "w") as f:
            f.write(llm_out.stdout)

        result_json = parse_llm_json(llm_out.stdout)

        if result_json is None:

            status = "LLM_OUTPUT_PARSE_ERROR"
            passed = 0
            energy_joules = 0
            test_returncode = -1

        else:

            energy_joules = result_json.get("energy_joules", 0)

            raw_output = result_json.get("output", "")

            cleaned_code, error = clean_llm_output(raw_output, block_name)

            if error:

                status = f"LLM_STRUCTURE_ERROR_{error}"
                passed = 0
                test_returncode = -1

            else:

                with open(buggy_path, "w") as f:
                    f.write(cleaned_code)

                run_test_script = os.path.abspath(os.path.join(
                                        args.bugsinpy_projects_dir,
                                        project,
                                        "bugs",
                                        str(bug_id),
                                        "run_test.sh",
                                    ))

                if os.path.exists(run_test_script):

                    result = run_cmd(
                        f". {venv_path}/bin/activate && PYTEST_ADDOPTS='--assert=plain' bash {run_test_script}",
                        cwd=repo,
                    )

                    # Fallback if pytest collected nothing
                    if "collected 0 items" in result.stdout or "collected 0 items" in result.stderr:
                        print("Fallback: running full pytest suite", flush=True)

                        result = run_cmd(
                            f". {venv_path}/bin/activate && pytest -q  --maxfail=1",
                            cwd=repo,
                        )

                else:

                    result = run_cmd(
                        f". {venv_path}/bin/activate && pytest -q",
                        cwd=repo,
                    )
                # Save test logs
                with open(os.path.join(eval_dir, "test_stdout.txt"), "w") as f:
                    f.write(result.stdout)

                with open(os.path.join(eval_dir, "test_stderr.txt"), "w") as f:
                    f.write(result.stderr)

                passed = 1 if result.returncode == 0 else 0
                status = "LLM_OK_TEST_PASS" if passed else "LLM_OK_TEST_FAIL"
                test_returncode = result.returncode

    ensure_csv_header(args.results_file)

    with open(args.results_file, "a", newline="") as f:

        csv.writer(f).writerow([
            datetime.now().isoformat(),
            project,
            bug_id,
            model,
            run_id,
            status,
            passed,
            duration,
            energy_joules,
            "cpu",
            "NA",
            0,
            0,
            0,
            0,
            0,
            test_returncode,
            socket.gethostname(),
            os.cpu_count(),
            platform.platform(),
        ])

    print("→ Done:", status)


# ------------------------------------------------------------
# Run All Selected Bugs
# ------------------------------------------------------------

def run_all_selected(args):

    with open(args.selected_bugs_file) as f:
        selected = json.load(f)

    for domain, projects in selected.items():

        print(f"\n=== DOMAIN: {domain} ===")

        for project, bugs in projects.items():

            for bug in bugs:

                single_args = argparse.Namespace(
                    project=project,
                    bug=bug,
                    model=args.model,
                    run_id=args.run_id,
                    bug_python=args.bug_python,
                    llm_python=args.llm_python,
                    bugsinpy_projects_dir=args.bugsinpy_projects_dir,
                    eval_root=args.eval_root,
                    results_file=args.results_file,
                )

                try:
                    run_single(single_args)

                except Exception as e:

                    print(f"ERROR running {project} bug {bug}: {e}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--project")
    parser.add_argument("--bug", type=int)

    parser.add_argument("--all_selected", action="store_true")
    parser.add_argument("--selected_bugs_file", default="selected_bugs.json")

    parser.add_argument("--model", required=True)
    parser.add_argument("--run_id", type=int, required=True)

    parser.add_argument("--bug-python", required=True)
    parser.add_argument("--llm-python", default=sys.executable)

    parser.add_argument("--bugsinpy_projects_dir", default=DEFAULT_PROJECTS)
    parser.add_argument("--eval_root", default=PIPE)
    parser.add_argument("--results_file", default=RESULTS_FILE)

    args = parser.parse_args()

    if args.all_selected:
        run_all_selected(args)
    else:
        if args.project is None or args.bug is None:
            print("ERROR: For single run you must provide --project and --bug")
            sys.exit(1)

        run_single(args)