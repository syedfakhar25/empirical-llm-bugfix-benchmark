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
import textwrap
from datetime import datetime

from extract_block import extract_buggy_block


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, "pipeline")
DEFAULT_PROJECTS = os.path.join(ROOT, "..", "BugsInPy", "projects")
RESULTS_FILE = os.path.join(ROOT, "results/results.csv")
RUN_LLM = os.path.join(PIPE, "run_llm.py")


# ------------------------------------------------
# run command
# ------------------------------------------------
def run_cmd(cmd, cwd=None, extra_env=None):
    env = os.environ.copy()

    if cwd:
        env["PYTHONPATH"] = cwd + ":" + os.path.join(cwd, "src")

    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result


# ------------------------------------------------
# create virtualenv
# ------------------------------------------------
def create_virtualenv(venv_path, python_exec):
    if os.path.exists(venv_path):
        shutil.rmtree(venv_path)

    subprocess.run(f"{python_exec} -m venv {venv_path}", shell=True, check=False)

    pip = os.path.join(venv_path, "bin", "pip")

    subprocess.run(f"{pip} install --upgrade pip", shell=True, check=False)

    subprocess.run(
        f"{pip} install setuptools wheel pytest pytest-xdist nose mock hypothesis",
        shell=True,
        check=False
    )


# ------------------------------------------------
# install dependencies
# ------------------------------------------------
def install_dependencies(repo, pip, project, bug_id, bugsinpy_projects_dir, bug_python, eval_dir):
    logs = []

    def install_and_log(cmd, cwd=None):
        r = run_cmd(cmd, cwd=cwd)
        logs.append(f"\n$ {cmd}\n")
        logs.append(r.stdout)
        logs.append(r.stderr)
        return r

    bug_req = os.path.join(
        bugsinpy_projects_dir,
        project,
        "bugs",
        str(bug_id),
        "requirements.txt"
    )

    if os.path.exists(bug_req):
        install_and_log(f"{pip} install -r {bug_req} || true", cwd=repo)

    for fname in ["requirements.txt", "requirements-dev.txt", "dev-requirements.txt", "test-requirements.txt"]:
        path = os.path.join(repo, fname)
        if os.path.exists(path):
            install_and_log(f"{pip} install -r {fname} || true", cwd=repo)

    if os.path.exists(os.path.join(repo, "setup.py")):
        install_and_log(f"{pip} install -e . || true", cwd=repo)

    install_and_log(
            f"{pip} install "
            f"numpy scipy cython requests "
            f"pytest pytest-xdist pytest-cov "
            f"nose mock hypothesis "
            f"pytz python-dateutil "
            f"freezegun faker click jinja2 "
            f"pluggy packaging attrs "
            f"itsdangerous markupsafe "
            f"|| true",
            cwd=repo
        )

    # project-specific fallback
    if project == "pandas" and os.path.exists(os.path.join(repo, "setup.py")):
        install_and_log(f"{bug_python} setup.py build_ext --inplace --force || true", cwd=repo)

    with open(os.path.join(eval_dir, "dependency_install_log.txt"), "w") as f:
        f.write("".join(logs))


# ------------------------------------------------
# parse bug.info
# ------------------------------------------------
def parse_bug_info(bug_info_path):
    info = {
        "buggy_commit_id": None,
        "fixed_commit_id": None,
        "test_file": None,
        "python_version": None,
    }

    with open(bug_info_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("buggy_commit_id"):
                info["buggy_commit_id"] = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("fixed_commit_id"):
                info["fixed_commit_id"] = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("test_file"):
                info["test_file"] = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("python_version"):
                info["python_version"] = line.split("=", 1)[1].strip().strip('"')

    return info


# ------------------------------------------------
# locate declared test file
# ------------------------------------------------
def resolve_test_file(repo, declared_test_file):
    if not declared_test_file:
        return None, 0

    exact = os.path.join(repo, declared_test_file)
    if os.path.exists(exact):
        return declared_test_file, 1

    filename = os.path.basename(declared_test_file)
    for root, _, files in os.walk(repo):
        if filename in files:
            full = os.path.join(root, filename)
            rel = os.path.relpath(full, repo)
            return rel, 1

    return None, 0


# ------------------------------------------------
# robust JSON parse
# ------------------------------------------------
def parse_llm_json(output):
    if not output:
        return None

    try:
        return json.loads(output)
    except Exception:
        pass

    matches = re.findall(r"\{.*\}", output, re.DOTALL)
    for m in reversed(matches):
        try:
            return json.loads(m)
        except Exception:
            continue

    return None

# ADD THIS HELPER FUNCTION (put after parse_llm_json)

def extract_llm_metrics(result_json):
    if not result_json:
        return -1, -1, -1

    return (
        result_json.get("energy_joules", -1),
        result_json.get("input_tokens", -1),
        result_json.get("tokens_generated", -1)
    )

# ------------------------------------------------
# clean LLM output
# ------------------------------------------------
def clean_llm_output(raw_output, expected_name):
    if not raw_output:
        return "", "EMPTY_OUTPUT"

    text = raw_output.replace("```python", "").replace("```", "")
    text = textwrap.dedent(text).strip()

    # remove leading comments / blank lines
    lines = text.splitlines()
    while lines and (not lines[0].strip() or lines[0].strip().startswith("#")):
        lines.pop(0)
    text = "\n".join(lines).strip()

    if not text:
        return "", "EMPTY_OUTPUT"

    pattern = rf"(def|class)\s+{re.escape(expected_name)}\b"
    match = re.search(pattern, text)

    if not match:
        return text, "NO_MATCHING_BLOCK"

    candidate = text[match.start():]
    candidate_lines = candidate.splitlines()

    for i in range(len(candidate_lines), 0, -1):
        snippet = "\n".join(candidate_lines[:i]).rstrip()
        try:
            tree = ast.parse(snippet)
            if len(tree.body) > 0:
                node = tree.body[0]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == expected_name:
                    final_lines = snippet.splitlines()
                    return "\n".join(final_lines[node.lineno - 1: node.end_lineno]), None
        except Exception:
            continue

    return candidate, "AST_PARSE_FAIL"


# ------------------------------------------------
# validate candidate patch
# ------------------------------------------------
def validate_patch(cleaned_code, block_name, block_type):
    if not cleaned_code.strip():
        return False, "EMPTY_OUTPUT"

    stripped = cleaned_code.lstrip()

    if block_type == "function":
        if not (stripped.startswith(f"def {block_name}") or stripped.startswith(f"async def {block_name}")):
            return False, "INVALID_FUNCTION_HEADER"
    elif block_type == "class":
        if not stripped.startswith(f"class {block_name}"):
            return False, "INVALID_CLASS_HEADER"

    try:
        ast.parse(cleaned_code)
    except Exception:
        return False, "AST_PARSE_FAIL"

    return True, "NONE"


# ------------------------------------------------
# replace function/class in file
# ------------------------------------------------
def replace_function_in_file(file_path, block_name, new_code):
    import ast

    with open(file_path, encoding="utf-8", errors="ignore") as f:
        source = f.read()

    tree = ast.parse(source)

    target_node = None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == block_name:
                target_node = node
                break

    if target_node is None:
        raise RuntimeError(f"Block '{block_name}' not found")

    # get exact lines
    lines = source.splitlines()

    start = target_node.lineno - 1
    end = target_node.end_lineno

    # preserve indentation
    indent = len(lines[start]) - len(lines[start].lstrip())

    new_lines = new_code.splitlines()
    new_lines = [(" " * indent) + line if line.strip() else line for line in new_lines]

    updated = lines[:start] + new_lines + lines[end:]

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(updated))


# ------------------------------------------------
# CSV header
# ------------------------------------------------
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
            "failure_reason",

            "declared_test_file",
            "resolved_test_file",
            "test_file_found",

            "pre_test_pass",
            "pre_total",

            "post_test_pass",
            "post_total",

            "full_total_pass",
            "full_total",

            "llm_energy_joules",
            "input_tokens",
            "tokens_generated",

            "duration_seconds",
            "hostname",
            "cpu_count",
            "platform"
        ])


# ------------------------------------------------
# build pytest command
# ------------------------------------------------
def build_pytest_cmd(activate, target=None):
    base = (
        f"{activate} && "
        f"PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
        f"pytest -q --tb=short -c /dev/null"
    )
    if target:
        base += f" {target}"
    return base


# ------------------------------------------------
# NEW: extract pytest counts correctly
# ------------------------------------------------
def extract_test_counts(stdout, stderr):
    text = f"{stdout}\n{stderr}"

    passed = 0
    failed = 0
    errors = 0

    m = re.search(r"(\d+)\s+passed", text)
    if m:
        passed = int(m.group(1))

    f = re.search(r"(\d+)\s+failed", text)
    if f:
        failed = int(f.group(1))

    e = re.search(r"(\d+)\s+errors?", text)
    if e:
        errors = int(e.group(1))

    total = passed + failed + errors
    return passed, total


# ------------------------------------------------
# detect whether tests ran
# ------------------------------------------------
def parse_pytest_run(stdout, stderr):
    passed, total = extract_test_counts(stdout, stderr)

    if total > 0:
        return True, total

    text = f"{stdout}\n{stderr}"
    if "no tests ran" in text:
        return False, 0

    return False, 0


# ------------------------------------------------
# classify pytest result
# ------------------------------------------------
def classify_pytest_result(result, tests_ran):
    stderr = result.stderr
    stdout = result.stdout

    passed, total = extract_test_counts(stdout, stderr)

    if (
        "ModuleNotFoundError" in stderr
        or "ImportError" in stderr
    ):
        return "DEPENDENCY_ERROR", "MISSING_PACKAGE", passed

    if "fixture '" in stderr and "not found" in stderr:
        return "DEPENDENCY_ERROR", "MISSING_TEST_FIXTURE", passed

    if "SyntaxError" in stderr:
        return "TEST_EXECUTION_ERROR", "SYNTAX_ERROR", passed

    if "ERROR collecting" in stderr or "ERROR collecting" in stdout:
        return "TEST_EXECUTION_ERROR", "TEST_COLLECTION_ERROR", passed

    if total == 0:
        return "TEST_EXECUTION_ERROR", "NO_TESTS_COLLECTED", 0

    if passed == total:
        return "PASS", "NONE", passed

    return "FAIL", "TEST_ASSERTION_FAIL", passed
# ------------------------------------------------
# multi-run wrapper
# ------------------------------------------------
def run_multi(args):
    runs = args.runs
    sleep_time = args.sleep

    for i in range(runs):
        print(f"\n===== RUN {i+1}/{runs} =====")

        args.run_id = f"{args.bug}_{i+1}"
        run_single(args)

        if i < runs - 1:
            print(f"Sleeping {sleep_time}s...\n")
            time.sleep(sleep_time)

# ------------------------------------------------
# run all selected bugs
# ------------------------------------------------
def run_all_selected(args):
    ALL_MODELS = [
        "Qwen/Qwen2.5-Coder-1.5B",
        "stabilityai/stablecode-3b",
    ]
    if not args.selected_bugs_file:
        raise ValueError("selected_bugs_file required for --mode all_selected")

    with open(args.selected_bugs_file) as f:
        selected = json.load(f)
    for model in ALL_MODELS:
        print(f"\n\n######## MODEL: {model} ########")
        args.model = model
        for entry in selected:
            project = entry["project"]
            bug_id = entry["bug_id"]

            args.project = project
            args.bug = bug_id

            print(f"\n### {project} Bug {bug_id} ###")

            if args.runs > 1:
                run_multi(args)
            else:
                run_single(args)

# ------------------------------------------------
# MAIN RUN
# ------------------------------------------------
def run_single(args):
    project = args.project
    bug_id = args.bug
    model = args.model
    run_id = args.run_id

    model_safe = model.replace("/", "_")

    eval_dir = os.path.join(
        args.eval_root,
        f"eval_{project}_{bug_id}_{model_safe}_{run_id}",
    )

    shutil.rmtree(eval_dir, ignore_errors=True)
    os.makedirs(eval_dir)

    print(f"\n=== {project} Bug {bug_id} | {model} | Run {run_id} ===")

    project_info = os.path.join(args.bugsinpy_projects_dir, project, "project.info")
    with open(project_info, encoding="utf-8", errors="ignore") as f:
        url = [l.split("=")[1].strip().strip('"') for l in f if l.startswith("github_url")][0]

    repo = os.path.join(eval_dir, project)
    clone = run_cmd(f"git clone {url} {repo}")
    if clone.returncode != 0:
        status = "SETUP_ERROR"
        failure_reason = "GIT_CLONE_FAILED"
        declared_test_file = ""
        resolved_test_file = ""
        test_file_found = 0
        pre_test_ran = pre_test_pass = 0
        post_test_ran = post_test_pass = 0
        full_test_ran = full_test_pass = 0
        duration = 0

        ensure_csv_header(args.results_file)
        with open(args.results_file, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(), project, bug_id, model, run_id,
                status, failure_reason, declared_test_file, resolved_test_file,
                test_file_found, pre_test_ran, pre_test_pass, post_test_ran,
                post_test_pass, full_test_ran, full_test_pass, duration,
                socket.gethostname(), os.cpu_count(), platform.platform()
            ])

        print(f"→ Done: {status} | Reason: {failure_reason}")
        return

    bug_info_path = os.path.join(
        args.bugsinpy_projects_dir, project, "bugs", str(bug_id), "bug.info"
    )
    bug_meta = parse_bug_info(bug_info_path)
    commit = bug_meta["buggy_commit_id"]
    declared_test_file = bug_meta["test_file"] or ""

    run_cmd(f"git checkout {commit}", cwd=repo)

    venv_path = os.path.join(eval_dir, "venv")
    create_virtualenv(venv_path, args.bug_python)
    pip = os.path.join(venv_path, "bin", "pip")
    activate = f". {venv_path}/bin/activate"

    install_dependencies(
        repo,
        pip,
        project,
        bug_id,
        args.bugsinpy_projects_dir,
        args.bug_python,
        eval_dir
    )

    resolved_test_file, test_file_found = resolve_test_file(repo, declared_test_file)
    resolved_test_file = resolved_test_file or ""

    patch_file = os.path.join(
        args.bugsinpy_projects_dir, project, "bugs", str(bug_id), "bug_patch.txt"
    )

    diff_line = [l for l in open(patch_file, encoding="utf-8", errors="ignore") if l.startswith("diff --git")][0]
    buggy_rel = re.search(r"a/(.+?) ", diff_line).group(1)
    buggy_path = os.path.join(repo, buggy_rel)

    with open(buggy_path, encoding="utf-8", errors="ignore") as f:
        source = f.read()

    with open(os.path.join(eval_dir, "original_buggy_file.py"), "w") as f:
        f.write(source)

    block_info = extract_buggy_block(source, patch_file)
    if not block_info:
        status = "SETUP_ERROR"
        failure_reason = "BLOCK_DETECTION_FAILED"
        pre_test_ran = pre_test_pass = 0
        post_test_ran = post_test_pass = 0
        full_test_ran = full_test_pass = 0
        duration = 0

        ensure_csv_header(args.results_file)
        with open(args.results_file, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(), project, bug_id, model, run_id,
                status, failure_reason, declared_test_file, resolved_test_file,
                test_file_found, pre_test_ran, pre_test_pass, post_test_ran,
                post_test_pass, full_test_ran, full_test_pass, duration,
                socket.gethostname(), os.cpu_count(), platform.platform()
            ])

        print(f"→ Done: {status} | Reason: {failure_reason}")
        return

    block_type, block_name, buggy_block = block_info

    with open(os.path.join(eval_dir, "buggy_block.py"), "w") as f:
        f.write(buggy_block)

    # -------------------------
    # pre-patch bug-specific test
    # -------------------------
    pre_test_ran = 0
    pre_test_pass = 0
    pre_total = 0

    if test_file_found and resolved_test_file:
        pre_cmd = build_pytest_cmd(activate, resolved_test_file)
        pre_result = run_cmd(pre_cmd, cwd=repo)

        with open(os.path.join(eval_dir, "pre_test_stdout.txt"), "w") as f:
            f.write(pre_result.stdout)
        with open(os.path.join(eval_dir, "pre_test_stderr.txt"), "w") as f:
            f.write(pre_result.stderr)

        pre_tests_ran, pre_total = parse_pytest_run(pre_result.stdout, pre_result.stderr)
        pre_test_ran = pre_total
        _, _, pre_test_pass = classify_pytest_result(pre_result, pre_tests_ran)
        pre_test_pass = pre_test_pass

    # -------------------------
    # prompt
    # -------------------------
    prompt = f"""You are a Python expert.

Fix the following {block_type} named '{block_name}'.

STRICT RULES:
- Output ONLY valid Python code
- DO NOT explain anything
- DO NOT include markdown
- DO NOT include text before or after code
- The output MUST start with: def {block_name} OR class {block_name}
- The code MUST be complete and executable

BUGGY CODE:
{buggy_block}

FIXED CODE:
"""

    prompt_file = os.path.join(eval_dir, "prompt.txt")
    with open(prompt_file, "w") as f:
        f.write(prompt)

    cmd = (
        f"{args.llm_python} {RUN_LLM} "
        f"--model \"{model}\" "
        f"--prompt_file {prompt_file}"
    )

    start = time.time()
    llm_out = run_cmd(cmd, cwd=eval_dir)
    duration = time.time() - start
    
    # DEBUG: print stderr if exists
    if llm_out.stderr.strip():
        print("\n[LLM STDERR START]")
        print(llm_out.stderr[:2000])  # limit so output explode na ho
        print("[LLM STDERR END]\n")
        
    with open(os.path.join(eval_dir, "llm_raw_output.txt"), "w") as f:
        f.write(llm_out.stdout)
    with open(os.path.join(eval_dir, "llm_stderr.txt"), "w") as f:
        f.write(llm_out.stderr)

    # default result fields
    post_test_ran = 0
    post_test_pass = 0
    full_test_ran = 0
    full_test_pass = 0
    post_total = 0
    full_total = 0
    status = "UNKNOWN"
    failure_reason = "UNKNOWN"

    result_json = parse_llm_json(llm_out.stdout)
    llm_energy = -1
    input_tokens = -1
    tokens_generated = -1

    if result_json is not None and isinstance(result_json, dict):
        raw_output = result_json.get("output", "")
        llm_energy, input_tokens, tokens_generated = extract_llm_metrics(result_json)
    else:
        raw_output = llm_out.stdout.strip()

    # -------------------------
    # SAVE LLM OUTPUT
    # -------------------------
    output_dir = os.path.join(args.eval_root, "llm_outputs")
    os.makedirs(output_dir, exist_ok=True)

    model_clean = model.replace("/", "_")

    filename = f"{project}_{bug_id}_{model_clean}_run{run_id}.py"
    output_path = os.path.join(output_dir, filename)

    cleaned_code = raw_output.strip()

    if not cleaned_code:
        print("⚠ Empty LLM output — inserting dummy")
        cleaned_code = f"def {block_name}(*args, **kwargs):\n    pass"

    with open(output_path, "w") as f:
        f.write(f"# Project: {project}\n")
        f.write(f"# Bug: {bug_id}\n")
        f.write(f"# Model: {model}\n")
        f.write(f"# Run: {run_id}\n\n")
        f.write(cleaned_code)

    try:
        shutil.copy(buggy_path, buggy_path + ".bak")
        replace_function_in_file(buggy_path, block_name, cleaned_code)
    except Exception as e:
        print("Replace failed — stopping run")
        print(str(e))
        status = "SETUP_ERROR"
        failure_reason = "PATCH_INSERTION_FAILED"
        return
        with open(buggy_path, "a") as f:
            f.write("\n\n# LLM PATCH\n")
            f.write(cleaned_code)


    # -------------------------
    # ALWAYS RUN TESTS
    # -------------------------
    if test_file_found and resolved_test_file:

        # bug SPECIFIC TEST
        post_cmd = build_pytest_cmd(activate, resolved_test_file)
        post_result = run_cmd(post_cmd, cwd=repo)

        with open(os.path.join(eval_dir, "post_test_stdout.txt"), "w") as f:
            f.write(post_result.stdout)
        with open(os.path.join(eval_dir, "post_test_stderr.txt"), "w") as f:
            f.write(post_result.stderr)

        post_tests_ran, post_total = parse_pytest_run(post_result.stdout, post_result.stderr)
        post_test_ran = post_total

        post_status, post_reason, post_test_pass = classify_pytest_result(post_result, post_tests_ran)


        # FULL TEST SUITE
        full_cmd = build_pytest_cmd(activate)
        full_result = run_cmd(full_cmd, cwd=repo)

        with open(os.path.join(eval_dir, "full_test_stdout.txt"), "w") as f:
            f.write(full_result.stdout)
        with open(os.path.join(eval_dir, "full_test_stderr.txt"), "w") as f:
            f.write(full_result.stderr)

        full_tests_ran, full_total = parse_pytest_run(full_result.stdout, full_result.stderr)
        full_test_ran = full_total

        full_status, full_reason, full_total_pass = classify_pytest_result(full_result, full_tests_ran)
        full_test_pass = full_total_pass


        # -------------------------
        # FINAL STATUS LOGIC
        # -------------------------
        if post_status == "DEPENDENCY_ERROR":
            status = "DEPENDENCY_ERROR"
            failure_reason = post_reason

        elif post_status == "TEST_EXECUTION_ERROR":
            status = "TEST_EXECUTION_ERROR"
            failure_reason = post_reason

        elif post_test_pass == 0:
            status = "BUG_TEST_FAIL"
            failure_reason = post_reason

        elif full_status == "DEPENDENCY_ERROR":
            status = "DEPENDENCY_ERROR"
            failure_reason = full_reason

        elif full_status == "TEST_EXECUTION_ERROR":
            status = "TEST_EXECUTION_ERROR"
            failure_reason = full_reason

        elif full_test_pass < full_test_ran:
            status = "REGRESSION_FAIL"
            failure_reason = full_reason

        else:
            status = "SUCCESS"
            failure_reason = "NONE"
    else:
        status = "TEST_EXECUTION_ERROR"
        failure_reason = "DECLARED_TEST_FILE_NOT_FOUND"

    ensure_csv_header(args.results_file)

    with open(args.results_file, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(),
            project,
            bug_id,
            model,
            run_id,
            status,
            failure_reason,

            declared_test_file,
            resolved_test_file,
            test_file_found,

            pre_test_pass,
            pre_total,

            post_test_pass,
            post_total,

            full_test_pass,
            full_total,

            llm_energy,
            input_tokens,
            tokens_generated,

            duration,
            socket.gethostname(),
            os.cpu_count(),
            platform.platform()
        ])

    print(
        f"→ Done: {status} | Reason: {failure_reason} | "
        f"Pre: {pre_test_pass}/{pre_total} | "
        f"Post: {post_test_pass}/{post_total} | "
        f"Full: {full_test_pass}/{full_total} | "
        f"Energy: {llm_energy:.2f}J | Tokens: {tokens_generated}"
    )
    # -------------------------
    # CLEANUP
    # -------------------------
    try:
        shutil.rmtree(eval_dir)
    except Exception:
        pass


# ------------------------------------------------
# ENTRY
# ------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--project")
    parser.add_argument("--bug", type=int)
    parser.add_argument("--model")
    parser.add_argument("--run_id", type=int, default=1)

    parser.add_argument("--bug-python", required=True)
    parser.add_argument("--llm-python", default=sys.executable)

    parser.add_argument("--bugsinpy_projects_dir", default=DEFAULT_PROJECTS)
    parser.add_argument("--eval_root", default=PIPE)
    parser.add_argument("--results_file", default=RESULTS_FILE)

    # NEW FLAGS
    parser.add_argument("--mode", choices=["single", "all_selected", "multi_run"], default="single")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--sleep", type=int, default=5)
    parser.add_argument("--selected_bugs_file")

    args = parser.parse_args()

    if args.mode == "single":
        if not args.project or args.bug is None:
            raise ValueError("project and bug required for single mode")
        run_single(args)

    elif args.mode == "multi_run":
        if not args.project or args.bug is None:
            raise ValueError("project and bug required for multi_run mode")
        run_multi(args)

    elif args.mode == "all_selected":
        run_all_selected(args)