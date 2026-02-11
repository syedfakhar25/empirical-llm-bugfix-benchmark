import ast
import re

def extract_buggy_block(source_code, patch_file):
    """
    Detect buggy function/class from patch file and extract AST block.
    """

    # --------------------------------------------
    # 1. LOAD PATCH FILE  (YOUR BUG WAS HERE)
    # --------------------------------------------
    with open(patch_file, "r") as f:
        patch = f.read()

    # --------------------------------------------
    # 2. Extract name of function/class from patch
    # --------------------------------------------
    name_match = re.search(r"@@.*def\s+([A-Za-z0-9_]+)", patch)
    type_match = "function"

    if not name_match:
        name_match = re.search(r"@@.*class\s+([A-Za-z0-9_]+)", patch)
        if name_match:
            type_match = "class"

    # fallback: any def in patch
    if not name_match:
        name_match = re.search(r"def\s+([A-Za-z0-9_]+)\s*\(", patch)
        type_match = "function"

    # fallback 2: patch modifies only inside body
    if not name_match:
        for line in patch.splitlines():
            if line.startswith("-") or line.startswith("+"):
                m = re.search(r"def\s+([A-Za-z0-9_]+)\s*\(", source_code)
                if m:
                    name_match = m
                    type_match = "function"
                    break

    if not name_match:
        print("ERROR: cannot detect function/class name from patch")
        return None

    target_name = name_match.group(1)
    target_type = type_match

    # --------------------------------------------
    # 3. Parse AST
    # --------------------------------------------
    tree = ast.parse(source_code)

    # --------------------------------------------
    # 4. Find matching block
    # --------------------------------------------
    for node in tree.body:

        if isinstance(node, ast.ClassDef) and node.name == target_name:
            return "class", target_name, ast.get_source_segment(source_code, node)

        if isinstance(node, ast.FunctionDef) and node.name == target_name:
            return "function", target_name, ast.get_source_segment(source_code, node)

        # function inside class → return full class
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == target_name:
                    return "class", node.name, ast.get_source_segment(source_code, node)

    return None
