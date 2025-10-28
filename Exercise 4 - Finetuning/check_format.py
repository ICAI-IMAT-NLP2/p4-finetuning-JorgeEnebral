
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_format.py — Format checker for Exercise Sheet 4 (Parameter-efficient Fine-tuning)

Validates ONLY the **format** of the required submission files:

1) peft config.txt (or peft_config.txt)
   Must contain the following integer variables (order not enforced):
       r                      # even, >= 0
       num_trainable_lora     # int, >= 0
       P                      # even, >= 0
       num_trainable_soft     # int, >= 0
       d_a                    # even, >= 0
       num_trainable_adapters # int, >= 0
       num_trainable_ia3      # int, >= 0

2) peft.txt
   Must contain:
       b       = [b_hi, b_hello, b_bye, b_regards]         # length-4 vector (numbers)
       A       = [[a1], [a2]]                               # 2x1 numeric matrix
       B       = [[b_hi, b_hello, b_bye, b_regards]]        # 1x4 numeric matrix
       Wprime  = [[..., ..., ..., ...],
                  [..., ..., ..., ...]]                     # 2x4 numeric matrix

Notes:
- This script is **format-only**: it checks presence, types, shapes, evenness for r/P/d_a,
  and that all numeric elements are numbers. It does not verify parameter counts or algebra.

Usage:
    python check_format.py /path/to/folder
    # or in current directory:
    python check_format.py
"""

import os
import sys
import ast
from typing import Any, Dict, List, Tuple, Optional

# ---------- Utils ----------

def is_numeric(x: Any) -> bool:
    return isinstance(x, (int, float))

def check_vector(vec: Any, length: int) -> Tuple[bool, str]:
    if not isinstance(vec, list) or len(vec) != length:
        return False, f"Expected a list of length {length}."
    for i, el in enumerate(vec):
        if not is_numeric(el):
            return False, f"Element at position {i} is not numeric: {repr(el)}"
    return True, ""

def check_matrix(mat: Any, rows: int, cols: int) -> Tuple[bool, str]:
    if not isinstance(mat, list) or len(mat) != rows:
        return False, f"Expected a list with {rows} rows."
    for r_idx, row in enumerate(mat):
        if not isinstance(row, list) or len(row) != cols:
            return False, f"Row {r_idx} must be a list with {cols} elements."
        for c_idx, el in enumerate(row):
            if not is_numeric(el):
                return False, f"Non-numeric at ({r_idx},{c_idx}): {repr(el)}"
    return True, ""

def read_assignments_multiline(path: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Parse entire file as Python and collect NAME = <python-literal> assignments.
    Supports multi-line dict/list literals. Returns (vars_dict, error_or_None).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except FileNotFoundError:
        return {}, f"File not found: {path}"
    except Exception as e:
        return {}, f"Error opening {path}: {e}"

    try:
        tree = ast.parse(src, mode="exec")
    except Exception as e:
        return {}, f"Could not parse file as Python: {e}"

    out: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            try:
                val = ast.literal_eval(node.value)
            except Exception as e:
                return {}, f"Could not parse value for {name}: {e}"
            out[name] = val
    return out, None

# ---------- File checks ----------

def check_peft_config(path: str) -> List[str]:
    """
    Format checks for 'peft config.txt' (or 'peft_config.txt').
    Required integer variables (>=0). r, P, d_a must be even.
    """
    errors: List[str] = []
    vars_out, err = read_assignments_multiline(path)
    if err:
        return [err]

    required = [
        "r",
        "num_trainable_lora",
        "P",
        "num_trainable_soft",
        "d_a",
        "num_trainable_adapters",
        "num_trainable_ia3",
    ]

    for k in required:
        if k not in vars_out:
            errors.append(f"Missing variable: {k}")

    if errors:
        return errors

    # Type and basic constraints
    def _assert_int_nonneg(name: str):
        v = vars_out[name]
        if not isinstance(v, int):
            errors.append(f"{name} must be an integer (got {type(v).__name__}).")
        elif v < 0:
            errors.append(f"{name} must be >= 0 (got {v}).")

    for name in ["r", "P", "d_a",
                 "num_trainable_lora", "num_trainable_soft",
                 "num_trainable_adapters", "num_trainable_ia3"]:
        _assert_int_nonneg(name)

    # Evenness for r, P, d_a (spec asks for largest even values)
    for name in ["r", "P", "d_a"]:
        if name in vars_out and isinstance(vars_out[name], int) and vars_out[name] % 2 != 0:
            errors.append(f"{name} must be even (got {vars_out[name]}).")

    return errors

def check_peft(path: str) -> List[str]:
    """
    Format checks for 'peft.txt'
      - b:       vector length 4 (numbers)
      - A:       2x1 numeric matrix
      - B:       1x4 numeric matrix
      - Wprime:  2x4 numeric matrix
    """
    errors: List[str] = []
    vars_out, err = read_assignments_multiline(path)
    if err:
        return [err]

    required = ["b", "A", "B", "Wprime"]
    for k in required:
        if k not in vars_out:
            errors.append(f"Missing variable: {k}")

    if errors:
        return errors

    # b: [4]
    ok, msg = check_vector(vars_out["b"], 4)
    if not ok:
        errors.append(f"b must be a numeric list of length 4: {msg}")

    # A: 2x1
    ok, msg = check_matrix(vars_out["A"], 2, 1)
    if not ok:
        errors.append(f"A must be a 2x1 numeric matrix: {msg}")

    # B: 1x4
    ok, msg = check_matrix(vars_out["B"], 1, 4)
    if not ok:
        errors.append(f"B must be a 1x4 numeric matrix: {msg}")

    # Wprime: 2x4
    ok, msg = check_matrix(vars_out["Wprime"], 2, 4)
    if not ok:
        errors.append(f"Wprime must be a 2x4 numeric matrix: {msg}")

    return errors

# ---------- Main ----------

def main(folder: str) -> int:
    # Accept both "peft config.txt" and "peft_config.txt"
    candidates_cfg = ["peft config.txt", "peft_config.txt"]
    path_cfg = None
    for cand in candidates_cfg:
        p = os.path.join(folder, cand)
        if os.path.exists(p):
            path_cfg = p
            break

    any_errors = False

    if path_cfg is None:
        print("❌ peft config.txt / peft_config.txt: File not found")
        any_errors = True
    else:
        errs = check_peft_config(path_cfg)
        if not errs:
            print(f"✅ {os.path.basename(path_cfg)}: OK")
        else:
            any_errors = True
            print(f"❌ {os.path.basename(path_cfg)}:")
            for e in errs:
                print(f"   - {e}")

    # peft.txt
    path_peft = os.path.join(folder, "peft.txt")
    if not os.path.exists(path_peft):
        print("❌ peft.txt: File not found")
        any_errors = True
    else:
        errs = check_peft(path_peft)
        if not errs:
            print("✅ peft.txt: OK")
        else:
            any_errors = True
            print("❌ peft.txt:")
            for e in errs:
                print(f"   - {e}")

    if not any_errors:
        print("\nAll files are correctly formatted ✅")
        return 0
    else:
        print("\nFormatting issues detected ❗")
        return 1

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    sys.exit(main(folder))
