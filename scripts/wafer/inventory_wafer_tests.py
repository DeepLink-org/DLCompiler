#!/usr/bin/env python3
"""Inventory repository tests statically, without importing or running them."""

import argparse
import ast
import csv
import json
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[2]


def test_functions(tree):
    functions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            functions.append(node)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            functions.extend(
                child for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test")
            )
    return functions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execution-summary", type=Path,
                        help="Optional run_wafer_example_suite.py summary to join with the static inventory")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Include new files before the final stage is committed and all project test
    # trees (including python/dlBLAS), while respecting ignored build outputs.
    files = sorted(set(subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO, text=True,
    ).splitlines()))
    config = ast.parse((REPO / "third_party/wafer/examples/conftest.py").read_text())
    ignored = set()
    for node in config.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "collect_ignore" for t in node.targets):
            ignored.update(ast.literal_eval(node.value))
    executions = {}
    if args.execution_summary:
        run = json.loads(args.execution_summary.read_text())
        for result in run['files']:
            name = result['file']
            if not name.startswith(('test/', 'third_party/')):
                name = 'third_party/wafer/examples/' + name
            executions[name] = run.get('execution', run.get('suite', 'unknown')) + ':' + result['status']
    rows = []
    for name in sorted(files):
        path = Path(name)
        if path.suffix != ".py" or not (path.name.startswith("test_") or path.name.endswith("_test.py")):
            continue
        text = (REPO / path).read_text()
        tree = ast.parse(text)
        tests = test_functions(tree)
        if name.startswith("test/wafer/"):
            group, status = "wafer_regression", executions.get(name, "execution_not_inferred_by_static_scan")
        elif name.startswith("third_party/wafer/examples/"):
            group, status = "wafer_examples", executions.get(name, "execution_not_inferred_by_static_scan")
        elif name.startswith("third_party/wafer/third_party/"):
            group, status = "bundled_dependency_examples", "not_executed"
        else:
            group, status = "/".join(path.parts[:2]), "not_executed_for_wafer"
        imports = sorted({
            node.module for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module and
            any(part in node.module for part in ("ztc", "cuda", "npu", "testing", "benchmark"))
        })
        assertions = sum(
            isinstance(node, ast.Assert) or (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr.startswith("assert")
            ) for node in ast.walk(tree)
        )
        rows.append({
            "file": name,
            "group": group,
            "execution_scope": status,
            "static_test_functions": len(tests),
            "test_names": ",".join(node.name for node in tests),
            "ignored_by_wafer_examples_conftest": group == "wafer_examples" and path.name in ignored,
            "assertion_sites_in_file": assertions,
            "special_imports": ",".join(imports),
            "has_main_entry": any(isinstance(node, ast.If) and "__name__" in ast.unparse(node.test) for node in tree.body),
        })
    upstream = {}
    for directory, paths in {
        "third_party/triton": ("test", "python/test"),
        "third_party/ascendnpu-ir": ("bishengir/test",),
    }.items():
        tracked = subprocess.check_output(
            ["git", "ls-files", "--", *paths], cwd=REPO / directory, text=True
        ).splitlines()
        upstream[directory] = {
            "tracked_files_in_test_trees": len(tracked),
            "mlir_fixtures": sum(name.endswith(".mlir") for name in tracked),
            "python_test_files": sum(Path(name).name.startswith("test_") and name.endswith(".py") for name in tracked),
            "execution_scope": "not_executed_as_upstream_suites",
        }
    summary = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "method": "Static AST inventory; function counts do not expand parametrization or prove execution.",
        "file_scope": "Project Python test_*.py/*_test.py files from git ls-files; upstream submodules counted separately.",
        "execution_summary": str(args.execution_summary) if args.execution_summary else None,
        "groups": {
            group: {
                "files": sum(row["group"] == group for row in rows),
                "static_test_functions": sum(row["static_test_functions"] for row in rows if row["group"] == group),
            } for group in sorted({row["group"] for row in rows})
        },
        "wafer_examples_collect_ignore": sorted(ignored),
        "wafer_examples_existing_ignored_files": sum(row["ignored_by_wafer_examples_conftest"] for row in rows),
        "wafer_examples_files_without_detected_assertions": [
            row["file"] for row in rows if row["group"] == "wafer_examples" and not row["assertion_sites_in_file"]
        ],
        "wafer_mlir_fixtures": [name for name in files if name.startswith("third_party/wafer/") and name.endswith(".mlir")],
        "upstream_submodules": upstream,
    }
    with (args.output_dir / "repository-tests.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "test-inventory.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
