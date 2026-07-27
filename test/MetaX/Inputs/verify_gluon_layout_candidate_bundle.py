#!/usr/bin/env python3

import argparse
import hashlib
import re
import tempfile

from triton._C.libtriton import ir, metax


SHA256 = re.compile(r"[0-9a-f]{64}")
FORBIDDEN = (
    "#gluon.auto_encoding",
    "#gluon.no_verify_encoding",
    "gluon.set_auto_layout",
    "gluon.require_layout",
    "gluon.release_layout",
    "layout-candidate-selection",
    "layout-selected-candidate",
    "layout-fallback-variant",
    "layout-variant-output-dir",
)


def fail(message):
    raise RuntimeError(message)


def build(path):
    context = ir.context()
    ir.load_dialects(context)
    metax.load_dialects(context)
    module = ir.parse_mlir_module(path, context)
    return metax.build_gluon_layout_candidate_bundle(module, 80)


def verify_bundle(name, path, expect_alternatives):
    first = build(path)
    second = build(path)
    expected_keys = {
        "version",
        "digest",
        "fallback",
        "fallback_only",
        "runtime_contract",
        "variants",
    }
    if set(first) != expected_keys or first["version"] != 1:
        fail(f"{name}: invalid bundle schema")
    if first != second:
        fail(f"{name}: candidate construction is not deterministic")
    if not SHA256.fullmatch(first["digest"]):
        fail(f"{name}: invalid domain digest")
    if not SHA256.fullmatch(first["fallback"]):
        fail(f"{name}: invalid fallback digest")

    variants = first["variants"]
    if not variants or first["fallback_only"] != (len(variants) == 1):
        fail(f"{name}: invalid fallback cardinality")
    if variants[0]["digest"] != first["fallback"]:
        fail(f"{name}: fallback is not first")
    if expect_alternatives != (len(variants) > 1):
        fail(f"{name}: unexpected alternative count {len(variants)}")
    digests = [variant["digest"] for variant in variants]
    if len(digests) != len(set(digests)):
        fail(f"{name}: duplicate finalized variant")

    for variant in variants:
        if set(variant) != {"digest", "source"}:
            fail(f"{name}: invalid variant schema")
        digest = variant["digest"]
        source = variant["source"]
        if not SHA256.fullmatch(digest):
            fail(f"{name}: invalid variant digest")
        leaked = [token for token in FORBIDDEN if token in source]
        if leaked:
            fail(f"{name}/{digest}: leaked temporary protocol {leaked}")
        if source.count("ttg.gluon.layout-domain-digest") != 1:
            fail(f"{name}/{digest}: missing unique domain identity")
        if source.count("ttg.gluon.layout-variant-digest") != 1:
            fail(f"{name}/{digest}: missing unique variant identity")
        if source.count("ttg.gluon.gvm-finalized") != 1:
            fail(f"{name}/{digest}: missing finalization identity")
        if f"__gluon_layout_{digest}" not in source:
            fail(f"{name}/{digest}: public entry was not variant-qualified")
        if hashlib.sha256(source.encode()).hexdigest() == digest:
            fail(f"{name}/{digest}: semantic identity depends on printed source")
        if name == "two-dot":
            if source.count("tt.dot") != 2:
                fail(f"{name}/{digest}: producer chain lost a dot")
            if "ttg.convert_layout" not in source:
                fail(f"{name}/{digest}: producer chain lost its local layout boundary")

        context = ir.context()
        ir.load_dialects(context)
        metax.load_dialects(context)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", encoding="utf-8"
        ) as temporary:
            temporary.write(source)
            temporary.flush()
            ir.parse_mlir_module(temporary.name, context)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-dot", required=True)
    parser.add_argument("--single-dot", required=True)
    parser.add_argument("--two-dot", required=True)
    args = parser.parse_args()

    verify_bundle("no-dot", args.no_dot, expect_alternatives=False)
    verify_bundle("single-dot", args.single_dot, expect_alternatives=True)
    verify_bundle("two-dot", args.two_dot, expect_alternatives=True)


if __name__ == "__main__":
    main()
