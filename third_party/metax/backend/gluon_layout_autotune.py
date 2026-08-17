"""Post-TTGIR C500 Gluon layout candidate export.

Candidate formulas and IR cloning live in the C++ passes.  This module only
persists verified standalone TTGIR leaves and records the minimal runtime
manifest on the already compiled C0 kernel.
"""

from __future__ import annotations

import hashlib

from triton._C.libtriton import gluon_ir, ir, passes
from triton.runtime.cache import get_cache_manager


def _digest(source: str, target, options) -> str:
    payload = "\n".join((source, repr(target), options.hash()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _contains_call(module) -> bool:
    found = False

    def visit(operation):
        nonlocal found
        found |= operation.get_name() == "tt.call"

    module.walk(visit)
    return found


def export_layout_candidates(module, metadata, options, capability):
    """Expand MMA, Shared and Blocked assignments and persist final leaves."""
    if _contains_call(module):
        return

    pm = ir.pass_manager(module.context)
    pm.enable_debug()
    passes.gluon.metax.add_mma_layout_candidates(pm, capability)
    passes.gluon.metax.add_expand_layout_candidates(pm, 0)
    passes.gluon.metax.add_shared_layout_candidates(pm)
    passes.gluon.metax.add_expand_layout_candidates(pm, 1)
    passes.gluon.metax.add_blocked_layout_candidates(pm)
    passes.gluon.metax.add_expand_layout_candidates(pm, 2)
    pm.run(module, "gluon_metax_layout_autotune")

    leaves = gluon_ir.export_metax_layout_candidates(module)
    if not leaves:
        return

    cache = get_cache_manager(metadata["hash"])
    target = metadata["target"]
    encoded_leaves = [
        (_digest(source, target, options), source, lineage)
        for source, lineage in leaves
    ]
    fallback_digests = [
        digest
        for digest, _, lineage in encoded_leaves
        if lineage == "0.0.0"
    ]
    if len(fallback_digests) != 1:
        raise RuntimeError("Gluon layout candidate bundle must have exactly one C0 leaf")
    fallback_digest = fallback_digests[0]

    variants = []
    seen = {fallback_digest}
    for digest, source, lineage in encoded_leaves:
        if lineage == "0.0.0" or digest in seen:
            continue
        seen.add(digest)
        filename = f"{digest}.ttgir"
        cache.put(source, filename, binary=False)
        variants.append({"digest": digest, "cache_file": filename, "lineage": lineage})
    domain_payload = fallback_digest + "\n" + "\n".join(
        item["digest"] for item in variants
    )
    metadata["gluon_layout_manifest"] = {
        "domain_digest": hashlib.sha256(domain_payload.encode("utf-8")).hexdigest(),
        "fallback_digest": fallback_digest,
        "variants": variants,
    }
