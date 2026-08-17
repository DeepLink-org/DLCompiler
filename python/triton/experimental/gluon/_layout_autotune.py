"""Runtime selection for post-TTGIR C500 Gluon layout candidates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from queue import Queue
from threading import Thread
import hashlib
import json

import triton
from triton import knobs
from triton.runtime.cache import get_cache_manager
from triton.runtime.driver import driver


_STOP = object()
_COMPILE_BATCH_SIZE = 4
_DISPATCH_RANGE_FACTOR = 128


def _dispatch_value(value):
    if isinstance(value, bool):
        return ("value", value)
    if isinstance(value, int) and value > 0:
        lower = 1
        upper = _DISPATCH_RANGE_FACTOR
        while value >= upper:
            lower = upper
            upper *= _DISPATCH_RANGE_FACTOR
        return ("range", lower, upper)
    if isinstance(value, (int, float, str)):
        return ("value", value)
    # Pointer/tensor identity must not create a new tuning workload. Its type,
    # dtype and alignment policy are already represented by the JIT key.
    return ("jit",)


def _dispatch_key(bound_args, arg_names):
    return tuple(
        (name, _dispatch_value(bound_args[name]))
        for name in arg_names
    )


def _workload_digest(jit_key, manifest, target, device, dispatch):
    payload = repr(
        (
            jit_key,
            manifest["domain_digest"],
            target,
            device,
            dispatch,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LayoutAutotuneRuntime:
    """One small owner for winner caching, candidate compilation and timing."""

    def __init__(self):
        self._winners = {}
        self._compiled = {}

    def _compile(self, baseline, variant, options, device):
        compiled_key = (baseline.metadata.hash, device, variant["digest"])
        if compiled_key in self._compiled:
            return self._compiled[compiled_key]

        cache = get_cache_manager(baseline.metadata.hash)
        path = cache.get_file(variant["cache_file"])
        if path is None:
            raise FileNotFoundError(f"missing Gluon layout candidate {variant['cache_file']}")
        compile_options = options if isinstance(options, dict) else options.__dict__
        kernel = triton.compile(
            path,
            target=baseline.metadata.target,
            options=compile_options,
        )
        # The standalone TTGIR has the same physical FuncOp signature as C0,
        # but IRSource no longer carries the original AST constants map. Reuse
        # C0's source so the existing launcher filters folded arguments exactly
        # as it does for the baseline binary.
        kernel.src = baseline.src
        self._compiled[compiled_key] = kernel
        return kernel

    def _compile_batch(self, pool, baseline, variants, options, device):
        futures = [
            pool.submit(self._compile, baseline, variant, options, device)
            for variant in variants
        ]
        wait(futures)

        compiled = []
        for variant, future in zip(variants, futures):
            try:
                compiled.append((variant, future.result()))
            except Exception as error:
                if knobs.autotuning.print:
                    print(
                        "Gluon layout candidate "
                        f"{variant['digest'][:12]} compile failed: {error}"
                    )
        return compiled

    def _benchmark(self, kernel, grid, stream, bound_args):
        arguments = tuple(bound_args.values())

        def call():
            kernel[grid](*arguments, stream=stream)

        result = driver.active.get_benchmarker()(call, quantiles=(0.5, 0.2, 0.8))
        return result[0] if isinstance(result, (tuple, list)) else result

    def _winner_cache(self, workload):
        cache = get_cache_manager(workload)
        return cache, "gluon-layout-winner.json"

    def _read_winner(self, workload):
        cache, filename = self._winner_cache(workload)
        path = cache.get_file(filename)
        if path is None:
            return None
        try:
            with open(path, encoding="utf-8") as winner_file:
                return json.load(winner_file).get("digest")
        except (OSError, ValueError):
            return None

    def _write_winner(self, workload, digest, dispatch):
        cache, filename = self._winner_cache(workload)
        cache.put(
            json.dumps({"digest": digest, "dispatch": dispatch}),
            filename,
            binary=False,
        )

    def prepare(
        self,
        baseline,
        *,
        jit_key,
        grid,
        stream,
        bound_args,
        options,
        device,
        do_not_specialize,
    ):
        manifest = getattr(baseline.metadata, "gluon_layout_manifest", None)
        if not manifest or not manifest.get("variants"):
            return baseline

        dispatch = _dispatch_key(bound_args, do_not_specialize)
        workload = _workload_digest(
            jit_key,
            manifest,
            baseline.metadata.target,
            device,
            dispatch,
        )
        if workload in self._winners:
            return self._winners[workload]

        variants = manifest["variants"]
        by_digest = {item["digest"]: item for item in variants}
        cached_digest = self._read_winner(workload)
        if cached_digest == manifest["fallback_digest"]:
            if knobs.autotuning.print:
                print(
                    "Gluon layout autotune cache hit: "
                    f"selected C0 {cached_digest[:12]}"
                )
            self._winners[workload] = baseline
            return baseline
        if cached_digest in by_digest:
            try:
                winner = self._compile(
                    baseline, by_digest[cached_digest], options, device
                )
                if knobs.autotuning.print:
                    print(
                        "Gluon layout autotune cache hit: "
                        f"selected {cached_digest[:12]} "
                        f"lineage={by_digest[cached_digest]['lineage']}"
                    )
                self._winners[workload] = winner
                return winner
            except Exception:
                pass

        ready = Queue(maxsize=1)

        def producer():
            try:
                max_workers = min(_COMPILE_BATCH_SIZE, len(variants))
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    for batch_index, begin in enumerate(
                        range(0, len(variants), _COMPILE_BATCH_SIZE)
                    ):
                        batch = variants[begin:begin + _COMPILE_BATCH_SIZE]
                        if knobs.autotuning.print:
                            print(
                                "Gluon layout compile batch "
                                f"{batch_index} start: candidates={len(batch)}"
                            )
                        compiled = self._compile_batch(
                            pool, baseline, batch, options, device
                        )
                        if knobs.autotuning.print:
                            print(
                                "Gluon layout compile batch "
                                f"{batch_index} ready: compiled={len(compiled)}/"
                                f"{len(batch)}"
                            )
                        ready.put((batch_index, compiled))
            except Exception as error:
                if knobs.autotuning.print:
                    print(f"Gluon layout compile producer failed: {error}")
            finally:
                ready.put(_STOP)

        winner = baseline
        winner_digest = manifest["fallback_digest"]
        if knobs.autotuning.print:
            print(
                "Gluon layout autotune start: "
                f"domain={manifest['domain_digest'][:12]} "
                f"candidates={len(variants)} dispatch={dispatch}"
            )
        try:
            best = self._benchmark(baseline, grid, stream, bound_args)
            if knobs.autotuning.print:
                print(
                    "Gluon layout candidate C0 "
                    f"digest={winner_digest[:12]} elapsed={best:.6f} ms"
                )
        except Exception as error:
            if knobs.autotuning.print:
                print(f"Gluon layout C0 benchmark failed: {error}")
            return baseline

        thread = Thread(target=producer, name="gluon-layout-compile", daemon=True)
        thread.start()

        while True:
            compiled_batch = ready.get()
            if compiled_batch is _STOP:
                break
            batch_index, compiled = compiled_batch
            if knobs.autotuning.print:
                print(
                    "Gluon layout benchmark batch "
                    f"{batch_index} start: candidates={len(compiled)}"
                )
            for item, kernel in compiled:
                try:
                    elapsed = self._benchmark(kernel, grid, stream, bound_args)
                except Exception as error:
                    if knobs.autotuning.print:
                        print(
                            "Gluon layout candidate "
                            f"{item['digest'][:12]} benchmark failed: {error}"
                        )
                    continue
                if knobs.autotuning.print:
                    print(
                        "Gluon layout candidate "
                        f"digest={item['digest'][:12]} "
                        f"lineage={item['lineage']} elapsed={elapsed:.6f} ms"
                    )
                if elapsed < best:
                    best = elapsed
                    winner = kernel
                    winner_digest = item["digest"]

        thread.join()
        self._winners[workload] = winner
        try:
            self._write_winner(workload, winner_digest, dispatch)
        except OSError as error:
            # Winner persistence is an optimization. A full target benchmark
            # has already produced a valid in-process winner, so a cache-space
            # failure must not turn the imminent kernel launch into a failure.
            if knobs.autotuning.print:
                print(f"Gluon layout winner cache write failed: {error}")
        if knobs.autotuning.print:
            lineage = by_digest.get(winner_digest, {}).get("lineage", "C0")
            print(
                "Gluon layout autotune selected "
                f"{winner_digest[:12]} lineage={lineage} at {best:.6f} ms"
            )
        return winner
