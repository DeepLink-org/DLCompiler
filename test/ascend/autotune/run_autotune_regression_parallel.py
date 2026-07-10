from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from regression_config import (
    CASE_DIR,
    PARALLEL_DEVICES,
    case_cmd,
    case_env,
    load_regression_specs,
    prepare_clean_cache,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-root",
        default="/tmp/dlc_autotune_parallel_devices_20260629_trust_region_final",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    root = Path(args.log_root)
    root.mkdir(parents=True, exist_ok=True)

    specs = load_regression_specs()
    procs = []
    start = time.time()
    for name, device in PARALLEL_DEVICES.items():
        spec = specs[name]
        case_root = root / name
        case_root.mkdir(parents=True, exist_ok=True)
        cache_root = case_root / "triton_cache"
        prepare_clean_cache(cache_root)
        log_path = case_root / "run.log"
        log_file = log_path.open("w")
        cmd = case_cmd(CASE_DIR / spec["script"])
        proc = subprocess.Popen(
            cmd,
            cwd=str(CASE_DIR),
            env=case_env(os.environ, cache_root, name, device),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        procs.append(
            {
                "name": name,
                "proc": proc,
                "log_file": log_file,
                "log_path": log_path,
                "timeout_s": int(spec.get("timeout_s", 600)),
                "start": time.time(),
                "device": device,
            }
        )
        print(f"started {name} device={device} pid={proc.pid}", flush=True)

    remaining = {item["name"] for item in procs}
    worst_rc = 0
    while remaining:
        for item in procs:
            name = item["name"]
            if name not in remaining:
                continue
            proc = item["proc"]
            rc = proc.poll()
            elapsed = time.time() - item["start"]
            if rc is None and elapsed > item["timeout_s"]:
                proc.kill()
                rc = proc.wait()
                item["log_file"].write(f"\nTIMEOUT after {item['timeout_s']}s\n")
            if rc is not None:
                item["log_file"].close()
                remaining.remove(name)
                worst_rc = worst_rc or rc
                print(
                    f"finished {name} rc={rc} elapsed={elapsed:.1f}s "
                    f"log={item['log_path']}",
                    flush=True,
                )
        if remaining:
            print(
                "running "
                + ", ".join(sorted(remaining))
                + f" elapsed={time.time() - start:.1f}s",
                flush=True,
            )
            time.sleep(args.poll_seconds)

    print(f"all_done root={root}", flush=True)
    return worst_rc


if __name__ == "__main__":
    raise SystemExit(main())
