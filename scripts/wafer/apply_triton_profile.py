#!/usr/bin/env python3
"""Apply one pinned patch profile; replacing tracked edits requires --force."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "third_party/wafer/patches/triton/profiles.json"
FORCE_EFFECTS = (
    "--force discards ALL uncommitted changes to tracked files in the selected "
    "Triton source, including staged edits, edits outside the patch set and any "
    "previously applied profile. No automatic backup is made. Untracked and "
    "ignored files are preserved; conflicting files cause an error. "
    "The pinned Triton commit is still required."
)


def git(source, *args, env=None, data=None):
    return subprocess.check_output(
        ["git", "-C", str(source), *args], env=env, input=data, stderr=subprocess.PIPE
    )


def restore_tracked_source(source, base, tree):
    """Reset tracked files only after checking for untracked obstructions."""
    if Path(git(source, "rev-parse", "--show-toplevel").decode().strip()).resolve() != source:
        raise RuntimeError("--force requires --source to name the Triton repository root")
    # Restore may otherwise overwrite an untracked file left by a staged
    # deletion. Check both the base tree and the new profile before any reset.
    targets = set()
    for revision in (base, tree):
        targets.update(os.fsdecode(p) for p in git(
            source, "ls-tree", "-r", "--name-only", "-z", revision
        ).split(b"\0") if p)
    parents = {str(parent) for p in targets for parent in Path(p).parents}
    # No --exclude-standard: ignored build files need the same protection.
    untracked = [os.fsdecode(p) for p in git(source, "ls-files", "--others", "-z").split(b"\0") if p]
    conflicts = [p for p in untracked if p in targets or p in parents or
                 any(str(parent) in targets for parent in Path(p).parents)]
    if conflicts:
        raise RuntimeError(
            "--force would overwrite untracked or ignored files; no tracked files were reset. "
            "Move these files aside before retrying: " + ", ".join(sorted(conflicts))
        )
    print(f"WARNING: {FORCE_EFFECTS}\nSource: {source}", file=sys.stderr)
    git(source, "restore", f"--source={base}", "--staged", "--worktree", "--", ".")


def apply_profile(source, profile, check=False, force=False):
    if check and force:
        raise RuntimeError("--check and --force cannot be used together")
    source = Path(source).resolve()
    catalog = json.loads(CATALOG.read_text())
    base = catalog["triton_commit"]
    if git(source, "rev-parse", "HEAD").decode().strip() != base:
        raise RuntimeError(f"{profile} requires Triton {base}: {source}")
    patches = [ROOT / p for p in catalog["profiles"][profile]]
    identity = hashlib.sha256()
    for path in patches:
        identity.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        identity.update(path.read_bytes())

    # Validate the entire profile in a temporary index before any destructive
    # operation. A bad patch must not discard the caller's existing edits.
    with tempfile.TemporaryDirectory(prefix="triton-profile-") as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        git(source, "read-tree", base, env=env)
        for path in patches:
            git(source, "apply", "--cached", "--whitespace=nowarn", str(path), env=env)
        tree = git(source, "write-tree", env=env).decode().strip()
        paths = git(source, "diff", "--name-only", base, tree).decode().splitlines()
        expected = {p: git(source, "show", f"{tree}:{p}") for p in paths}
        dirty = set(git(source, "diff", "--name-only", "HEAD").decode().splitlines())
        matches = dirty <= set(paths) and all(
            (source / p).is_file() and (source / p).read_bytes() == data
            for p, data in expected.items()
        )
        if force or not matches:
            if check:
                raise RuntimeError(f"Source does not match {profile}: {source}")
            if dirty and not force:
                command = shlex.join([
                    sys.executable, str(Path(__file__).resolve()), "--source", str(source),
                    "--profile", profile, "--force",
                ])
                raise RuntimeError(
                    f"Refusing to replace modified Triton source at {source}; no reset was performed.\n"
                    f"To replace these changes explicitly, run:\n  {command}\n"
                    f"WARNING: {FORCE_EFFECTS}"
                )
            delta = git(source, "diff", "--binary", base, tree)
            if force:
                restore_tracked_source(source, base, tree)
            git(source, "apply", "--check", "-", data=delta)
            git(source, "apply", "-", data=delta)
    return {"profile": profile, "triton_commit": base, "patch_sha256": identity.hexdigest(),
            "source": str(source), "files": {
                p: hashlib.sha256(data).hexdigest() for p, data in expected.items()
            }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "third_party/triton")
    parser.add_argument("--profile", choices=json.loads(CATALOG.read_text())["profiles"], required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify the profile without changing source files")
    mode.add_argument("--force", action="store_true", help=FORCE_EFFECTS)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    try:
        result = apply_profile(args.source, args.profile, check=args.check, force=args.force)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.decode(errors="replace") if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise SystemExit(detail) from None
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Triton profile verified: {args.profile} ({result['patch_sha256'][:12]})")


if __name__ == "__main__":
    main()
