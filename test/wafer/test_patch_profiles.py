"""Check profile composition against the actual pinned Triton Git objects."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("apply_triton_profile", ROOT / "scripts/wafer/apply_triton_profile.py")
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)


@pytest.fixture
def triton_source(tmp_path):
    source = tmp_path / "triton"
    subprocess.run(["git", "clone", "--shared", "--no-checkout", str(ROOT / "third_party/triton"), str(source)],
                   check=True, capture_output=True)
    commit = "c3c476f357f1e9768ea4e45aa5c17528449ab9ef"
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", commit], check=True, capture_output=True)
    return source


@pytest.mark.parametrize("profile", ["ascend", "wafer-tools", "wafer-frontend"])
def test_profiles_apply_idempotently_and_preserve_edits(triton_source, profile):
    source = triton_source
    first = profiles.apply_profile(source, profile)
    diff = profiles.git(source, "diff", "--binary")
    assert profiles.apply_profile(source, profile) == first
    assert profiles.apply_profile(source, profile, check=True) == first
    assert profiles.git(source, "diff", "--binary") == diff
    edited = source / "python/src/ir.cc"
    edited.write_text(edited.read_text() + "\n// independent user change\n")
    with pytest.raises(RuntimeError, match="Refusing to replace modified") as error:
        profiles.apply_profile(source, profile)
    assert "--force" in str(error.value)
    assert "including staged edits" in str(error.value)
    assert "No automatic backup" in str(error.value)
    assert edited.read_text().endswith("// independent user change\n")


@pytest.mark.parametrize("previous,target", [
    ("wafer-frontend", "ascend"), ("ascend", "wafer-tools"),
])
def test_force_replaces_profile_and_tracked_edits_only(triton_source, tmp_path, previous, target):
    source = triton_source
    profiles.apply_profile(source, previous)
    original_readme = profiles.git(source, "show", "HEAD:README.md")
    (source / "README.md").write_text("staged edit outside the profile\n")
    (source / "staged-only.txt").write_text("new tracked file\n")
    profiles.git(source, "add", "README.md", "staged-only.txt")
    (source / "README.md").write_text("unstaged edit outside the profile\n")
    (source / "python/src/ir.cc").write_text("independent edit inside a profile\n")
    (source / "force-untracked.txt").write_text("keep untracked\n")
    exclude = source / ".git/info/exclude"
    exclude.write_text(exclude.read_text() + "\nforce-ignored.txt\n")
    (source / "force-ignored.txt").write_text("keep ignored\n")
    record = tmp_path / "profile.json"
    outside = tmp_path / "outside-source.txt"
    outside.write_text("keep the caller's other files\n")

    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/wafer/apply_triton_profile.py"),
        "--source", str(source), "--profile", target, "--force", "--record", str(record),
    ], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "WARNING:" in result.stderr and "No automatic backup" in result.stderr
    assert str(source) in result.stderr
    assert json.loads(record.read_text()) == profiles.apply_profile(source, target, check=True)
    assert (source / "README.md").read_bytes() == original_readme
    assert not (source / "staged-only.txt").exists()
    assert profiles.git(source, "diff", "--cached", "--binary") == b""
    assert (source / "force-untracked.txt").read_text() == "keep untracked\n"
    assert (source / "force-ignored.txt").read_text() == "keep ignored\n"
    assert outside.read_text() == "keep the caller's other files\n"


@pytest.mark.parametrize("ignored", [False, True])
def test_force_refuses_untracked_obstructions_without_resetting(triton_source, ignored):
    source = triton_source
    profiles.git(source, "rm", "--cached", "README.md")
    (source / "README.md").write_text("keep the untracked replacement\n")
    if ignored:
        exclude = source / ".git/info/exclude"
        exclude.write_text(exclude.read_text() + "\n/README.md\n")
    (source / "python/src/ir.cc").write_text("keep the tracked edit too\n")
    working = profiles.git(source, "diff", "--binary")
    staged = profiles.git(source, "diff", "--cached", "--binary")

    with pytest.raises(RuntimeError, match="untracked or ignored files"):
        profiles.apply_profile(source, "ascend", force=True)

    assert profiles.git(source, "diff", "--binary") == working
    assert profiles.git(source, "diff", "--cached", "--binary") == staged
    assert (source / "README.md").read_text() == "keep the untracked replacement\n"


def test_force_does_not_bypass_pinned_commit(triton_source):
    source = triton_source
    # The source cache can be shallow, so create a different revision locally.
    profiles.git(source, "-c", "user.name=Profile Test", "-c", "user.email=profile-test@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "Different test revision")
    head = profiles.git(source, "rev-parse", "HEAD")
    (source / "README.md").write_text("keep edits on the wrong revision\n")
    with pytest.raises(RuntimeError, match="requires Triton"):
        profiles.apply_profile(source, "ascend", force=True)
    assert profiles.git(source, "rev-parse", "HEAD") == head
    assert (source / "README.md").read_text() == "keep edits on the wrong revision\n"


def test_force_cannot_be_combined_with_check(triton_source):
    source = triton_source
    (source / "README.md").write_text("check must not discard this\n")
    with pytest.raises(RuntimeError, match="--check and --force"):
        profiles.apply_profile(source, "ascend", check=True, force=True)
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/wafer/apply_triton_profile.py"),
        "--source", str(source), "--profile", "ascend", "--check", "--force",
    ], capture_output=True, text=True)
    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr
    assert (source / "README.md").read_text() == "check must not discard this\n"


def test_force_validates_patches_before_discarding_edits(triton_source, tmp_path, monkeypatch):
    source = triton_source
    config_root = tmp_path / "invalid-profile"
    patch_dir = config_root / "patch/triton"
    patch_dir.mkdir(parents=True)
    (patch_dir / "invalid.patch").write_text("not a valid patch\n")
    catalog = config_root / "profiles.json"
    catalog.write_text(json.dumps({
        "triton_commit": profiles.git(source, "rev-parse", "HEAD").decode().strip(),
        "profiles": {"ascend": ["patch/triton/invalid.patch"]},
    }))
    monkeypatch.setattr(profiles, "ROOT", config_root)
    monkeypatch.setattr(profiles, "CATALOG", catalog)
    (source / "README.md").write_text("keep staged edit\n")
    profiles.git(source, "add", "README.md")
    (source / "README.md").write_text("keep unstaged edit\n")
    working = profiles.git(source, "diff", "--binary")
    staged = profiles.git(source, "diff", "--cached", "--binary")

    with pytest.raises(subprocess.CalledProcessError):
        profiles.apply_profile(source, "ascend", force=True)

    assert profiles.git(source, "diff", "--binary") == working
    assert profiles.git(source, "diff", "--cached", "--binary") == staged


def test_force_rejects_a_source_inside_the_repository(triton_source):
    source = triton_source
    (source / "README.md").write_text("keep changes outside the selected subdirectory\n")
    with pytest.raises(RuntimeError, match="repository root"):
        profiles.apply_profile(source / "python", "ascend", force=True)
    assert (source / "README.md").read_text() == "keep changes outside the selected subdirectory\n"
