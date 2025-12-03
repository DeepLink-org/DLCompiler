import setuptools
import os
import shutil
import sys
import glob
from setuptools.command.build_py import build_py
from setuptools.command.install import install
import importlib.util
import subprocess
import site

setuptools._distutils_hack = lambda *args, **kwargs: None
from setuptools.dist import Distribution

# Monkey-patch: 让 bdist_egg 认为包永远不安全（从而生成目录）
_original_zip_safe = getattr(Distribution, "zip_safe", None)


def _always_false_zip_safe(self):
    return False


Distribution.zip_safe = property(_always_false_zip_safe)

ORIGIN_TRITON_PATH = None


# check triton
def check_triton_package():
    try:
        spec = importlib.util.find_spec("triton")
        if spec is not None:
            print("Triton package found.")
            global ORIGIN_TRITON_PATH
            ORIGIN_TRITON_PATH = spec.origin
            print(f"ORIGIN_TRITON_PATH: {ORIGIN_TRITON_PATH}")
            if ORIGIN_TRITON_PATH.endswith("__init__.py"):
                ORIGIN_TRITON_PATH = ORIGIN_TRITON_PATH[:-12]
            print(f"ORIGIN_TRITON_PATH: {ORIGIN_TRITON_PATH}")

        else:
            print("Triton package not found.")
            assert False, "Triton package not found, please choose env with triton."
    except ImportError:
        print("Triton package not found.")
        assert False, "Triton package not found, please choose env with triton."


def copy_triton_package():
    global ORIGIN_TRITON_PATH
    source_dir = ORIGIN_TRITON_PATH
    backup_dir = os.path.join(site.getsitepackages()[0], "triton-ori")
    if os.path.exists(source_dir):
        if os.path.exists(backup_dir):
            print(f"{backup_dir} already exists, use it {backup_dir}.")
        else:
            shutil.copytree(
                source_dir,
                backup_dir,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

    source_dir = backup_dir

    target_dir = "./triton"
    if not os.path.exists(target_dir):
        print(f"Copying {source_dir} to {target_dir}")
        shutil.copytree(
            source_dir,
            target_dir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    else:
        print(f"{target_dir} already exists, skipping.")

    if not os.listdir(target_dir):
        assert False, f"{target_dir} is empty, please check."


def copy_backend_files():
    source_dir = "./backend"
    target_dir = "./triton/backends/metax"
    print(f"source_dir: {os.path.realpath(source_dir)}")

    if not os.path.exists(target_dir):
        print(f"target_dir realpath: {os.path.realpath(target_dir)}")
        # 创建对应目录
        os.makedirs(target_dir)
        # assert False, f"Target directory {target_dir} does not exist, please check the path."

    for filename in ["compiler.py", "driver.py", "maca.py", "utils.py", "maca.c"]:
        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(target_dir, filename)

        if os.path.exists(src_path):
            print(f"Copying {src_path} to {dest_path}")
            shutil.copy2(src_path, dest_path)
        else:
            realpath = os.path.realpath(src_path)
            assert (
                False
            ), f"Source file {realpath} does not exist, please check the path."

    # 拷贝lib目录文件，到dest_path目录
    lib_dir = os.path.join(source_dir, "lib")
    dest_lib_dir = os.path.join(target_dir, "lib")
    if os.path.exists(lib_dir):
        print(
            f"Copying {os.path.realpath(lib_dir)} to {os.path.realpath(dest_lib_dir)}"
        )
        shutil.copytree(
            lib_dir, dest_lib_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )

    # 拷贝bin目录
    bin_dir = os.path.join(source_dir, "bin")
    dest_bin_dir = os.path.join(target_dir, "bin")
    if os.path.exists(bin_dir):
        print(
            f"Copying {os.path.realpath(bin_dir)} to {os.path.realpath(dest_bin_dir)}"
        )
        shutil.copytree(
            bin_dir, dest_bin_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )


def modify_backend_name():
    maca_dir = "./triton/backends/metax"
    dicp_triton_dir = "./triton/backends/dicp_triton"
    if os.path.exists(dicp_triton_dir):
        shutil.rmtree(dicp_triton_dir)

    if os.path.exists(maca_dir):
        print(f"Renaming {maca_dir} to {dicp_triton_dir}")
        shutil.move(maca_dir, dicp_triton_dir)
    else:
        assert (
            False
        ), f"Source directory {maca_dir} does not exist, please check the path."


# pip uninstall triton -y
def uninstall_triton_package():
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "uninstall", "triton", "-y"]
        )
        print("Triton package uninstalled successfully.")
    except subprocess.CalledProcessError:
        print("Failed to uninstall Triton package.")
        assert False, "Failed to uninstall Triton package."


# 清理旧的 egg 文件
def clean_old_egg():
    site_packages = site.getsitepackages()[0]
    egg_pattern = os.path.join(site_packages, "triton-*.egg")
    egg_files = glob.glob(egg_pattern)
    print(f"Found old egg files: {egg_files}")

    for egg_file in egg_files:
        print(f"Removing old egg file: {egg_file}")
        if os.path.isdir(egg_file):
            shutil.rmtree(egg_file)
        else:
            os.remove(egg_file)


# 清理旧的triton目录
def clean_old_triton():
    site_packages = site.getsitepackages()[0]
    triton_pattern = os.path.join(site_packages, "triton")
    triton_files = glob.glob(triton_pattern)
    print(f"Found old triton files: {triton_files}")

    for triton_file in triton_files:
        print(f"Removing old triton file: {triton_file}")
        if os.path.isdir(triton_file):
            shutil.rmtree(triton_file)
        else:
            os.remove(triton_file)


def clean_build_artifacts():
    # 清理构建目录
    build_dirs = ["build", "dist"]
    for dir_name in build_dirs:
        if os.path.exists(dir_name):
            print(f"Removing {dir_name} directory")
            shutil.rmtree(dir_name)

    # 清理 egg-info 目录
    egg_info_dirs = glob.glob("*.egg-info")
    for dir_name in egg_info_dirs:
        print(f"Removing {dir_name} directory")
        shutil.rmtree(dir_name)


clean_build_artifacts()
clean_old_egg()

check_triton_package()
copy_triton_package()
copy_backend_files()
modify_backend_name()
uninstall_triton_package()
clean_old_egg()
clean_old_triton()


def clean_build_dir():
    build_dir = os.path.join(os.getcwd(), "build")
    if os.path.exists(build_dir):
        print(f"Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir)


class CustomBuildPy(build_py):
    def run(self):
        super().run()

        self.copy_files("libtriton*.so", "triton/_C")
        self.copy_files("mlir-opt", "triton/backends/dicp_triton/bin")
        self.copy_files("ext_maca_mathlib.bc", "triton/backends/dicp_triton/lib")
        # self.copy_files("*.h", "triton/_C/include")
        # self.copy_files("*.hpp", "triton/_C/include")
        # self.copy_files("*.c", "triton/backends/dicp_triton")

    def copy_files(self, pattern, dest_subdir):
        dest_dir = os.path.join(self.build_lib, dest_subdir)
        os.makedirs(dest_dir, exist_ok=True)

        source_files = glob.glob(os.path.join("**", pattern), recursive=True)

        print(f"[DEBUG] Searching for files matching pattern: {pattern}")
        print(f"[DEBUG] Found source files: {source_files}")

        for src_path in source_files:
            if not os.path.isfile(src_path):
                continue

            dest_path = os.path.join(dest_dir, os.path.basename(src_path))

            if os.path.abspath(src_path) != os.path.abspath(dest_path):
                print(f"Copying {src_path} to {dest_path}")
                shutil.copy2(src_path, dest_path)
            else:
                print(f"Skipping copy (source and destination same): {src_path}")


packages = setuptools.find_packages(where=".")

package_data = {
    "triton": [
        "_C/*.so",  # 包含所有共享库
        "_C/include/*",  # 包含头文件
        "backends/**/*",  # 包含后端文件（包括 .bc）
        "backends/dicp_triton/bin/*",
        "backends/dicp_triton/lib/*.bc",  # 显式包含 bitcode 文件
        "compiler/**/*",
        "language/**/*",
        "ops/**/*",
        "runtime/**/*",
        "tools/**/*",
        # 'tutorials/*',
        # 'tutorials/**/*',
        # '**/*.bc',       # 包含所有位置的 bitcode 文件
        # '**/*.h',        # 包含所有头文件
        # '**/*.hpp'       # 包含所有 C++ 头文件
    ]
}

clean_build_dir()

setuptools.setup(
    name="triton",
    version="0.0.1",
    description="A language and compiler for custom Deep Learning operations on MACA backend",
    long_description="A language and compiler for custom Deep Learning operations on MACA backend",
    long_description_content_type="text/markdown",
    url="https://github.com/DeepLink-org/DLCompiler.git",
    packages=packages,
    package_data=package_data,
    include_package_data=True,
    # zip_safe=False,
    # options={
    #     'bdist_wheel': {'universal': False},
    #     # 'bdist_egg': {'zip_safe': lambda: False},
    # },
    python_requires=">=3.10",
    cmdclass={"build_py": CustomBuildPy},
)

print(f"Python executable: {sys.executable}")
print(f"Install prefix: {sys.prefix}")

print(f"finish.....")
