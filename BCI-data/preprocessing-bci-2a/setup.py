import os
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

# =============================================================================
# 1. Locate the directory containing this setup.py
# =============================================================================
here = os.path.abspath(os.path.dirname(__file__))

# =============================================================================
# 2. Define compiler and linker flags for OpenMP, depending on platform
# =============================================================================
extra_compile_args = ["-O3", "-std=c++17"]
extra_link_args = []

if sys.platform == "darwin":
    # macOS + Homebrew’s libomp
    #   -Xpreprocessor -fopenmp : tell clang’s preprocessor to accept OpenMP pragmas
    #   -lomp : link against libomp (installed by `brew install libomp`)
    extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
    extra_link_args    += ["-lomp"]
elif sys.platform.startswith("linux"):
    # Linux + GCC or Clang (Ubuntu, etc.)
    extra_compile_args += ["-fopenmp"]
    extra_link_args    += ["-fopenmp"]
# (On Windows/MSVC, you might add "/openmp" instead, but we omit that here.)

# =============================================================================
# 3. Construct the list of include directories
# =============================================================================
include_dirs = [
    # 3.1 pybind11’s include directories (non-user and user)
    pybind11.get_include(),
    pybind11.get_include(user=True),

    # 3.2 Eigen headers (Homebrew on macOS, apt on Ubuntu)
    #     If you installed Eigen to a different prefix, adjust this path.
    "/usr/local/include/eigen3",

    # 3.3 Local tqdm.hpp: point to tqdm.cpp/include so that
    #     #include <tqdm/tqdm.h> resolves correctly.
    os.path.join(here, "tqdm.cpp", "include"),
]

# =============================================================================
# 4. Define the Pybind11Extension
# =============================================================================
ext_modules = [
    Pybind11Extension(
        name="betaprime_cpp",                 # name of the Python module
        sources=["betaprime_cpp.cpp"],        # your C++ source file
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++"
    ),
]

# =============================================================================
# 5. Call setup()
# =============================================================================
setup(
    name="betaprime_cpp",
    version="0.1",
    author="Rishabh Kumar",
    description="Optimized C++ backend for Betaprime kernel (using Eigen, OpenMP, tqdm)",
    long_description=(
        "This module provides a highly optimized implementation of the "
        "pairwise Betaprime kernel using Eigen::Map (zero-copy), OpenMP (multithreading), "
        "and tqdm for C++ progress bars. It exports a function `pairwise_kernel(X, Y, alpha)` "
        "that accepts two NumPy arrays of shape (n, d, d) and (m, d, d), respectively, "
        "and returns an (n, m) NumPy array of kernel values."
    ),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    # Optional: declare at least Python 3.7+ if you rely on certain Pybind11 features.
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
)