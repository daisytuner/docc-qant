# docc-qant

A plugin for [DOCC](https://github.com/daisytuner/docc) (Daisytuner Optimizing Compiler Collection) that adds support for the Q.ANT native computing toolkit, enabling efficient execution of numerical computations with bfloat16 precision.

For Python/PyPI usage, see [`python/README.md`](python/README.md).

## Prerequisites

### System Requirements
- Linux (Ubuntu 24.04 or similar)
- Git with LFS support
- CMake 3.15+
- Ninja build system
- Clang 19
- Python 3.11, 3.12, 3.13, or 3.14

### System Dependencies

Install required packages:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    git-lfs \
    cmake \
    ninja-build \
    clang-19 \
    libgmp-dev \
    libcurl4-gnutls-dev \
    libisl-dev \
    nlohmann-json3-dev \
    libopenblas-dev
```

Optional development tools:

```bash
sudo apt install -y clang-format-19
```

Install DLPack v1.2 headers:

```bash
cd /tmp
wget https://github.com/dmlc/dlpack/archive/refs/tags/v1.2.tar.gz
tar -xzf v1.2.tar.gz
cd dlpack-1.2
sudo cp -r include/dlpack /usr/local/include/
```

You will also need the Q.ANT Native Computing Toolkit (proprietary):
```bash
sudo apt install ./qant-native-computing-toolkit.2.1.0-1-amd64.deb
```

## Building from Source

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/your-org/docc-qant.git
cd docc-qant
git lfs pull
```

### 2. Build C++ Plugin

```bash
mkdir build && cd build

cmake -G Ninja \
  -DCMAKE_C_COMPILER=clang-19 \
  -DCMAKE_CXX_COMPILER=clang++-19 \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DCMAKE_BUILD_TYPE=Release \
  ..

ninja -j$(nproc)
sudo ninja install
```

**Build Options:**
- `CMAKE_BUILD_TYPE`: `Release` (optimized) or `Debug` (with debug symbols)
- `DOCC_QANT_ENABLE_COVERAGE`: Enable code coverage (default: OFF)
- `BUILD_TESTS`: Build C++ tests (default: OFF)

### 3. Install Python Bindings

```bash
cd ..  # Back to repository root

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install build dependencies
pip install scikit-build-core pybind11

# Install DOCC Python bindings (includes numpy, scipy, ml_dtypes)
pip install -e 3rdParty/docc/python/

# Install docc-qant Python package
pip install -e python/

# Install pytorch support (docc-ai, requires additional system packages)
pip install torch==2.10.0+cpu torchvision==0.25.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch-mlir==20260512.810 -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
pip install -e 3rdParty/docc/mlir/

# Optional: Install development tools
pip install pytest  # For running tests
pip install black   # For code formatting
```

#### MLIR Support

Building the `docc-ai` (MLIR) Python package for pytorch support requires MLIR/LLVM 19 development packages:

```bash
sudo apt install -y libmlir-19-dev mlir-19-tools
```

These provide the MLIR CMake config files and headers needed by `3rdParty/docc/mlir/`. The MLIR wheel also requires `torch-mlir` (only supported on Python 3.11 and 3.12).

### 4. Verify Installation

```bash
# Set library path
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Test Python import
python -c "from docc.qant import register_docc_plugin; print('Success!')"

# Run unit tests (optional)
cd build
./plugin/tests/docc-qant-plugin_test
```


## Development

### Running Tests

### C++ Tests

```bash
cd build
./plugin/tests/docc-qant-plugin_test
```

### Python Tests

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
pytest -v python/tests/
```

### Code Formatting

The project uses consistent code formatting:

**C++ (clang-format):**
```bash
# Check formatting
clang-format-19 -style=file --dry-run --Werror plugin/**/*.h plugin/**/*.cpp

# Auto-format
clang-format-19 -style=file -i plugin/**/*.h plugin/**/*.cpp
```

**Python (black):**
```bash
# Check formatting
black --check python/docc/ python/tests/

# Auto-format
black python/docc/ python/tests/
```
