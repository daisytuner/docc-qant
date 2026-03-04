# docc-qant

A plugin for DOCC (Daiytuner Optimizing Compiler Collection) that adds support for the Qant native computing toolkit, enabling efficient execution of numerical computations with bfloat16 precision. The `docc-qant` plugin extends DOCC with Qant target support, allowing you to compile and execute Python functions with numpy operations on Qant hardware.

## Quick Start

```python
import numpy as np
import ml_dtypes
from docc.python import native
from docc.qant import register_docc_plugin

# Register the Qant target
register_docc_plugin()

# Decorate functions to run on Qant hardware
@native(target="qant")
def matmul_bf16(a, b):
    return a @ b

# Use standard numpy arrays with bfloat16 precision
M, K, N = 32, 48, 64
A = np.random.rand(M, K).astype(ml_dtypes.bfloat16)
B = np.random.rand(K, N).astype(ml_dtypes.bfloat16)

# Automatically compiled and executed on Qant hardware
C = matmul_bf16(A, B)
```

## Installation

### Prerequisites

#### System Requirements
- Linux (Ubuntu 24.04 or similar)
- Git with LFS support
- CMake 3.15+
- Ninja build system
- Clang 19
- Python 3.11, 3.12, 3.13, or 3.14

#### System Dependencies

Install required packages:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    libdlpack-dev \
    libxtensor-blas-dev \
    xtensor-dev \
    cmake \
    ninja-build \
    clang-19 \
    git-lfs
```

Install DLPack v1.2 headers:

```bash
cd /tmp
wget https://github.com/dmlc/dlpack/archive/refs/tags/v1.2.tar.gz
tar -xzf v1.2.tar.gz
cd dlpack-1.2
sudo cp -r include/dlpack /usr/local/include/
```

You will also need the Qant Native Computing Toolkit (proprietary):
```bash
# Install the Qant toolkit
sudo apt install ./qant-native-computing-toolkit.2.1.0-1-amd64.deb
```

### Building from Source

#### 1. Clone the Repository

```bash
git clone --recursive https://github.com/your-org/docc-qant.git
cd docc-qant
git lfs pull
```

#### 3. Build C++ Plugin

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

#### 4. Install Python Bindings

```bash
cd ..  # Back to repository root

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install numpy scipy ml_dtypes pybind11 scikit-build-core

# Install DOCC Python bindings
pip install -e 3rdParty/docc/python/

# Install docc-qant Python package
pip install -e python/
```

#### 5. Verify Installation

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

#### C++ Tests

```bash
cd build
./plugin/tests/docc-qant-plugin_test
```

#### Python Tests

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
