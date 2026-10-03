#!/bin/sh
# Run the same C++/Python checks in a disposable Linux container from macOS/Linux.
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"

# Copy only project inputs; do not mount the workspace or pass environment files.
tar --no-xattrs --exclude='__pycache__' -cf - CMakeLists.txt pyproject.toml README.md pysource cpp tests |
    docker run --rm -i python:3.12-slim sh -c '
        set -eu
        mkdir /work
        tar -xf - -C /work
        apt-get update -qq
        apt-get install -y -qq --no-install-recommends g++ make cmake > /tmp/install.log 2>&1 || {
            cat /tmp/install.log
            exit 1
        }
        cd /work
        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/local/bin/python3
        cmake --build build --parallel 2
        ctest --test-dir build --output-on-failure
    '
