# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""libfnx_cpu.so installed as a Python package (tt/cpu_experts/pyproject.toml): LIB is the library fnx.paths.CPU_LIB
names through FNX_CPU_LIB, CMAKELISTS the repository's CMakeLists.txt it was built from (the engine version, through
STRATA_CMAKELISTS), DATA a copy of the repository's data/*.bin (through STRATA_DATA)."""
from pathlib import Path

LIB = Path(__file__).with_name("libfnx_cpu.so")
CMAKELISTS = Path(__file__).with_name("strata-CMakeLists.txt")
DATA = Path(__file__).with_name("data")
