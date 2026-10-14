#!/usr/bin/env python3
from __future__ import annotations

import json

from gamebench.systems.hardware import cpu_capabilities


if __name__ == "__main__":
    print(json.dumps(cpu_capabilities(), indent=2))
