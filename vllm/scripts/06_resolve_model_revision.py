#!/usr/bin/env python3

import sys

from huggingface_hub import HfApi


if len(sys.argv) != 2:
    print(
        "Usage: 06_resolve_model_revision.py <MODEL>",
        file=sys.stderr,
    )
    sys.exit(1)


model = sys.argv[1]

info = HfApi().model_info(
    model,
    revision="main",
)

print(info.sha)