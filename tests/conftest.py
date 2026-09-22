import os

import mlx.core as mx

# HEMI_DEVICE=cpu runs the tests on the CPU, leaving the GPU free for training.
if os.environ.get("HEMI_DEVICE") == "cpu":
    mx.set_default_device(mx.cpu)
