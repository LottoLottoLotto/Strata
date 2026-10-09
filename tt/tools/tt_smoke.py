"""P150 smoke: open the device, report grid / DRAM, run one bf16 and one bfp4 matmul against torch."""
import time

import torch
import ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=24576)
try:
    g = dev.compute_with_storage_grid_size()
    print(f"arch {dev.arch()}  compute grid {g.x}x{g.y}")
    torch.manual_seed(0)
    x = torch.randn(32, 2560)
    w = torch.randn(2560, 640) * 0.02
    ref = x @ w
    for name, dt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b), ("bfp4", ttnn.bfloat4_b)):
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        tw = ttnn.from_torch(w, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        ttnn.matmul(tx, tw)  # compile
        t0 = time.time()
        out = ttnn.to_torch(ttnn.matmul(tx, tw)).float()
        dt_ms = (time.time() - t0) * 1e3
        rel = ((out - ref).norm() / ref.norm()).item()
        print(f"matmul 32x2560x640 weight {name}: rel err {rel:.4f}, {dt_ms:.2f} ms incl. readback")
finally:
    ttnn.close_device(dev)
print("tt_smoke OK")
