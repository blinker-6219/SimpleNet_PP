"""Params/latency/FPS. usage: python scripts/benchmark.py repvit 224"""
import sys, time, torch
sys.path.insert(0, ".")
from models.simplenet import SimpleNetPP
key, size = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 224
m = SimpleNetPP(key).cuda().eval()
n = sum(p.numel() for p in m.parameters()) / 1e6
x = torch.randn(1, 3, size, size, device="cuda")
for _ in range(50): m.predict(x)
torch.cuda.synchronize(); t = time.time()
for _ in range(300): m.predict(x)
torch.cuda.synchronize(); dt = (time.time() - t) / 300
print(f"{key}: params={n:.2f}M latency={dt*1e3:.2f}ms FPS={1/dt:.0f} ({size}px, batch 1)")
