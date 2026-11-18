import argparse, os, json, random, time, yaml, numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.simplenet import SimpleNetPP
from utils.data import MVTec, CATS
from utils.metrics import image_auroc, pixel_auroc, pro_score

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); amaps, scores, masks, ys = [], [], [], []
    for x, m, y in tqdm(loader, desc="  inference", leave=False):
        a, s = model.predict(x.to(dev))
        amaps.append(a.cpu().numpy()); scores.append(s.cpu().numpy())
        masks.append(m[:, 0].numpy()); ys.append(y.numpy())
    amaps, scores, masks, ys = map(np.concatenate, (amaps, scores, masks, ys))
    print("  computing I-AUROC / P-AUROC ...", flush=True)
    i_au, p_au = image_auroc(ys, scores) * 100, pixel_auroc(masks, amaps) * 100
    print("  computing PRO (slowest metric, be patient) ...", flush=True)
    pro = pro_score(masks, amaps) * 100
    return dict(i_auroc=i_au, p_auroc=p_au, pro=pro)

def run(cfg, cat, seed, dev):
    seed_all(seed)
    tr_ds = MVTec(cfg["data_root"], cat, "train", cfg["resize"], cfg["imagesize"])
    te_ds = MVTec(cfg["data_root"], cat, "test", cfg["resize"], cfg["imagesize"])
    print(f"  [{cat}] train images: {len(tr_ds)} | test images: {len(te_ds)}", flush=True)
    assert len(tr_ds) > 0, f"No training images found for '{cat}' under {cfg['data_root']}"
    tr = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, drop_last=True)
    te = DataLoader(te_ds, batch_size=8, num_workers=4)
    model = SimpleNetPP(cfg["backbone"], cfg["embed_dim"], cfg["sigma"], cfg["th"]).to(dev)
    n_train = sum(p.numel() for p in list(model.adaptor.parameters()) + list(model.disc.parameters()))
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  params: total={n_all/1e6:.2f}M trainable={n_train/1e6:.2f}M", flush=True)
    opt_p = torch.optim.AdamW(model.adaptor.parameters(), lr=cfg["lr_proj"], weight_decay=cfg["wd"])
    opt_d = torch.optim.Adam(model.disc.parameters(), lr=cfg["lr_disc"], weight_decay=cfg["wd"])
    t0 = time.time()
    bar = tqdm(range(cfg["meta_epochs"]), desc=f"  training {cat}", unit="meta-ep")
    for meta in bar:
        model.adaptor.train(); model.disc.train()
        losses = []
        for _ in range(cfg["gan_epochs"]):
            for x, _, _ in tr:
                opt_p.zero_grad(); opt_d.zero_grad()
                loss = model.train_loss(x.to(dev)); loss.backward()
                opt_p.step(); opt_d.step()
                losses.append(loss.item())
        bar.set_postfix(loss=f"{np.mean(losses):.4f}")
    print(f"  training done in {time.time()-t0:.0f}s", flush=True)
    res = evaluate(model, te, dev)   # final-epoch metrics: no test-set checkpoint selection
    print(f"  [{cat}] I-AUROC={res['i_auroc']:.2f}  P-AUROC={res['p_auroc']:.2f}  PRO={res['pro']:.2f}", flush=True)
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--backbone"); ap.add_argument("--cats", nargs="*", default=CATS)
    ap.add_argument("--seeds", nargs="*", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.backbone: cfg["backbone"] = a.backbone
    seeds = a.seeds if a.seeds is not None else cfg["seeds"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else "  <-- CPU will be very slow"))
    print(f"backbone: {cfg['backbone']} | data_root: {cfg['data_root']}")
    print(f"categories: {len(a.cats)} | seeds: {seeds}")
    print(f"config: {cfg}")
    print("=" * 60, flush=True)
    out = os.path.join(cfg["out_dir"], cfg["backbone"]); os.makedirs(out, exist_ok=True)
    for si, seed in enumerate(seeds, 1):
        print(f"\n##### Seed {seed} ({si}/{len(seeds)}) #####", flush=True)
        res = {}
        for ci, c in enumerate(tqdm(a.cats, desc=f"seed {seed} categories", unit="cat"), 1):
            print(f"\n>>> Category {ci}/{len(a.cats)}: {c}", flush=True)
            res[c] = run(cfg, c, seed, dev)
        json.dump(res, open(f"{out}/seed{seed}.json", "w"), indent=1)
        print(f"\nSeed {seed} summary (saved to {out}/seed{seed}.json):")
        for k in ("i_auroc", "p_auroc", "pro"):
            print(f"  mean {k}: {np.mean([r[k] for r in res.values()]):.2f}")