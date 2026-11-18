import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.ndimage import label

def image_auroc(labels, scores): return roc_auc_score(labels, scores)
def pixel_auroc(masks, amaps): return roc_auc_score(masks.ravel() > 0.5, amaps.ravel())

def pro_score(masks, amaps, fpr_max=0.3, n_thr=200):
    """Per-Region Overlap AUC up to FPR=fpr_max (normalised), as in MVTec AD."""
    masks = masks > 0.5
    regions = []
    for i, m in enumerate(masks):
        lab, n = label(m)
        for r in range(1, n + 1):
            regions.append((i, lab == r))
    neg = ~masks
    thr = np.linspace(amaps.min(), amaps.max(), n_thr)[::-1]
    fprs, pros = [], []
    for t in thr:
        pred = amaps >= t
        fprs.append((pred & neg).sum() / max(neg.sum(), 1))
        pros.append(np.mean([(pred[i] & reg).sum() / reg.sum() for i, reg in regions]))
    fprs, pros = np.array(fprs), np.array(pros)
    keep = fprs <= fpr_max
    return np.trapz(pros[keep], fprs[keep]) / fpr_max
