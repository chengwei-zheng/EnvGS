# mirror-region SSIM / LPIPS (masked spatial maps) + shift-tolerant PSNR, on eval_npz (1296x972) vs GT persp
import os, sys, glob, json, cv2, numpy as np, torch, lpips
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
P = os.environ['GT_PERSP']; E = '/home/yiinqiang/SMBC/EnvGS/EnvGS/data/result/envgs/' + os.environ.get('RESULT_GROUP', 'mirror_b') + '/envgs_'  # RESULT_GROUP=corridor for c480 (2026-10-29)
tj = json.load(open(P + '/transforms.json')); sub = lambda a: a[1::2, 1::2][:972, :1296]
dev = 'cuda'; LA = lpips.LPIPS(net='alex', spatial=True, verbose=False).to(dev); LV = lpips.LPIPS(net='vgg', spatial=True, verbose=False).to(dev)
T = lambda x: torch.as_tensor(x).permute(2, 0, 1)[None].float().to(dev)
S = 4  # shift search radius (px)
EXCL = os.environ.get('EXCLUDE_LIGHT', '0') == '1'  # drop mirror pixels whose reflected ray hits the camera-invisible area light (+ dilation)
_c, _n = np.array([1.298, 0.249, 1.3]), np.array([0.8660254, 0.5, 0.])
_C4 = np.array([(0.66, -0.67, 2.71), (0.66, 0.81, 2.71), (2.14, -0.67, 2.65), (2.14, 0.81, 2.65)]); _o, _ex, _ey = _C4[0], _C4[2] - _C4[0], _C4[1] - _C4[0]; _pn = np.cross(_ex, _ey)
def light_mask(fr):
    T = np.array(fr['transform_matrix']); ys, xs = np.mgrid[0:tj['h'], 0:tj['w']]
    D = np.stack([(xs + 0.5 - tj['cx']) / tj['fl_x'], -(ys + 0.5 - tj['cy']) / tj['fl_y'], -np.ones_like(xs, dtype=float)], -1) @ T[:3, :3].T
    D /= np.linalg.norm(D, axis=-1, keepdims=True); P = T[:3, 3] + (((_c - T[:3, 3]) @ _n) / (D @ _n))[..., None] * D
    R = D - 2 * (D @ _n)[..., None] * _n; s = ((_o - P) @ _pn) / (R @ _pn); X = P + s[..., None] * R - _o; u = X @ _ex / (_ex @ _ex); v = X @ _ey / (_ey @ _ey)
    m = sub(((s > 0) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)).astype(np.uint8))
    return cv2.dilate(m, np.ones((7, 7), np.uint8)) > 0
print(f'{"run":26s} | {"mirror: SSIM":>12s} {"LPIPS-a":>8s} {"LPIPS-v":>8s} {"PSNR":>6s} {"PSNR±4px":>9s} {"|shift|":>7s} | {"opaque: SSIM":>12s} {"LPIPS-a":>8s} {"LPIPS-v":>8s} | {"full: SSIM":>10s} {"LPIPS-a":>8s} {"LPIPS-v":>8s}')
for r in sys.argv[1:]:
    acc = {k: [] for k in 'ms ma mv mp mps msh os oa ov fs fa fv'.split()}
    for f in sorted(glob.glob(E + r + '/eval_npz/*.npz')):
        fid = int(os.path.basename(f)[4:8]); fr = tj['frames'][fid]; pr = np.load(f)
        g = cv2.resize(cv2.imread(P + '/' + fr['file_path'])[..., ::-1], (1296, 972), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        p = pr['rgb'].astype(np.float32) / 255
        mask = sub(cv2.imread(P + '/' + fr['mask_path'], -1)); obj = sub(cv2.imread(P + '/' + fr['obj_id_path'], -1))
        mm = mask == 3
        if EXCL: lm = light_mask(fr); mm = mm & ~lm; acc.setdefault('lf', []).append(lm[mask == 3].mean() if (mask == 3).any() else 0)
        om = (mask != 3) & (obj > 0) & (mask == 0) if False else (mask != 3)
        with torch.no_grad():
            tp, tg = T(p), T(g)
            _, smap = ssim_fn(tp, tg, data_range=1.0, return_full_image=True); smap = smap[0].mean(0).cpu().numpy()
            la = LA(tp * 2 - 1, tg * 2 - 1)[0, 0].cpu().numpy(); lv = LV(tp * 2 - 1, tg * 2 - 1)[0, 0].cpu().numpy()
        acc['fs'].append(smap.mean()); acc['fa'].append(la.mean()); acc['fv'].append(lv.mean())
        acc['os'].append(smap[om].mean()); acc['oa'].append(la[om].mean()); acc['ov'].append(lv[om].mean())
        mi = cv2.erode(mm.astype(np.uint8), np.ones((2 * S + 1, 2 * S + 1))) > 0  # interior, so shifts stay inside the mirror
        if mm.sum() < 2000 or mi.sum() < 1000: continue
        acc['ms'].append(smap[mm].mean()); acc['ma'].append(la[mm].mean()); acc['mv'].append(lv[mm].mean())
        se = lambda dy, dx: ((np.roll(p, (dy, dx), (0, 1)) - g) ** 2).mean(-1)[mi].mean()
        e0 = se(0, 0); best = min(((se(dy, dx), dy, dx) for dy in range(-S, S + 1) for dx in range(-S, S + 1)))
        acc['mp'].append(-10 * np.log10(e0)); acc['mps'].append(-10 * np.log10(best[0])); acc['msh'].append(np.hypot(best[1], best[2]))
    m = {k: np.mean(v) for k, v in acc.items() if len(v)}
    print(f'{r:26s} | {m["ms"]:12.4f} {m["ma"]:8.4f} {m["mv"]:8.4f} {m["mp"]:6.2f} {m["mps"]:9.2f} {m["msh"]:7.2f} | {m["os"]:12.4f} {m["oa"]:8.4f} {m["ov"]:8.4f} | {m["fs"]:10.4f} {m["fa"]:8.4f} {m["fv"]:8.4f}   ({len(acc["ms"])} mirror views' + (f', light px {100*np.mean([x for x in acc["lf"] if x > 0]):.1f}% of mirror in {sum(x > 0 for x in acc["lf"])} views' if EXCL else '') + ')', flush=True)
