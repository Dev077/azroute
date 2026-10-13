"""Draw docs/architecture.svg. The boxes are written by hand from model.py / features.py: update them when the network changes.   python docs/draw_architecture.py"""
import sys
from xml.sax.saxutils import escape

import os
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.svg")
LH, PAD_T, PAD_B = 16, 26, 10
STY = {
    "in":   ("#f1f3f5", "#868e96"),
    "cnn":  ("#e7f0fb", "#3b6fb6"),
    "net":  ("#f1eafa", "#7048b8"),
    "head": ("#e8f6ec", "#2f8a4c"),
    "pool": ("#fff6e5", "#c28a17"),
}
parts, boxes = [], {}


def box(key, x, y, w, title, lines, style, loss=None):
    fill, stroke = STY[style]
    n = len(lines) + (1 if loss else 0)
    h = PAD_T + LH * n + PAD_B if n else 34
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    parts.append(f'<text x="{x + 10}" y="{y + 21}" class="t">{escape(title)}</text>')
    for i, ln in enumerate(lines):
        parts.append(f'<text x="{x + 10}" y="{y + PAD_T + 14 + i * LH}" class="l">{escape(ln)}</text>')
    if loss:
        parts.append(f'<text x="{x + 10}" y="{y + PAD_T + 14 + len(lines) * LH}" class="loss">{escape(loss)}</text>')
    boxes[key] = (x, y, w, h)
    return y + h


def side(key, s):
    x, y, w, h = boxes[key]
    return {"t": (x + w / 2, y), "b": (x + w / 2, y + h), "l": (x, y + h / 2), "r": (x + w, y + h / 2)}[s]


def arrow(points, label=None):
    pts = " ".join(f"{px},{py}" for px, py in points)
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#495057" stroke-width="1.4" marker-end="url(#ah)"/>')
    if label:
        (x1, y1), (x2, y2) = points[-2], points[-1]
        parts.append(f'<text x="{(x1 + x2) / 2 + 6}" y="{(y1 + y2) / 2 - 4}" class="lab">{escape(label)}</text>')


def down(a, b, label=None):
    (x1, y1), (x2, y2) = side(a, "b"), side(b, "t")
    arrow([(x1, y1), (x1, y1 + (y2 - y1) / 2), (x2, y1 + (y2 - y1) / 2), (x2, y2)] if x1 != x2 else [(x1, y1), (x2, y2)], label)



L, M, R = 30, 470, 960
LW, MW, RW = 380, 410, 330

parts.append(f'<text x="{L}" y="52" class="sec">GRID STREAM (multi-scale CNN)</text>')
parts.append(f'<text x="{M}" y="52" class="sec">NET STREAM (one node per net, one pair per net pair)</text>')
parts.append(f'<text x="{R}" y="52" class="sec">WHOLE-REROUTE</text>')

# ---- inputs ----
y = box("canvas", L, 66, LW, "Region canvas  (uint8 bit field)",
        ["6 layers x 64 x 64 (region max-pooled if > 64 cells)",
         "bit 0  occupied by a fixed net", "bit 1  occupied by an already-rerouted net",
         "bit 2  pin of a fixed net", "bit 3  pin of a pending net",
         "bit 4  previous route of a pending net (KEEP)", "bit 5  inside the region"], "in")
box("delays", L, y + 14, LW, "Layer + via delays", ["7 values / max delay"], "in")
y = box("nfeat", M, 66, MW, "Net features   16 x 24",
        ["pending / rerouted / seed flags, log fanout, dies,",
         "box size & centre, ideal share, log(old / ideal),",
         "log(new / ideal), mode one-hot (7), demand, valid"], "in")
y = box("boxes", M, y + 12, MW, "Net boxes   16 x 4", ["pin bounding box on the canvas"], "in")
y = box("pairs", M, y + 12, MW, "Pair conflicts   16 x 16 x 3  (exact, from the router)",
        ["share of net i's ideal tree that net j", "  occupies now / would occupy if kept /",
         "  blocks in the via columns above its pins"], "in")
box("legal", M, y + 12, MW, "Legal decisions   16 x 7", ["pending nets x 7 modes"], "in")
box("glob", R, 66, RW, "Reroute features   6",
    ["log(old / ideal), slack, progress,", "canvas scale, #pending, #nets"], "in")

# ---- grid stream ----
y = box("planes", L, 330, LW, "Input planes   46 x 64 x 64",
        ["36 bit planes, 7 delay planes, 1 demand,", "2 coordinates"], "cnn")
y = box("e1", L, y + 22, LW, "64 x 64   stem + 2 residual blocks, 96 ch",
        ["conv 3x3 46 -> 96; blocks: conv 3x3, BN, ReLU, conv 3x3, BN, + skip"], "cnn")
y = box("e2", L, y + 22, LW, "32 x 32   stride-2 conv + 3 blocks, 160 ch",
        ["blocks add region-wide context (mean & max pool -> bias)"], "cnn")
y = box("e3", L, y + 22, LW, "16 x 16   stride-2 conv + 3 blocks, 224 ch",
        ["each cell sees the whole region; + region context"], "cnn")
y = box("d2", L, y + 22, LW, "32 x 32   upsample + skip + 1 block, 160 ch",
        ["-> grid read by the nets (2 x 2 cells per token)"], "cnn")
y = box("d1", L, y + 22, LW, "64 x 64   upsample + skip + 1 block, 96 ch",
        ["full-resolution features; backbone 6.12 M params"], "cnn")

# ---- net stream ----
y0 = boxes["e2"][1]
y = box("roi", M, y0, MW, "Box-mean pooling", ["each net: mean of 64 x 64 features over its box -> 96"], "pool")
y = box("init", M, y + 22, MW, "Initial states",
        ["nets  h: [net 24 | reroute 6 | pooled 96] -> 256",
         "pairs z: [geometry 6 | conflict i,j 3 | j,i 3] -> 64"], "net")
y = box("inter", M, y + 22, MW, "Net interaction layer  x 3     (8 heads)",
        ["1  grid read: net attends over the 1024 grid tokens,",
         "   biased to its own box (learned outside penalty)",
         "2  pair update: z_ij += f(h_i) + g(h_j) + overlap of",
         "   i's and j's grid attention;  z += MLP(z)",
         "3  net exchange: attention among nets with bias(z_ij);",
         "   message from j to i = v_j + W z_ij",
         "4  feed-forward 256 -> 1024 -> 256",
         "pre-LayerNorm residuals; 3.36 M params"], "net")
box("emb", M, y + 22, MW, "Net embeddings   16 x 256   (+ pair vectors 16 x 16 x 64)", [], "net")

# ---- whole-reroute column ----
box("gtok", R, 404, RW, "Grid tokens   32 x 32 = 1024", ["from the 32 x 32 decoder: Linear 160 -> 256", "+ learned position"], "pool")
box("rpool", R, boxes["e3"][1], RW, "Region mean pooling", ["mean of 16 x 16 features -> 224"], "pool")
ey = boxes["emb"][1]
box("trunk", R, ey - 60, RW, "Trunk",
    ["[attention-pool nets 256 | max nets 256 |", " region pool 224 | reroute 6] = 742",
     "-> 256 -> 256 (GELU)"], "head")

# ---- heads ----
hy = max(boxes["emb"][1] + boxes["emb"][3], boxes["trunk"][1] + boxes["trunk"][3]) + 60
box("future", L, hy, LW, "Future-occupancy head",
    ["conv 3x3 96 -> 96, ReLU, conv 1x1 -> 6", "6 x 64 x 64 logits: cell used by a later net",
     "sigmoid -> price canvas for LEARNED_PRICE"], "head", "loss: 0.5 x BCE per region cell")
box("policy", M, hy, 200, "Decision head",
    ["per net 256->256->7", "illegal -> -inf", "flatten -> 112 logits"], "head", "loss: KL(pi' || p)")
box("excess", M + 210, hy, 200, "Net-excess head",
    ["per net 256->128->1", "log(final / ideal)", ""], "head", "loss: 0.25 x MSE")
box("value", R, hy, 160, "Value head",
    ["Linear 256 -> 1", "y = standardised", "log(D_rem / LB_rem)"], "head", "loss: MSE")
box("blocked", R + 170, hy, 160, "Blocked head",
    ["Linear 256 -> 1", "logit: reroute", "ends blocked"], "head", "loss: 0.5 x BCE")

# ---- arrows: grid stream ----
cx, cy, cw, ch = boxes["canvas"]
arrow([(L, cy + ch / 2), (L - 14, cy + ch / 2), (L - 14, boxes["planes"][1] + 20), (L, boxes["planes"][1] + 20)])
down("delays", "planes")
down("planes", "e1"); down("e1", "e2"); down("e2", "e3"); down("e3", "d2"); down("d2", "d1")
for a, b, lane in (("e1", "d1", L - 22), ("e2", "d2", L - 10)):
    ax_, ay_, aw_, ah_ = boxes[a]; bx_, by_, bw_, bh_ = boxes[b]
    arrow([(L, ay_ + ah_ / 2), (lane, ay_ + ah_ / 2), (lane, by_ + bh_ / 2), (L, by_ + bh_ / 2)], None)
bx, by, bw, bh = boxes["boxes"]
arrow([(bx, by + bh / 2), (445, by + bh / 2), (445, boxes["planes"][1] + 40), (L + LW, boxes["planes"][1] + 40)], "demand")

# ---- arrows: grid -> net stream / whole-reroute ----
d1x, d1y, d1w, d1h = boxes["d1"]
rx, ry, rw, rh = boxes["roi"]
arrow([(L + 30, d1y + d1h), (L + 30, hy)])
arrow([(L + LW, d1y + d1h / 2), (440, d1y + d1h / 2), (440, ry + rh / 2), (M, ry + rh / 2)])
d2x, d2y, d2w, d2h = boxes["d2"]
gx_, gy_, gw_, gh_ = boxes["gtok"]
gin = gy_ + 22
arrow([(L + LW, d2y + d2h / 2), (450, d2y + d2h / 2), (450, gin), (R, gin)])
e3x, e3y, e3w, e3h = boxes["e3"]
px, py, pw, ph = boxes["rpool"]
lane16 = (gy_ + gh_ + boxes["roi"][1]) / 2
arrow([(L + LW, e3y + e3h / 2), (458, e3y + e3h / 2), (458, lane16), (R + 40, lane16), (R + 40, py)])

# ---- arrows: net stream ----
down("roi", "init")
nx, ny, nw, nh = boxes["nfeat"]
ix, iy, iw, ih = boxes["init"]
arrow([(nx + nw, ny + 24), (M + MW + 18, ny + 24), (M + MW + 18, iy + 18), (M + MW, iy + 18)])
pqx, pqy, pqw, pqh = boxes["pairs"]
arrow([(pqx + pqw, pqy + pqh / 2), (M + MW + 32, pqy + pqh / 2), (M + MW + 32, iy + 36), (M + MW, iy + 36)])
gbx, gby, gbw, gbh = boxes["glob"]
arrow([(gbx + 30, gby + gbh), (gbx + 30, gby + gbh + 14), (R - 14, gby + gbh + 14), (R - 14, iy + 8), (M + MW, iy + 8)], "reroute")
down("init", "inter")
inx, iny, inw, inh = boxes["inter"]
arrow([(bx + bw, by + bh / 2), (M + MW + 46, by + bh / 2), (M + MW + 46, iny + 24), (M + MW, iny + 24)], "boxes")
arrow([(gx_ + gw_, gy_ + gh_ / 2), (R + RW + 14, gy_ + gh_ / 2), (R + RW + 14, iny + 60), (M + MW, iny + 60)], "grid tokens")
down("inter", "emb")
mx, my, mw, mh = boxes["emb"]
arrow([(mx + 100, my + mh), (mx + 100, hy)])
arrow([(mx + 310, my + mh), (mx + 310, hy)])
tx, ty, tw, th = boxes["trunk"]
arrow([(mx + mw, my + mh / 2), (R, my + mh / 2)])
arrow([(px + pw / 2, py + ph), (px + pw / 2, ty)])
arrow([(gbx + gbw, gby + gbh / 2), (R + RW + 30, gby + gbh / 2), (R + RW + 30, ty + 20), (R + RW, ty + 20)])
lx, ly, lw, lh = boxes["legal"]
arrow([(lx + lw, ly + lh / 2), (M + MW + 60, ly + lh / 2), (M + MW + 60, hy - 24), (M + 100, hy - 24), (M + 100, hy)], "mask")
arrow([(R + 80, ty + th), (R + 80, hy)])
arrow([(R + 250, ty + th), (R + 250, hy)])

# ---- footer ----
fy2 = hy + 150
parts.append(f'<rect x="{L}" y="{fy2}" width="{R + RW - L}" height="92" rx="8" fill="#fff4e6" stroke="#d9480f" stroke-width="1.6"/>')
foot = ["Total loss = KL(pi' || p) + (y_std - y)^2 + 0.5 BCE(blocked) + 0.5 BCE(future) + 0.25 (net excess)^2,  AdamW weight decay 1e-4",
        "value, future and excess terms only on reroutes that completed;  pi' = search's improved decision distribution",
        "In search:  q = (1 - P_blocked) * clip((old - (rerouted + LB_pending * exp(y))) / slack, -1, 1) - P_blocked",
        "Total: 10.33 M parameters (backbone 6.12 M, net interaction 3.36 M, grid tokens 0.30 M, inputs/heads/trunk 0.54 M);  11.9 GFLOPs per sample"]
for i, ln in enumerate(foot):
    parts.append(f'<text x="{L + 12}" y="{fy2 + 24 + i * 19}" class="{"lf" if i else "lfb"}">{escape(ln)}</text>')
H = fy2 + 92 + 24
W = R + RW + 50
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#495057"/></marker>
<style>.t{{font-size:14px;font-weight:700;fill:#212529}} .l{{font-size:12px;fill:#343a40}} .loss{{font-size:12px;font-weight:700;fill:#d9480f}}
.sec{{font-size:12px;font-weight:700;letter-spacing:1px;fill:#868e96}} .lab{{font-size:11px;fill:#495057;font-style:italic}}
.lf{{font-size:13px;fill:#343a40}} .lfb{{font-size:13px;font-weight:700;fill:#212529}} .h{{font-size:20px;font-weight:700;fill:#212529}}</style></defs>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{L}" y="30" class="h">azroute network (current defaults)</text>
{chr(10).join(parts)}
</svg>'''
open(OUT, "w").write(svg)
print("wrote", OUT, W, H)
