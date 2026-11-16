"""누끼(그릇 분할) 모듈: DINO(그릇 찾기) → 파인튜닝 SAM 2.1(그릇마다 마스크) → 가벼운 후처리

FastAPI 서버(app/main.py)에서 쓰는 방법
    from pipeline import segment
    segment.load()                                   # 서버 켤 때 한 번
    mask = segment.segment(img)                      # 사진 → 흑백 마스크 (L, 사진과 같은 크기)
    cands = segment.candidates(img)                  # 사진 → 그릇 후보 목록 (사용자 선택용)
    mask = segment.merge(img, cands, [0, 2])         # 사용자가 고른 그릇만 합친 마스크
    mask = segment.refine(img, box, points, labels)  # 클릭 보정 (labels: 1 포함, 0 제외)

필요한 것
    pip install git+https://github.com/facebookresearch/sam2.git transformers torchvision opencv-python scipy pillow pillow-heif
    파인튜닝 가중치: 환경 변수 SAM_FT_WEIGHTS (기본 weights/sam2_ft.pt). 없으면 원래 SAM을 씀
    (가중치 파일은 GitHub에 올리지 않고 드라이브에서 받아서 씀)
"""
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy import ndimage

SAM_MODEL = 'facebook/sam2.1-hiera-small'
SAM_FT_WEIGHTS = os.environ.get('SAM_FT_WEIGHTS', 'weights/sam2_ft.pt')
GD_ID = 'IDEA-Research/grounding-dino-tiny'
MAX_SIDE = 2048            # 이보다 큰 사진은 줄여서 처리하고, 마스크는 원래 크기로 되돌림

# 1_pipeline.ipynb와 같은 값 (11/16 test IoU 0.906, BF 0.827)
DINO_CFG = dict(
    dish_words=['plate', 'bowl', 'dish', 'pot'],            # 그릇 (기본 선택)
    other_words=['cup', 'glass', 'spoon', 'chopsticks'],    # 후보로만 (기본 선택 해제)
    food_words=['food'],                                    # 그릇 없이 놓인 음식
    box_th=0.3, nms_iou=0.6, max_area=0.9, food_max_area=0.5,
)
CAND = dict(
    utensil_words=['spoon', 'chopsticks'],
    utensil_overlap=0.2,   # 수저의 20% 이상이 그릇 외곽 안에 있으면 걸친 것 (원래 SAM일 때만 합침)
    part_min=0.1,          # 마스크 조각 중 가장 큰 조각의 10% 이상만 남김
    solidity=0.9,          # 원래 SAM일 때: 그릇이 볼록 외곽의 90% 미만이면 외곽으로 채움
    inside_drop=0.9,       # 다른 그릇 안에 90% 이상 들어간 후보는 중복으로 제거
    sam_chunk=4,           # SAM을 박스 몇 개씩 돌릴지 (메모리 절약)
)

_m = {}   # 불러온 모델: device, gd_proc, gd, sam, finetuned


# ---------------- 불러오기 ----------------
def _device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load(weights=SAM_FT_WEIGHTS):
    """모델 불러오기 (서버 켤 때 한 번). 반환: 파인튜닝 가중치를 썼는지"""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()   # 아이폰 HEIC
    except ImportError:
        pass
    dev = _device()
    _m['device'] = dev
    _m['gd_proc'] = AutoProcessor.from_pretrained(GD_ID)
    _m['gd'] = AutoModelForZeroShotObjectDetection.from_pretrained(GD_ID).to(dev).eval()
    sam = SAM2ImagePredictor.from_pretrained(SAM_MODEL, device=dev)
    _m['finetuned'] = bool(weights) and os.path.exists(weights)
    if _m['finetuned']:
        res = sam.model.load_state_dict(torch.load(weights, map_location=dev), strict=False)
        assert not res.missing_keys and not res.unexpected_keys, 'sam2_ft.pt와 SAM_MODEL이 맞지 않음'
    _m['sam'] = sam
    _m['sam_img'] = None   # set_image 한 사진 (클릭 보정 때 다시 계산하지 않으려고)
    return _m['finetuned']


def _ready():
    if 'sam' not in _m:
        load()


def _prep(img):
    """PIL 사진 → (처리용 RGB 사진, 원래 크기). EXIF 회전 반영, 긴 변 MAX_SIDE로"""
    img = ImageOps.exif_transpose(img).convert('RGB')
    size = img.size
    if max(size) > MAX_SIDE:
        img = img.copy()
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img, size


def _to_size(m, size):
    """bool 마스크 → 원래 크기의 흑백 PIL (L, 0/255)"""
    out = Image.fromarray((m * 255).astype(np.uint8))
    return out if out.size == size else out.resize(size, Image.NEAREST)


def _set_image(img_np):
    key = (img_np.shape, int(img_np[::97, ::89].sum()))
    if _m['sam_img'] != key:
        _m['sam'].set_image(img_np)
        _m['sam_img'] = key


# ---------------- DINO ----------------
def _box_area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


@torch.no_grad()
def dino_boxes(img):
    """사진 → [{'box', 'label', 'score', 'kind'(dish/other/food), 'dish'(기본 선택)}]"""
    from torchvision.ops import nms
    C, proc, gd = DINO_CFG, _m['gd_proc'], _m['gd']
    words = C['dish_words'] + C['other_words'] + C['food_words']
    kinds = ['dish'] * len(C['dish_words']) + ['other'] * len(C['other_words']) + ['food'] * len(C['food_words'])
    inputs = proc(images=img, text='. '.join(words) + '.', return_tensors='pt').to(_m['device'])
    out = gd(**inputs)
    probs = out.logits[0].sigmoid().float().cpu()
    ids = inputs['input_ids'][0].tolist()
    cols = []
    for w in words:
        sub = proc.tokenizer(w, add_special_tokens=False)['input_ids']
        pos = [i + j for i in range(len(ids) - len(sub) + 1) if ids[i:i + len(sub)] == sub for j in range(len(sub))]
        cols.append(probs[:, pos].max(1).values if pos else torch.zeros(len(probs)))
    score, best = torch.stack(cols, 1).max(1)
    W, H = img.size
    cx, cy, bw, bh = out.pred_boxes[0].float().cpu().unbind(-1)
    boxes = torch.stack([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H], -1)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H)
    keep = [i for i in range(len(boxes)) if score[i] > C['box_th'] and _box_area(boxes[i].tolist()) <
            (C['food_max_area'] if kinds[int(best[i])] == 'food' else C['max_area']) * W * H]
    if not keep:
        return []
    keep = torch.tensor(keep)
    k = keep[nms(boxes[keep], score[keep], C['nms_iou'])].tolist()
    return [{'box': boxes[i].tolist(), 'label': words[int(best[i])], 'score': float(score[i]),
             'kind': kinds[int(best[i])], 'dish': kinds[int(best[i])] != 'other'} for i in k]


# ---------------- 마스크 도우미 ----------------
def _box_mask(shape, box, expand=0.1):
    x0, y0, x1, y1 = box
    ex, ey = expand * (x1 - x0), expand * (y1 - y0)
    out = np.zeros(shape, bool)
    out[max(int(y0 - ey), 0):max(int(np.ceil(y1 + ey)), 0), max(int(x0 - ex), 0):max(int(np.ceil(x1 + ex)), 0)] = True
    return out


def _local(fn, m, pad=0):
    out = np.zeros_like(m)
    ys, xs = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    if not len(ys):
        return out
    y0, y1 = max(ys[0] - pad, 0), min(ys[-1] + 1 + pad, m.shape[0])
    x0, x1 = max(xs[0] - pad, 0), min(xs[-1] + 1 + pad, m.shape[1])
    out[y0:y1, x0:x1] = fn(m[y0:y1, x0:x1])
    return out


def _hull(m):
    def f(c):
        out = np.zeros(c.shape, np.uint8)
        cv2.fillConvexPoly(out, cv2.convexHull(cv2.findNonZero(c.astype(np.uint8))), 1)
        return out.astype(bool)
    return _local(f, m)


def _close_fill(m):
    return _local(lambda c: ndimage.binary_fill_holes(ndimage.binary_closing(c, iterations=3)), m, pad=4)


def _keep_parts(c, box):
    lab, n = ndimage.label(c)
    if n <= 1:
        return c
    idx = np.arange(1, n + 1)
    sizes = ndimage.sum(c, lab, idx)
    inside = ndimage.sum(_box_mask(c.shape, box), lab, idx)
    keep = idx[(sizes >= CAND['part_min'] * sizes.max()) & (inside >= 0.5 * sizes)]
    return np.isin(lab, keep if len(keep) else idx[[sizes.argmax()]])


def _crop(m, box, fn, pad=4):
    ys, xs = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    if not len(ys):
        return m
    y0, y1 = max(ys[0] - pad, 0), min(ys[-1] + 1 + pad, m.shape[0])
    x0, x1 = max(xs[0] - pad, 0), min(xs[-1] + 1 + pad, m.shape[1])
    out = np.zeros_like(m)
    out[y0:y1, x0:x1] = fn(m[y0:y1, x0:x1], [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0])
    return out


def _pick_whole(ms, ss, box, mm_score=0.6, mm_inside=0.9):
    ok = [j for j in range(len(ms)) if ms[j].any() and ss[j] >= mm_score * ss.max()]
    bm = _box_mask(ms[0].shape, box)
    ok = [j for j in ok if (ms[j] & bm).sum() >= mm_inside * ms[j].sum()]
    return ms[max(ok, key=lambda j: ms[j].sum())] if ok else ms[int(ss.argmax())]


def _clean_dish(m, box):
    out = _crop(m, box, lambda c, b: ndimage.binary_fill_holes(ndimage.binary_closing(_keep_parts(c, b), iterations=3)))
    if not out.any():
        return out
    h = _hull(out) & (_box_mask(m.shape, box, 0.05) | out)
    return h if out.sum() < CAND['solidity'] * h.sum() else out


# ---------------- SAM + 후처리 ----------------
@torch.no_grad()
def _sam_masks(img_np, boxes, kinds):
    """파인튜닝 SAM은 마스크 1개, 원래 SAM은 3개 중 전체 물체를 고름"""
    _set_image(img_np)
    H, W = img_np.shape[:2]
    multi = not _m['finetuned']
    out = []
    for s0 in range(0, len(boxes), CAND['sam_chunk']):
        b = np.array(boxes[s0:s0 + CAND['sam_chunk']], np.float32)
        m, sc, _ = _m['sam'].predict(box=b, multimask_output=multi)
        m = np.asarray(m).reshape(len(b), -1, H, W) > 0.5
        if not multi:
            out += list(m[:, 0])
            continue
        sc = np.asarray(sc).reshape(len(b), -1)
        for k in range(len(b)):
            out.append(m[k, int(sc[k].argmax())] if kinds[s0 + k] == 'other' else _pick_whole(m[k], sc[k], boxes[s0 + k]))
    return out


def _tidy(cands, light):
    for c in cands:
        if c['dish'] and c['kind'] != 'food':
            c['mask'] = _close_fill(_crop(c['mask'], c['box'], _keep_parts, pad=0)) if light else _clean_dish(c['mask'], c['box'])
    dishes = sorted([c for c in cands if c['dish'] and c['kind'] != 'food'], key=lambda c: -c['mask'].sum())
    for i, small in enumerate(dishes):
        for big in dishes[:i]:
            if 'dup_of' not in big and (small['mask'] & big['mask']).sum() >= CAND['inside_drop'] * max(small['mask'].sum(), 1):
                small['dish'], small['dup_of'] = False, big['id']
                break
    kept = [d for d in dishes if 'dup_of' not in d]
    union = np.zeros(cands[0]['mask'].shape, bool)
    for d in kept:
        union |= d['mask']
    for f in [c for c in cands if c['kind'] == 'food']:
        f['mask'] = _close_fill(_crop(f['mask'], f['box'], _keep_parts, pad=0))
        a = f['mask'].sum()
        swallows = any((f['mask'] & d['mask']).sum() > 0.3 * d['mask'].sum() for d in kept)
        if a == 0 or (f['mask'] & union).sum() >= 0.5 * a or swallows:
            f['dish'], f['dup_of'] = False, -1
    utensils = [u for u in cands if u['label'] in CAND['utensil_words'] and u['mask'].any()]
    if utensils and not light:
        hulls = {c['id']: _hull(c['mask']) for c in cands if c['dish'] and 'dup_of' not in c and c['mask'].any()}
        for u in utensils:
            if not hulls:
                break
            best = max(hulls, key=lambda i: (u['mask'] & hulls[i]).sum())
            if (u['mask'] & hulls[best]).sum() > CAND['utensil_overlap'] * u['mask'].sum():
                d = next(c for c in cands if c['id'] == best)
                d['mask'] = _close_fill(d['mask'] | u['mask'])
                u['merged_into'] = best
    return [c for c in cands if 'dup_of' not in c]


# ---------------- 바깥에서 쓰는 함수 ----------------
def candidates(img):
    """사진 → 그릇 후보 목록. 박스는 원래 사진 좌표, mask는 원래 크기의 흑백 PIL
    [{'id', 'label', 'score', 'kind'(dish/other/food), 'dish'(기본 선택), 'box'[x0,y0,x1,y1], 'mask'}]
    그릇을 하나도 못 찾으면 [] (서비스에서는 사용자가 직접 박스를 그리게 → refine)"""
    _ready()
    work, size = _prep(img)
    found = dino_boxes(work)
    if not any(f['dish'] for f in found):
        return []
    masks = _sam_masks(np.array(work), [f['box'] for f in found], [f['kind'] for f in found])
    for i, (f, m) in enumerate(zip(found, masks)):
        f.update(id=i, mask=m)
    cands = _tidy(found, light=_m['finetuned'])
    s = size[0] / work.size[0]
    return [{'id': c['id'], 'label': c['label'], 'score': round(c['score'], 3), 'kind': c['kind'], 'dish': c['dish'],
             'box': [round(v * s, 1) for v in c['box']], 'mask': _to_size(c['mask'], size)} for c in cands]


def merge(img, cands, ids=None):
    """후보 중 고른 것만 합친 마스크 (ids=None이면 기본 선택된 그릇 전부)"""
    W, H = ImageOps.exif_transpose(img).size
    out = np.zeros((H, W), bool)
    for c in cands:
        if (c['id'] in ids) if ids is not None else c['dish']:
            out |= np.asarray(c['mask']) > 127
    return _to_size(out, (W, H))


def segment(img):
    """팀 공통 함수 모양 (KAN-5): RGB 사진 → 흑백 마스크 (L, 사진과 같은 크기, 255 = 음식+그릇)"""
    return merge(img, candidates(img))


@torch.no_grad()
def refine(img, box, points=None, labels=None):
    """클릭 보정: 박스(원래 사진 좌표) + 클릭 점(1 포함, 0 제외) → 그 그릇 하나의 마스크 (L, 원래 크기)"""
    _ready()
    work, size = _prep(img)
    s = work.size[0] / size[0]
    _set_image(np.array(work))
    kw = dict(box=np.array([v * s for v in box], np.float32), multimask_output=False)
    if points:
        kw.update(point_coords=np.array(points, np.float32) * s, point_labels=np.array(labels, np.int32))
    m, _, _ = _m['sam'].predict(**kw)
    return _to_size(np.asarray(m)[0] > 0.5, size)


if __name__ == '__main__':   # python segment.py 사진.jpg → 사진_mask.png
    import sys
    load()
    for p in sys.argv[1:]:
        im = Image.open(p)
        cs = candidates(im)
        merge(im, cs).save(os.path.splitext(p)[0] + '_mask.png')
        print(p, '그릇 후보', sum(c['dish'] for c in cs), '개 | 파인튜닝:', _m['finetuned'])
