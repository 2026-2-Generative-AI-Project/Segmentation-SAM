r"""AdCanvas 누끼 모델 비교 (PyCharm용, 파일 하나)

[설치] PyCharm 아래쪽 Terminal 에서 한 번:
    pip install torch torchvision transformers timm kornia einops huggingface_hub "rembg[cpu]" pillow pillow-heif numpy scipy scikit-image pandas matplotlib psutil ultralytics opencv-python
    huggingface-cli login        # RMBG-2.0 사용 시 (허깅페이스에서 모델 승인 먼저)

[사진] C:\Users\jungj\pythonAdcanvas\images 폴더에 넣기 (.jpg .png .webp .heic .heif 등, 아이폰 사진 그대로 OK)
       처음 실행하면 images, outputs, gt 폴더가 자동으로 생깁니다.

[실행] PyCharm에서 이 파일을 열고 Run(▶) → 메뉴에서 번호 선택
       1 모델 실행  2 정답 마스크 만들기  3 점수 계산
   또는 터미널:  python adcanvas_nukki.py run --pick IMG_1190 IMG_9902
                 python adcanvas_nukki.py label [사진이름 ...]
                 python adcanvas_nukki.py eval
"""
import argparse
import gc
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()  # 아이폰 사진(.heic/.heif)을 PIL로 열 수 있게 등록

# =====================================================================
# 공통 설정
# =====================================================================
BASE = Path(os.environ.get('NUKKI_BASE', r'C:\Users\jungj\pythonAdcanvas'))   # 작업 폴더 (바꾸려면 여기)
IN_DIR = BASE / 'images'    # 입력 사진
OUT_DIR = BASE / 'outputs'  # 모델 결과
GT_DIR = BASE / 'gt'        # 정답 마스크 (사진과 같은 이름의 .png)
for _d in (IN_DIR, OUT_DIR, GT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 아이폰: .heic/.heif(기본 '고효율성'), .jpg('높은 호환성' 설정), .png(스크린샷)
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif', '.bmp', '.tif', '.tiff')


def list_images(folder=IN_DIR):
    """폴더 안 사진 목록 (확장자 대소문자 무시, 맥의 '._' 숨김 파일 제외)."""
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith('.'))


def load_image(path, max_side=None):
    """EXIF 회전 반영 후 RGB로. max_side를 주면 비율 유지하며 축소."""
    img = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    if max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def get_device():
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'  # 애플 실리콘 맥
    return 'cpu'


def sync(device):
    import torch
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def empty_cache(device):
    import torch
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()


def show():
    plt.show(block=False)
    plt.pause(0.3)


# =====================================================================
# 1. 모델 실행 (rembg, RMBG-2.0, BiRefNet)
# =====================================================================
MAX_SIDE = 2048


def ram():
    import psutil
    m = psutil.virtual_memory()
    return f'RAM {m.used / 1e9:.1f}/{m.total / 1e9:.1f}GB'


def make_loaders(device):
    import torch
    from torchvision import transforms

    seg_tf = transforms.Compose([   # BiRefNet 계열 공통 전처리
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    @torch.no_grad()
    def run_seg(model, img):
        dtype = next(model.parameters()).dtype
        x = seg_tf(img).unsqueeze(0).to(device, dtype)
        pred = model(x)[-1].sigmoid()[0, 0].float().cpu()   # 0~1 확률 마스크
        return transforms.functional.to_pil_image(pred).resize(img.size, Image.BILINEAR)

    # 각 함수는 (사진 -> 마스크 함수, 지울 모델 객체, 표시 이름)을 돌려줌
    def load_rembg():
        from rembg import remove, new_session
        sess = new_session('u2net')
        return (lambda img: remove(img, session=sess, only_mask=True)), sess, 'rembg'

    def load_family(repo, label):
        from transformers import AutoModelForImageSegmentation
        m = AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True)
        m = m.to(device).eval()
        return (lambda img: run_seg(m, img)), m, label

    return [load_rembg,
            lambda: load_family('briaai/RMBG-2.0', 'RMBG-2.0'),
            lambda: load_family('ZhengPeng7/BiRefNet', 'BiRefNet')]


def on_gray(img, mask):
    return Image.composite(img, Image.new('RGB', img.size, (128, 128, 128)), mask)


def plot_grid(results, names, models, save_path, cell=4, dpi=100, title_rows=False):
    cols = ['original'] + models
    fig, axes = plt.subplots(len(names), len(cols),
                             figsize=(cell * len(cols), cell * len(names)), squeeze=False)
    for r, name in enumerate(names):
        res = results[name]
        for c, col in enumerate(cols):
            ax = axes[r][c]
            ax.imshow(res['original'] if col == 'original' else on_gray(res['original'], res[col]))
            ax.axis('off')
            if title_rows:
                if r == 0:
                    ax.set_title(col, fontsize=14)
            else:
                ax.set_title(f'{name} | {col}' if c == 0 else col, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print('저장:', save_path)


def cmd_run(pick=()):
    import torch
    device = get_device()
    print('device:', device, torch.cuda.get_device_name(0) if device == 'cuda' else '')
    if os.environ.get('HF_TOKEN'):
        from huggingface_hub import login
        login(token=os.environ['HF_TOKEN'])

    paths = list_images()
    if not paths:
        print('사진이 없습니다. 여기에 넣어 주세요 →', IN_DIR)
        return
    print(f'{len(paths)}장:', [p.name for p in paths])

    images = {p.stem: load_image(p, MAX_SIDE) for p in paths}
    results = {name: {'original': img} for name, img in images.items()}
    records, models = [], []

    # 모델을 하나씩: 불러오기 -> 모든 사진 처리 -> 메모리에서 지우기
    for i, loader in enumerate(make_loaders(device)):
        try:
            fn, holder, model_name = loader()
            fn(next(iter(images.values())))   # 워밍업 1회 (시간 측정 제외)
        except Exception as e:
            print(f'{i + 1}번째 모델 건너뜀:', type(e).__name__, str(e)[:200])
            continue
        models.append(model_name)
        print(f'[{model_name}] 불러옴', ram())

        d = OUT_DIR / model_name
        d.mkdir(parents=True, exist_ok=True)
        for name, img in images.items():
            sync(device); t0 = time.perf_counter()
            mask = fn(img).convert('L').resize(img.size, Image.BILINEAR)
            sync(device); dt = time.perf_counter() - t0

            mask.save(d / f'{name}_mask.png')
            cut = img.copy(); cut.putalpha(mask); cut.save(d / f'{name}_cut.png')
            results[name][model_name] = mask
            records.append({'image': name, 'model': model_name,
                            'sec': round(dt, 3), 'size': f'{img.width}x{img.height}'})
            print(f'  {name}: {dt:.2f}s')

        del fn, holder
        gc.collect()
        empty_cache(device)
        print(f'[{model_name}] 완료, 메모리 정리 후', ram())

    if not models:
        print('실행된 모델이 없습니다.')
        return

    df = pd.DataFrame(records)
    print(df.groupby('model')['sec'].agg(['mean', 'min', 'max']).round(3).to_string())
    df.to_csv(OUT_DIR / 'inference_time.csv', index=False)

    plot_grid(results, list(results), models, OUT_DIR / 'comparison.png')
    ok = [n for n in pick if n in results]
    for n in set(pick) - set(ok):
        print('pick에 없는 이름:', n)
    if ok:
        plot_grid(results, ok, models, OUT_DIR / 'slide_compare.png', cell=3, dpi=150, title_rows=True)
    plt.show()


# =====================================================================
# 2. 정답 마스크 만들기 (SAM 2.1 + 박스)
#    박스 창: 드래그 → ENTER/SPACE(박스 확정) → 다음 박스 … → ESC(끝), 확정 전 취소는 C
# =====================================================================
DISPLAY = 900   # 박스 그리는 창의 긴 변 길이(px). 화면에 맞게 조절

HELP = """명령:  s 저장하고 다음 | + 박스 더 그려서 추가 | - 박스로 영역 빼기
       h HULL  c CLOSE  f FILL  l LARGEST  (옵션 켜기/끄기)
       r 처음부터 | n 이 사진 건너뛰기 | q 종료"""


def draw_boxes(img):
    """OpenCV 창에서 여러 박스를 그리고 원본 좌표 [x1, y1, x2, y2] 목록을 돌려줌."""
    import cv2
    scale = min(1.0, DISPLAY / max(img.size))
    disp = img.resize((round(img.width * scale), round(img.height * scale)))
    disp = cv2.cvtColor(np.array(disp), cv2.COLOR_RGB2BGR)
    win = 'drag box -> ENTER/SPACE, repeat, ESC = done'
    rois = cv2.selectROIs(win, disp, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(win)
    cv2.waitKey(1)

    boxes = [[x / scale, y / scale, (x + w) / scale, (y + h) / scale] for x, y, w, h in rois]
    min_side = max(img.size) * 0.02   # 클릭 실수로 생긴 작은 박스 제외
    boxes = [b for b in boxes if (b[2] - b[0]) > min_side and (b[3] - b[1]) > min_side]
    W, H = img.size
    for i, b in enumerate(boxes):
        if (b[2] - b[0]) * (b[3] - b[1]) > 0.4 * W * H:
            print(f' {i}번 박스가 너무 큼 → 여러 그릇을 한 박스에 넣지 않았는지 확인')
    return boxes


def sam_box_masks(sam, sam_device, img, boxes, hull):
    """박스마다 SAM 실행 → 박스 밖 잘라내기 → (선택) 윤곽 채우기."""
    from skimage.morphology import convex_hull_image
    W, H = img.size
    out = []
    for bx in boxes:
        r = sam(img, bboxes=[bx], device=sam_device, verbose=False)[0]
        mb = r.masks.data.cpu().numpy().any(axis=0)
        mb = np.array(Image.fromarray(mb).resize((W, H), Image.NEAREST))
        x1, y1, x2, y2 = [int(round(v)) for v in bx]
        clip = np.zeros_like(mb)
        clip[max(y1, 0):min(y2, H), max(x1, 0):min(x2, W)] = True
        mb &= clip
        if hull and mb.any():
            mb = convex_hull_image(mb)
        out.append(mb)
    return out


def tint(arr, m):
    arr[m] = (arr[m] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    return arr


def preview_boxes(base, boxes, box_masks):
    n = len(boxes)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for i, (bx, mb) in enumerate(zip(boxes, box_masks)):
        x1, y1, x2, y2 = [int(round(v)) for v in bx]
        sl = (slice(max(y1, 0), y2), slice(max(x1, 0), x2))
        a = axs[i // cols][i % cols]
        a.imshow(tint(base[sl].copy(), mb[sl])); a.set_title(f'box {i}'); a.axis('off')
    for j in range(n, rows * cols):
        axs[j // cols][j % cols].axis('off')
    plt.tight_layout()
    show()


def refine(acc, opt, size):
    from scipy import ndimage
    m = acc.copy()
    if opt['close']:
        k = max(size) // 100
        m = ndimage.binary_closing(m, structure=np.ones((k, k)), iterations=2)
    if opt['fill']:
        m = ndimage.binary_fill_holes(m)
    if opt['largest']:
        lab, nlab = ndimage.label(m)
        if nlab > 1:
            sizes = ndimage.sum(m, lab, range(1, nlab + 1))
            m = lab == (np.argmax(sizes) + 1)
    return Image.fromarray((m * 255).astype(np.uint8))


def preview_final(img, base, mask, name):
    over = tint(base.copy(), np.array(mask) > 127)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, im, t in zip(ax, [img, mask, over], ['image', 'mask', 'overlay']):
        a.imshow(im, cmap='gray'); a.set_title(t); a.axis('off')
    fig.suptitle(name)
    show()


def label_one(sam, sam_device, path):
    """한 장 라벨링. 반환: 'saved' | 'skip' | 'quit'"""
    name, img = path.stem, load_image(path)   # GT는 원본 해상도로 저장
    base = np.array(img)
    W, H = img.size
    print(f'\n=== {path.name}  {img.size} ===')

    acc, mask, mode, need_draw = None, None, '+', True
    opt = {'hull': False, 'close': False, 'fill': True, 'largest': False}

    while True:
        if need_draw:
            print(f"박스 그리기 (모드 {mode}, HULL={'on' if opt['hull'] else 'off'})")
            boxes = draw_boxes(img)
            if boxes:
                print(f'사용할 박스 {len(boxes)}개, SAM 실행 중...')
                box_masks = sam_box_masks(sam, sam_device, img, boxes, opt['hull'])
                plt.close('all')
                preview_boxes(base, boxes, box_masks)
                s = input(f'실패한 박스 번호 (예: 2 5, 없으면 Enter): ').replace(',', ' ').split()
                drop = {int(t) for t in s if t.isdigit()}
                m_new = np.zeros((H, W), dtype=bool)
                for i, mb in enumerate(box_masks):
                    if i not in drop:
                        m_new |= mb
                acc = m_new if mode == '+' else acc & ~m_new
            else:
                print('박스가 없습니다.')
            need_draw = False

        if acc is not None:
            mask = refine(acc, opt, img.size)
            plt.close('all')
            preview_final(img, base, mask, name)
        print('옵션:', ' '.join(f'{k.upper()}={"on" if v else "off"}' for k, v in opt.items()))
        print(HELP)

        cmd = input('> ').strip().lower()
        if cmd == 's':
            if mask is None:
                print('저장할 마스크가 없습니다.'); continue
            mask.save(GT_DIR / f'{name}.png')
            print('저장:', GT_DIR / f'{name}.png')
            return 'saved'
        if cmd in ('+', '-'):
            if cmd == '-' and acc is None:
                print("빼기 전에 먼저 '+' 로 영역을 만들어 주세요"); continue
            mode, need_draw = cmd, True
        elif cmd in ('h', 'c', 'f', 'l'):
            key = {'h': 'hull', 'c': 'close', 'f': 'fill', 'l': 'largest'}[cmd]
            opt[key] = not opt[key]
            if key == 'hull':
                print('HULL은 다음에 그리는 박스부터 적용됩니다.')
        elif cmd == 'r':
            acc, mask, mode, need_draw = None, None, '+', True
        elif cmd == 'n':
            return 'skip'
        elif cmd == 'q':
            return 'quit'
        else:
            print('알 수 없는 명령:', cmd)


def cmd_label(names=()):
    import torch
    from ultralytics import SAM
    paths = list_images()
    if names:   # 이름을 주면 그 사진만 (이미 있으면 덮어씀)
        todo = [p for p in paths if p.stem in set(names)]
    else:
        todo = [p for p in paths if not (GT_DIR / f'{p.stem}.png').exists()]
    print(f'전체 {len(paths)}장 중 남은 것 {len(todo)}장  (입력 폴더: {IN_DIR})')
    if not todo:
        return

    sam = SAM('sam2.1_l.pt')   # 처음 실행 시 자동 다운로드
    # NVIDIA GPU가 있으면 사용, 맥(MPS)은 SAM 일부 연산이 불안정해 CPU 사용
    sam_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for p in todo:
        if label_one(sam, sam_device, p) == 'quit':
            break
    plt.close('all')
    print('끝')


# =====================================================================
# 3. 점수 계산 (IoU, MAE)
# =====================================================================
def iou_mae(pred_mask, gt_mask, th=0.5):
    p = np.asarray(pred_mask.convert('L').resize(gt_mask.size), dtype=np.float32) / 255
    g = (np.asarray(gt_mask.convert('L'), dtype=np.float32) / 255) > 0.5
    pb = p > th
    union = (pb | g).sum()
    iou = (pb & g).sum() / union if union else 1.0
    return float(iou), float(np.abs(p - g).mean())


def cmd_eval():
    models = sorted(d.name for d in OUT_DIR.iterdir() if d.is_dir())
    rows = []
    for gt_path in sorted(GT_DIR.glob('*.png')):
        gt = Image.open(gt_path)
        for model in models:
            pred_path = OUT_DIR / model / f'{gt_path.stem}_mask.png'
            if pred_path.exists():
                iou, mae = iou_mae(Image.open(pred_path), gt)
                rows.append({'image': gt_path.stem, 'model': model, 'IoU': iou, 'MAE': mae})

    if not rows:
        print('비교할 것이 없습니다. 먼저 1(모델 실행)과 2(정답 마스크)를 해 주세요.')
        return
    mdf = pd.DataFrame(rows)
    print(mdf.groupby('model')[['IoU', 'MAE']].mean().round(4).to_string())
    print('\n사진별 IoU:')
    print(mdf.pivot(index='image', columns='model', values='IoU').round(3).to_string())
    mdf.to_csv(OUT_DIR / 'metrics.csv', index=False)
    print('\n저장:', OUT_DIR / 'metrics.csv')


# =====================================================================
# 시작
# =====================================================================
def main():
    print('데이터 폴더:', BASE)
    if len(sys.argv) == 1:   # PyCharm Run 버튼으로 실행 → 메뉴
        choice = input('1 모델 실행 | 2 정답 마스크 만들기 | 3 점수 계산  > ').strip()
        if choice == '1':
            pick = input('슬라이드용 사진 이름 (띄어쓰기로 구분, 없으면 Enter): ').split()
            cmd_run(pick)
        elif choice == '2':
            cmd_label()
        elif choice == '3':
            cmd_eval()
        return

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('run').add_argument('--pick', nargs='*', default=[])
    sub.add_parser('label').add_argument('names', nargs='*')
    sub.add_parser('eval')
    a = ap.parse_args()
    if a.cmd == 'run':
        cmd_run(a.pick)
    elif a.cmd == 'label':
        cmd_label(a.names)
    else:
        cmd_eval()


if __name__ == '__main__':
    main()