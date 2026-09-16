"""
P&ID Digitizer — antarmuka Streamlit untuk model YOLO hasil training tile CVAT.

Menjalankan pipeline yang sama dengan notebook:
    halaman penuh -> potong jadi tile -> YOLO per tile -> gabung + NMS
                  -> OCR tag utuh per simbol -> tabel + ekspor

Jalankan:
    streamlit run app/pid_app.py
"""
from __future__ import annotations

import io
import json
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

# ----------------------------------------------------------------- KONSTANTA
# Harus SAMA dengan yang dipakai saat membuat tile & training.
TILE_SIZE = 512
STRIDE = 472          # overlap 40 px

# Parameter penyiapan halaman — direplikasi dari crop-code milik intern:
#   convertir.py     : PDF -> PNG pada 200 DPI
#   retirer_cadre.py : buang kop & bingkai gambar -> crop (74, 75, 2761, 2261)
# Model dilatih pada hasil crop ini (2687x2186), BUKAN pada raster mentah
# (3308x2339). Melewatkan tahap crop = memberi model gambar yang tidak pernah
# ia lihat saat training, plus satu kolom tile berisi kop gambar.
RASTER_DPI = 200
CROP_BOX = (74, 75, 2761, 2261)          # left, top, right, bottom
RASTER_SIZE = (3308, 2339)               # ukuran raster 200 DPI yang diharapkan
CROPPED_SIZE = (CROP_BOX[2] - CROP_BOX[0], CROP_BOX[3] - CROP_BOX[1])   # 2687x2186

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

st.set_page_config(page_title="P&ID Digitizer", page_icon="⚙", layout="wide")


# ============================================================ GEOMETRI TILING
def tile_origin(row: int, col: int, page_w: int, page_h: int,
                tile: int = TILE_SIZE, stride: int = STRIDE) -> tuple[int, int]:
    """Kiri-atas tile (1-indexed), di-clamp pada tepi kanan/bawah halaman."""
    return (max(min((col - 1) * stride, page_w - tile), 0),
            max(min((row - 1) * stride, page_h - tile), 0))


def grid_shape(page_w: int, page_h: int,
               tile: int = TILE_SIZE, stride: int = STRIDE) -> tuple[int, int]:
    ncol = 1 if page_w <= tile else math.ceil((page_w - tile) / stride) + 1
    nrow = 1 if page_h <= tile else math.ceil((page_h - tile) / stride) + 1
    return nrow, ncol


def _iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["xmin"], b["xmin"]), max(a["ymin"], b["ymin"])
    ix2, iy2 = min(a["xmax"], b["xmax"]), min(a["ymax"], b["ymax"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a["xmax"] - a["xmin"]) * (a["ymax"] - a["ymin"])
    ab = (b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"])
    return inter / (aa + ab - inter)


def _containment(a: dict, b: dict) -> float:
    """Porsi kotak TERKECIL yang tertelan kotak lain (0..1).

    IoU buta terhadap kotak bersarang: kotak kecil yang 100% berada di dalam
    kotak besar hanya ber-IoU 0.25 bila sisinya separuh — jadi lolos ambang 0.5
    dan tergambar bertumpuk. Ukuran ini menangkap kasus tersebut.
    """
    ix1, iy1 = max(a["xmin"], b["xmin"]), max(a["ymin"], b["ymin"])
    ix2, iy2 = min(a["xmax"], b["xmax"]), min(a["ymax"], b["ymax"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a["xmax"] - a["xmin"]) * (a["ymax"] - a["ymin"])
    ab = (b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"])
    smaller = min(aa, ab)
    return inter / smaller if smaller > 0 else 0.0


def merge_nms(dets: list[dict], iou_thr: float = 0.5,
              class_agnostic: bool = True,
              contain_thr: float = 0.80) -> list[dict]:
    """NMS global: confidence tertinggi menang.

    class_agnostic=False : duplikat hanya dibuang bila kelasnya sama. Satu simbol
        yang ditebak sebagai dua kelas berbeda (mis. TEMPERATURE ELEMENT vs
        PRESSURE TRANSMITTER — bentuknya sama-sama lingkaran tiga baris) akan
        lolos dua-duanya dan tergambar sebagai dua kotak bertumpuk.
    class_agnostic=True  : duplikat dibuang berdasarkan posisi saja. Satu lokasi
        menghasilkan satu simbol; kelas dgn confidence tertinggi yang menang.

    contain_thr : kotak dianggap duplikat juga bila >=80% luas kotak terkecil
        tertelan kotak lain, walau IoU-nya rendah. Tanpa ini, deteksi bubble
        kecil di dalam kotak simbol besar akan lolos dan tergambar bertumpuk
        (kotak 50% sisi yang tertelan penuh hanya ber-IoU 0.25).

    Kelas yang kalah tidak dibuang begitu saja — ia dicatat di field `alt_class`
    milik pemenang, supaya kasus ragu tetap bisa diaudit dari tabel hasil.
    """
    kept: list[dict] = []
    for d in sorted(dets, key=lambda x: x["confidence"], reverse=True):
        hit = None
        for k in kept:
            if not (class_agnostic or k["class_id"] == d["class_id"]):
                continue
            if _iou(k, d) >= iou_thr or _containment(k, d) >= contain_thr:
                hit = k
                break
        if hit is not None:
            # Simpan kandidat kelas lain yang kalah (hanya bila kelasnya beda).
            if hit["class_id"] != d["class_id"]:
                hit.setdefault("alt", []).append((d["class_name"], d["confidence"]))
            continue
        kept.append(d)
    return kept


# ================================================================ TAG PARSING
ISA_FIRST = set("FLPTAWSHJKMNQRUVXYZBCDGIEO")
ISA_SUCCEED = set("ITVGECSAYQZKXRDFHLMNPUWBOJ")

_TO_ALPHA = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "C": "0", "U": "0",
             "I": "1", "L": "1", "J": "1", "S": "5", "B": "8",
             "Z": "2", "G": "6", "T": "7", "A": "4", "E": "8"}

TAG_STOPWORDS = {"NOTE", "NOTES", "SIZE", "TYP", "MIN", "MAX", "DP", "FC", "FO",
                 "TSO", "DBB", "CONT", "TYPE", "REF", "SEE", "DWG"}

CLASS_TO_ISA = {
    "PRESSUREGAUGE": "PG", "PRESSURETRANSMITTER": "PT",
    "PRESSURECONTROLVALVE": "PCV", "PRESSURESAFETYVALVE": "PSV",
    "PRESSUREDIFFERENTIALTRANSMITTER": "PDT", "PRESSUREDIFFERENTIALVALVE": "PDV",
    "TEMPERATUREELEMENT": "TE", "TEMPERATURETRANSMITTER": "TT",
    "TEMPERATUREGAUGE": "TG", "TEMPERATURECONTROLVALVE": "TCV",
    "THERMALSAFETYVALVE": "TSV", "TERMOWELL": "TW",
    "FLOWELEMENT": "FE", "FLOWTRANSMITTER": "FT", "FLOWCONTROLVALVE": "FV",
    "FLOWSIGHTGLASS": "SG", "VARIABLEAREAFLOWMETER": "FI",
    "LEVELGAUGE": "LG", "LEVELTRANSMITTER": "LT", "LEVELCONTROLVALVE": "LCV",
    "ONOFFVALVE": "XV", "RESTRICTIONORIFICE": "RO", "SAMPLINGCONNECTION": "SC",
}


def isa_code_for_class(class_name: str) -> str:
    if not class_name:
        return ""
    return CLASS_TO_ISA.get(re.sub(r"[^A-Z]", "", str(class_name).upper()), "")


def _norm_digits(s): return "".join(_TO_DIGIT.get(c, c) for c in s)
def _norm_alpha(s): return "".join(_TO_ALPHA.get(c, c) for c in s)


def _is_stopword_line(s: str) -> bool:
    a = re.sub(r"[^A-Z]", "", s)
    if a in TAG_STOPWORDS:
        return True
    for w in TAG_STOPWORDS:
        if a.startswith(w) and len(a) <= len(w) + 2 and any(c.isdigit() for c in s):
            return True
    return False


def _coerce_number(s: str) -> str:
    if not s:
        return s
    m = re.match(r"^(.*?)([A-Z]{1,3})$", s)
    head, tail = (m.group(1), m.group(2)) if m else (s, "")
    head_n = _norm_digits(head)
    if tail and tail in {"A", "B", "C", "D"} and len(head_n) >= 3 and head_n.isdigit():
        return head_n + tail
    return _norm_digits(s)


def _looks_like_instrument_code(s: str) -> bool:
    """Kode instrumen ISA (2-4 huruf). String murni angka TIDAK PERNAH lolos."""
    if not (2 <= len(s) <= 4) or s.isdigit():
        return False
    n_alpha = sum(c.isalpha() for c in s)
    if n_alpha < len(s) - 1 or n_alpha == 0:
        return False
    a = _norm_alpha(s)
    if not a.isalpha():
        return False
    return a[0] in ISA_FIRST and all(c in ISA_SUCCEED for c in a[1:])


def parse_tag_components(tag: str, sep: str = "-") -> dict:
    if not tag:
        return {"tag_prefix": "", "tag_code": "", "tag_number_only": "", "tag_suffix": ""}
    parts = [p for p in tag.split(sep) if p]
    prefix = code = number = suffix = ""
    for p in parts:
        if not code and _looks_like_instrument_code(p):
            code = p
        elif p.isdigit():
            if not prefix and not code:
                prefix = p
            elif not number:
                number = p
            else:
                suffix = (suffix + p) if suffix else p
        else:
            suffix = (suffix + p) if suffix else p
    if not code:
        digits = [p for p in parts if p.isdigit()]
        if len(digits) >= 2:
            prefix, number = digits[0], digits[1]
        elif len(digits) == 1:
            number = digits[0]
    return {"tag_prefix": prefix, "tag_code": code,
            "tag_number_only": number, "tag_suffix": suffix}


def ocr_symbol_tag(reader, page_img, box, upscale=4, pad=6, min_conf=0.30,
                   expected_code=None, sep="-") -> str:
    """Baca SELURUH isi bbox simbol, rakit jadi tag utuh (mis. '069-TT-0025').

    Crop diperbesar `upscale` kali karena baris tengah bubble (kode instrumen)
    berukuran kecil dan menempel garis pembagi -- pada resolusi asli sering
    tidak terbaca sama sekali.
    """
    W, H = page_img.size
    x1 = max(int(box["xmin"]) - pad, 0)
    y1 = max(int(box["ymin"]) - pad, 0)
    x2 = min(int(box["xmax"]) + pad, W)
    y2 = min(int(box["ymax"]) + pad, H)
    if x2 <= x1 or y2 <= y1:
        return ""

    crop = page_img.crop((x1, y1, x2, y2))
    if upscale > 1:
        crop = crop.resize((crop.width * upscale, crop.height * upscale), Image.LANCZOS)

    allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    try:
        res = reader.readtext(np.array(crop), detail=1, paragraph=False, allowlist=allow)
    except Exception:
        res = reader.readtext(np.array(crop), detail=1, paragraph=False)

    frags = []
    for bbox, text, conf in res:
        t = re.sub(r"\s+", "", text).strip().upper()
        if not t or conf < min_conf:
            continue
        ys = [q[1] for q in bbox]
        xs = [q[0] for q in bbox]
        cy, cx = (min(ys) + max(ys)) / 2, (min(xs) + max(xs)) / 2
        if not (pad * upscale <= cx <= (x2 - x1 - pad) * upscale and
                pad * upscale <= cy <= (y2 - y1 - pad) * upscale):
            continue
        frags.append({"text": t, "conf": float(conf), "cy": cy, "cx": cx,
                      "h": max(ys) - min(ys)})
    if not frags:
        return ""

    frags.sort(key=lambda f: f["cy"])
    med_h = sorted(f["h"] for f in frags)[len(frags) // 2] or 1
    lines, cur = [], [frags[0]]
    for f in frags[1:]:
        if abs(f["cy"] - cur[-1]["cy"]) <= med_h * 0.6:
            cur.append(f)
        else:
            lines.append(cur); cur = [f]
    lines.append(cur)

    parts = []
    for ln in lines:
        ln.sort(key=lambda f: f["cx"])
        s = "".join(f["text"] for f in ln)
        if s:
            parts.append({"text": s, "conf": min(f["conf"] for f in ln)})
    parts = [p for p in parts if not _is_stopword_line(p["text"])]
    if not parts:
        return ""

    def _is_clean_number(t): return bool(re.fullmatch(r"\d{2,6}[A-D]?", t))
    def _is_clean_code(t): return bool(re.fullmatch(r"[A-Z]{1,4}", t))

    code_idx = next((i for i, p in enumerate(parts)
                     if _looks_like_instrument_code(p["text"])), None)
    if code_idx is None:
        code_idx = next((i for i, p in enumerate(parts)
                         if i > 0 and _is_clean_code(p["text"])), None)

    if expected_code:
        if code_idx is None and len(parts) >= 2:
            parts = parts[:1] + [{"text": expected_code, "conf": 0.0}] + parts[1:]
            code_idx = 1
        elif code_idx is not None:
            cur_t = parts[code_idx]["text"]
            if (cur_t == expected_code or expected_code.startswith(cur_t)
                    or cur_t.startswith(expected_code)
                    or (len(cur_t) == 1 and cur_t in expected_code)):
                parts[code_idx] = {"text": expected_code, "conf": parts[code_idx]["conf"]}

    fixed = []
    for i, p in enumerate(parts):
        t = p["text"]
        if code_idx is not None and i == code_idx:
            t = t if t.isalpha() else _norm_alpha(t)
        elif _is_clean_number(t):
            pass
        elif i == len(parts) - 1 or (code_idx is not None and i > code_idx):
            t = _coerce_number(t)
        elif i == 0 and not t.isdigit():
            t = _norm_digits(t)
        fixed.append(t)
    return sep.join(f for f in fixed if f)


# =================================================================== RESOURCE
@st.cache_resource(show_spinner=False)
def load_model(model_bytes: bytes | None, model_path: str | None):
    """Muat YOLO. Di-cache agar tidak reload tiap interaksi."""
    from ultralytics import YOLO
    if model_bytes is not None:
        tmp = APP_DIR / ".cache_model.pt"
        tmp.write_bytes(model_bytes)
        return YOLO(str(tmp))
    return YOLO(str(model_path))


@st.cache_resource(show_spinner=False)
def load_ocr(use_gpu: bool):
    import easyocr
    return easyocr.Reader(["en"], gpu=use_gpu, verbose=False)


# ================================================================== INFERENSI
def predict_page(model, page: Image.Image, conf=0.25, iou=0.45, merge_iou=0.5,
                 device="cpu", progress=None, class_agnostic=True) -> list[dict]:
    """Sliding-window: potong halaman -> YOLO tiap tile -> gabung + NMS."""
    W, H = page.size
    nrow, ncol = grid_shape(W, H)
    total = nrow * ncol
    raw: list[dict] = []
    done = 0
    for r in range(1, nrow + 1):
        for c in range(1, ncol + 1):
            ox, oy = tile_origin(r, c, W, H)
            crop = page.crop((ox, oy, ox + TILE_SIZE, oy + TILE_SIZE))
            res = model.predict(source=np.array(crop), conf=conf, iou=iou,
                                imgsz=TILE_SIZE, device=device, verbose=False)[0]
            if res.boxes is not None and len(res.boxes):
                for box, cf, ci in zip(res.boxes.xyxy.cpu().numpy(),
                                       res.boxes.conf.cpu().numpy(),
                                       res.boxes.cls.cpu().numpy().astype(int)):
                    raw.append({
                        "class_id": int(ci),
                        "class_name": model.names[int(ci)],
                        "confidence": float(cf),
                        "xmin": float(box[0]) + ox, "ymin": float(box[1]) + oy,
                        "xmax": float(box[2]) + ox, "ymax": float(box[3]) + oy,
                        "src_tile": f"r{r:02d}_c{c:02d}",
                    })
            done += 1
            if progress:
                progress(done / total, f"Tile {done}/{total} — {len(raw)} deteksi mentah")
    merged = merge_nms(raw, iou_thr=merge_iou, class_agnostic=class_agnostic)
    return merged, len(raw), (nrow, ncol)


# Palet stabil per class_id (tidak bergantung matplotlib).
_PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
]


def color_for(cid: int) -> tuple[int, int, int]:
    return _PALETTE[cid % len(_PALETTE)]


def label_lines(r: dict, style: str = "class_tag", show_conf: bool = False) -> list[str]:
    """Isi label untuk satu deteksi, sebagai daftar baris.

    style:
      "class_tag" : nama kelas di baris 1, tag di baris 2  (default)
      "isa_tag"   : "PSV - 069-PSV-3100" satu baris
      "tag"       : tag saja (perilaku lama)
      "class"     : nama kelas saja
    Bila tag tidak terbaca ("N/A"/kosong), baris tag dilewati supaya tidak
    muncul label "N/A" di gambar.
    """
    cls = str(r.get("class_name", "") or "")
    tag = str(r.get("tag_number", "") or "")
    if tag in ("N/A", "nan"):
        tag = ""

    if style == "tag":
        lines = [tag or cls]
    elif style == "class":
        lines = [cls]
    elif style == "isa_tag":
        isa = str(r.get("tag_code", "") or "") or isa_code_for_class(cls)
        # Tanpa tag, kode ISA sendirian ("TT") kurang informatif dibanding nama
        # kelas -- jatuh ke nama kelas seperti gaya lain.
        lines = [f"{isa} - {tag}" if (isa and tag) else (tag or cls)]
    else:  # class_tag
        lines = [x for x in (cls, tag) if x] or [cls]

    if show_conf and lines:
        try:
            lines[-1] = f"{lines[-1]}  {float(r['confidence']):.2f}".strip()
        except (KeyError, TypeError, ValueError):
            pass
    return [l for l in lines if l]


def draw_detections(page: Image.Image, rows: list[dict], show_tag=True,
                    show_conf=False, lw=3, style="class_tag") -> Image.Image:
    out = page.convert("RGB").copy()
    dr = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    # Dua tahap: SEMUA kotak dulu, baru SEMUA label. Label punya latar solid dan
    # sering lebih lebar dari kotaknya sendiri; kalau digambar selang-seling,
    # label satu simbol menimpa garis kotak simbol tetangga sehingga kotak itu
    # terlihat "terpotong setengah".
    W, H = out.size
    for r in rows:
        col = color_for(r.get("class_id", 0))
        dr.rectangle([r["xmin"], r["ymin"], r["xmax"], r["ymax"]], outline=col, width=lw)

    # Label ditempatkan menghindari label lain. Latar label bersifat solid dan
    # biasanya JAUH lebih lebar dari kotaknya (mis. "069-PSV-3100" ~100px vs
    # kotak simbol ~50px), sehingga label yang ditempel begitu saja menutupi
    # simbol & teks tetangga. Empat kandidat posisi dicoba berurutan; yang
    # paling sedikit bertabrakan yang dipakai.
    placed: list[tuple[float, float, float, float]] = []

    def _overlap_area(box, others):
        x1, y1, x2, y2 = box
        tot = 0.0
        for o in others:
            ix1, iy1 = max(x1, o[0]), max(y1, o[1])
            ix2, iy2 = min(x2, o[2]), min(y2, o[3])
            if ix2 > ix1 and iy2 > iy1:
                tot += (ix2 - ix1) * (iy2 - iy1)
        return tot

    for r in rows:
        col = color_for(r.get("class_id", 0))
        if not show_tag:
            continue
        lines = label_lines(r, style=style, show_conf=show_conf)
        if not lines:
            continue
        # Kotak label memuat beberapa baris: lebar = baris terlebar.
        dims = [dr.textbbox((0, 0), l, font=font) for l in lines]
        widths = [d[2] - d[0] for d in dims]
        line_h = max(d[3] - d[1] for d in dims) + 2
        tw, th = max(widths), line_h * len(lines)
        bw, bh = tw + 6, th + 5

        # atas-kiri, bawah-kiri, atas-kanan (rata kanan kotak), di dalam kotak
        cands = [
            (r["xmin"], r["ymin"] - bh),
            (r["xmin"], r["ymax"]),
            (r["xmax"] - bw, r["ymin"] - bh),
            (r["xmin"], r["ymin"]),
        ]
        best, best_cost = None, None
        for cx, cy in cands:
            cx = min(max(cx, 0), max(W - bw, 0))
            cy = min(max(cy, 0), max(H - bh, 0))
            box = (cx, cy, cx + bw, cy + bh)
            cost = _overlap_area(box, placed)
            if best_cost is None or cost < best_cost:
                best, best_cost = box, cost
            if cost == 0:
                break

        dr.rectangle(list(best), fill=col)
        for li, ln in enumerate(lines):
            dr.text((best[0] + 3, best[1] + 2 + li * line_h), ln,
                    fill=(255, 255, 255), font=font)
        placed.append(best)
    return out


def build_excel(df: pd.DataFrame) -> bytes:
    """Workbook multi-sheet; kolom tag dipaksa teks agar nol depan tidak hilang."""
    df = df.copy()
    for c in ("tag_number", "tag_prefix", "tag_code", "tag_number_only", "tag_suffix",
              "alt_class", "alt_all"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    tagged = df[df["tag_number"] != "N/A"] if "tag_number" in df.columns else df.iloc[0:0]

    per_page = (df.assign(_t=(df["tag_number"] != "N/A").astype(int))
                  .groupby("image_name")
                  .agg(jumlah_simbol=("class_name", "size"),
                       simbol_bertag=("_t", "sum"),
                       conf_rata2=("confidence", "mean")).reset_index())
    per_page["persen_bertag"] = (per_page["simbol_bertag"] /
                                 per_page["jumlah_simbol"] * 100).round(1)
    per_page["conf_rata2"] = per_page["conf_rata2"].round(3)

    per_class = (df.groupby("class_name")
                   .agg(jumlah=("class_name", "size"),
                        conf_rata2=("confidence", "mean"),
                        conf_min=("confidence", "min"),
                        conf_max=("confidence", "max"))
                   .reset_index().sort_values("jumlah", ascending=False))
    for c in ("conf_rata2", "conf_min", "conf_max"):
        per_class[c] = per_class[c].round(3)

    tcols = [c for c in ("tag_number", "tag_prefix", "tag_code", "tag_number_only",
                         "class_name", "image_name") if c in df.columns]
    tag_reg = (tagged[tcols].drop_duplicates(subset=["tag_number"]).sort_values("tag_number")
               .reset_index(drop=True) if len(tagged) else pd.DataFrame(columns=tcols))

    sheets = {"Symbols": df, "Ringkasan": per_page,
              "Per Kelas": per_class, "Tag Register": tag_reg}
    TEXT_COLS = {"tag_number", "tag_prefix", "tag_code", "tag_number_only",
                 "tag_suffix", "src_tile", "image_name", "class_name",
                 "alt_class", "alt_all"}

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, sdf in sheets.items():
            sdf.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            if len(sdf):
                ws.auto_filter.ref = ws.dimensions
            for j, col in enumerate(sdf.columns, 1):
                letter = ws.cell(row=1, column=j).column_letter
                body = sdf[col].astype(str)
                width = max(len(str(col)), int(body.str.len().max()) if len(body) else 0)
                ws.column_dimensions[letter].width = min(max(width + 2, 10), 42)
                if col in TEXT_COLS:
                    for row in range(2, len(sdf) + 2):
                        ws.cell(row=row, column=j).number_format = "@"
    return buf.getvalue()


def pdf_to_images(data: bytes, dpi: int, skip_cover: bool = True) -> list[Image.Image]:
    """Rasterisasi PDF -> list gambar halaman (setara convertir.py).

    `skip_cover` melewati halaman pertama (page de garde), persis seperti
    convertir.py yang memulai dari doc[1]. Halaman ke-2 PDF menjadi p01.
    """
    import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    zoom = dpi / 72
    pages = []
    start = 1 if (skip_cover and len(doc) > 1) else 0
    for p in doc[start:]:
        pix = p.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return pages


def crop_frame(img: Image.Image, box: tuple[int, int, int, int] = CROP_BOX
               ) -> tuple[Image.Image, str]:
    """Buang kop & bingkai gambar (setara retirer_cadre.py).

    Kotak crop (74, 75, 2761, 2261) ditentukan pada raster 200 DPI berukuran
    3308x2339. Bila halaman datang pada ukuran lain (DPI berbeda, atau PNG yang
    sudah dipotong), kotak diskalakan proporsional agar tetap membuang bagian
    yang sama secara relatif — memakai angka absolut pada gambar berukuran lain
    akan memotong di tempat yang salah.

    Mengembalikan (gambar, catatan) di mana catatan menjelaskan apa yang terjadi
    supaya bisa ditampilkan ke pengguna.
    """
    W, H = img.size
    exp_w, exp_h = RASTER_SIZE

    # Sudah seukuran hasil crop -> kemungkinan besar sudah dipotong sebelumnya.
    if (W, H) == CROPPED_SIZE:
        return img, "sudah berukuran hasil crop — dilewati"

    if (W, H) == RASTER_SIZE:
        return img.crop(box), f"crop {box} -> {CROPPED_SIZE[0]}x{CROPPED_SIZE[1]}"

    # Ukuran lain: skalakan kotak crop mengikuti rasio halaman.
    sx, sy = W / exp_w, H / exp_h
    scaled = (round(box[0] * sx), round(box[1] * sy),
              round(box[2] * sx), round(box[3] * sy))
    scaled = (max(0, scaled[0]), max(0, scaled[1]),
              min(W, scaled[2]), min(H, scaled[3]))
    if scaled[2] - scaled[0] < TILE_SIZE or scaled[3] - scaled[1] < TILE_SIZE:
        return img, f"{W}x{H}: hasil crop terlalu kecil — dilewati"
    out = img.crop(scaled)
    return out, (f"{W}x{H} != {exp_w}x{exp_h}, kotak diskalakan {scaled} "
                 f"-> {out.size[0]}x{out.size[1]}")


# ======================================================================== UI
st.title("⚙ P&ID Digitizer")
st.caption("Deteksi simbol P&ID + ekstraksi tag number — sliding-window YOLO pada tile 512.")

with st.sidebar:
    st.header("Model")
    default_models = sorted(ROOT.glob("*.pt")) + sorted(APP_DIR.glob("*.pt"))
    default_models = [p for p in default_models if not p.name.startswith(".cache_")]

    src = st.radio("Sumber model", ["File di folder project", "Upload .pt"],
                   index=0 if default_models else 1)
    model_bytes = model_path = None
    if src == "File di folder project":
        if default_models:
            sel = st.selectbox("Pilih model", [p.name for p in default_models])
            model_path = str(next(p for p in default_models if p.name == sel))
        else:
            st.warning("Tidak ada file .pt di folder project.")
    else:
        up = st.file_uploader("Upload best.pt", type=["pt"])
        if up:
            model_bytes = up.read()

    st.divider()
    st.header("Parameter deteksi")
    conf = st.slider("Confidence minimum", 0.05, 0.95, 0.25, 0.05,
                     help="Turunkan bila simbol terlewat; naikkan bila banyak deteksi palsu.")
    iou = st.slider("IoU NMS (dalam tile)", 0.1, 0.9, 0.45, 0.05)
    merge_iou = st.slider("IoU penggabungan antar-tile", 0.1, 0.9, 0.5, 0.05,
                          help="Membuang duplikat di zona overlap 40 px.")
    class_agnostic = st.checkbox(
        "Gabungkan deteksi antar-kelas", value=True,
        help="Satu lokasi = satu simbol. Bila model menebak dua kelas berbeda "
             "untuk objek yang sama (mis. TEMPERATURE ELEMENT vs PRESSURE "
             "TRANSMITTER — sama-sama lingkaran tiga baris), hanya confidence "
             "tertinggi yang dipakai; kelas kalah dicatat di kolom alt_class. "
             "Matikan bila di gambar Anda memang ada simbol berbeda yang sah "
             "bertumpuk erat.")

    st.divider()
    st.header("Ekstraksi tag (OCR)")
    do_ocr = st.checkbox("Aktifkan OCR tag", value=True)
    upscale = st.slider("Upscale crop", 1, 8, 4, 1,
                        help="Kode instrumen (baris tengah bubble) kecil — "
                             "perbesaran membuatnya terbaca.")
    use_class_hint = st.checkbox("Lengkapi kode dari kelas terdeteksi", value=True,
                                 help="Mis. OCR baca 'T', kelas TEMPERATURE TRANSMITTER -> 'TT'.")
    use_gpu = st.checkbox("Pakai GPU untuk OCR", value=False)

    st.divider()
    device = st.selectbox("Device YOLO", ["cpu", "0"], index=0)
    st.caption(f"Tile {TILE_SIZE} px · stride {STRIDE} px · overlap {TILE_SIZE - STRIDE} px")

tab_run, tab_help = st.tabs(["Inferensi", "Panduan"])

with tab_help:
    st.markdown(f"""
### Cara kerja

Model dilatih pada **tile {TILE_SIZE}×{TILE_SIZE}**, bukan halaman penuh. Karena itu
halaman utuh tidak bisa langsung dimasukkan ke model — simbol akan menyusut drastis
dan tidak terdeteksi.

Aplikasi ini menjalankan pipeline penuh, dari PDF mentah sampai tabel akhir —
tahap 1–2 menirukan `crop-code`, tahap 3 seterusnya menirukan notebook:

```
PDF  ->  rasterisasi {RASTER_DPI} DPI (lewati sampul)      [convertir.py]
     ->  buang kop & bingkai: crop {CROP_BOX}   [retirer_cadre.py]
     ->  potong jadi tile (stride {STRIDE})                  [tuiles.py]
     ->  YOLO tiap tile  ->  geser koordinat ke halaman
     ->  NMS buang duplikat overlap
     ->  OCR isi tiap bubble  ->  rakit tag utuh
```

### Kenapa tahap crop tidak boleh dilewati

Model dilatih pada gambar **hasil crop** ({CROPPED_SIZE[0]}×{CROPPED_SIZE[1]}), bukan
raster mentah ({RASTER_SIZE[0]}×{RASTER_SIZE[1]}). Bila kop gambar dibiarkan:

- grid berubah dari **5×6 = 30 tile** menjadi **5×7 = 35 tile**;
- satu kolom tile berisi kop gambar — tabel, logo, dan teks revisi yang tidak
  pernah ada di data training, sumber deteksi palsu;
- posisi tile bergeser, sehingga simbol muncul di bagian tile yang berbeda dari
  saat training.

### Kenapa tag butuh upscale

Bubble instrumen berisi tiga baris: `069` / `TT` / `0025`. Baris tengah berukuran
kecil dan menempel garis pembagi, sehingga pada resolusi asli **sering tidak terbaca
sama sekali**. Memperbesar crop 4× membuatnya muncul dengan confidence ~1.00.

### Format input

| Input | Catatan |
|---|---|
| PNG / JPG | Paling langsung. Pakai gambar dengan resolusi asli (jangan diperkecil). |
| PDF | Dirasterisasi dulu; atur DPI agar ukuran simbol wajar (~50 px). |

### Batas yang perlu diketahui

- Kelas yang tidak ada di `obj.names` tidak akan pernah terdeteksi — mis. instrumen
  **Indicator** (`PI`/`TI`/`FI`) bila model hanya dilatih untuk **Transmitter**.
- Simbol berukuran jauh lebih kecil/besar dari data training akan sulit dikenali.
- Kolom tag di CSV sebaiknya dibuka lewat file **XLSX**; Excel mengubah `069` jadi
  `69` bila CSV dibuka langsung.
""")

with tab_run:
    up_files = st.file_uploader(
        "Unggah gambar halaman P&ID (PNG/JPG) atau PDF",
        type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)

    has_pdf = bool(up_files) and any(f.name.lower().endswith(".pdf") for f in up_files)

    st.markdown("**Penyiapan halaman** — samakan dengan cara data training dibuat.")
    cpre1, cpre2 = st.columns(2)
    with cpre1:
        do_crop = st.checkbox(
            "Buang kop & bingkai gambar", value=True,
            help=f"Crop {CROP_BOX} seperti retirer_cadre.py. Model dilatih pada "
                 f"hasil crop ({CROPPED_SIZE[0]}×{CROPPED_SIZE[1]}); tanpa ini kop "
                 "gambar ikut diproses dan koordinat bergeser.")
    with cpre2:
        skip_cover = st.checkbox(
            "Lewati halaman sampul PDF", value=True, disabled=not has_pdf,
            help="convertir.py melewati halaman pertama; halaman ke-2 PDF menjadi p01.")

    dpi = RASTER_DPI
    if has_pdf:
        dpi = st.slider("DPI rasterisasi PDF", 100, 400, RASTER_DPI, 25,
                        help="200 DPI = ukuran yang dipakai saat membuat data training.")
        if dpi != RASTER_DPI:
            st.warning(f"DPI {dpi} berbeda dari {RASTER_DPI} DPI yang dipakai saat "
                       "training — ukuran simbol ikut berubah dan deteksi bisa memburuk.")

    ready = (model_path or model_bytes) and up_files
    if not ready:
        st.info("Pilih model di sidebar dan unggah minimal satu file untuk memulai.")

    if ready and st.button("Jalankan deteksi", type="primary", use_container_width=True):
        try:
            with st.spinner("Memuat model…"):
                model = load_model(model_bytes, model_path)
            reader = None
            if do_ocr:
                with st.spinner("Memuat EasyOCR (pertama kali bisa lama)…"):
                    reader = load_ocr(use_gpu)
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            st.stop()

        # --- kumpulkan halaman (rasterisasi -> crop kop/bingkai) -----------
        # Urutan tahap sengaja dibuat identik dengan crop-code:
        #   convertir.py (PDF->PNG 200 DPI) -> retirer_cadre.py (buang kop)
        # sehingga gambar yang masuk ke model sama bentuknya dgn data training.
        pages: list[tuple[str, Image.Image]] = []
        prep_notes: list[str] = []
        for f in up_files:
            data = f.read()
            raw: list[tuple[str, Image.Image]] = []
            if f.name.lower().endswith(".pdf"):
                try:
                    for i, im in enumerate(pdf_to_images(data, dpi, skip_cover), 1):
                        raw.append((f"{Path(f.name).stem}_p{i:02d}.png", im))
                except Exception as e:
                    st.error(f"Gagal membaca PDF {f.name}: {e}")
            else:
                raw.append((f.name, Image.open(io.BytesIO(data)).convert("RGB")))

            for name, im in raw:
                if do_crop:
                    im, note = crop_frame(im)
                    prep_notes.append(f"{name}: {note}")
                pages.append((name, im))

        if prep_notes:
            with st.expander(f"Penyiapan halaman ({len(prep_notes)})", expanded=False):
                st.code("\n".join(prep_notes), language=None)

        if not pages:
            st.warning("Tidak ada halaman yang bisa diproses.")
            st.stop()

        all_rows: list[dict] = []
        results: list[tuple[str, Image.Image, list[dict]]] = []
        bar = st.progress(0.0, "Mulai…")

        for pi, (name, page) in enumerate(pages, 1):
            W, H = page.size
            if W < TILE_SIZE or H < TILE_SIZE:
                st.warning(f"{name}: {W}×{H} lebih kecil dari tile {TILE_SIZE} — dilewati.")
                continue

            def prog(frac, msg, _pi=pi, _n=name):
                bar.progress(((_pi - 1) + frac) / len(pages),
                             f"[{_pi}/{len(pages)}] {_n} — {msg}")

            dets, n_raw, (nr, nc) = predict_page(
                model, page, conf=conf, iou=iou, merge_iou=merge_iou,
                device=device, progress=prog, class_agnostic=class_agnostic)

            rows = []
            for i, d in enumerate(dets):
                tag = ""
                if do_ocr and reader is not None:
                    hint = isa_code_for_class(d["class_name"]) if use_class_hint else None
                    try:
                        tag = ocr_symbol_tag(reader, page, d, upscale=upscale,
                                             expected_code=hint)
                    except Exception:
                        tag = ""
                comp = parse_tag_components(tag)
                # Kelas lain yang kalah di lokasi ini (bila ada) -- sinyal bahwa
                # model ragu antar kelas yang bentuknya mirip. Berguna untuk
                # menentukan kelas mana yang perlu tambahan contoh training.
                alt = sorted(d.get("alt", []), key=lambda z: -z[1])
                rows.append({
                    "image_name": name, "symbol_id": i,
                    "class_id": d["class_id"], "class_name": d["class_name"],
                    "confidence": round(d["confidence"], 4),
                    "tag_number": tag if tag else "N/A", **comp,
                    "alt_class": alt[0][0] if alt else "",
                    "alt_confidence": round(alt[0][1], 4) if alt else "",
                    "alt_all": "; ".join(f"{n}:{c:.2f}" for n, c in alt),
                    "xmin": round(d["xmin"], 1), "ymin": round(d["ymin"], 1),
                    "xmax": round(d["xmax"], 1), "ymax": round(d["ymax"], 1),
                    "src_tile": d["src_tile"],
                })
            all_rows.extend(rows)
            results.append((name, page, rows))
            bar.progress(pi / len(pages), f"[{pi}/{len(pages)}] {name} selesai")

        bar.empty()

        if not all_rows:
            st.warning("Tidak ada simbol terdeteksi. Coba turunkan confidence.")
            st.stop()

        df = pd.DataFrame(all_rows)
        st.session_state["df"] = df
        st.session_state["results"] = results

        # --- laporan keraguan antar-kelas ---------------------------------
        # Setiap baris ber-alt_class = satu lokasi yang ditebak >1 kelas.
        # Tanpa penggabungan antar-kelas, tiap kasus ini akan tergambar
        # sebagai dua kotak bertumpuk pada simbol yang sama.
        n_amb = int((df["alt_class"] != "").sum()) if "alt_class" in df.columns else 0
        if n_amb:
            if class_agnostic:
                st.info(f"{n_amb} simbol ditebak lebih dari satu kelas — "
                        f"kotak ganda sudah digabung, kelas kalah tercatat di "
                        f"kolom `alt_class`.")
            else:
                st.warning(f"{n_amb} lokasi ditebak lebih dari satu kelas dan "
                           f"penggabungan antar-kelas SEDANG MATI — simbol "
                           f"tersebut tergambar sebagai kotak bertumpuk.")
            amb = df[df["alt_class"] != ""]
            pair = (amb.groupby(["class_name", "alt_class"])
                       .size().reset_index(name="jumlah")
                       .sort_values("jumlah", ascending=False))
            with st.expander(f"Pasangan kelas yang tertukar ({len(pair)})",
                             expanded=False):
                st.caption("Kelas kiri menang (confidence lebih tinggi). Pasangan "
                           "yang sering muncul = kandidat penambahan data training.")
                st.dataframe(pair, use_container_width=True, hide_index=True)

    # ---------------------------------------------------- tampilkan hasil
    if "df" in st.session_state:
        df: pd.DataFrame = st.session_state["df"]
        results = st.session_state["results"]

        full = df[(df["tag_code"] != "") & (df["tag_number_only"] != "")]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Halaman", len(results))
        c2.metric("Simbol terdeteksi", len(df))
        c3.metric("Tag terbaca", int((df["tag_number"] != "N/A").sum()))
        c4.metric("Tag lengkap", f"{len(full)} ({len(full)/len(df)*100:.0f}%)")

        st.subheader("Visualisasi")
        vc1, vc2, vc3, vc4 = st.columns([2, 2, 1, 1])
        page_names = [r[0] for r in results]
        pick = vc1.selectbox("Halaman", page_names)
        LABEL_STYLES = {
            "Nama kelas + tag (2 baris)": "class_tag",
            "Kode ISA + tag (1 baris)": "isa_tag",
            "Tag saja": "tag",
            "Nama kelas saja": "class",
        }
        style_name = vc2.selectbox("Format label", list(LABEL_STYLES), index=0)
        style = LABEL_STYLES[style_name]
        show_tag = vc3.checkbox("Tampilkan label", value=True)
        show_conf = vc4.checkbox("Confidence", value=False)

        name, page, rows = next(r for r in results if r[0] == pick)
        vis = draw_detections(page, rows, show_tag=show_tag, show_conf=show_conf,
                              style=style)
        st.image(vis, use_container_width=True,
                 caption=f"{name} — {len(rows)} simbol")

        buf = io.BytesIO()
        vis.save(buf, format="PNG")
        st.download_button("Unduh gambar beranotasi", buf.getvalue(),
                           file_name=f"annotated_{Path(name).stem}.png", mime="image/png")

        st.subheader("Tabel hasil")
        f1, f2 = st.columns(2)
        classes = sorted(df["class_name"].unique())
        pick_cls = f1.multiselect("Filter kelas", classes, default=classes)
        only_tag = f2.checkbox("Hanya yang punya tag", value=False)

        view = df[df["class_name"].isin(pick_cls)]
        if only_tag:
            view = view[view["tag_number"] != "N/A"]
        st.dataframe(
            view[["image_name", "class_name", "tag_number", "tag_prefix", "tag_code",
                  "tag_number_only", "confidence", "src_tile"]],
            use_container_width=True, height=380)

        st.subheader("Ekspor")
        e1, e2, e3 = st.columns(3)
        e1.download_button("CSV", df.to_csv(index=False).encode("utf-8"),
                           "pid_hasil.csv", "text/csv", use_container_width=True)
        try:
            e2.download_button(
                "Excel (4 sheet)", build_excel(df), "pid_hasil.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)
        except Exception as e:
            e2.error(f"XLSX gagal: {e}")
        e3.download_button("JSON", df.to_json(orient="records", indent=2).encode("utf-8"),
                           "pid_hasil.json", "application/json", use_container_width=True)

        with st.expander("Ringkasan per kelas"):
            summ = (df.groupby("class_name")
                      .agg(jumlah=("class_name", "size"),
                           conf_rata2=("confidence", "mean"))
                      .reset_index().sort_values("jumlah", ascending=False))
            summ["conf_rata2"] = summ["conf_rata2"].round(3)
            st.dataframe(summ, use_container_width=True, hide_index=True)
