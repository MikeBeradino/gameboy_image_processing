# gbc_vector_gui.py
# Game Boy Camera raster (2-bit) -> 1-bit masks + plotter-friendly SVGs.
# Modes: Horizontal hatch, Vertical hatch, Nested boxes (pen down / pen lift), Dots.
# Features: linked scaling (pixel mm <-> canvas W/H mm), nib mm, edge margin mm,
# Zig-zag for H/V (serpentine within runs), CairoSVG preview (~500 px),
# labeled layers (deep/mid/light/white), per-image output dirs with per-mode subfolders.

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import io, sys, os, re, xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import svgwrite
import tkinter as tk
from tkinter import ttk, messagebox

# ---------------- macOS Cairo hint (for preview) ----------------
if sys.platform == "darwin" and "CAIRO" not in os.environ:
    for p in ("/opt/homebrew/opt/cairo/lib/libcairo.2.dylib",
              "/usr/local/opt/cairo/lib/libcairo.2.dylib"):
        if Path(p).exists():
            os.environ["CAIRO"] = p
            break
# ----------------------------------------------------------------

# ---------------- config ----------------
TONE_DISPLAY = ["deep", "mid", "light", "white"]
# Darkest..lightest class index -> label
IDX_TO_LABEL = {0: "deep", 1: "mid", 2: "light", 3: "white"}

OUT_DIR = Path("./out"); OUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_TARGET_PX = 500
ALLOWED_EXTS = {".png", ".bmp", ".gif", ".jpg", ".jpeg", ".tif", ".tiff"}
EPS = 1e-6
# ---------------------------------------

# -------------- core logic --------------
def load_gray_u8(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)

def find_four_levels(gray: np.ndarray) -> np.ndarray:
    uniq = np.unique(gray)
    if len(uniq) <= 4:
        levels = np.sort(uniq)
        if len(levels) < 4:
            levels = np.linspace(levels.min(), levels.max(), 4).round().astype(np.uint8)
        return levels.astype(np.uint8)
    bins = np.linspace(0, 256, 5)
    idx = np.digitize(gray, bins[1:-1], right=False)
    levels = []
    for b in range(4):
        sel = gray[idx == b]
        if sel.size == 0:
            lo, hi = int(bins[b]), int(bins[b+1]-1)
            levels.append((lo + hi)//2)
        else:
            levels.append(int(np.median(sel)))
    return np.array(sorted(levels), dtype=np.uint8)

def classify_to_4(gray: np.ndarray, levels: np.ndarray) -> np.ndarray:
    levels = np.sort(levels)
    dists = np.abs(gray[..., None].astype(np.int16) - levels[None, None, :].astype(np.int16))
    return np.argmin(dists, axis=2).astype(np.uint8)

def labels_to_layer_masks(labels: np.ndarray):
    return [(labels == i).astype(np.uint8) for i in range(4)]

def save_1bit_bmp(mask: np.ndarray, path: Path):
    """Save binary mask as 1-bit BMP (ink = black, paper = white)."""
    img = np.where(mask == 1, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img, mode="L").convert("1")
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path, format="BMP")

def _binary_runs_1d(vec: np.ndarray) -> List[Tuple[int, int]]:
    runs = []; in_run = False; start = 0
    for i, v in enumerate(vec):
        if v and not in_run:
            in_run = True; start = i
        elif (not v) and in_run:
            runs.append((start, i - 1)); in_run = False
    if in_run: runs.append((start, len(vec) - 1))
    return runs

# ---- Dots mode ----
def stroke_dots_svg(mask: np.ndarray, svg_path: Path, pixel_mm: float, nib_mm: float, dot_mm: float):
    """
    One tiny horizontal segment (length = dot_mm) centered in each filled pixel.
    Stroke width = nib_mm (scaled into user units).
    """
    h, w = mask.shape
    nib_u = nib_mm / pixel_mm
    half_dot_u = max(1e-6, 0.5 * (dot_mm / pixel_mm))  # clamp to something > 0

    dwg = svgwrite.Drawing(str(svg_path),
                           size=(f"{w*pixel_mm}mm", f"{h*pixel_mm}mm"),
                           profile="tiny")
    dwg.viewbox(0, 0, w, h)
    dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill="white"))

    for j in range(h):
        row = mask[j, :]
        xs = np.where(row == 1)[0]
        if xs.size == 0:
            continue
        y = j + 0.5
        for i in xs:
            cx = i + 0.5
            dwg.add(dwg.line((cx - half_dot_u, y), (cx + half_dot_u, y),
                             stroke="black",
                             stroke_width=nib_u,
                             stroke_linecap="butt"))  # butt keeps marks tight
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    dwg.save()

# -------- H/V hatch (constant lines per pixel, overlap allowed, edge margin, zig-zag) --------
def stroke_hatch_horizontal_svg(mask, svg_path, pixel_mm, nib_mm, edge_mm, zigzag):
    import svgwrite
    h, w = mask.shape
    nib_u  = nib_mm  / pixel_mm
    edge_u = min(0.5, max(0.0, edge_mm / pixel_mm))   # inset left/right
    N      = max(1, int(np.ceil(pixel_mm / nib_mm)))  # lines per pixel (allow overlap)
    s      = 1.0 / N                                  # vertical spacing between line centers

    dwg = svgwrite.Drawing(str(svg_path),
                           size=(f"{w*pixel_mm}mm", f"{h*pixel_mm}mm"),
                           profile="tiny")
    dwg.viewbox(0, 0, w, h)
    dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill="white"))

    def row_segments(vec):
        segs = []
        in_run = False; start = 0
        for i, v in enumerate(vec):
            if v and not in_run:
                in_run = True; start = i
            elif (not v) and in_run:
                xL = start + edge_u; xR = i - edge_u
                if xR > xL + EPS: segs.append((xL, xR))
                in_run = False
        if in_run:
            xL = start + edge_u; xR = len(vec) - edge_u
            if xR > xL + EPS: segs.append((xL, xR))
        return segs

    for j in range(h):
        segs = row_segments(mask[j, :])
        if not segs: continue
        y_centers = [j + (k + 0.5) * s for k in range(N)]
        if not zigzag:
            for k, y in enumerate(y_centers):
                seq = segs if (k % 2 == 0) else [(b, a) for (a, b) in segs[::-1]]
                for (xL, xR) in seq:
                    dwg.add(dwg.line((xL, y), (xR, y),
                                     stroke="black",
                                     stroke_width=nib_u,
                                     stroke_linecap="butt"))
        else:
            for (xL, xR) in segs:
                d = [f"M {xL:.6f} {y_centers[0]:.6f}",
                     f"L {xR:.6f} {y_centers[0]:.6f}"]
                at_right = True
                for k in range(1, N):
                    yk = y_centers[k]
                    if at_right:
                        d.append(f"L {xR:.6f} {yk:.6f}")
                        d.append(f"L {xL:.6f} {yk:.6f}")
                    else:
                        d.append(f"L {xL:.6f} {yk:.6f}")
                        d.append(f"L {xR:.6f} {yk:.6f}")
                    at_right = not at_right
                dwg.add(dwg.path(" ".join(d),
                                 fill="none",
                                 stroke="black",
                                 stroke_width=nib_u,
                                 stroke_linecap="butt"))
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    dwg.save()

def stroke_hatch_vertical_svg(mask, svg_path, pixel_mm, nib_mm, edge_mm, zigzag):
    import svgwrite
    h, w = mask.shape
    nib_u  = nib_mm  / pixel_mm
    edge_u = min(0.5, max(0.0, edge_mm / pixel_mm))   # inset top/bottom
    N      = max(1, int(np.ceil(pixel_mm / nib_mm)))
    s      = 1.0 / N                                  # horizontal spacing between line centers

    dwg = svgwrite.Drawing(str(svg_path),
                           size=(f"{w*pixel_mm}mm", f"{h*pixel_mm}mm"),
                           profile="tiny")
    dwg.viewbox(0, 0, w, h)
    dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill="white"))

    def col_segments(vec):
        segs = []
        in_run = False; start = 0
        for j, v in enumerate(vec):
            if v and not in_run:
                in_run = True; start = j
            elif (not v) and in_run:
                yT = start + edge_u; yB = j - edge_u
                if yB > yT + EPS: segs.append((yT, yB))
                in_run = False
        if in_run:
            yT = start + edge_u; yB = len(vec) - edge_u
            if yB > yT + EPS: segs.append((yT, yB))
        return segs

    for i in range(w):
        segs = col_segments(mask[:, i])
        if not segs: continue
        x_centers = [i + (k + 0.5) * s for k in range(N)]
        if not zigzag:
            for k, x in enumerate(x_centers):
                seq = segs if (k % 2 == 0) else [(b, a) for (a, b) in segs[::-1]]
                for (yT, yB) in seq:
                    dwg.add(dwg.line((x, yT), (x, yB),
                                     stroke="black",
                                     stroke_width=nib_u,
                                     stroke_linecap="butt"))
        else:
            for (yT, yB) in segs:
                d = [f"M {x_centers[0]:.6f} {yT:.6f}",
                     f"L {x_centers[0]:.6f} {yB:.6f}"]
                at_bottom = True
                for k in range(1, N):
                    xk = x_centers[k]
                    if at_bottom:
                        d.append(f"L {xk:.6f} {yB:.6f}")
                        d.append(f"L {xk:.6f} {yT:.6f}")
                    else:
                        d.append(f"L {xk:.6f} {yT:.6f}")
                        d.append(f"L {xk:.6f} {yB:.6f}")
                    at_bottom = not at_bottom
                dwg.add(dwg.path(" ".join(d),
                                 fill="none",
                                 stroke="black",
                                 stroke_width=nib_u,
                                 stroke_linecap="butt"))
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    dwg.save()

# ---- Nested boxes (nib-aware, fallback when nib > 0.5*pixel) ----
def _ring_offsets_units_for_pixel(pixel_mm: float, nib_mm: float, allow_edge_touch: bool = False) -> List[float]:
    nib_u = nib_mm / pixel_mm
    r_u   = (nib_mm / 2.0) / pixel_mm
    P_u   = 1.0
    o = r_u + (0.0 if allow_edge_touch else nib_u / 2.0)
    offsets = []
    while True:
        w = P_u - 2.0 * o
        if w < 0.0: break
        offsets.append(o)
        o += nib_u
    if not offsets and allow_edge_touch and nib_u < 1.0:
        offsets = [r_u]
    return offsets

def _append_rect_loop_path(d: List[str], xL: float, xR: float, yT: float, yB: float, xc: float):
    d.append(f"L {xR:.6f} {yT:.6f}")
    d.append(f"L {xR:.6f} {yB:.6f}")
    d.append(f"L {xL:.6f} {yB:.6f}")
    d.append(f"L {xL:.6f} {yT:.6f}")
    d.append(f"L {xc:.6f} {yT:.6f}")

def stroke_nested_boxes_svg(mask: np.ndarray, svg_path: Path, pixel_mm: float, nib_mm: float, pen_lift: bool):
    h, w = mask.shape
    nib_u = nib_mm / pixel_mm

    dwg = svgwrite.Drawing(str(svg_path),
                           size=(f"{w*pixel_mm}mm", f"{h*pixel_mm}mm"),
                           profile="tiny")
    dwg.viewbox(0, 0, w, h)
    dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill="white"))

    edge_touch = (nib_u > 0.5 + 1e-9)
    base_offsets = _ring_offsets_units_for_pixel(pixel_mm, nib_mm, allow_edge_touch=edge_touch)
    join_style = "bevel" if edge_touch else "miter"

    for j in range(h):
        row = mask[j, :]
        for i, v in enumerate(row):
            if not v or not base_offsets: continue
            offsets = list(reversed(base_offsets))  # inside→out default
            def ring_rect(o: float) -> Tuple[float, float, float, float]:
                return i + o, (i + 1) - o, j + o, (j + 1) - o
            xc, yc = i + 0.5, j + 0.5

            if pen_lift:
                for o in offsets:
                    xL, xR, yT, yB = ring_rect(o)
                    d = [f"M {xc:.6f} {yT:.6f}",
                         f"L {xR:.6f} {yT:.6f}",
                         f"L {xR:.6f} {yB:.6f}",
                         f"L {xL:.6f} {yB:.6f}",
                         f"L {xL:.6f} {yT:.6f}",
                         f"L {xc:.6f} {yT:.6f}"]
                    dwg.add(dwg.path(" ".join(d), fill="none", stroke="black",
                                     stroke_width=nib_u, stroke_linecap="butt",
                                     stroke_linejoin=join_style, stroke_miterlimit=2))
            else:
                d: List[str] = [f"M {xc:.6f} {yc:.6f}"]
                o0 = offsets[0]; xL, xR, yT, yB = ring_rect(o0)
                d.append(f"L {xc:.6f} {yT:.6f}")
                _append_rect_loop_path(d, xL, xR, yT, yB, xc)
                for o in offsets[1:]:
                    xL, xR, yT, yB = ring_rect(o)
                    d.append(f"L {xc:.6f} {yT:.6f}")  # vertical bridge
                    _append_rect_loop_path(d, xL, xR, yT, yB, xc)
                dwg.add(dwg.path(" ".join(d), fill="none", stroke="black",
                                 stroke_width=nib_u, stroke_linecap="butt",
                                 stroke_linejoin=join_style, stroke_miterlimit=2))
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    dwg.save()

# ---------------------------------------

def _fmt_mm(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s if s else "0"

def process_image_file(path: str, mode: str, pixel_mm: float, nib_mm: float,
                       pen_lift: bool, edge_mm: float, zigzag: bool, dot_mm: float):
    """
    mode ∈ {"Horizontal","Vertical","Nested boxes","Dots"}.
    Returns (results, wpx, hpx, base_dir).
    """
    stem = Path(path).stem
    gray = load_gray_u8(path); h, w = gray.shape
    levels = find_four_levels(gray)
    labels = classify_to_4(gray, levels)
    masks = labels_to_layer_masks(labels)

    # Per-image directories
    base_dir = OUT_DIR / stem
    bmps_dir = base_dir / "bmps"
    svgs_dir = base_dir / "svgs"
    bmps_dir.mkdir(parents=True, exist_ok=True)
    svgs_dir.mkdir(parents=True, exist_ok=True)

    # SVG subfolder + mode tag for filenames
    if mode == "Horizontal":
        svg_mode_dir = svgs_dir / "horizontal"; mode_tag = "horizontal"
    elif mode == "Vertical":
        svg_mode_dir = svgs_dir / "vertical"; mode_tag = "vertical"
    elif mode == "Dots":
        svg_mode_dir = svgs_dir / "dots"; mode_tag = "dots"
    else:
        if pen_lift:
            svg_mode_dir = svgs_dir / "nested_pen_lift"; mode_tag = "nested_pen_lift"
        else:
            svg_mode_dir = svgs_dir / "nested_pen_down"; mode_tag = "nested_pen_down"
    svg_mode_dir.mkdir(parents=True, exist_ok=True)

    px_tag   = f"px-{_fmt_mm(pixel_mm)}mm"
    nib_tag  = f"nib-{_fmt_mm(nib_mm)}mm"
    edge_tag = f"edge-{_fmt_mm(edge_mm)}mm" if mode in ("Horizontal", "Vertical") else None
    zz_tag   = "zigzag" if (mode in ("Horizontal","Vertical") and zigzag) else None
    dot_tag  = f"dot-{_fmt_mm(dot_mm)}mm" if mode == "Dots" else None

    results = []
    for i in range(4):
        label = IDX_TO_LABEL[i]
        mask = masks[i]

        bmp_path = bmps_dir / f"{stem}_{label}_1bit.bmp"

        name_parts = [stem, label, mode_tag, px_tag, nib_tag]
        if edge_tag: name_parts.append(edge_tag)
        if zz_tag:   name_parts.append(zz_tag)
        if dot_tag:  name_parts.append(dot_tag)
        svg_name = "_".join(name_parts) + ".svg"
        svg_path = svg_mode_dir / svg_name

        # write BMP
        save_1bit_bmp(mask, bmp_path)

        # write SVG by mode
        if mode == "Horizontal":
            stroke_hatch_horizontal_svg(mask, svg_path, pixel_mm=pixel_mm, nib_mm=nib_mm,
                                        edge_mm=edge_mm, zigzag=zigzag)
        elif mode == "Vertical":
            stroke_hatch_vertical_svg(mask, svg_path, pixel_mm=pixel_mm, nib_mm=nib_mm,
                                      edge_mm=edge_mm, zigzag=zigzag)
        elif mode == "Dots":
            stroke_dots_svg(mask, svg_path, pixel_mm=pixel_mm, nib_mm=nib_mm, dot_mm=dot_mm)
        else:
            stroke_nested_boxes_svg(mask, svg_path, pixel_mm=pixel_mm, nib_mm=nib_mm, pen_lift=pen_lift)

        results.append({"name": label, "mask_path": bmp_path, "svg_path": svg_path, "mask": mask})

    # 4-tone preview
    prev = np.zeros_like(gray)
    for i, lvl in enumerate(levels):
        prev[labels == i] = lvl
    Image.fromarray(prev, mode="L").save(base_dir / f"{stem}_preview_4level.png")

    return results, w, h, base_dir

# -------- SVG → PIL (preview ~500px) --------
def _parse_svg_size(svg_bytes: bytes) -> tuple[Optional[float], Optional[float]]:
    try:
        root = ET.fromstring(svg_bytes)
    except Exception:
        return (None, None)
    w_attr = root.get("width"); h_attr = root.get("height")
    def _f(s: Optional[str]) -> Optional[float]:
        if not s: return None
        m = re.match(r'^\s*([0-9]*\.?[0-9]+)', s)
        return float(m.group(1)) if m else None
    w = _f(w_attr); h = _f(h_attr)
    if w and h: return (w, h)
    vb = root.get("viewBox")
    if vb:
        parts = [p for p in re.split(r'[\s,]+', vb.strip()) if p]
        if len(parts) == 4:
            try: return (float(parts[2]), float(parts[3]))
            except Exception: pass
    return (None, None)

def _scale_to_max(img: Image.Image, target: int, resample=Image.NEAREST) -> Image.Image:
    w, h = img.size
    if max(w, h) == target: return img
    r = target / float(max(w, h))
    return img.resize((max(1, int(round(w*r))), max(1, int(round(h*r)))), resample)

def render_svg_to_pillow(svg_path: Path) -> Image.Image:
    try:
        import cairosvg
    except ImportError:
        raise RuntimeError("Install cairosvg for SVG preview: pip install cairosvg")
    svg_bytes = svg_path.read_bytes()
    w, h = _parse_svg_size(svg_bytes)
    if w and h and w > 0 and h > 0:
        png_data = cairosvg.svg2png(bytestring=svg_bytes,
                                    output_width=PREVIEW_TARGET_PX if w >= h else None,
                                    output_height=PREVIEW_TARGET_PX if h > w else None)
    else:
        png_data = cairosvg.svg2png(bytestring=svg_bytes)
    img = Image.open(io.BytesIO(png_data)).convert("RGBA")
    return _scale_to_max(img, PREVIEW_TARGET_PX, resample=Image.NEAREST)
# -------------------------------------

# ------------------ GUI ----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GBC → 1-bit masks → SVG (H/V/Nested/Dots) + Scaling + Nib + Edge + Zig-zag + Pen-lift")
        self.geometry("1140x980"); self.minsize(940, 780)

        # state
        self.folder = str(Path.cwd()); self.files: Dict[str, Path] = {}
        self.selected_path: Optional[Path] = None
        self.photo_refs: List[ImageTk.PhotoImage] = []
        self.auto_process = tk.BooleanVar(value=False)

        # mode selector
        self.mode = tk.StringVar(value="Horizontal")

        # linked scaling
        self.pixel_mm_var = tk.StringVar(value="1.0")
        self.canvas_w_var = tk.StringVar(value="")
        self.canvas_h_var = tk.StringVar(value="")
        self._link_guard = False

        # nib (mm), edge margin (mm), dot length (mm)
        self.nib_mm_var = tk.StringVar(value="0.3")
        self.edge_mm_var = tk.StringVar(value="")      # blank = auto nib/2
        self.dot_mm_var  = tk.StringVar(value="0.005") # tiny move for dots

        # nested option: pen lift between rings
        self.nested_pen_lift = tk.BooleanVar(value=False)

        # zig-zag option for H/V
        self.zigzag = tk.BooleanVar(value=True)

        # current image size
        self._wpx = None; self._hpx = None

        self.build_ui(); self.scan_folder()
        self.bind("<Return>", lambda *_: self.on_process())

    def build_ui(self):
        top = ttk.Frame(self, padding=8); top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Folder:").pack(side=tk.LEFT)
        self.folder_entry = ttk.Entry(top, width=46); self.folder_entry.insert(0, self.folder)
        self.folder_entry.pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Scan", command=self.scan_folder).pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="File:").pack(side=tk.LEFT, padx=(12, 0))
        self.file_combo = ttk.Combobox(top, state="readonly", width=36, values=[])
        self.file_combo.pack(side=tk.LEFT, padx=6); self.file_combo.bind("<<ComboboxSelected>>", self.on_select_file)
        ttk.Checkbutton(top, text="Auto-process on select", variable=self.auto_process).pack(side=tk.LEFT, padx=(12, 6))

        modebar = ttk.Frame(self, padding=(8, 0)); modebar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(modebar, text="Mode:").pack(side=tk.LEFT)
        self.dir_combo = ttk.Combobox(modebar, state="readonly", width=20,
                                      values=["Horizontal", "Vertical", "Nested boxes", "Dots"],
                                      textvariable=self.mode)
        self.dir_combo.pack(side=tk.LEFT, padx=6); self.dir_combo.current(0)

        opts = ttk.Frame(self, padding=(8, 0)); opts.pack(side=tk.TOP, fill=tk.X)
        ttk.Checkbutton(opts, text="Zig-zag (connect within runs)", variable=self.zigzag).pack(side=tk.LEFT)
        ttk.Checkbutton(opts, text="Pen lift between rings (Nested)", variable=self.nested_pen_lift).pack(side=tk.LEFT, padx=(12,0))

        scale = ttk.Frame(self, padding=(8, 2)); scale.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(scale, text="Pixel size (mm/px):").grid(row=0, column=0, sticky="w")
        self.pixel_entry = ttk.Entry(scale, width=10, textvariable=self.pixel_mm_var)
        self.pixel_entry.grid(row=0, column=1, padx=(4, 16), sticky="w")

        ttk.Label(scale, text="Canvas W (mm):").grid(row=0, column=2, sticky="w")
        self.canvas_w_entry = ttk.Entry(scale, width=10, textvariable=self.canvas_w_var)
        self.canvas_w_entry.grid(row=0, column=3, padx=(4, 16), sticky="w")

        ttk.Label(scale, text="Canvas H (mm):").grid(row=0, column=4, sticky="w")
        self.canvas_h_entry = ttk.Entry(scale, width=10, textvariable=self.canvas_h_var)
        self.canvas_h_entry.grid(row=0, column=5, padx=(4, 16), sticky="w")

        ttk.Label(scale, text="Nib (mm):").grid(row=0, column=6, sticky="w")
        self.nib_entry = ttk.Entry(scale, width=8, textvariable=self.nib_mm_var)
        self.nib_entry.grid(row=0, column=7, padx=(4, 12), sticky="w")

        ttk.Label(scale, text="Edge margin (mm):").grid(row=0, column=8, sticky="w")
        self.edge_entry = ttk.Entry(scale, width=10, textvariable=self.edge_mm_var)
        self.edge_entry.grid(row=0, column=9, padx=(4, 0), sticky="w")

        ttk.Label(scale, text="Dot length (mm):").grid(row=0, column=10, sticky="w")
        self.dot_entry = ttk.Entry(scale, width=10, textvariable=self.dot_mm_var)
        self.dot_entry.grid(row=0, column=11, padx=(4, 0), sticky="w")

        # auto-fill edge = nib/2 if left blank when nib changes
        def _maybe_autofill_edge(*_):
            val = (self.edge_mm_var.get() or "").strip()
            nib = self.parse_float(self.nib_mm_var.get())
            if (not val) and nib:
                self.edge_mm_var.set(f"{nib/2:.3f}")
        self.nib_entry.bind("<FocusOut>", _maybe_autofill_edge)
        self.nib_entry.bind("<Return>",   _maybe_autofill_edge)

        # link updates
        self.pixel_entry.bind("<FocusOut>", lambda e: self._scale_from("pixel"))
        self.pixel_entry.bind("<Return>",   lambda e: self._scale_from("pixel"))
        self.canvas_w_entry.bind("<FocusOut>", lambda e: self._scale_from("cw"))
        self.canvas_w_entry.bind("<Return>",   lambda e: self._scale_from("cw"))
        self.canvas_h_entry.bind("<FocusOut>", lambda e: self._scale_from("ch"))
        self.canvas_h_entry.bind("<Return>",   lambda e: self._scale_from("ch"))

        runbar = ttk.Frame(self, padding=(8, 2)); runbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(runbar, text="▶ Run (Generate outputs)", command=self.on_process).pack(side=tk.LEFT, padx=4, pady=6)

        self.size_label = ttk.Label(self, text=""); self.size_label.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0,6))
        self.status = ttk.Label(self, text=""); self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)

        self.tabs = ttk.Notebook(self); self.tabs.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.mask_labels = {}; self.svg_labels = {}
        for name in TONE_DISPLAY:
            frame = ttk.Frame(self.tabs, padding=8); self.tabs.add(frame, text=name)
            frame.columnconfigure(0, weight=1); frame.columnconfigure(1, weight=1); frame.rowconfigure(0, weight=1)
            left = ttk.LabelFrame(frame, text="1-bit BMP", padding=6); left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
            right = ttk.LabelFrame(frame, text="SVG (preview)", padding=6); right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
            ml = ttk.Label(left); ml.pack(expand=True); sl = ttk.Label(right); sl.pack(expand=True)
            self.mask_labels[name] = ml; self.svg_labels[name] = sl

    # helpers
    def parse_float(self, s: str) -> Optional[float]:
        s = (s or "").strip()
        if not s: return None
        try:
            v = float(s); return v if v > 0 else None
        except ValueError: return None

    def _edge_mm(self) -> float:
        v = self.parse_float(self.edge_mm_var.get())
        if v is not None:
            return max(0.0, v)
        nib = self.parse_float(self.nib_mm_var.get()) or 0.3
        return nib / 2.0

    def _scale_from(self, source: str):
        if self._wpx is None or self._hpx is None or self._link_guard: return
        self._link_guard = True
        try:
            wpx, hpx = self._wpx, self._hpx
            P  = self.parse_float(self.pixel_mm_var.get())
            CW = self.parse_float(self.canvas_w_var.get())
            CH = self.parse_float(self.canvas_h_var.get())
            if source == "pixel" and P:
                self.canvas_w_var.set(f"{wpx*P:.3f}")
                self.canvas_h_var.set(f"{hpx*P:.3f}")
            elif source == "cw" and CW:
                P = CW / wpx; self.pixel_mm_var.set(f"{P:.3f}")
                self.canvas_h_var.set(f"{hpx*P:.3f}")
            elif source == "ch" and CH:
                P = CH / hpx; self.pixel_mm_var.set(f"{P:.3f}")
                self.canvas_w_var.set(f"{wpx*P:.3f}")
            P_final = self.parse_float(self.pixel_mm_var.get())
            if P_final:
                self.size_label.config(text=f"Image: {wpx}×{hpx} px  |  Pixel: {P_final:.4g} mm/px  |  Output: {wpx*P_final:.2f} × {hpx*P_final:.2f} mm")
        finally:
            self._link_guard = False

    # actions
    def scan_folder(self):
        folder = Path(self.folder_entry.get()).expanduser()
        if not folder.exists() or not folder.is_dir():
            messagebox.showwarning("Folder not found", f"{folder} is not a directory."); return
        self.folder = str(folder)
        files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]
        self.files = {p.name: p for p in files}; names = list(self.files.keys())
        self.file_combo["values"] = names
        if names:
            self.file_combo.current(0); self.on_select_file()
            self.status.config(text=f"Found {len(names)} image(s) in {folder}")
        else:
            self.selected_path = None; self.status.config(text=f"No images found in {folder}"); self.clear_previews()

    def on_select_file(self, *_):
        name = self.file_combo.get(); self.selected_path = self.files.get(name)
        if self.selected_path:
            self.status.config(text=f"Selected: {self.selected_path}")
            gray = load_gray_u8(str(self.selected_path))
            self._hpx, self._wpx = gray.shape
            self._scale_from("pixel")
            if self.auto_process.get(): self.on_process()

    def on_process(self):
        if not self.selected_path:
            messagebox.showwarning("No file", "Pick an image from the File dropdown."); return
        P     = self.parse_float(self.pixel_mm_var.get()) or 1.0
        nib   = self.parse_float(self.nib_mm_var.get()) or 0.3
        edge  = self._edge_mm()
        zz    = bool(self.zigzag.get())
        dotmm = self.parse_float(self.dot_mm_var.get()) or 0.005
        if nib > P and self.mode.get() != "Dots":
            self.status.config(text=f"Note: nib {nib:.3g} mm > pixel {P:.3g} mm — some pixels may fit no hatch/rings.")
        try:
            self.status.config(text="Processing… (writing BMP+SVG into /out/<name>/bmps and /svgs/<mode>)"); self.update_idletasks()
            results, wpx, hpx, base_dir = process_image_file(
                str(self.selected_path),
                mode=self.mode.get(),
                pixel_mm=P, nib_mm=nib,
                pen_lift=self.nested_pen_lift.get(),
                edge_mm=edge, zigzag=zz, dot_mm=dotmm
            )
            self.update_previews(results)
            self.size_label.config(text=f"Image: {wpx}×{hpx} px  |  Pixel: {P:.4g} mm  |  Nib: {nib:.4g} mm  |  Edge: {edge:.4g} mm  |  Dot: {dotmm:.4g} mm  |  Zig-zag: {zz}  |  Output: {wpx*P:.2f} × {hpx*P:.2f} mm  |  Dir: {base_dir.resolve()}")
            self.status.config(text=f"Done. Outputs in {base_dir.resolve()}")
        except Exception as e:
            messagebox.showerror("Error", str(e)); self.status.config(text=f"Error: {e}")

    def clear_previews(self):
        for name in TONE_DISPLAY:
            self.mask_labels[name].config(image="", text="")
            self.svg_labels[name].config(image="", text="")
        self.photo_refs.clear()

    def update_previews(self, results):
        self.photo_refs.clear()
        for item in results:
            name = item["name"]  # deep/mid/light/white
            # mask
            mask_img = Image.open(item["mask_path"]).convert("L")
            mask_img = _scale_to_max(mask_img, PREVIEW_TARGET_PX, resample=Image.NEAREST)
            mask_photo = ImageTk.PhotoImage(mask_img)
            self.mask_labels[name].config(image=mask_photo); self.photo_refs.append(mask_photo)
            # svg
            try:
                img = render_svg_to_pillow(item["svg_path"])
                svg_photo = ImageTk.PhotoImage(img)
                self.svg_labels[name].config(image=svg_photo); self.photo_refs.append(svg_photo)
            except Exception as e:
                placeholder = Image.new("L", (520, 220), 255)
                ImageDraw.Draw(placeholder).multiline_text((10, 10),
                    "SVG preview unavailable.\nInstall: pip install cairosvg\n\n"+str(e), fill=0, spacing=4)
                svg_photo = ImageTk.PhotoImage(placeholder.convert("RGB"))
                self.svg_labels[name].config(image=svg_photo); self.photo_refs.append(svg_photo)

if __name__ == "__main__":
    App().mainloop()
