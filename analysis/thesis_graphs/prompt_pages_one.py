from __future__ import annotations
import json, re, textwrap, base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from importlib import resources


# ---------- styling ----------
def _apply_style():
    try:
        with resources.as_file(
            resources.files(__package__).joinpath("thesis.mplstyle")
        ) as p:
            plt.style.use(str(p))
    except Exception:
        pass


# ---------- small helpers ----------
def _wrap(s: str | None, width: int = 76) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n")
    out: List[str] = []
    for para in s.split("\n\n"):
        lines: List[str] = []
        for line in para.splitlines():
            lines.extend(textwrap.wrap(line, width=width) or [""])
        out.append("\n".join(lines))
    return "\n\n".join(out)


def _decode_data_url_to_image(url: str) -> Optional[Image.Image]:
    try:
        if not (isinstance(url, str) and url.startswith("data:")):
            return None
        b64 = url.split(",", 1)[1]
        data = base64.b64decode(b64)
        return Image.open(BytesIO(data)).convert("RGBA")
    except Exception:
        return None


def _normalize_user_to_text_and_image(
    user_content_text: str,
) -> Tuple[str, Optional[Image.Image]]:
    """
    User content is a JSON list like:
      [{"type":"text","text":...}, {"type":"image_url","image_url":{"url": "..."}}, ...]
    Return (concatenated text, first decoded image if the URL is a base64 data URL).
    We do *not* fetch remote HTTP(S) URLs.
    """
    if not isinstance(user_content_text, str):
        return "", None
    s = user_content_text.strip()
    if not s.startswith("["):  # plain text
        return s, None

    try:
        parts = json.loads(s)
    except Exception:
        return s, None

    texts: List[str] = []
    first_img: Optional[Image.Image] = None

    for p in parts:
        if not isinstance(p, dict):
            continue
        t = p.get("type")
        if t == "text":
            txt = p.get("text", "")
            if txt:
                texts.append(txt)
        elif t in {"image_url", "input_image", "image"}:
            # schema uses {"type":"image_url","image_url":{"url": media}}
            img_url = None
            if "image_url" in p and isinstance(p["image_url"], dict):
                img_url = p["image_url"].get("url")
            elif "url" in p:
                img_url = p.get("url")
            if isinstance(img_url, str) and img_url.startswith("data:"):
                first_img = first_img or _decode_data_url_to_image(img_url)
            # (ignore http(s) urls)

    return "\n".join(t for t in texts if t).strip(), first_img


# ---------- base modality selection ----------
def _detect_exp_id_column(df: pd.DataFrame) -> Optional[str]:
    for c in ("exp_id", "experiment_id", "experiment", "run_id"):
        if c in df.columns:
            return c
    return None


def _infer_exp_id_from_path(row: pd.Series) -> Optional[str]:
    # e.g. ".../runs/base_text/..." or "... base_3_text_image ..."
    val = str(row.get("result_file", "")) or str(row.get("csv_path", ""))
    m = re.search(r"(base(?:_\d+)?)_(text_image|text|image)", val)
    return m.group(0) if m else None


def _filter_base(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    col = _detect_exp_id_column(df)
    key = f"base_{modality}"
    if col is not None:
        sub = df[df[col] == key]
        if not sub.empty:
            return sub.copy()
    tmp = df.copy()
    tmp["_exp_guess"] = tmp.apply(_infer_exp_id_from_path, axis=1)
    sub = tmp[tmp["_exp_guess"] == key].drop(columns=["_exp_guess"])
    return sub.copy()


# ---------- pick one interaction ----------
def _pick_first_interaction(
    df_mod: pd.DataFrame,
) -> Tuple[str, str, Optional[Image.Image]]:
    """
    From a modality-filtered DataFrame, pick the *first* sample_id,
    return (system_text, user_text, user_image).
    """
    # robust column names
    sample_col = "sample_id" if "sample_id" in df_mod.columns else None
    role_col = "role"
    text_col = "content_text"
    turn_col = "turn_idx" if "turn_idx" in df_mod.columns else None

    if (
        sample_col is None
        or role_col not in df_mod.columns
        or text_col not in df_mod.columns
    ):
        return "", "", None

    # stable order
    if turn_col:
        df_mod = df_mod.sort_values([sample_col, turn_col])
    else:
        df_mod = df_mod.sort_values([sample_col])

    sid = pd.unique(df_mod[sample_col]).tolist()
    if not sid:
        return "", "", None
    g = df_mod[df_mod[sample_col] == sid[0]]

    sys_series = g[g[role_col] == "system"][text_col].dropna()
    usr_series = g[g[role_col] == "user"][text_col].dropna()

    sys_txt = sys_series.iloc[0].strip() if not sys_series.empty else ""
    usr_raw = usr_series.iloc[0] if not usr_series.empty else ""
    usr_text, usr_img = _normalize_user_to_text_and_image(usr_raw)
    return sys_txt, usr_text, usr_img


# ---------- rendering ----------
from matplotlib.patches import FancyBboxPatch


def _draw_single_page(ax, system_text: str, user_text: str, user_img, wrap: int):
    """
    Minimal layout: headings + monospaced text. If an image is present,
    it goes on the right. No boxes/frames.
    """
    ax.set_axis_off()

    # areas in axes coords (x, y, w, h)
    sys_area = (0.02, 0.62, 0.96, 0.28)
    usr_area = (0.02, 0.14, 0.52 if user_img is not None else 0.96, 0.32)
    img_area = (0.56, 0.14, 0.42, 0.32)  # only used if user_img exists

    # headings
    ax.text(
        sys_area[0],
        sys_area[1] + sys_area[3] + 0.02,
        "System",
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        usr_area[0],
        usr_area[1] + usr_area[3] + 0.02,
        "User",
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        transform=ax.transAxes,
    )

    # text blocks (selectable, no bbox)
    ax.text(
        sys_area[0],
        sys_area[1] + sys_area[3],
        _wrap(system_text, wrap) if system_text else "(missing)",
        ha="left",
        va="top",
        family="monospace",
        fontsize=9,
        transform=ax.transAxes,
    )

    ax.text(
        usr_area[0],
        usr_area[1] + usr_area[3],
        _wrap(user_text, wrap) if user_text else "(missing)",
        ha="left",
        va="top",
        family="monospace",
        fontsize=9,
        transform=ax.transAxes,
    )

    # optional image on the right
    if user_img is not None:
        try:
            ax_img = ax.inset_axes(list(img_area))  # x, y, w, h (axes fraction)
            ax_img.imshow(user_img)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_frame_on(False)
            ax_img.set_title("User image", fontsize=8)
        except Exception:
            pass


# ---------- public API ----------
def make_prompt_pages_from_master_one_interaction(
    master_parquet: str | Path,
    out_pdf: str | Path,
    *,
    title_prefix: str = "Base prompts",
    wrap: int = 76,
):
    """
    Builds a 3-page PDF from a single master turns parquet:
      Page 1 : Text      (base_text)      — first system+user turn, image omitted
      Page 2 : Image     (base_image)     — first system+user turn, render image if present
      Page 3 : Text+Image(base_text_image)— first system+user turn, render image if present
    """
    _apply_style()
    df = pd.read_parquet(master_parquet)

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    modalities = [("text", "Text"), ("image", "Image"), ("text_image", "Text+Image")]

    with PdfPages(out_pdf) as pdf:
        for mod, label in modalities:
            sub = _filter_base(df, mod)
            sys_txt, usr_txt, usr_img = _pick_first_interaction(sub)

            fig, ax = plt.subplots(1, 1, figsize=(10.8, 6.4), constrained_layout=True)
            fig.suptitle(f"{title_prefix}: {label}", y=0.99)
            _draw_single_page(ax, sys_txt, usr_txt, usr_img, wrap)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
