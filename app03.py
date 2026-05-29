"""
╔══════════════════════════════════════════════════════════════════════╗
║  ระบบวิเคราะห์คำร้องเรียนอัจฉริยะ                                    ║
║  Smart City Complaint Intelligence Platform                           ║
║  เทศบาลนครขอนแก่น  ·  Production Version 3.0                         ║
╚══════════════════════════════════════════════════════════════════════╝

Architecture:
  config.py   → constants, color palette, CSS
  data.py     → loading, cleaning, feature engineering
  models.py   → NLP tokenizer, ML classifiers
  charts.py   → reusable Plotly chart builders
  pages/      → one function per dashboard page
  app.py      → wiring: sidebar → page router

Usage:
  pip install -r requirements.txt
  streamlit run app.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

# ── Map page module ────────────────────────────────────────────────────────────
# Gracefully import the geospatial command center
try:
    from map_page import render_map_page as _render_map_page
    _MAP_PAGE_AVAILABLE = True
except ImportError:
    _MAP_PAGE_AVAILABLE = False

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ── DESIGN TOKENS & GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Palette ───────────────────────────────────────────────────────────────────
C = dict(
    primary   = "#0057FF",   # electric indigo-blue  (action, brand)
    secondary = "#00C2CB",   # teal                  (data accent)
    violet    = "#7B2FFF",   # violet                (AI / predictive)
    success   = "#00B37E",   # emerald
    warning   = "#E8960C",   # amber
    danger    = "#E5484D",   # red
    neutral   = "#64748B",   # slate
    # Backgrounds
    bg_page   = "#F0F4F8",   # cool near-white
    bg_card   = "#FFFFFF",
    bg_muted  = "#F8FAFC",
    bg_sidebar= "#FFFFFF",
    # Typography
    tx_dark   = "#0D1117",
    tx_mid    = "#334155",
    tx_soft   = "#94A3B8",
    tx_xsoft  = "#CBD5E1",
    # Borders
    border    = "#E2E8F0",
    border_focus = "#0057FF",
)

# Sequential scales for Plotly
SCALE_BLUE   = [[0, "#EFF6FF"], [1, C["primary"]]]
SCALE_TEAL   = [[0, "#ECFEFF"], [1, C["secondary"]]]
SCALE_VIOLET = [[0, "#F5F3FF"], [1, C["violet"]]]
SCALE_HEAT   = [[0, "#FEE2E2"], [0.5, "#FEF3C7"], [1, "#D1FAE5"]]
PALETTE      = [C["primary"], C["secondary"], C["violet"], C["success"],
                C["warning"], "#F59E0B", "#EC4899", "#14B8A6"]

# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(f"""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Noto+Sans+Thai:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {{
  font-family: 'Plus Jakarta Sans', 'Noto Sans Thai', sans-serif;
}}
.main .block-container {{
  background: {C['bg_page']};
  padding: 1.75rem 2.5rem 4rem;
  max-width: 1480px;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
  background: {C['bg_sidebar']};
  border-right: 1px solid {C['border']};
}}
[data-testid="stSidebar"] section {{
  padding-top: 0;
}}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}

/* ── KPI Card ── */
.kpi {{
  background: {C['bg_card']};
  border: 1px solid {C['border']};
  border-radius: 16px;
  padding: 1.5rem 1.6rem 1.4rem;
  position: relative;
  overflow: hidden;
  transition: box-shadow .18s ease, transform .18s ease;
  cursor: default;
}}
.kpi:hover {{
  box-shadow: 0 8px 32px rgba(0,87,255,.10);
  transform: translateY(-2px);
}}
.kpi::after {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--kc, {C['primary']});
  border-radius: 16px 16px 0 0;
}}
.kpi-icon  {{ font-size: 1.5rem; margin-bottom: .5rem; opacity: .85; }}
.kpi-val   {{
  font-size: 2.5rem; font-weight: 800; line-height: 1;
  color: {C['tx_dark']}; font-variant-numeric: tabular-nums;
  letter-spacing: -.02em;
}}
.kpi-lbl   {{
  font-size: .72rem; font-weight: 700; letter-spacing: .09em;
  text-transform: uppercase; color: {C['tx_soft']};
  margin-top: .4rem;
}}
.kpi-sub   {{ font-size: .82rem; color: {C['tx_mid']}; margin-top: .5rem; }}
.kpi-bg    {{
  position: absolute; right: -10px; bottom: -10px;
  font-size: 4.5rem; opacity: .06; pointer-events: none;
  font-variant: normal;
}}

/* ── Section header ── */
.sec {{
  display: flex; align-items: center; gap: .65rem;
  margin: 2rem 0 1.25rem;
  padding-bottom: .8rem;
  border-bottom: 2px solid {C['border']};
}}
.sec h2 {{
  font-size: 1.2rem; font-weight: 800;
  color: {C['tx_dark']}; margin: 0; letter-spacing: -.01em;
}}
.sec-badge {{
  font-size: .68rem; font-weight: 700;
  background: {C['bg_muted']}; color: {C['primary']};
  border: 1px solid {C['border']}; border-radius: 20px;
  padding: .15rem .65rem; text-transform: uppercase; letter-spacing: .07em;
}}

/* ── Chart card ── */
.cc {{
  background: {C['bg_card']};
  border: 1px solid {C['border']};
  border-radius: 16px;
  padding: 1.3rem 1.5rem 1rem;
  margin-bottom: 1.2rem;
}}
.cc-title {{
  font-size: .9rem; font-weight: 700;
  color: {C['tx_dark']}; margin: 0 0 .15rem;
}}
.cc-sub {{
  font-size: .75rem; color: {C['tx_soft']}; margin-bottom: .75rem;
}}

/* ── Alert / info banners ── */
.banner {{
  border-radius: 10px; padding: .75rem 1.1rem;
  font-size: .85rem; margin: .5rem 0 1rem;
  display: flex; align-items: flex-start; gap: .55rem;
}}
.banner-info    {{ background:#EFF6FF; border:1px solid #BFDBFE; color:#1E40AF; }}
.banner-success {{ background:#F0FDF4; border:1px solid #BBF7D0; color:#065F46; }}
.banner-warning {{ background:#FFFBEB; border:1px solid #FDE68A; color:#92400E; }}
.banner-danger  {{ background:#FEF2F2; border:1px solid #FECACA; color:#991B1B; }}

/* ── Tag / badge ── */
.tag {{
  display:inline-block; padding:.2rem .7rem;
  border-radius:20px; font-size:.75rem; font-weight:700;
  font-family:'Plus Jakarta Sans',sans-serif;
}}
.tag-blue   {{ background:#DBEAFE; color:#1E40AF; }}
.tag-teal   {{ background:#CCFBF1; color:#0F766E; }}
.tag-violet {{ background:#EDE9FE; color:#5B21B6; }}
.tag-green  {{ background:#D1FAE5; color:#065F46; }}
.tag-amber  {{ background:#FEF3C7; color:#92400E; }}
.tag-red    {{ background:#FEE2E2; color:#991B1B; }}
.tag-gray   {{ background:#F1F5F9; color:#475569; }}

/* ── AI result card ── */
.ai-card {{
  background: {C['bg_card']};
  border: 1px solid {C['border']};
  border-radius: 14px;
  padding: 1.1rem 1.4rem;
}}
.ai-card-label {{
  font-size: .68rem; font-weight: 700; letter-spacing: .09em;
  text-transform: uppercase; color: {C['tx_soft']}; margin-bottom: .35rem;
}}
.ai-card-value {{
  font-size: 1.15rem; font-weight: 700; color: {C['tx_dark']};
}}
.ai-card-sub {{ font-size: .78rem; color: {C['tx_mid']}; margin-top: .25rem; }}

/* ── Routing card ── */
.routing-card {{
  background: linear-gradient(135deg,#EFF6FF 0%,#F5F3FF 100%);
  border: 1.5px solid #C7D2FE;
  border-radius: 16px;
  padding: 1.4rem 1.8rem;
  margin-top: 1rem;
}}

/* ── Data table tweaks ── */
[data-testid="stDataFrame"] thead th {{
  background: {C['bg_muted']} !important;
  font-size: .75rem !important; font-weight: 700 !important;
  text-transform: uppercase; letter-spacing: .06em;
  color: {C['tx_soft']} !important;
}}
[data-testid="stDataFrame"] tbody td {{
  font-size: .85rem; color: {C['tx_dark']};
}}

/* ── Sidebar nav radio ── */
[data-testid="stSidebar"] .stRadio > label {{
  font-size: .88rem; color: {C['tx_mid']};
}}
[data-testid="stSidebar"] .stRadio > div > label {{
  padding: .4rem .6rem !important;
  border-radius: 8px !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {C['border']}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ── LAYOUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def kpi_card(icon, value, label, sub="", color=None):
    color = color or C["primary"]
    st.markdown(f"""
    <div class="kpi" style="--kc:{color}">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-val">{value}</div>
      <div class="kpi-lbl">{label}</div>
      {"<div class='kpi-sub'>"+sub+"</div>" if sub else ""}
      <div class="kpi-bg">{icon}</div>
    </div>""", unsafe_allow_html=True)


def section_header(icon, title, badge=""):
    b = f"<span class='sec-badge'>{badge}</span>" if badge else ""
    st.markdown(f"""
    <div class="sec">
      <span style="font-size:1.25rem">{icon}</span>
      <h2>{title}</h2>{b}
    </div>""", unsafe_allow_html=True)


def chart_card(title="", sub=""):
    """Context manager-style: call open, yield, call close manually."""
    hdr = (f"<div class='cc-title'>{title}</div>" if title else "") + \
          (f"<div class='cc-sub'>{sub}</div>" if sub else "")
    st.markdown(f"<div class='cc'>{hdr}", unsafe_allow_html=True)


def chart_card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def banner(msg, kind="info"):
    icons = {"info":"ℹ️","success":"✅","warning":"⚠️","danger":"🚨"}
    st.markdown(f"<div class='banner banner-{kind}'>{icons[kind]} {msg}</div>",
                unsafe_allow_html=True)


def spacer(h=1):
    st.markdown(f"<div style='margin:{h}rem 0'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ── CHART BUILDERS  (pure functions, return fig)
# ═══════════════════════════════════════════════════════════════════════════════

_BASE = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(family="'Plus Jakarta Sans','Noto Sans Thai',sans-serif",
                         color=C["tx_mid"], size=12),
    margin        = dict(t=40, b=20, l=10, r=10),
    title_font    = dict(size=14, color=C["tx_dark"], family="'Plus Jakarta Sans',sans-serif"),
    hoverlabel    = dict(bgcolor="white", bordercolor=C["border"],
                         font=dict(size=12, color=C["tx_dark"])),
    legend        = dict(font=dict(size=11, color=C["tx_mid"]),
                         bgcolor="rgba(0,0,0,0)"),
)

_AX = dict(
    showgrid=True, gridcolor=C["bg_page"], gridwidth=1,
    showline=False, zeroline=False,
    tickfont=dict(size=11, color=C["tx_mid"]),
)


def _style(fig, h=380):
    fig.update_layout(**_BASE, height=h)
    fig.update_xaxes(**_AX)
    fig.update_yaxes(**_AX)
    return fig


def hbar(df, x, y, title="", color_scale=None, h=380, text_fmt=None):
    """Generic sorted horizontal bar chart."""
    scale = color_scale or SCALE_BLUE
    df = df.sort_values(x, ascending=True)
    texts = df[x].apply(text_fmt) if text_fmt else df[x].astype(str)
    fig = px.bar(df, x=x, y=y, orientation="h",
                 color=x, color_continuous_scale=scale,
                 text=texts,
                 labels={x: "", y: ""})
    fig.update_traces(textposition="outside", textfont_size=11,
                      marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(title=title)
    return _style(fig, h)


def trend_line(df, x, y, title="", h=340):
    ma = df[y].rolling(3, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[x], y=df[y], name="จำนวนคำร้อง",
        marker_color=C["secondary"], marker_opacity=.55,
        hovertemplate="%{y} รายการ<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df[x], y=ma, name="ค่าเฉลี่ย 3 เดือน",
        mode="lines", line=dict(color=C["primary"], width=2.5),
        hovertemplate="MA: %{y:.0f}<extra></extra>",
    ))
    fig.update_layout(title=title,
                      barmode="overlay", bargap=.25,
                      legend=dict(orientation="h", y=1.08, x=0))
    return _style(fig, h)


def sla_bar(dept_df, h=400):
    """Coloured horizontal bar: green=fast, amber=ok, red=slow."""
    df = dept_df.sort_values("mean", ascending=True)
    colors = df["mean"].apply(
        lambda v: C["success"] if v <= 7 else (C["warning"] if v <= 14 else C["danger"])
    )
    fig = go.Figure(go.Bar(
        x=df["mean"], y=df["department"],
        orientation="h",
        marker_color=colors,
        text=df["mean"].apply(lambda v: f"{v:.1f} วัน"),
        textposition="outside",
        textfont_size=11,
        hovertemplate="<b>%{y}</b><br>เฉลี่ย: %{x:.1f} วัน<extra></extra>",
    ))
    fig.add_vline(x=7, line_dash="dot", line_color=C["success"],
                  annotation_text="เป้าหมาย 7 วัน",
                  annotation_font=dict(color=C["success"], size=11),
                  annotation_position="top right")
    return _style(fig, h)


def heatmap_chart(pivot, title="", h=360):
    fig = px.imshow(
        pivot,
        color_continuous_scale=[[0, C["bg_muted"]], [0.5, C["secondary"]], [1, C["primary"]]],
        text_auto=True, aspect="auto",
        labels={"color": "จำนวน"},
    )
    fig.update_traces(textfont_size=11)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(title=title)
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=11))
    return _style(fig, h)


def treemap_chart(df, path, values, title="", h=460):
    fig = px.treemap(
        df, path=path, values=values,
        color=values,
        color_continuous_scale=[[0, "#EFF6FF"], [0.5, C["secondary"]], [1, C["primary"]]],
    )
    fig.update_traces(textfont_size=13, textinfo="label+value",
                      marker_line_width=2, marker_line_color="white")
    fig.update_coloraxes(showscale=False)
    fig.update_layout(title=title)
    return _style(fig, h)


def histogram_chart(series, title="", h=360):
    fig = px.histogram(series, nbins=28,
                       color_discrete_sequence=[C["secondary"]],
                       opacity=.8,
                       labels={"value": "วัน", "count": "จำนวน"})
    med = series.median()
    fig.add_vline(x=med, line_dash="dot", line_color=C["violet"],
                  annotation_text=f"Median {med:.0f} วัน",
                  annotation_font=dict(color=C["violet"], size=11))
    fig.update_layout(title=title, showlegend=False)
    return _style(fig, h)


def map_bubble(df, lat, lon, size, color, hover_name, title="", h=440):
    fig = px.scatter_mapbox(
        df, lat=lat, lon=lon,
        size=size, color=color,
        color_continuous_scale=[[0,"#FEE2E2"],[0.5,"#FEF3C7"],[1,"#D1FAE5"]],
        size_max=55, zoom=12.2,
        mapbox_style="carto-positron",
        hover_name=hover_name,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0,b=0,l=0,r=0),
        height=h,
        coloraxis_colorbar=dict(title="Complete %", tickformat=".0f"),
    )
    return fig


def confidence_bar(labels, values, h=280):
    df = pd.DataFrame({"label": labels, "prob": [v*100 for v in values]})
    df = df.sort_values("prob", ascending=True)
    colors = [C["primary"] if i == len(df)-1 else C["bg_page"] for i in range(len(df))]
    border = [C["primary"] if i == len(df)-1 else C["border"] for i in range(len(df))]
    fig = go.Figure(go.Bar(
        x=df["prob"], y=df["label"], orientation="h",
        marker_color=colors,
        marker_line_color=border, marker_line_width=1.5,
        text=df["prob"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        textfont=dict(size=11,
                      color=[C["primary"] if i==len(df)-1 else C["tx_soft"]
                             for i in range(len(df))]),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 115])
    return _style(fig, h)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ── DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_clean(file_obj=None) -> pd.DataFrame:
    """Load Excel, normalise columns, parse Thai dates, engineer features."""

    # ── Load ──────────────────────────────────────────────────────────────────
    if file_obj is not None:
        raw = pd.read_excel(file_obj)
    else:
        candidates = [
            "ข้อมูลคำร้อง_sampled.xlsx",
            "ข_อม_ลคำร_อง_sampled__1_.xlsx",
            "data.xlsx",
        ]
        for p in candidates:
            if os.path.exists(p):
                raw = pd.read_excel(p)
                break
        else:
            st.error("❌ ไม่พบไฟล์ข้อมูล — กรุณาอัปโหลดไฟล์ Excel ในแถบซ้าย")
            st.stop()

    # ── Rename ────────────────────────────────────────────────────────────────
    rename = {
        "ส่วนงาน":        "division",
        "ฝ่าย":           "department",
        "เลขคำร้อง":      "cid",
        "เรื่องร้องทุกข์": "text",
        "ประเภทคำร้อง":   "category",
        "เขต":            "district",
        "ชุมชน":          "community",
        "วันที่รับเรื่อง": "received",
        "วันที่เสร็จ":    "completed",
        "สถานะ":          "status",
    }
    df = raw.rename(columns=rename)

    # ── Thai Buddhist calendar date parser ────────────────────────────────────
    def _parse(s):
        if pd.isna(s) or str(s).strip() in ("", "nan"):
            return pd.NaT
        try:
            p = str(s).strip().split("/")
            d, m, y = int(p[0]), int(p[1]), int(p[2])
            if y > 2500:
                y -= 543
            return pd.Timestamp(y, m, d)
        except Exception:
            return pd.NaT

    df["received"]  = df["received"].apply(_parse)
    df["completed"] = df["completed"].apply(_parse)

    # ── Derived columns ───────────────────────────────────────────────────────
    df["days"]    = (df["completed"] - df["received"]).dt.days.clip(lower=0)
    df["done"]    = df["status"].str.contains("เสร็จ", na=False)
    df["month"]   = df["received"].dt.to_period("M").astype(str)
    df["year"]    = df["received"].dt.year
    df["weekday"] = df["received"].dt.day_name()
    df["district"] = df["district"].replace(
        {"ไม่ระบุ": "ไม่ระบุเขต", "อาคารเขต 7": "เขต 7"}
    )

    # ── Priority scoring (keyword rule-based) ─────────────────────────────────
    _hi  = ["ไฟดับ","น้ำท่วม","ถนนพัง","ท่อแตก","อุบัติเหตุ","ฉุกเฉิน","ชำรุด","แตก","พัง"]
    _med = ["ซ่อม","ท่อระบาย","กิ่งไม้","ไฟฟ้า","จราจร","อุดตัน"]

    def _priority(t):
        t = str(t).lower()
        if any(k in t for k in _hi):  return "🔴 สูง"
        if any(k in t for k in _med): return "🟡 กลาง"
        return "🟢 ต่ำ"

    # ── Sentiment scoring ─────────────────────────────────────────────────────
    _neg = ["เสีย","พัง","ดับ","ชำรุด","อันตราย","แตก","เน่า","กีดขวาง","ท่วม","บาดเจ็บ"]

    def _sentiment(t):
        sc = sum(1 for k in _neg if k in str(t).lower())
        return "เชิงลบมาก" if sc >= 2 else ("เชิงลบ" if sc == 1 else "เป็นกลาง")

    df["priority"]  = df["text"].apply(_priority)
    df["sentiment"] = df["text"].apply(_sentiment)

    # ── Text correction dictionary (abbreviations → full words) ───────────────
    _corrections = {
        "ถ."   : "ถนน",   "ซ."  : "ซอย",   "ต."  : "ตำบล",
        "อ."   : "อำเภอ", "จ."  : "จังหวัด",
        "ไฟฝ้า": "ไฟฟ้า", "ทอ"  : "ท่อ",   "รถไฟ": "ไฟฟ้า",
    }
    def _correct(t):
        for abbr, full in _corrections.items():
            t = str(t).replace(abbr, full)
        return t

    df["text_clean"] = df["text"].apply(_correct)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 ── NLP & ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════

_STOPWORDS = {
    "ของ","ใน","ที่","และ","มี","ให้","กับ","ได้","จาก","เป็น",
    "ไม่","มา","แล้ว","จะ","ก็","นั้น","อยู่","ต้อง","ว่า","ไป",
    "ๆ","nan","","ใน","บน","ตาม","โดย","กา","การ","ราย","นะ","ครับ","ค่ะ",
}


@st.cache_resource(show_spinner=False)
def get_tokenizer():
    try:
        from pythainlp.tokenize import word_tokenize
        def _tok(text: str) -> list[str]:
            tokens = word_tokenize(str(text), engine="newmm", keep_whitespace=False)
            return [t.strip() for t in tokens if t.strip() and len(t) > 1
                    and t.strip() not in _STOPWORDS]
        return _tok
    except ImportError:
        return None


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    """Train category + department classifiers. Returns (cat_pipe, dept_pipe, acc, tok_fn)."""
    tok_fn = get_tokenizer()
    if tok_fn is None:
        return None, None, 0.0, None

    try:
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        df2 = df.dropna(subset=["text_clean","category","department"]).copy()
        df2 = df2[df2["category"].map(df2["category"].value_counts()) >= 3]

        def _tok_str(t): return " ".join(tok_fn(t))
        X  = df2["text_clean"].apply(_tok_str)
        yc = df2["category"]
        yd = df2["department"]

        Xtr,Xte,yc_tr,yc_te,yd_tr,yd_te = train_test_split(
            X, yc, yd, test_size=.2, random_state=42, stratify=yc
        )

        def _pipe():
            return Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1,2), max_features=8000,
                    sublinear_tf=True, min_df=2,
                )),
                ("clf", LogisticRegression(
                    max_iter=600, C=1.5,
                    class_weight="balanced", solver="lbfgs",
                )),
            ])

        cp = _pipe(); cp.fit(Xtr, yc_tr)
        dp = _pipe(); dp.fit(Xtr, yd_tr)
        acc = accuracy_score(yc_te, cp.predict(Xte))

        return cp, dp, acc, lambda t: " ".join(tok_fn(t))

    except ImportError:
        return None, None, 0.0, None


@st.cache_data(show_spinner=False)
def get_token_stats(texts: tuple) -> dict:
    tok_fn = get_tokenizer()
    if tok_fn is None:
        return {}
    all_tok = []
    for t in texts:
        all_tok.extend(tok_fn(str(t)))
    freq = Counter(all_tok)
    bigrams = Counter(
        all_tok[i]+" "+all_tok[i+1]
        for i in range(len(all_tok)-1)
    )
    return {"freq": freq, "bigrams": bigrams, "tokens": all_tok}


def predict_days(cat, dept, df_done: pd.DataFrame) -> int:
    """Historical median lookup with fallback."""
    m1 = df_done[(df_done["category"]==cat) & df_done["days"].notna()]
    if len(m1) >= 3:
        return int(m1["days"].median())
    m2 = df_done[(df_done["department"]==dept) & df_done["days"].notna()]
    if len(m2) >= 3:
        return int(m2["days"].median())
    return int(df_done["days"].median()) if len(df_done) else 7


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 ── PAGE RENDERERS
# ═══════════════════════════════════════════════════════════════════════════════

def page_overview(df: pd.DataFrame):
    # ── Title ──────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-bottom:1.75rem">
      <h1 style="font-size:1.8rem;font-weight:800;color:{C['tx_dark']};
                 margin:0;letter-spacing:-.02em">
        ภาพรวมระบบคำร้องเรียน
      </h1>
      <p style="font-size:.9rem;color:{C['tx_mid']};margin:.25rem 0 0">
        เทศบาลนครขอนแก่น &nbsp;·&nbsp; Smart City Complaint Intelligence Platform
      </p>
    </div>
    """, unsafe_allow_html=True)

    df_done = df[df["done"] & df["days"].notna() & (df["days"] >= 0)]
    avg_sla  = df_done["days"].mean() if len(df_done) else 0
    pending  = int((~df["done"]).sum())
    rate     = df["done"].mean() * 100
    slowest  = (df_done.groupby("department")["days"].mean()
                .idxmax() if len(df_done) else "—")
    top_cat  = df["category"].value_counts().index[0] if len(df) else "—"

    # ── KPI row ────────────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: kpi_card("📋", f"{len(df):,}", "คำร้องทั้งหมด", color=C["primary"])
    with c2: kpi_card("✅", f"{df['done'].sum():,}", "ดำเนินการเสร็จ", color=C["success"])
    with c3: kpi_card("⏳", f"{pending:,}", "รอดำเนินการ", color=C["warning"])
    with c4: kpi_card("⏱️", f"{avg_sla:.1f}", "เฉลี่ย SLA (วัน)", color=C["secondary"])
    with c5: kpi_card("🏆", (top_cat[:12]+"…") if len(top_cat)>12 else top_cat,
                      "ประเภทที่พบมากสุด", color=C["violet"])
    with c6: kpi_card("🚦", f"{rate:.0f}%", "Completion Rate",
                      color=C["success"] if rate>=90 else C["warning"])

    spacer()

    # ── Row: Trend + Status ────────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        chart_card("📈 แนวโน้มคำร้องรายเดือน",
                   "จำนวนคำร้องที่เข้ามาแต่ละเดือน พร้อมเส้นค่าเฉลี่ย 3 เดือน")
        trend = df.groupby("month").size().reset_index(name="count").sort_values("month")
        st.plotly_chart(trend_line(trend,"month","count"), use_container_width=True)
        chart_card_close()

    with right:
        chart_card("📋 สถานะคำร้อง", "การกระจายตามสถานะปัจจุบัน")
        st_df = (df["status"].value_counts().reset_index()
                   .rename(columns={"status":"s","count":"c"})
                   .sort_values("c"))
        fig = hbar(st_df, "c", "s", color_scale=SCALE_BLUE, h=340)
        st.plotly_chart(fig, use_container_width=True)
        chart_card_close()

    # ── Row: Priority + Sentiment + Completion Rate ────────────────────────────
    c1,c2,c3 = st.columns(3, gap="large")

    with c1:
        chart_card("⚡ ระดับความสำคัญ", "จัดลำดับจากคำร้องเร่งด่วน")
        pr = df["priority"].value_counts().reset_index().rename(columns={"priority":"p","count":"c"})
        cm = {"🔴 สูง":C["danger"],"🟡 กลาง":C["warning"],"🟢 ต่ำ":C["success"]}
        fig = px.bar(pr.sort_values("c"), x="c", y="p", orientation="h",
                     color="p", color_discrete_map=cm,
                     text="c", labels={"c":"","p":""})
        fig.update_traces(textposition="outside", textfont_size=13, showlegend=False)
        fig.update_layout(**{**_BASE,"height":240,"showlegend":False})
        fig.update_xaxes(**_AX); fig.update_yaxes(**{**_AX,"tickfont":dict(size=13)})
        st.plotly_chart(fig, use_container_width=True)
        chart_card_close()

    with c2:
        chart_card("💬 Sentiment", "อารมณ์จากเนื้อหาคำร้อง")
        se = df["sentiment"].value_counts().reset_index().rename(columns={"sentiment":"s","count":"c"})
        sm = {"เชิงลบมาก":C["danger"],"เชิงลบ":C["warning"],"เป็นกลาง":C["success"]}
        fig = px.bar(se.sort_values("c"), x="c", y="s", orientation="h",
                     color="s", color_discrete_map=sm,
                     text="c", labels={"c":"","s":""})
        fig.update_traces(textposition="outside", textfont_size=13, showlegend=False)
        fig.update_layout(**{**_BASE,"height":240,"showlegend":False})
        fig.update_xaxes(**_AX); fig.update_yaxes(**{**_AX,"tickfont":dict(size=13)})
        st.plotly_chart(fig, use_container_width=True)
        chart_card_close()

    with c3:
        chart_card("✅ Completion Rate ตามเขต", "อัตราดำเนินการเสร็จแต่ละพื้นที่")
        cr = (df.groupby("district")
                .agg(total=("done","count"), done=("done","sum")).reset_index())
        cr["rate"] = (cr["done"] / cr["total"] * 100).round(1)
        cr = cr.sort_values("rate", ascending=True)
        fig = px.bar(cr, x="rate", y="district", orientation="h",
                     color="rate",
                     color_continuous_scale=[[0,"#FEE2E2"],[.5,"#FEF9C3"],[1,"#D1FAE5"]],
                     text=cr["rate"].apply(lambda v: f"{v:.0f}%"),
                     labels={"rate":"","district":""})
        fig.update_traces(textposition="outside", textfont_size=12)
        fig.update_coloraxes(showscale=False)
        fig.update_xaxes(range=[0,115])
        fig.update_layout(**{**_BASE,"height":240})
        fig.update_xaxes(**_AX); fig.update_yaxes(**{**_AX,"tickfont":dict(size=12)})
        st.plotly_chart(fig, use_container_width=True)
        chart_card_close()


# ─────────────────────────────────────────────────────────────────────────────
def page_complaints(df: pd.DataFrame):
    section_header("📊", "การวิเคราะห์ประเภทคำร้อง", f"{len(df):,} รายการ")

    # Top 15 bar
    chart_card("🏆 Top 15 ประเภทคำร้อง",
               "ประเภทที่ประชาชนร้องเรียนมากที่สุด — เรียงจากมากไปน้อย")
    top15 = (df["category"].value_counts().head(15)
               .reset_index().rename(columns={"category":"cat","count":"c"})
               .sort_values("c"))
    st.plotly_chart(hbar(top15,"c","cat",color_scale=SCALE_BLUE,h=520),
                    use_container_width=True)
    chart_card_close()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        chart_card("🏢 คำร้องตามฝ่าย", "จำนวนคำร้องที่แต่ละฝ่ายรับผิดชอบ")
        d = (df["department"].value_counts().reset_index()
               .rename(columns={"department":"dep","count":"c"})
               .sort_values("c"))
        st.plotly_chart(hbar(d,"c","dep",color_scale=SCALE_VIOLET,h=380),
                        use_container_width=True)
        chart_card_close()

    with col2:
        chart_card("📍 คำร้องตามเขต", "การกระจายตามพื้นที่")
        d = (df["district"].value_counts().reset_index()
               .rename(columns={"district":"dist","count":"c"})
               .sort_values("c"))
        st.plotly_chart(hbar(d,"c","dist",color_scale=SCALE_TEAL,h=380),
                        use_container_width=True)
        chart_card_close()

    # Treemap
    chart_card("🗂️ Treemap — ฝ่าย × ประเภทคำร้อง",
               "สัดส่วนคำร้องทุกมิติในแผนภาพเดียว")
    tm = df.groupby(["department","category"]).size().reset_index(name="n")
    fig_tm = treemap_chart(
        tm, [px.Constant("ทั้งหมด"),"department","category"], "n", h=480
    )
    st.plotly_chart(fig_tm, use_container_width=True)
    chart_card_close()

    # NLP section
    section_header("🔤", "NLP Analysis — หัวข้อสำคัญ", "Text Mining")

    tok_stats = get_token_stats(
        tuple(df["text_clean"].dropna().astype(str).tolist())
    )

    if not tok_stats:
        banner("ติดตั้ง <code>pythainlp</code> เพื่อดูผล NLP: "
               "<code>pip install pythainlp</code>", "warning")
        return

    freq    = tok_stats["freq"]
    bigrams = tok_stats["bigrams"]

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        chart_card("🔑 Top 20 Keywords", "คำที่ปรากฏในคำร้องบ่อยที่สุด")
        kw = (pd.DataFrame(freq.most_common(20), columns=["w","f"])
                .sort_values("f"))
        st.plotly_chart(hbar(kw,"f","w",color_scale=[[0,"#FFF7ED"],[1,C["warning"]]],h=500),
                        use_container_width=True)
        chart_card_close()

    with col_b:
        chart_card("🔗 Top 15 Bigrams", "คู่คำที่ปรากฏบ่อยที่สุด (N-gram = 2)")
        bg = (pd.DataFrame(bigrams.most_common(15), columns=["bg","f"])
                .sort_values("f"))
        st.plotly_chart(hbar(bg,"f","bg",color_scale=SCALE_VIOLET,h=500),
                        use_container_width=True)
        chart_card_close()

    # WordCloud
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        import io as _io

        chart_card("☁️ Word Cloud", "ภาพรวมคำสำคัญจากคำร้องทั้งหมด")
        thai_font = next(
            (fp for fp in fm.findSystemFonts()
             if any(k in fp.lower() for k in ["noto","thai","sarabun","garuda"])),
            None
        )
        wc_kw = dict(background_color="white", width=1400, height=440,
                     max_words=90, colormap="Blues", prefer_horizontal=.8)
        if thai_font:
            wc_kw["font_path"] = thai_font
        wc = WordCloud(**wc_kw).generate_from_frequencies(freq)
        fig_wc, ax = plt.subplots(figsize=(14, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig_wc.patch.set_facecolor("white")
        plt.tight_layout(pad=0)
        buf = _io.BytesIO()
        fig_wc.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        st.image(buf, use_column_width=True)
        plt.close()
        chart_card_close()
    except ImportError:
        banner("ติดตั้ง <code>wordcloud</code> เพื่อดู Word Cloud", "warning")


# ─────────────────────────────────────────────────────────────────────────────
def page_performance(df: pd.DataFrame):
    section_header("⏱️", "Performance Analytics & SLA",
                   f"{df['done'].sum():,} รายการที่เสร็จแล้ว")

    df_done = df[df["done"] & df["days"].notna() & (df["days"] >= 0)]
    avg_sla = df_done["days"].mean() if len(df_done) else 0
    rate    = df["done"].mean() * 100

    # KPIs
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi_card("⏱️", f"{avg_sla:.1f}", "เฉลี่ยวันดำเนินการ", color=C["primary"])
    with c2: kpi_card("📊", f"{df_done['days'].median():.0f}", "Median (วัน)", color=C["secondary"])
    with c3: kpi_card("🚨", f"{df_done['days'].max():.0f}", "สูงสุด (วัน)", color=C["danger"])
    with c4: kpi_card("✅", f"{rate:.1f}%", "Completion Rate",
                      color=C["success"] if rate>=90 else C["warning"])

    spacer()

    # SLA by dept + distribution
    col1, col2 = st.columns(2, gap="large")

    with col1:
        chart_card("🏢 Average SLA ตามฝ่าย",
                   "🟢 ≤7 วัน  ·  🟡 ≤14 วัน  ·  🔴 >14 วัน")
        ds = (df_done.groupby("department")["days"]
                .agg(mean="mean", median="median", count="count")
                .reset_index())
        st.plotly_chart(sla_bar(ds, h=420), use_container_width=True)
        chart_card_close()

    with col2:
        chart_card("📊 การกระจายวันดำเนินการ",
                   "Histogram แสดงความถี่ของจำนวนวัน")
        st.plotly_chart(histogram_chart(df_done["days"], h=420),
                        use_container_width=True)
        chart_card_close()

    # Backlog + Category SLA
    col3, col4 = st.columns(2, gap="large")

    with col3:
        chart_card("🚨 Backlog ตามฝ่าย", "จำนวนคำร้องที่ยังไม่ดำเนินการ")
        bl = (df[~df["done"]].groupby("department").size()
                .reset_index(name="bl").rename(columns={"department":"dep"})
                .sort_values("bl"))
        if len(bl):
            st.plotly_chart(
                hbar(bl,"bl","dep",
                     color_scale=[[0,"#FEF3C7"],[1,C["danger"]]],h=380),
                use_container_width=True
            )
        else:
            banner("🎉 ไม่มี Backlog ในขณะนี้", "success")
        chart_card_close()

    with col4:
        chart_card("📋 SLA ตามประเภทคำร้อง",
                   "ประเภทที่ใช้เวลาดำเนินการนานที่สุด (Top 12)")
        cs = (df_done[df_done["category"].map(
                  df_done["category"].value_counts()) >= 3]
              .groupby("category")["days"].mean()
              .reset_index().rename(columns={"category":"cat","days":"mean"})
              .sort_values("mean").tail(12))
        st.plotly_chart(
            hbar(cs,"mean","cat",
                 color_scale=[[0,"#ECFDF5"],[.5,C["warning"]],[1,C["danger"]]],
                 h=380, text_fmt=lambda v: f"{v:.0f}"),
            use_container_width=True
        )
        chart_card_close()

    # SLA summary table
    section_header("📋", "ตาราง SLA สรุปรายฝ่าย")
    tbl = df_done.groupby("department").agg(
        คำร้อง    = ("days","count"),
        เฉลี่ยวัน = ("days","mean"),
        Median    = ("days","median"),
        Min       = ("days","min"),
        Max       = ("days","max"),
    ).round(1).reset_index().rename(columns={"department":"ฝ่าย"})
    tbl["สถานะ SLA"] = tbl["เฉลี่ยวัน"].apply(
        lambda v: "✅ ดีเยี่ยม" if v<=7 else ("⚠️ พอใช้" if v<=14 else "❌ ล่าช้า")
    )
    st.dataframe(tbl.sort_values("เฉลี่ยวัน"),
                 use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
def page_district(df: pd.DataFrame):
    section_header("🗺️", "District & Community Analysis", "Geospatial")

    dist = (df.groupby("district")
              .agg(total=("done","count"), done=("done","sum"))
              .reset_index())
    dist["rate"]    = (dist["done"]/dist["total"]*100).round(1)
    dist["pending"] = dist["total"] - dist["done"]

    # District × Category heatmap
    chart_card("🔥 Heatmap: เขต × ประเภทคำร้อง",
               "ความหนาแน่นของแต่ละประเภทคำร้องในแต่ละเขต")
    top10_cats = df["category"].value_counts().head(10).index
    dc = df[df["category"].isin(top10_cats)].groupby(
        ["district","category"]).size().reset_index(name="n")
    pivot = dc.pivot(index="district",columns="category",values="n").fillna(0)
    st.plotly_chart(heatmap_chart(pivot, h=340), use_container_width=True)
    chart_card_close()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        chart_card("📊 จำนวนคำร้องตามเขต", "")
        d = dist.sort_values("total")[["district","total"]].rename(
            columns={"district":"dist","total":"c"})
        st.plotly_chart(hbar(d,"c","dist",color_scale=SCALE_BLUE,h=360),
                        use_container_width=True)
        chart_card_close()

    with col2:
        chart_card("✅ Completion Rate ตามเขต", "")
        cr = dist.sort_values("rate")[["district","rate"]].rename(
            columns={"district":"dist"})
        fig = px.bar(cr, x="rate", y="dist", orientation="h",
                     color="rate",
                     color_continuous_scale=[[0,"#FEE2E2"],[.5,"#FEF9C3"],[1,"#D1FAE5"]],
                     text=cr["rate"].apply(lambda v: f"{v:.0f}%"),
                     labels={"rate":"","dist":""})
        fig.update_traces(textposition="outside", textfont_size=12)
        fig.update_coloraxes(showscale=False)
        fig.update_xaxes(range=[0,115])
        fig = _style(fig, 360)
        fig.update_xaxes(**_AX); fig.update_yaxes(**{**_AX,"tickfont":dict(size=12)})
        st.plotly_chart(fig, use_container_width=True)
        chart_card_close()

    # Top communities
    chart_card("🏘️ Top 15 ชุมชนที่มีคำร้องมากที่สุด", "")
    comm = (df["community"].value_counts().head(15)
              .reset_index().rename(columns={"community":"c","count":"n"})
              .sort_values("n"))
    st.plotly_chart(hbar(comm,"n","c",color_scale=SCALE_VIOLET,h=480),
                    use_container_width=True)
    chart_card_close()

    # Map
    chart_card("🗺️ แผนที่ความหนาแน่นคำร้อง — Khon Kaen", "")
    COORDS = {
        "เขต 1":(16.4419,102.8360), "เขต 2":(16.4280,102.8420),
        "เขต 3":(16.4180,102.8300), "เขต 4":(16.4350,102.8200),
        "เขต 7":(16.4500,102.8150), "ไม่ระบุเขต":(16.4330,102.8340),
    }
    map_df = dist[dist["district"].isin(COORDS)].copy()
    map_df["lat"] = map_df["district"].map(lambda d: COORDS.get(d,(16.43,102.83))[0])
    map_df["lon"] = map_df["district"].map(lambda d: COORDS.get(d,(16.43,102.83))[1])
    fig_map = map_bubble(map_df,"lat","lon","total","rate","district",h=460)
    fig_map.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "คำร้องทั้งหมด: %{marker.size}<br>"
            "Completion: %{marker.color:.1f}%<extra></extra>"
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)
    chart_card_close()


# ─────────────────────────────────────────────────────────────────────────────
def page_ai(df: pd.DataFrame):
    section_header("🤖", "AI Complaint Intelligence", "Live Demo")

    df_done = df[df["done"] & df["days"].notna() & (df["days"] >= 0)]

    # Load models
    with st.spinner("🧠 โหลดโมเดล AI..."):
        cat_pipe, dept_pipe, acc, tok_fn = train_models(df)

    # Status banner
    if cat_pipe:
        st.markdown(f"""
        <div class="banner banner-success">
          ✅ <strong>โมเดล AI พร้อมใช้งาน</strong>
          &nbsp;·&nbsp; Accuracy: <strong>{acc:.1%}</strong>
          &nbsp;·&nbsp; Trained on {len(df):,} รายการ
        </div>
        """, unsafe_allow_html=True)
    else:
        banner("ติดตั้ง <code>pythainlp</code> และ <code>scikit-learn</code> "
               "เพื่อเปิดใช้ AI", "warning")

    # Example chips
    EXAMPLES = [
        "ไฟทางหน้าชุมชนดับมา 3 วันแล้ว คนเดินถนนไม่ปลอดภัย",
        "ท่อระบายน้ำอุดตัน น้ำท่วมขัง ส่งกลิ่นเหม็น",
        "ถนนเป็นหลุมบ่อขนาดใหญ่หน้าโรงเรียน",
        "กิ่งไม้โค่นขวางถนน รถผ่านไม่ได้",
        "ไฟสัญญาณจราจรแยกมิตรภาพดับหลายวัน",
    ]

    st.markdown(f"<p style='font-size:.82rem;color:{C['tx_soft']};margin:0 0 .5rem'>💡 คลิกตัวอย่างเพื่อทดลอง</p>",
                unsafe_allow_html=True)
    ex_cols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        with ex_cols[i]:
            if st.button(f"ตย. {i+1}", key=f"ex{i}",
                         use_container_width=True, help=ex):
                st.session_state["ai_text"] = ex

    spacer(.5)

    # Input area
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#EFF6FF,#F5F3FF);
                border:1.5px solid #C7D2FE;border-radius:14px;
                padding:1.25rem 1.5rem 1rem;margin-bottom:.75rem">
      <div style="font-size:.85rem;font-weight:700;color:{C['primary']};margin-bottom:.5rem">
        📝 พิมพ์คำร้องเรียน (ภาษาไทย)
      </div>
    """, unsafe_allow_html=True)

    text_input = st.text_area(
        label="",
        value=st.session_state.get("ai_text",""),
        height=110,
        placeholder="เช่น: ไฟทางหน้าชุมชนดับมา 3 วันแล้ว คนเดินถนนไม่ปลอดภัยตอนกลางคืน...",
        label_visibility="collapsed",
    )
    btn = st.button("🔍  วิเคราะห์ด้วย AI", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    # Run prediction
    if btn and text_input.strip():
        if not cat_pipe:
            banner("ต้องการ pythainlp และ scikit-learn", "danger")
            return

        with st.spinner("🧠 กำลังวิเคราะห์..."):
            tok    = tok_fn(text_input)
            p_cat  = cat_pipe.predict([tok])[0]
            p_dept = dept_pipe.predict([tok])[0]
            proba_cat  = cat_pipe.predict_proba([tok])[0]
            proba_dept = dept_pipe.predict_proba([tok])[0]
            conf_cat   = max(proba_cat)
            conf_dept  = max(proba_dept)
            est = predict_days(p_cat, p_dept, df_done)

            _hi  = ["ไฟดับ","น้ำท่วม","พัง","แตก","อุบัติเหตุ","ฉุกเฉิน","ชำรุด"]
            _med = ["ซ่อม","ท่อระบาย","กิ่งไม้","ไฟฟ้า","จราจร"]
            t = text_input.lower()
            if any(k in t for k in _hi):
                prio, ptag = "สูง (High)",   "tag-red"
            elif any(k in t for k in _med):
                prio, ptag = "กลาง (Medium)","tag-amber"
            else:
                prio, ptag = "ต่ำ (Low)",    "tag-green"

        spacer(.5)
        st.markdown(f"""
        <div style="background:{C['bg_muted']};border-left:3px solid {C['primary']};
                    border-radius:0 10px 10px 0;padding:.7rem 1.1rem;
                    margin-bottom:1rem;font-size:.85rem;color:{C['tx_mid']}">
          🤖 <strong>ผลการวิเคราะห์สำหรับ:</strong>
          "{text_input[:90]}{'...' if len(text_input)>90 else ''}"
        </div>
        """, unsafe_allow_html=True)

        r1,r2,r3,r4 = st.columns(4)
        cards = [
            (r1, "📂 ประเภทคำร้อง", p_cat,
             f"Confidence: {conf_cat:.0%}", None),
            (r2, "🏢 ฝ่ายที่รับผิดชอบ",
             f"<span class='tag tag-blue'>{p_dept}</span>",
             f"Confidence: {conf_dept:.0%}", None),
            (r3, "📅 คาดว่าจะเสร็จใน",
             f"<span style='color:{C['primary']};font-size:1.4rem;font-weight:800'>{est} วัน</span>",
             "จากข้อมูลประวัติดำเนินการ", None),
            (r4, "⚡ ระดับความสำคัญ",
             f"<span class='tag {ptag}'>{prio}</span>",
             "ประเมินจากเนื้อหาคำร้อง", None),
        ]
        for col, lbl, val, sub, _ in cards:
            with col:
                st.markdown(f"""
                <div class="ai-card">
                  <div class="ai-card-label">{lbl}</div>
                  <div class="ai-card-value">{val}</div>
                  <div class="ai-card-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        spacer(.75)

        # Confidence chart
        chart_card("📊 AI Confidence — Top 5 ประเภทที่เป็นไปได้",
                   "ความน่าจะเป็นของแต่ละประเภทคำร้องจากโมเดล NLP")
        top5_idx = np.argsort(proba_cat)[::-1][:5]
        fig_c = confidence_bar(
            [cat_pipe.classes_[i] for i in top5_idx],
            [proba_cat[i] for i in top5_idx],
            h=260,
        )
        st.plotly_chart(fig_c, use_container_width=True)
        chart_card_close()

        # Routing card
        st.markdown(f"""
        <div class="routing-card">
          <div style="font-size:.95rem;font-weight:800;color:{C['primary']};
                      margin-bottom:.6rem;letter-spacing:-.01em">
            🚀 Smart Routing Recommendation
          </div>
          <p style="color:{C['tx_mid']};margin:0;line-height:1.75;font-size:.9rem">
            ส่งคำร้องนี้ไปยัง
            <strong style="color:{C['tx_dark']}">{p_dept}</strong>
            โดยอัตโนมัติ<br>
            คาดว่าจะดำเนินการแล้วเสร็จภายใน
            <strong style="color:{C['primary']}">{est} วัน</strong>
            นับจากวันที่รับเรื่อง<br>
            ระดับความสำคัญ: <span class="tag {ptag}">{prio}</span>
          </p>
        </div>
        """, unsafe_allow_html=True)

    elif btn:
        banner("กรุณาพิมพ์คำร้องเรียนก่อนกดวิเคราะห์", "warning")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 ── SIDEBAR + MAIN ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Complaint Intelligence · Khon Kaen",
        page_icon="🏙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:1.25rem 0 1rem;text-align:center">
          <div style="font-size:2.2rem">🏙️</div>
          <div style="font-size:.95rem;font-weight:800;color:{C['tx_dark']};
                      margin-top:.3rem;letter-spacing:-.01em">
            Complaint Intelligence
          </div>
          <div style="font-size:.72rem;color:{C['tx_soft']};margin-top:.15rem">
            เทศบาลนครขอนแก่น · v3.0
          </div>
        </div>
        <hr style="border:none;border-top:1px solid {C['border']};margin:.5rem 0 1rem">
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "📂 อัปโหลดไฟล์ข้อมูล (.xlsx)",
            type=["xlsx","xls"],
        )

        # Load data
        with st.spinner("โหลดข้อมูล..."):
            df_all = load_and_clean(uploaded)

        total = len(df_all)
        done  = df_all["done"].sum()
        st.markdown(f"""
        <div class="banner banner-success" style="margin:.5rem 0 1rem">
          ✅ โหลดสำเร็จ · <strong>{total:,}</strong> รายการ
          (<strong>{done:,}</strong> เสร็จแล้ว)
        </div>
        """, unsafe_allow_html=True)

        # Filters
        st.markdown(f"<p style='font-size:.72rem;font-weight:700;color:{C['tx_soft']};"
                    f"text-transform:uppercase;letter-spacing:.08em;margin:.5rem 0 .3rem'>"
                    f"🔍 ตัวกรอง</p>", unsafe_allow_html=True)

        dist_opts = ["ทั้งหมด"] + sorted(df_all["district"].dropna().unique())
        dept_opts = ["ทั้งหมด"] + sorted(df_all["department"].dropna().unique())
        cat_opts  = ["ทั้งหมด"] + sorted(df_all["category"].dropna().unique())
        stat_opts = ["ทั้งหมด"] + sorted(df_all["status"].dropna().unique())

        sel_dist = st.selectbox("เขต", dist_opts)
        sel_dept = st.selectbox("ฝ่าย", dept_opts)
        sel_cat  = st.selectbox("ประเภทคำร้อง", cat_opts)
        sel_stat = st.selectbox("สถานะ", stat_opts)

        st.markdown(f"<hr style='border:none;border-top:1px solid {C['border']};margin:.75rem 0'>",
                    unsafe_allow_html=True)

        # Navigation
        NAV = [
            "🏠  ภาพรวม",
            "📊  วิเคราะห์คำร้อง",
            "⏱️  Performance & SLA",
            "🗺️  District & Map",
            "🤖  AI Demo",
        ]
        page = st.radio("เมนู", NAV, label_visibility="collapsed")

        # Footer
        st.markdown(f"""
        <div style="position:absolute;bottom:1.5rem;left:1rem;right:1rem;
                    text-align:center;font-size:.7rem;color:{C['tx_xsoft']}">
          Smart City Intelligence Platform<br>
          GovTech Innovation · Khon Kaen
        </div>
        """, unsafe_allow_html=True)

    # ── Apply filters ──────────────────────────────────────────────────────────
    df = df_all.copy()
    if sel_dist != "ทั้งหมด": df = df[df["district"]  == sel_dist]
    if sel_dept != "ทั้งหมด": df = df[df["department"] == sel_dept]
    if sel_cat  != "ทั้งหมด": df = df[df["category"]  == sel_cat]
    if sel_stat != "ทั้งหมด": df = df[df["status"]     == sel_stat]

    if len(df) == 0:
        banner("ไม่พบข้อมูลที่ตรงกับตัวกรองที่เลือก กรุณาปรับตัวกรอง", "warning")
        return

    # ── Route to page ──────────────────────────────────────────────────────────
    if   "ภาพรวม"    in page: page_overview(df)
    elif "วิเคราะห์" in page: page_complaints(df)
    elif "Performance" in page: page_performance(df)
    elif "District"   in page: page_district(df)
    elif "AI"         in page: page_ai(df)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center;padding:2.5rem 0 1rem;
                border-top:1px solid {C['border']};margin-top:2.5rem;
                font-size:.75rem;color:{C['tx_xsoft']}">
      🏙️ Smart City Complaint Intelligence Platform &nbsp;·&nbsp;
      เทศบาลนครขอนแก่น &nbsp;·&nbsp; v3.0 &nbsp;·&nbsp;
      Powered by AI + NLP
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
