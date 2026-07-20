"""
Collection Trend Analysis — Streamlit dashboard.

Wraps the analysis functions from `generate_library_analysis_fy23_26.py` in an
interactive UI. Upload the physical-usage CSV (and optionally a holdings CSV),
click Run, and browse LC / subject / geographic shifts, tiered weeding
candidates, and e-book targets. All tables are downloadable as CSVs.

Run locally:
    streamlit run analysis_dashboard.py
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Import all pure primitives from the analysis script — parsers, constants,
# and the trend-fitting pivot. No file I/O happens on import.
from generate_library_analysis_fy23_26 import (  # noqa: E402
    DISCIPLINE_ORDER,
    FY_COLS,
    FIRST_FY,
    LAST_FY,
    LC_CLASS_DESC,
    LC_SUBCLASS_DESC,
    _classify_geography_region,
    categorize_discipline,
    get_class,
    get_subclass,
    normalize_title,
    parse_full_headings,
    parse_geographic_terms,
    pivot_by_year,
)

# ---------------------------------------------------------------------------
# Streamlit page config + styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Collection Trend Analysis — FY23–FY26",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

POS_COLOR = "#3E8E4F"
NEG_COLOR = "#B84343"
NEUTRAL_COLOR = "#8A8A8A"

# ---------------------------------------------------------------------------
# Data loading + prep
# ---------------------------------------------------------------------------

REQUIRED_CIRC_COLS = {
    "Title", "Loan Fiscal Year", "Author", "Call Number",
    "Subjects", "Loans (In House + Not In House)", "Publication Year",
}


@st.cache_data(show_spinner=False)
def load_circulation(file_bytes: bytes) -> pd.DataFrame:
    """Read the circulation CSV and attach LC/discipline metadata."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    missing = REQUIRED_CIRC_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["LC_Class"] = df["Call Number"].apply(get_class)
    df["LC_Subclass"] = df["Call Number"].apply(get_subclass)
    df["Discipline"] = df["LC_Subclass"].apply(categorize_discipline)
    return df


@st.cache_data(show_spinner=False)
def load_holdings(file_bytes: bytes) -> pd.DataFrame:
    """Read the (optional) holdings CSV and attach LC/discipline metadata."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    title_col = "Title (Normalized)" if "Title (Normalized)" in df.columns else "Title"
    if "Call Number" not in df.columns or title_col not in df.columns:
        raise ValueError("Holdings CSV needs 'Call Number' plus 'Title' or 'Title (Normalized)'")
    df["LC_Subclass"] = df["Call Number"].apply(get_subclass)
    df["Discipline"] = df["LC_Subclass"].apply(categorize_discipline)
    df["title_key"] = df[title_col].apply(normalize_title)
    return df


# ---------------------------------------------------------------------------
# Cached compute — each returns a DataFrame with trend metrics ready to display
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_lc_class_shifts(df: pd.DataFrame) -> pd.DataFrame:
    out = pivot_by_year(df, "LC_Class")
    out["Description"] = out["LC_Class"].map(
        lambda c: LC_CLASS_DESC.get(c, f"{c} - Class {c}")
    )
    out["Discipline"] = out["LC_Class"].apply(categorize_discipline)
    return out[
        ["Discipline", "LC_Class", "Description", *FY_COLS,
         "Mean Annual Loans", "Trend Slope (loans/yr)", "Trend R^2",
         "Cumulative % Change (trend)", "Endpoint % Change (FY23->FY26)",
         "Absolute Change (FY23 to FY26)"]
    ]


@st.cache_data(show_spinner=False)
def compute_lc_subclass_shifts(df: pd.DataFrame) -> pd.DataFrame:
    out = pivot_by_year(df, "LC_Subclass")
    out["Description"] = out["LC_Subclass"].map(
        lambda s: LC_SUBCLASS_DESC.get(s, f"{s} - Subclass {s}")
    )
    out["Discipline"] = out["LC_Subclass"].apply(categorize_discipline)
    return out[
        ["Discipline", "LC_Subclass", "Description", *FY_COLS,
         "Mean Annual Loans", "Trend Slope (loans/yr)", "Trend R^2",
         "Cumulative % Change (trend)", "Endpoint % Change (FY23->FY26)",
         "Absolute Change (FY23 to FY26)"]
    ]


@st.cache_data(show_spinner=False)
def compute_subject_shifts(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["FullHeadings"] = df2["Subjects"].apply(parse_full_headings)
    exploded = df2.explode("FullHeadings").rename(columns={"FullHeadings": "Subject"})
    exploded = exploded[exploded["Subject"].notna() & (exploded["Subject"] != "")]
    subj = pivot_by_year(exploded, "Subject")

    disc_totals = (
        exploded.groupby(["Subject", "Discipline"])["Loans (In House + Not In House)"]
        .sum().reset_index()
    )
    dominant = (
        disc_totals.sort_values("Loans (In House + Not In House)", ascending=False)
        .drop_duplicates("Subject")[["Subject", "Discipline"]]
    )
    subj = subj.merge(dominant, on="Subject", how="left")
    return subj[
        ["Discipline", "Subject", *FY_COLS,
         "Mean Annual Loans", "Trend Slope (loans/yr)", "Trend R^2",
         "Cumulative % Change (trend)", "Endpoint % Change (FY23->FY26)",
         "Absolute Change (FY23 to FY26)"]
    ]


@st.cache_data(show_spinner=False)
def compute_geographic_shifts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df2 = df.copy()
    df2["Geographies"] = df2["Subjects"].apply(parse_geographic_terms)
    exploded = df2.explode("Geographies").rename(columns={"Geographies": "Geography"})
    exploded = exploded[exploded["Geography"].notna() & (exploded["Geography"] != "")]
    if exploded.empty:
        return pd.DataFrame(), pd.DataFrame()
    exploded["Region"] = exploded["Geography"].apply(_classify_geography_region)

    geo = pivot_by_year(exploded, "Geography")
    place_regions = (
        exploded[["Geography", "Region"]].drop_duplicates()
        .set_index("Geography")["Region"]
    )
    geo["Region"] = geo["Geography"].map(place_regions)
    geo = geo[
        ["Region", "Geography", *FY_COLS,
         "Mean Annual Loans", "Trend Slope (loans/yr)", "Trend R^2",
         "Cumulative % Change (trend)", "Endpoint % Change (FY23->FY26)",
         "Absolute Change (FY23 to FY26)"]
    ]

    region = pivot_by_year(exploded, "Region")
    region = region[
        ["Region", *FY_COLS,
         "Mean Annual Loans", "Trend Slope (loans/yr)", "Trend R^2",
         "Cumulative % Change (trend)", "Endpoint % Change (FY23->FY26)",
         "Absolute Change (FY23 to FY26)"]
    ]
    return geo, region


@st.cache_data(show_spinner=False)
def compute_weeding_candidates(
    df: pd.DataFrame,
    strong: int = 10,
    medium: int = 5,
    weak: int = 3,
    protect_recent_pub_years: int = 5,
) -> pd.DataFrame:
    """Usage-decay weeding, tiered by early demand, with pub-year gate."""
    from datetime import date

    title_yr = (
        df.groupby(["Title", "Author", "Publication Year", "Call Number",
                    "LC_Subclass", "Discipline", "Loan Fiscal Year"])
          ["Loans (In House + Not In House)"].sum().unstack(fill_value=0)
    )
    for fy in FY_COLS:
        if fy not in title_yr.columns:
            title_yr[fy] = 0
    early = [FY_COLS[0], FY_COLS[1]]
    recent = [FY_COLS[2], FY_COLS[3]]
    title_yr["Early Loans"] = title_yr[early].sum(axis=1)
    title_yr["Recent Loans"] = title_yr[recent].sum(axis=1)
    title_yr["Total Loans"] = title_yr[FY_COLS].sum(axis=1)
    title_yr = title_yr.reset_index()

    pub_num = pd.to_numeric(title_yr["Publication Year"], errors="coerce")
    cutoff = date.today().year - protect_recent_pub_years
    protected = pub_num >= cutoff

    def tier(row):
        if row["Recent Loans"] != 0:
            return None
        if row["Early Loans"] >= strong:
            return "Strong"
        if row["Early Loans"] >= medium:
            return "Medium"
        if row["Early Loans"] >= weak:
            return "Weak"
        return None

    title_yr["Weeding Tier"] = title_yr.apply(tier, axis=1)
    title_yr.loc[protected, "Weeding Tier"] = None

    out = title_yr[title_yr["Weeding Tier"].notna()].copy()
    tier_dtype = pd.CategoricalDtype(["Strong", "Medium", "Weak"], ordered=True)
    out["Weeding Tier"] = out["Weeding Tier"].astype(tier_dtype)
    out = out.sort_values(["Weeding Tier", "Early Loans"],
                          ascending=[True, False])
    return out[
        ["Weeding Tier", "Discipline", "LC_Subclass", "Title", "Author",
         "Publication Year", "Call Number",
         *FY_COLS, "Early Loans", "Recent Loans", "Total Loans"]
    ]


@st.cache_data(show_spinner=False)
def compute_ebook_candidates(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    title_yr = (
        df.groupby(["Title", "Author", "Call Number", "LC_Subclass",
                    "Discipline", "Loan Fiscal Year"])
          ["Loans (In House + Not In House)"].sum().unstack(fill_value=0)
    )
    for fy in FY_COLS:
        if fy not in title_yr.columns:
            title_yr[fy] = 0
    title_yr["Total Loans"] = title_yr[FY_COLS].sum(axis=1)
    title_yr["Active Years"] = (title_yr[FY_COLS] > 0).sum(axis=1)
    title_yr["Early Loans"] = title_yr[[FY_COLS[0], FY_COLS[1]]].sum(axis=1)
    title_yr["Recent Loans"] = title_yr[[FY_COLS[-2], FY_COLS[-1]]].sum(axis=1)

    def rationale(row):
        rising = row["Recent Loans"] > row["Early Loans"]
        sustained = row["Active Years"] >= 3
        if rising and sustained:
            return "Sustained, rising demand — priority e-book candidate; digital license unlocks multi-user simultaneous access."
        if sustained:
            return "Consistent multi-year demand — strong e-book candidate to reduce shelf wear and support remote access."
        if rising:
            return "Recent demand spike — e-book license would meet current course/research pressure quickly."
        return "High historical volume — evaluate whether digital access would better serve current use."

    ranked = title_yr.sort_values("Total Loans", ascending=False).head(top_n).reset_index()
    ranked["Strategic Case for E-Book Acquisition"] = ranked.apply(rationale, axis=1)
    return ranked[
        ["Title", "Author", "Call Number", "Discipline", *FY_COLS,
         "Early Loans", "Recent Loans", "Active Years", "Total Loans",
         "Strategic Case for E-Book Acquisition"]
    ]


@st.cache_data(show_spinner=False)
def compute_holdings_weeding(circ: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    """Classic weeding: subclasses with the highest % uncirculated holdings."""
    circ_local = circ.copy()
    circ_local["title_key"] = circ_local["Title"].apply(normalize_title)
    per_title = (
        circ_local.groupby("title_key")["Loans (In House + Not In House)"]
        .sum().reset_index()
    )
    merged = holdings.merge(per_title, on="title_key", how="left")
    merged["Loans (In House + Not In House)"] = (
        merged["Loans (In House + Not In House)"].fillna(0)
    )

    stats = merged.groupby(["Discipline", "LC_Subclass"]).agg(
        Total_Holdings=("title_key", "count"),
        Uncirculated_Holdings=(
            "Loans (In House + Not In House)", lambda x: (x == 0).sum()
        ),
    ).reset_index()
    stats["Percent_Uncirculated"] = (
        stats["Uncirculated_Holdings"] / stats["Total_Holdings"] * 100
    ).round(2)
    stats["Description"] = stats["LC_Subclass"].map(
        lambda s: LC_SUBCLASS_DESC.get(s, f"{s} - Subclass {s}")
    )
    return stats.sort_values(
        ["Discipline", "Percent_Uncirculated"], ascending=[True, False]
    )


# ---------------------------------------------------------------------------
# Plotly rendering
# ---------------------------------------------------------------------------

def _bar_colors(values):
    return [POS_COLOR if v > 0 else (NEG_COLOR if v < 0 else NEUTRAL_COLOR)
            for v in values]


def make_shift_chart(df: pd.DataFrame, label_col: str, pct_col: str,
                     title: str, r2_col: Optional[str] = "Trend R^2") -> go.Figure:
    plot_df = df.copy()
    plot_df[pct_col] = pd.to_numeric(plot_df[pct_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[pct_col])
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data to display for these filters.",
                           x=0.5, y=0.5, showarrow=False)
        return fig
    plot_df = plot_df.sort_values(pct_col, ascending=True)

    colors = _bar_colors(plot_df[pct_col])
    text_labels = [f"{v:+.1f}%" for v in plot_df[pct_col]]

    hover_lines = [f"<b>{lbl}</b><br>Trend %: {v:+.1f}%"
                   for lbl, v in zip(plot_df[label_col], plot_df[pct_col])]
    if r2_col and r2_col in plot_df.columns:
        hover_lines = [
            f"{h}<br>R²: {r:.3f}" if pd.notna(r) else h
            for h, r in zip(hover_lines, plot_df[r2_col])
        ]
    # Add raw FY loans to hover
    fy_lines = []
    for _, row in plot_df.iterrows():
        parts = " · ".join(f"{fy}: {int(row[fy])}" for fy in FY_COLS if fy in row)
        fy_lines.append(parts)
    hover_lines = [f"{h}<br>{fy}" for h, fy in zip(hover_lines, fy_lines)]

    fig = go.Figure(go.Bar(
        y=plot_df[label_col],
        x=plot_df[pct_col],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=text_labels,
        textposition="outside",
        textfont=dict(size=11),
        hovertext=hover_lines,
        hoverinfo="text",
    ))
    max_abs = float(plot_df[pct_col].abs().max())
    limit = max_abs * 1.25 if max_abs > 0 else 10
    fig.add_vline(x=0, line_dash="dash", line_color="#444", line_width=1)
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left",
                   font=dict(size=15, family="system-ui")),
        xaxis=dict(
            title=f"Cumulative % Change (trend fit across {FIRST_FY}–{LAST_FY})",
            range=[-limit, limit],
            gridcolor="#eee",
        ),
        yaxis=dict(title="", automargin=True),
        height=max(360, 30 * len(plot_df) + 120),
        plot_bgcolor="white",
        margin=dict(l=0, r=80, t=60, b=50),
        showlegend=False,
    )
    return fig


def top_risers_and_decliners(df: pd.DataFrame, pct_col: str,
                             top_up: int, top_down: int,
                             baseline_col: Optional[str] = None,
                             baseline_min: int = 0) -> pd.DataFrame:
    d = df.copy()
    if baseline_col is not None:
        d = d[d[baseline_col] >= baseline_min]
    d[pct_col] = pd.to_numeric(d[pct_col], errors="coerce")
    d = d.dropna(subset=[pct_col])
    risers = d[d[pct_col] > 0].nlargest(top_up, pct_col)
    decliners = d[d[pct_col] < 0].nsmallest(top_down, pct_col)
    return pd.concat([risers, decliners])


def download_button_for_df(df: pd.DataFrame, label: str, filename: str,
                            key: Optional[str] = None):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename,
                       mime="text/csv", key=key)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("📚 Collection Trend Analysis")
st.caption(f"Fiscal-year window: {FIRST_FY} → {LAST_FY}  ·  "
           "trend metrics fit a line through all four years, "
           "so interior spikes and dips register in the shift number.")

# ----- Sidebar --------------------------------------------------------------
with st.sidebar:
    st.header("Data Input")
    circ_file = st.file_uploader(
        "Physical usage CSV (required)",
        type=["csv"],
        help="One row per Title × Fiscal Year with a 'Loans (In House + Not In House)' column.",
    )
    holdings_file = st.file_uploader(
        "Holdings CSV (optional)",
        type=["csv"],
        help="Enables the classic uncirculated-holdings weeding analysis. "
             "Needs 'Call Number' plus 'Title' or 'Title (Normalized)'.",
    )
    st.markdown("---")

    with st.expander("Analysis parameters", expanded=False):
        top_up = st.slider("Risers to show", 5, 25, 12, key="top_up")
        top_down = st.slider("Decliners to show", 5, 25, 12, key="top_down")
        min_baseline_subj = st.number_input(
            "Subject baseline: min FY-2023 loans",
            min_value=1, max_value=200, value=25,
            help="Filters out noise from headings with tiny baselines.",
        )
        min_baseline_geo = st.number_input(
            "Geography baseline: min FY-2023 loans",
            min_value=1, max_value=200, value=20,
        )
        min_baseline_sub = st.number_input(
            "LC subclass baseline: min FY-2023 loans",
            min_value=1, max_value=500, value=50,
        )
        st.markdown("**Weeding tier thresholds** (early-window loans):")
        weed_strong = st.number_input("Strong ≥", 1, 100, 10)
        weed_medium = st.number_input("Medium ≥", 1, 100, 5)
        weed_weak = st.number_input("Weak ≥", 1, 100, 3)
        weed_pub_grace = st.number_input(
            "Protect items published within last N years", 0, 20, 5,
        )

    with st.expander("About this tool", expanded=False):
        st.markdown(
            f"""
            **What it does.** Reads a physical-circulation CSV and produces
            trend analyses for LC classifications, subject headings (full LC
            headings), and geographic mentions across the fiscal-year window,
            plus tiered weeding and e-book candidate lists.

            **Metric.** Every "% change" is a cumulative trend fit — a linear
            regression through all four fiscal-year totals, expressed as a
            percentage of the {FIRST_FY} baseline. This means a subject that
            went 100 → 200 → 200 → 100 shows near-zero movement (returned to
            baseline) rather than the "0% change" you'd get from an endpoint
            comparison. R² tells you how well the straight line actually fits
            the four points — trust the direction when R² is high, treat noisy
            trends as directional guidance rather than gospel.
            """
        )

if circ_file is None:
    st.info("👈 Upload a physical-usage CSV in the sidebar to begin.")
    st.markdown("### Expected columns")
    st.code(", ".join(sorted(REQUIRED_CIRC_COLS)), language="text")
    st.stop()

# ----- Load data ------------------------------------------------------------
try:
    circ_df = load_circulation(circ_file.getvalue())
except Exception as exc:
    st.error(f"Could not read circulation CSV: {exc}")
    st.stop()

holdings_df = None
if holdings_file is not None:
    try:
        holdings_df = load_holdings(holdings_file.getvalue())
    except Exception as exc:
        st.warning(f"Holdings CSV skipped: {exc}")

# ----- Header stats ---------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Circulation rows", f"{len(circ_df):,}")
c2.metric("Unique titles", f"{circ_df['Title'].nunique():,}")
c3.metric(
    "Total loans (all FYs)",
    f"{int(circ_df['Loans (In House + Not In House)'].sum()):,}",
)
c4.metric(
    "Holdings loaded",
    f"{len(holdings_df):,}" if holdings_df is not None else "—",
)

# ----- Compute --------------------------------------------------------------
with st.spinner("Running trend analysis…"):
    lc_class_df = compute_lc_class_shifts(circ_df)
    lc_sub_df = compute_lc_subclass_shifts(circ_df)
    subj_df = compute_subject_shifts(circ_df)
    geo_df, region_df = compute_geographic_shifts(circ_df)
    weeding_df = compute_weeding_candidates(
        circ_df,
        strong=weed_strong, medium=weed_medium, weak=weed_weak,
        protect_recent_pub_years=weed_pub_grace,
    )
    ebook_df = compute_ebook_candidates(circ_df)
    holdings_weed_df = (
        compute_holdings_weeding(circ_df, holdings_df)
        if holdings_df is not None else None
    )

# ----- Tabs -----------------------------------------------------------------
tab_labels = [
    "🏠 Overview",
    "LC Class",
    "LC Subclass",
    "Subject Headings",
    "Geographic",
    "Weeding",
    "E-Book Candidates",
]
if holdings_weed_df is not None:
    tab_labels.append("Holdings Weeding")

tabs = st.tabs(tab_labels)

# --- Overview ---------------------------------------------------------------
with tabs[0]:
    st.subheader("Highlights")

    def _top_movers_summary(df, label_col, name):
        d = df.copy()
        d["Cumulative % Change (trend)"] = pd.to_numeric(
            d["Cumulative % Change (trend)"], errors="coerce"
        )
        d = d.dropna(subset=["Cumulative % Change (trend)"])
        if d.empty:
            return
        riser = d.nlargest(1, "Cumulative % Change (trend)").iloc[0]
        decliner = d.nsmallest(1, "Cumulative % Change (trend)").iloc[0]
        cA, cB = st.columns(2)
        cA.metric(f"{name} — biggest riser",
                  f"{riser[label_col]}",
                  f"{riser['Cumulative % Change (trend)']:+.1f}%")
        cB.metric(f"{name} — biggest decliner",
                  f"{decliner[label_col]}",
                  f"{decliner['Cumulative % Change (trend)']:+.1f}%")

    _top_movers_summary(lc_class_df, "Description", "LC Class")
    _top_movers_summary(
        top_risers_and_decliners(subj_df, "Cumulative % Change (trend)",
                                 25, 25, FIRST_FY, min_baseline_subj),
        "Subject", "Subject Heading",
    )
    if not geo_df.empty:
        _top_movers_summary(
            top_risers_and_decliners(geo_df, "Cumulative % Change (trend)",
                                     25, 25, FIRST_FY, min_baseline_geo),
            "Geography", "Geography",
        )

    st.markdown("---")
    st.markdown("**Region-level roll-up** (continent buckets across all fiscal years)")
    if not region_df.empty:
        st.plotly_chart(
            make_shift_chart(
                region_df, "Region", "Cumulative % Change (trend)",
                "Geographic Mentions by Region — Cumulative Trend",
            ),
            use_container_width=True,
        )

# --- LC Class ---------------------------------------------------------------
with tabs[1]:
    st.subheader("LC Class Shifts")
    disc_choice = st.selectbox(
        "Filter by discipline",
        ["All disciplines"] + list(DISCIPLINE_ORDER),
        key="lc_class_disc",
    )
    d = lc_class_df if disc_choice == "All disciplines" \
        else lc_class_df[lc_class_df["Discipline"] == disc_choice]
    st.plotly_chart(
        make_shift_chart(d, "Description", "Cumulative % Change (trend)",
                         f"LC Class Shifts — {disc_choice}"),
        use_container_width=True,
    )
    st.dataframe(
        d.sort_values("Cumulative % Change (trend)", ascending=False),
        use_container_width=True, hide_index=True,
    )
    download_button_for_df(lc_class_df,
                           "⬇ Download LC Class shifts CSV",
                           "lc_class_circulation_shifts_fy23_fy26.csv",
                           key="dl_lc_class")

# --- LC Subclass ------------------------------------------------------------
with tabs[2]:
    st.subheader("LC Subclass Shifts — Greatest Cumulative Movement")
    disc_choice = st.selectbox(
        "Filter by discipline",
        ["All disciplines"] + list(DISCIPLINE_ORDER),
        key="lc_sub_disc",
    )
    d = lc_sub_df if disc_choice == "All disciplines" \
        else lc_sub_df[lc_sub_df["Discipline"] == disc_choice]

    top_movers = top_risers_and_decliners(
        d, "Cumulative % Change (trend)",
        top_up=8, top_down=8,
        baseline_col=FIRST_FY, baseline_min=min_baseline_sub,
    )
    st.plotly_chart(
        make_shift_chart(
            top_movers, "Description", "Cumulative % Change (trend)",
            f"LC Subclasses — Top Movers ({disc_choice})",
        ),
        use_container_width=True,
    )
    st.markdown("**Full subclass detail (all rows for this discipline):**")
    st.dataframe(
        d.sort_values("Cumulative % Change (trend)", ascending=False),
        use_container_width=True, hide_index=True,
    )
    download_button_for_df(lc_sub_df,
                           "⬇ Download LC Subclass shifts CSV",
                           "lc_subclass_circulation_shifts_fy23_fy26.csv",
                           key="dl_lc_sub")

# --- Subject Headings -------------------------------------------------------
with tabs[3]:
    st.subheader("Subject Heading Shifts (full LC headings)")
    st.caption("Full heading strings — 'Politics and government--United States' "
               "stays distinct from 'Politics and government--France'.")
    disc_choice = st.selectbox(
        "Filter by discipline",
        ["All disciplines"] + list(DISCIPLINE_ORDER),
        key="subj_disc",
    )
    d = subj_df if disc_choice == "All disciplines" \
        else subj_df[subj_df["Discipline"] == disc_choice]

    top_movers = top_risers_and_decliners(
        d, "Cumulative % Change (trend)",
        top_up=top_up, top_down=top_down,
        baseline_col=FIRST_FY, baseline_min=min_baseline_subj,
    )
    st.plotly_chart(
        make_shift_chart(
            top_movers, "Subject", "Cumulative % Change (trend)",
            f"Subject Headings — Top Risers & Decliners ({disc_choice})",
        ),
        use_container_width=True,
    )
    search_term = st.text_input(
        "🔍 Search subject headings",
        placeholder="e.g. 'African American', 'Louisiana', 'artificial'",
        key="subj_search",
    )
    display = d
    if search_term:
        display = display[display["Subject"].str.contains(
            search_term, case=False, na=False)]
    st.dataframe(
        display.sort_values("Cumulative % Change (trend)", ascending=False),
        use_container_width=True, hide_index=True,
    )
    download_button_for_df(subj_df,
                           "⬇ Download subject-heading shifts CSV",
                           "subject_term_shifts_fy23_fy26.csv",
                           key="dl_subj")

# --- Geographic Trends ------------------------------------------------------
with tabs[4]:
    st.subheader("Geographic Trends")
    if geo_df.empty:
        st.info("No geographic terms detected in this dataset.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Region-level roll-up**")
            st.plotly_chart(
                make_shift_chart(
                    region_df, "Region", "Cumulative % Change (trend)",
                    "By Region",
                ),
                use_container_width=True,
            )
        with c2:
            region_options = ["All regions"] + sorted(
                geo_df["Region"].dropna().unique().tolist()
            )
            region_choice = st.selectbox("Filter places by region",
                                         region_options, key="geo_region")
            d = geo_df if region_choice == "All regions" \
                else geo_df[geo_df["Region"] == region_choice]
            top_movers = top_risers_and_decliners(
                d, "Cumulative % Change (trend)",
                top_up=top_up, top_down=top_down,
                baseline_col=FIRST_FY, baseline_min=min_baseline_geo,
            )
            st.plotly_chart(
                make_shift_chart(
                    top_movers, "Geography", "Cumulative % Change (trend)",
                    f"Places — Top Movers ({region_choice})",
                ),
                use_container_width=True,
            )

        st.markdown("**Full place-level detail:**")
        search_geo = st.text_input(
            "🔍 Search places",
            placeholder="e.g. 'Louisiana', 'New Orleans', 'France'",
            key="geo_search",
        )
        display = geo_df
        if search_geo:
            display = display[display["Geography"].str.contains(
                search_geo, case=False, na=False)]
        st.dataframe(
            display.sort_values("Cumulative % Change (trend)", ascending=False),
            use_container_width=True, hide_index=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            download_button_for_df(
                geo_df, "⬇ Download place-level shifts CSV",
                "geographic_term_shifts_fy23_fy26.csv", key="dl_geo",
            )
        with c2:
            download_button_for_df(
                region_df, "⬇ Download region-level shifts CSV",
                "geographic_region_shifts_fy23_fy26.csv", key="dl_geo_r",
            )

# --- Weeding ----------------------------------------------------------------
with tabs[5]:
    st.subheader("Weeding Candidates (Usage Decay)")
    st.caption(
        "Titles with meaningful early-window use that went silent in the "
        "recent window. Recently published titles are protected from the list."
    )
    tier_counts = weeding_df["Weeding Tier"].value_counts().reindex(
        ["Strong", "Medium", "Weak"], fill_value=0
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strong tier", int(tier_counts["Strong"]))
    c2.metric("Medium tier", int(tier_counts["Medium"]))
    c3.metric("Weak tier", int(tier_counts["Weak"]))
    c4.metric("Total flagged", int(tier_counts.sum()))

    c1, c2 = st.columns(2)
    with c1:
        tier_filter = st.multiselect(
            "Tier", ["Strong", "Medium", "Weak"], default=["Strong"],
            key="weed_tier",
        )
    with c2:
        disc_filter = st.multiselect(
            "Discipline", list(DISCIPLINE_ORDER), default=list(DISCIPLINE_ORDER),
            key="weed_disc",
        )
    display = weeding_df[
        weeding_df["Weeding Tier"].isin(tier_filter)
        & weeding_df["Discipline"].isin(disc_filter)
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)
    download_button_for_df(
        weeding_df, "⬇ Download weeding candidates CSV",
        "weeding_candidates_fy23_fy26.csv", key="dl_weed",
    )

# --- E-Book Candidates ------------------------------------------------------
with tabs[6]:
    st.subheader("E-Book Purchase Candidates")
    st.caption(
        "Ranked by total 4-year loans, with a rationale keyed to whether "
        "demand is sustained (loans in 3+ FYs) and/or rising (recent > early)."
    )
    st.dataframe(ebook_df, use_container_width=True, hide_index=True)
    download_button_for_df(
        ebook_df, "⬇ Download e-book candidates CSV",
        "ebook_purchase_candidates_fy23_fy26.csv", key="dl_ebook",
    )

# --- Holdings weeding (conditional) -----------------------------------------
if holdings_weed_df is not None:
    with tabs[7]:
        st.subheader("Holdings-Based Weeding: % Uncirculated by LC Subclass")
        st.caption(
            "Requires the holdings CSV. Subclasses ranked by the share of "
            "items that had zero loans across the whole fiscal-year window."
        )
        min_holdings = st.slider(
            "Minimum holdings per subclass",
            10, 500, 150, key="holdings_min",
        )
        top_display = st.slider(
            "How many subclasses to chart",
            5, 30, 15, key="holdings_top",
        )
        top = (
            holdings_weed_df[holdings_weed_df["Total_Holdings"] >= min_holdings]
            .sort_values("Percent_Uncirculated", ascending=False)
            .head(top_display)
        )
        fig = go.Figure(go.Bar(
            y=top["Description"],
            x=top["Percent_Uncirculated"],
            orientation="h",
            marker=dict(color=NEG_COLOR),
            text=[f"{v:.1f}%" for v in top["Percent_Uncirculated"]],
            textposition="outside",
        ))
        fig.add_vline(x=50, line_dash="dot", line_color="#777")
        fig.add_vline(x=70, line_dash="dash", line_color="#333",
                       annotation_text="High-priority (70%)",
                       annotation_position="top")
        fig.update_layout(
            title=f"Highest % Uncirculated (min {min_holdings} holdings)",
            xaxis=dict(title="% of Holdings with Zero Loans",
                       range=[0, 100], gridcolor="#eee"),
            yaxis=dict(title="", automargin=True),
            plot_bgcolor="white",
            height=max(400, 32 * len(top) + 120),
            margin=dict(l=0, r=60, t=60, b=50),
            showlegend=False,
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Full subclass-level uncirculated stats:**")
        st.dataframe(
            holdings_weed_df.sort_values(
                ["Discipline", "Percent_Uncirculated"], ascending=[True, False]
            ),
            use_container_width=True, hide_index=True,
        )
        download_button_for_df(
            holdings_weed_df, "⬇ Download uncirculated-by-subclass CSV",
            "uncirculated_by_subclass_fy23_fy26.csv", key="dl_holdings",
        )
