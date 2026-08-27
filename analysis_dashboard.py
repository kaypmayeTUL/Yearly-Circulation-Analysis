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

# Import primitives from the analysis script. FY_COLS is a mutable list that
# configure_fy_window() populates in-place at load time; FIRST_FY / LAST_FY
# are strings that get reassigned in the script module, so we access them via
# the module namespace (`script.FIRST_FY`) rather than a `from ... import`
# copy which would freeze at empty-string.
import generate_library_analysis_fy23_26 as script
from generate_library_analysis_fy23_26 import (  # noqa: E402
    DISCIPLINE_ORDER,
    FY_COLS,           # mutable list — reads reflect the currently loaded data
    LC_CLASS_DESC,
    LC_SUBCLASS_DESC,
    _classify_geography_region,
    categorize_discipline,
    configure_fy_window,
    get_class,
    get_subclass,
    normalize_title,
    parse_full_headings,
    parse_geographic_terms,
    pivot_by_year,
)


def _first_fy() -> str:
    """Current FY_COLS[0], or empty string if no data loaded yet."""
    return FY_COLS[0] if FY_COLS else ""


def _last_fy() -> str:
    return FY_COLS[-1] if FY_COLS else ""


def _fy_label() -> str:
    """'FY-2023 → FY-2026' for headers, or empty string before load."""
    if not FY_COLS:
        return ""
    if len(FY_COLS) == 1:
        return FY_COLS[0]
    return f"{FY_COLS[0]} \u2192 {FY_COLS[-1]}"


# ---------------------------------------------------------------------------
# Streamlit page config + styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Collection Trend Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

POS_COLOR = "#3E8E4F"
NEG_COLOR = "#B84343"
NEUTRAL_COLOR = "#8A8A8A"

# ---------------------------------------------------------------------------
# LC subclass captions
# ---------------------------------------------------------------------------
# Transcribed from the Library of Congress Classification Outline
# (https://www.loc.gov/catdir/cpso/lcco/), one entry per subclass across all
# 21 classes. This is a FALLBACK only: whatever the analysis script's
# LC_SUBCLASS_DESC already defines wins, so local caption choices are never
# overridden. Its job is to stop unfamiliar subclasses from rendering as
# "XX - Subclass XX" in the charts.
#
# Two caveats worth knowing before you read a K caption literally:
#   * K's KJ-KKZ and KL-KWX spans are labeled at the grouped span level the
#     LOC outline uses ("Law of Europe", "Law of Asia and Eurasia, Africa,
#     Pacific Area, and Antarctica") rather than per-country. Fine for
#     grouping, too coarse for a law-collection review.
#   * E and F have no subclasses in LCC; the entries below are class-level
#     captions so E/F call numbers still get a readable label.
LCC_SUBCLASS_OUTLINE: dict[str, str] = {
    # Class A — General Works
    "AC": "Collections. Series. Collected works",
    "AE": "Encyclopedias",
    "AG": "Dictionaries and other general reference works",
    "AI": "Indexes",
    "AM": "Museums. Collectors and collecting",
    "AN": "Newspapers",
    "AP": "Periodicals",
    "AS": "Academies and learned societies",
    "AY": "Yearbooks. Almanacs. Directories",
    "AZ": "History of scholarship and learning. The humanities",
    # Class B — Philosophy. Psychology. Religion
    "B": "Philosophy (General)",
    "BC": "Logic",
    "BD": "Speculative philosophy",
    "BF": "Psychology",
    "BH": "Aesthetics",
    "BJ": "Ethics",
    "BL": "Religions. Mythology. Rationalism",
    "BM": "Judaism",
    "BP": "Islam. Bahai Faith. Theosophy, etc.",
    "BQ": "Buddhism",
    "BR": "Christianity",
    "BS": "The Bible",
    "BT": "Doctrinal Theology",
    "BV": "Practical Theology",
    "BX": "Christian Denominations",
    # Class C — Auxiliary Sciences of History
    "C": "Auxiliary Sciences of History (General)",
    "CB": "History of Civilization",
    "CC": "Archaeology",
    "CD": "Diplomatics. Archives. Seals",
    "CE": "Technical Chronology. Calendar",
    "CJ": "Numismatics",
    "CN": "Inscriptions. Epigraphy",
    "CR": "Heraldry",
    "CS": "Genealogy",
    "CT": "Biography",
    # Class D — World History
    "D": "History (General)",
    "DA": "Great Britain",
    "DAW": "Central Europe",
    "DB": "Austria. Liechtenstein. Hungary. Czechoslovakia",
    "DC": "France. Andorra. Monaco",
    "DD": "Germany",
    "DE": "Greco-Roman World",
    "DF": "Greece",
    "DG": "Italy. Malta",
    "DH": "Low Countries. Benelux Countries",
    "DJ": "Netherlands (Holland)",
    "DJK": "Eastern Europe (General)",
    "DK": "Russia. Soviet Union. Former Soviet Republics. Poland",
    "DL": "Northern Europe. Scandinavia",
    "DP": "Spain. Portugal",
    "DQ": "Switzerland",
    "DR": "Balkan Peninsula",
    "DS": "Asia",
    "DT": "Africa",
    "DU": "Oceania (South Seas)",
    "DX": "Romanies",
    # Classes E-F — History of the Americas (no subclasses)
    "E": "History of the Americas (General). United States (General)",
    "F": "United States local history. Americas outside the United States",
    # Class G — Geography. Anthropology. Recreation
    "G": "Geography (General). Atlases. Maps",
    "GA": "Mathematical geography. Cartography",
    "GB": "Physical geography",
    "GC": "Oceanography",
    "GE": "Environmental sciences",
    "GF": "Human ecology. Anthropogeography",
    "GN": "Anthropology",
    "GR": "Folklore",
    "GT": "Manners and customs (General)",
    "GV": "Recreation. Leisure",
    # Class H — Social Sciences
    "H": "Social sciences (General)",
    "HA": "Statistics",
    "HB": "Economic theory. Demography",
    "HC": "Economic history and conditions",
    "HD": "Industries. Land use. Labor",
    "HE": "Transportation and communications",
    "HF": "Commerce",
    "HG": "Finance",
    "HJ": "Public finance",
    "HM": "Sociology (General)",
    "HN": "Social history and conditions. Social problems. Social reform",
    "HQ": "The family. Marriage. Women",
    "HS": "Societies: secret, benevolent, etc.",
    "HT": "Communities. Classes. Races",
    "HV": "Social pathology. Social and public welfare. Criminology",
    "HX": "Socialism. Communism. Anarchism",
    # Class J — Political Science
    "J": "General legislative and executive papers",
    "JA": "Political science (General)",
    "JC": "Political theory",
    "JF": "Political institutions and public administration",
    "JJ": "Political institutions and public administration (North America)",
    "JK": "Political institutions and public administration (United States)",
    "JL": "Political institutions and public administration (Canada, Latin America, etc.)",
    "JN": "Political institutions and public administration (Europe)",
    "JQ": "Political institutions and public administration (Asia, Africa, Australia, Pacific Area, etc.)",
    "JS": "Local government. Municipal government",
    "JV": "Colonies and colonization. Emigration and immigration. International migration",
    "JZ": "International relations",
    # Class K — Law
    "K": "Law in general. Comparative and uniform law. Jurisprudence",
    "KB": "Religious law in general. Comparative religious law. Jurisprudence",
    "KBM": "Jewish law",
    "KBP": "Islamic law",
    "KBR": "History of canon law",
    "KBU": "Law of the Roman Catholic Church. The Holy See",
    "KD": "Law of the United Kingdom and Ireland",
    "KDC": "Law of Scotland",
    "KDE": "Law of Northern Ireland",
    "KDG": "Law of the Isle of Man. Channel Islands",
    "KDK": "Law of Ireland (Eire)",
    "KDZ": "Law of America. North America",
    "KE": "Law of Canada",
    "KF": "Law of the United States",
    "KG": "Law of Latin America, Mexico, Central America, West Indies, Caribbean area",
    "KH": "Law of South America",
    "KJ": "Law of Europe",
    "KJV": "Law of France",
    "KK": "Law of Germany",
    "KKC": "Law of Germany",
    "KL": "Law of Asia and Eurasia, Africa, Pacific Area, and Antarctica",
    "KM": "Law of Asia and Eurasia, Africa, Pacific Area, and Antarctica",
    "KN": "Law of Asia and Eurasia, Africa, Pacific Area, and Antarctica",
    "KP": "Law of Asia and Eurasia, Africa, Pacific Area, and Antarctica",
    "KQ": "Law of Asia and Eurasia, Africa, Pacific Area, and Antarctica",
    "KR": "Law of Asia and Eurasia, Africa, Pacific Area, and Antarctica",
    "KS": "Law of Asia and Eurasia, Africa, Pacific Area, and Antarctica",
    "KT": "Law of Asia and Eurasia, Africa, Pacific Area, and Antarctica",
    "KU": "Law of Asia and Eurasia, Africa, Pacific Area, and Antarctica",
    "KV": "Law of Asia and Eurasia, Africa, Pacific Area, and Antarctica",
    "KW": "Law of Asia and Eurasia, Africa, Pacific Area, and Antarctica",
    "KWX": "Law of Asia and Eurasia, Africa, Pacific Area, and Antarctica",
    "KZ": "Law of nations",
    # Class L — Education
    "L": "Education (General)",
    "LA": "History of education",
    "LB": "Theory and practice of education",
    "LC": "Special aspects of education",
    "LD": "Individual institutions - United States",
    "LE": "Individual institutions - America (except United States)",
    "LF": "Individual institutions - Europe",
    "LG": "Individual institutions - Asia, Africa, Indian Ocean islands, Australia, New Zealand, Pacific islands",
    "LH": "College and school magazines and papers",
    "LJ": "Student fraternities and societies, United States",
    "LT": "Textbooks",
    # Class M — Music
    "M": "Music",
    "ML": "Literature on music",
    "MT": "Musical instruction and study",
    # Class N — Fine Arts
    "N": "Visual arts",
    "NA": "Architecture",
    "NB": "Sculpture",
    "NC": "Drawing. Design. Illustration",
    "ND": "Painting",
    "NE": "Print media",
    "NK": "Decorative arts",
    "NX": "Arts in general",
    # Class P — Language and Literature
    "P": "Philology. Linguistics",
    "PA": "Greek language and literature. Latin language and literature",
    "PB": "Modern languages. Celtic languages",
    "PC": "Romanic languages",
    "PD": "Germanic languages. Scandinavian languages",
    "PE": "English language",
    "PF": "West Germanic languages",
    "PG": "Slavic languages. Baltic languages. Albanian language",
    "PH": "Uralic languages. Basque language",
    "PJ": "Oriental languages and literatures",
    "PK": "Indo-Iranian languages and literatures",
    "PL": "Languages and literatures of Eastern Asia, Africa, Oceania",
    "PM": "Hyperborean, Indian, and artificial languages",
    "PN": "Literature (General)",
    "PQ": "French, Italian, Spanish, and Portuguese literature",
    "PR": "English literature",
    "PS": "American literature",
    "PT": "German, Dutch, Flemish, Afrikaans, Scandinavian, and Icelandic literature",
    "PZ": "Fiction and juvenile belles lettres",
    # Class Q — Science
    "Q": "Science (General)",
    "QA": "Mathematics",
    "QB": "Astronomy",
    "QC": "Physics",
    "QD": "Chemistry",
    "QE": "Geology",
    "QH": "Natural history. Biology",
    "QK": "Botany",
    "QL": "Zoology",
    "QM": "Human anatomy",
    "QP": "Physiology",
    "QR": "Microbiology",
    # Class R — Medicine
    "R": "Medicine (General)",
    "RA": "Public aspects of medicine",
    "RB": "Pathology",
    "RC": "Internal medicine",
    "RD": "Surgery",
    "RE": "Ophthalmology",
    "RF": "Otorhinolaryngology",
    "RG": "Gynecology and obstetrics",
    "RJ": "Pediatrics",
    "RK": "Dentistry",
    "RL": "Dermatology",
    "RM": "Therapeutics. Pharmacology",
    "RS": "Pharmacy and materia medica",
    "RT": "Nursing",
    "RV": "Botanic, Thomsonian, and eclectic medicine",
    "RX": "Homeopathy",
    "RZ": "Other systems of medicine",
    # Class S — Agriculture
    "S": "Agriculture (General)",
    "SB": "Plant culture",
    "SD": "Forestry",
    "SF": "Animal culture",
    "SH": "Aquaculture. Fisheries. Angling",
    "SK": "Hunting sports",
    # Class T — Technology
    "T": "Technology (General)",
    "TA": "Engineering (General). Civil engineering",
    "TC": "Hydraulic engineering. Ocean engineering",
    "TD": "Environmental technology. Sanitary engineering",
    "TE": "Highway engineering. Roads and pavements",
    "TF": "Railroad engineering and operation",
    "TG": "Bridge engineering",
    "TH": "Building construction",
    "TJ": "Mechanical engineering and machinery",
    "TK": "Electrical engineering. Electronics. Nuclear engineering",
    "TL": "Motor vehicles. Aeronautics. Astronautics",
    "TN": "Mining engineering. Metallurgy",
    "TP": "Chemical technology",
    "TR": "Photography",
    "TS": "Manufactures",
    "TT": "Handicrafts. Arts and crafts",
    "TX": "Home economics",
    # Class U — Military Science
    "U": "Military science (General)",
    "UA": "Armies: Organization, distribution, military situation",
    "UB": "Military administration",
    "UC": "Maintenance and transportation",
    "UD": "Infantry",
    "UE": "Cavalry. Armor",
    "UF": "Artillery",
    "UG": "Military engineering. Air forces",
    "UH": "Other services",
    # Class V — Naval Science
    "V": "Naval science (General)",
    "VA": "Navies: Organization, distribution, naval situation",
    "VB": "Naval administration",
    "VC": "Naval maintenance",
    "VD": "Naval seamen",
    "VE": "Marines",
    "VF": "Naval ordnance",
    "VG": "Minor services of navies",
    "VK": "Navigation. Merchant marine",
    "VM": "Naval architecture. Shipbuilding. Marine engineering",
    # Class Z — Bibliography. Library Science
    "Z": "Books (General). Writing. Paleography. Book industries and trade. Libraries. Bibliography",
    "ZA": "Information resources (General)",
}


def subclass_description(code: str) -> str:
    """Best available caption for an LC subclass code.

    Resolution order: the script's own LC_SUBCLASS_DESC, then the LOC outline
    above, then progressively shorter prefixes (so a three-letter code like
    KFA falls back to KF, then K), then a plain placeholder. The prefix walk
    matters because get_subclass() may hand back three letters for codes like
    DAW, DJK, KBM or the KF state schedules.
    """
    if not isinstance(code, str) or not code:
        return "(no LC subclass)"
    if code in LC_SUBCLASS_DESC:
        return LC_SUBCLASS_DESC[code]
    if code in LCC_SUBCLASS_OUTLINE:
        return LCC_SUBCLASS_OUTLINE[code]
    for cut in range(len(code) - 1, 0, -1):
        stem = code[:cut]
        if stem in LC_SUBCLASS_DESC:
            return f"{LC_SUBCLASS_DESC[stem]} (under {stem})"
        if stem in LCC_SUBCLASS_OUTLINE:
            return f"{LCC_SUBCLASS_OUTLINE[stem]} (under {stem})"
    return f"Subclass {code}"


# ---------------------------------------------------------------------------
# Data loading + prep
# ---------------------------------------------------------------------------

REQUIRED_CIRC_COLS = {
    "Title", "Loan Fiscal Year", "Author", "Call Number",
    "Subjects", "Loans (In House + Not In House)", "Publication Year",
}


@st.cache_data(show_spinner=False)
def load_circulation(file_bytes: bytes) -> pd.DataFrame:
    """Read the circulation CSV and attach LC/discipline metadata.

    Also detects the fiscal-year window and populates the script module's
    FY_COLS / FIRST_FY / LAST_FY / FY_SHORT constants in place, so every
    downstream compute function keys off the actual data.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    missing = REQUIRED_CIRC_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["LC_Class"] = df["Call Number"].apply(get_class)
    df["LC_Subclass"] = df["Call Number"].apply(get_subclass)
    df["Discipline"] = df["LC_Subclass"].apply(categorize_discipline)
    configure_fy_window(df)
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
         "Cumulative % Change (trend)", "Endpoint % Change",
         "Absolute Change"]
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
         "Cumulative % Change (trend)", "Endpoint % Change",
         "Absolute Change"]
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
         "Cumulative % Change (trend)", "Endpoint % Change",
         "Absolute Change"]
    ]


def split_heading_terms(heading: str) -> list[str]:
    """Split a full LC heading into its component terms.

    'Politics and government--United States--19th century' becomes
    ['Politics and government', 'United States', '19th century'].
    Kept local rather than imported so the dashboard doesn't depend on the
    script gaining a term-splitter. Empty fragments are dropped.
    """
    if not isinstance(heading, str):
        return []
    return [part.strip(" .,") for part in heading.split("--") if part.strip(" .,")]


# Level metadata: the two granularities the Subjects × LC tab can work at.
# Keeping the column name, caption resolver and label in one place means the
# compute + render path is identical for both — no parallel code paths.
LC_LEVELS = {
    "LC Class": {
        "col": "LC_Class",
        "describe": lambda c: LC_CLASS_DESC.get(c, f"{c} - Class {c}"),
        "noun": "class",
        "noun_plural": "classes",
    },
    "LC Subclass": {
        "col": "LC_Subclass",
        "describe": subclass_description,
        "noun": "subclass",
        "noun_plural": "subclasses",
    },
}


@st.cache_data(show_spinner=False)
def compute_subject_by_lc(df: pd.DataFrame, granularity: str = "Full headings",
                          level: str = "LC Class") -> pd.DataFrame:
    """Loans per (LC class-or-subclass, subject) pair.

    granularity="Full headings" keeps whole heading strings intact; "Terms"
    splits them on '--' so a term like 'United States' is counted once no
    matter which heading it came from. Either way a term is counted at most
    once per bib row, so a record carrying two headings that both include
    'United States' contributes its loans a single time.

    level picks the LC granularity — class (H) or subclass (HQ). Subclass is
    the sharper cut for collection work but spreads the same loans over ~10x
    the buckets, so cells thin out fast.

    Not mode-specific: this aggregates raw loans across whatever fiscal years
    are selected, so it works the same in snapshot and trend mode.
    """
    spec = LC_LEVELS[level]
    lc_col = spec["col"]

    d = df.copy()
    d["_row_id"] = np.arange(len(d))
    d["_subj"] = d["Subjects"].apply(parse_full_headings)
    exploded = d.explode("_subj")
    exploded = exploded[exploded["_subj"].notna() & (exploded["_subj"] != "")]

    if granularity == "Terms":
        exploded["_subj"] = exploded["_subj"].apply(split_heading_terms)
        exploded = exploded.explode("_subj")
        exploded = exploded[exploded["_subj"].notna() & (exploded["_subj"] != "")]

    # One count per (row, term) — protects against double-counting loans.
    exploded = exploded.drop_duplicates(subset=["_row_id", "_subj"])

    # Records with an unparseable call number have no bucket to sit under.
    exploded = exploded[exploded[lc_col].notna() & (exploded[lc_col] != "")]
    if exploded.empty:
        return pd.DataFrame(
            columns=["Discipline", lc_col, "LC Description",
                     "Subject", "Loans", "Titles"]
        )

    pairs = (
        exploded.groupby([lc_col, "_subj"])
        .agg(Loans=("Loans (In House + Not In House)", "sum"),
             Titles=("Title", "nunique"))
        .reset_index()
        .rename(columns={"_subj": "Subject"})
    )
    pairs["LC Description"] = pairs[lc_col].map(spec["describe"])
    pairs["Discipline"] = pairs[lc_col].apply(categorize_discipline)
    return pairs[
        ["Discipline", lc_col, "LC Description", "Subject", "Loans", "Titles"]
    ].sort_values([lc_col, "Loans"], ascending=[True, False])


@st.cache_data(show_spinner=False)
def compute_subject_breadth(pairs: pd.DataFrame,
                            level: str = "LC Class") -> pd.DataFrame:
    """Collapse the (class-or-subclass, subject) pairs to one row per subject.

    'LC Buckets' is how many distinct classes or subclasses the term shows up
    in — the breadth measure. '% in Top Bucket' says how concentrated it is:
    a term at 98% sits inside one bucket, a term at 30% is genuinely
    cross-cutting.
    """
    spec = LC_LEVELS[level]
    lc_col = spec["col"]
    count_label = f"LC {spec['noun_plural'].capitalize()}"
    top_label = f"Top {spec['noun'].capitalize()}"
    pct_label = f"% in Top {spec['noun'].capitalize()}"
    present_label = f"{spec['noun_plural'].capitalize()} Present"

    if pairs.empty:
        return pd.DataFrame(
            columns=["Subject", count_label, "Loans", "Titles",
                     top_label, f"{top_label} Description", pct_label,
                     present_label]
        )

    totals = (
        pairs.groupby("Subject")
        .agg(**{count_label: (lc_col, "nunique"),
                "Loans": ("Loans", "sum"),
                "Titles": ("Titles", "sum")})
        .reset_index()
    )
    top = (
        pairs.sort_values("Loans", ascending=False)
        .drop_duplicates("Subject")
        [["Subject", lc_col, "LC Description", "Loans"]]
        .rename(columns={lc_col: top_label,
                         "LC Description": f"{top_label} Description",
                         "Loans": "_top_loans"})
    )
    out = totals.merge(top, on="Subject", how="left")
    out[pct_label] = np.where(
        out["Loans"] > 0, out["_top_loans"] / out["Loans"] * 100.0, np.nan
    ).round(1)
    present = (
        pairs.groupby("Subject")[lc_col]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .rename(present_label).reset_index()
    )
    out = out.merge(present, on="Subject", how="left")
    return out[
        ["Subject", count_label, "Loans", "Titles",
         top_label, f"{top_label} Description", pct_label, present_label]
    ].sort_values([count_label, "Loans"], ascending=[False, False])


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
         "Cumulative % Change (trend)", "Endpoint % Change",
         "Absolute Change"]
    ]

    region = pivot_by_year(exploded, "Region")
    region = region[
        ["Region", *FY_COLS,
         "Mean Annual Loans", "Trend Slope (loans/yr)", "Trend R^2",
         "Cumulative % Change (trend)", "Endpoint % Change",
         "Absolute Change"]
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
    # Auto-split window: first half = early, second half = recent.
    # For a 4-year window this gives [FY0, FY1] / [FY2, FY3]; for an odd-numbered
    # window the later half gets the extra year.
    if len(FY_COLS) < 2:
        raise ValueError("Weeding needs at least 2 fiscal years")
    mid = len(FY_COLS) // 2
    early = list(FY_COLS[:mid])
    recent = list(FY_COLS[mid:])
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
    if len(FY_COLS) < 2:
        raise ValueError("E-book analysis needs at least 2 fiscal years")
    mid = len(FY_COLS) // 2
    title_yr["Early Loans"] = title_yr[list(FY_COLS[:mid])].sum(axis=1)
    title_yr["Recent Loans"] = title_yr[list(FY_COLS[mid:])].sum(axis=1)

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


# Snapshot compute functions — used when the user picks a single fiscal year.
# Trend metrics don't apply; instead we aggregate raw loans per category.

@st.cache_data(show_spinner=False)
def compute_snapshot(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Aggregate loans per group for a single-year view. No trend metrics."""
    out = (
        df.groupby(group_col)["Loans (In House + Not In House)"]
        .sum().reset_index().rename(columns={"Loans (In House + Not In House)": "Loans"})
    )
    return out.sort_values("Loans", ascending=False)


@st.cache_data(show_spinner=False)
def compute_subject_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["FullHeadings"] = d["Subjects"].apply(parse_full_headings)
    exploded = d.explode("FullHeadings").rename(columns={"FullHeadings": "Subject"})
    exploded = exploded[exploded["Subject"].notna() & (exploded["Subject"] != "")]
    out = (
        exploded.groupby(["Discipline", "Subject"])
        ["Loans (In House + Not In House)"].sum().reset_index()
        .rename(columns={"Loans (In House + Not In House)": "Loans"})
    )
    return out.sort_values("Loans", ascending=False)


@st.cache_data(show_spinner=False)
def compute_geo_snapshot(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["Geographies"] = d["Subjects"].apply(parse_geographic_terms)
    exploded = d.explode("Geographies").rename(columns={"Geographies": "Geography"})
    exploded = exploded[exploded["Geography"].notna() & (exploded["Geography"] != "")]
    if exploded.empty:
        return pd.DataFrame(), pd.DataFrame()
    exploded["Region"] = exploded["Geography"].apply(_classify_geography_region)
    places = (
        exploded.groupby(["Region", "Geography"])
        ["Loans (In House + Not In House)"].sum().reset_index()
        .rename(columns={"Loans (In House + Not In House)": "Loans"})
        .sort_values("Loans", ascending=False)
    )
    regions = (
        exploded.groupby("Region")["Loans (In House + Not In House)"]
        .sum().reset_index().rename(columns={"Loans (In House + Not In House)": "Loans"})
        .sort_values("Loans", ascending=False)
    )
    return places, regions


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
            title=f"Cumulative % Change (trend fit across {_first_fy()}–{_last_fy()})",
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


def make_snapshot_chart(df: pd.DataFrame, label_col: str, value_col: str,
                        title: str, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of raw loan counts — used in single-year snapshot mode."""
    plot_df = df.head(top_n).sort_values(value_col, ascending=True)
    fig = go.Figure(go.Bar(
        y=plot_df[label_col],
        x=plot_df[value_col],
        orientation="h",
        marker=dict(color=POS_COLOR, line=dict(width=0)),
        text=[f"{int(v):,}" for v in plot_df[value_col]],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left",
                   font=dict(size=15, family="system-ui")),
        xaxis=dict(title="Loans", gridcolor="#eee"),
        yaxis=dict(title="", automargin=True),
        height=max(360, 30 * len(plot_df) + 120),
        plot_bgcolor="white",
        margin=dict(l=0, r=80, t=60, b=50),
        showlegend=False,
    )
    return fig


def make_subject_lc_heatmap(pairs: pd.DataFrame, subjects: list[str],
                            title: str, level: str = "LC Class") -> go.Figure:
    """Subjects (rows) × LC classes or subclasses (columns), shaded by loans.

    Shows where a term's use actually sits: a row lit up in one column is a
    term owned by one bucket, a row with several lit columns is cross-cutting.
    """
    spec = LC_LEVELS[level]
    lc_col = spec["col"]

    def _empty(msg="No data to display for these filters."):
        fig = go.Figure()
        fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=360, plot_bgcolor="white")
        return fig

    if pairs.empty or not subjects:
        return _empty()

    sub = pairs[pairs["Subject"].isin(subjects)]
    mat = sub.pivot_table(index="Subject", columns=lc_col, values="Loans",
                          aggfunc="sum", fill_value=0)
    # Plot in the order handed in, reversed so the strongest lands at the top.
    mat = mat.reindex(index=[s for s in subjects if s in mat.index]).iloc[::-1]
    mat = mat.loc[:, mat.sum(axis=0) > 0]
    if mat.empty:
        return _empty()

    describe = spec["describe"]
    hover = [
        [f"<b>{subj}</b><br>{spec['noun'].capitalize()} {code} — "
         f"{describe(code)}<br>Loans: {int(val):,}"
         for code, val in zip(mat.columns, row)]
        for subj, row in zip(mat.index, mat.values)
    ]
    fig = go.Figure(go.Heatmap(
        z=mat.values,
        x=list(mat.columns),
        y=list(mat.index),
        colorscale=[[0, "#F5F5F5"], [1, POS_COLOR]],
        hovertext=hover,
        hoverinfo="text",
        colorbar=dict(title="Loans"),
    ))
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left",
                   font=dict(size=15, family="system-ui")),
        xaxis=dict(title=level, side="top", type="category",
                   tickangle=-45 if len(mat.columns) > 12 else 0),
        yaxis=dict(title="", automargin=True, type="category"),
        height=max(400, 24 * len(mat) + 160),
        # Wide subclass matrices need room for the rotated tick labels.
        width=None,
        plot_bgcolor="white",
        margin=dict(l=0, r=40, t=110 if len(mat.columns) > 12 else 90, b=40),
    )
    return fig


def download_button_for_df(df: pd.DataFrame, label: str, filename: str,
                            key: Optional[str] = None):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename,
                       mime="text/csv", key=key)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("📚 Collection Trend Analysis")
if FY_COLS:
    st.caption(f"Fiscal-year window: {_first_fy()} → {_last_fy()}  ·  "
               f"trend metrics fit a line through all {len(FY_COLS)} years, "
               "so interior spikes and dips register in the shift number.")
else:
    st.caption("Upload a physical-usage CSV to begin. Trend metrics fit a "
               "line through every fiscal year in the file, so interior "
               "spikes and dips register in the shift number.")

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
        _baseline_fy = _first_fy() or "first FY"
        min_baseline_subj = st.number_input(
            f"Subject baseline: min {_baseline_fy} loans",
            min_value=1, max_value=200, value=25,
            help="Filters out noise from headings with tiny baselines.",
        )
        min_baseline_geo = st.number_input(
            f"Geography baseline: min {_baseline_fy} loans",
            min_value=1, max_value=200, value=20,
        )
        min_baseline_sub = st.number_input(
            f"LC subclass baseline: min {_baseline_fy} loans",
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
        first_fy = _first_fy() or "the first fiscal year"
        n_fys = len(FY_COLS) if FY_COLS else "all"
        st.markdown(
            f"""
            **What it does.** Reads a physical-circulation CSV and produces
            trend analyses for LC classifications, subject headings (full LC
            headings), and geographic mentions across the fiscal-year window,
            plus tiered weeding and e-book candidate lists.

            **Metric.** Every "% change" is a cumulative trend fit — a linear
            regression through all {n_fys} fiscal-year totals, expressed as a
            percentage of the {first_fy} baseline. This means a subject that
            went 100 → 200 → 200 → 100 shows near-zero movement (returned to
            baseline) rather than the "0% change" you'd get from an endpoint
            comparison. R² tells you how well the straight line actually fits
            the yearly points — trust the direction when R² is high, treat noisy
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
    circ_full = load_circulation(circ_file.getvalue())
except Exception as exc:
    st.error(f"Could not read circulation CSV: {exc}")
    st.stop()

holdings_df = None
if holdings_file is not None:
    try:
        holdings_df = load_holdings(holdings_file.getvalue())
    except Exception as exc:
        st.warning(f"Holdings CSV skipped: {exc}")

# ----- Fiscal-year selection ------------------------------------------------
# Which years to include in the analysis. Defaults to the whole detected window
# so nothing changes for users who just want to look at everything at once.
all_years = sorted(circ_full["Loan Fiscal Year"].dropna().unique().tolist())

st.markdown("### Fiscal years to analyze")
c_years, c_mode = st.columns([3, 1])
with c_years:
    selected_years = st.multiselect(
        "Include these fiscal years",
        options=all_years,
        default=all_years,
        help="Pick any subset. Choose 2+ for trend analysis; choose 1 to see "
             "that year on its own as a snapshot.",
    )
with c_mode:
    if len(selected_years) >= 2:
        st.success(f"**Trend mode** · {len(selected_years)} years")
    elif len(selected_years) == 1:
        st.info(f"**Snapshot mode** · {selected_years[0]}")
    else:
        st.error("Pick at least one FY")

if not selected_years:
    st.stop()

# Filter to just the selected FYs and re-run window detection so every
# downstream compute uses the picked window (not the full uploaded window)
circ_df = circ_full[circ_full["Loan Fiscal Year"].isin(selected_years)].copy()
configure_fy_window(circ_df)
is_snapshot = len(selected_years) == 1

# ----- Header stats ---------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (selected FYs)", f"{len(circ_df):,}")
c2.metric("Unique titles", f"{circ_df['Title'].nunique():,}")
c3.metric(
    f"Total loans ({_fy_label() or 'selected FYs'})",
    f"{int(circ_df['Loans (In House + Not In House)'].sum()):,}",
)
c4.metric(
    "Holdings loaded",
    f"{len(holdings_df):,}" if holdings_df is not None else "—",
)

# ----- Compute --------------------------------------------------------------
if is_snapshot:
    with st.spinner("Building single-year snapshot…"):
        snap_lc_class = compute_snapshot(circ_df, "LC_Class")
        snap_lc_class["Description"] = snap_lc_class["LC_Class"].map(
            lambda c: LC_CLASS_DESC.get(c, f"{c} - Class {c}")
        )
        snap_lc_class["Discipline"] = snap_lc_class["LC_Class"].apply(categorize_discipline)
        snap_lc_sub = compute_snapshot(circ_df, "LC_Subclass")
        snap_lc_sub["Description"] = snap_lc_sub["LC_Subclass"].map(
            lambda s: LC_SUBCLASS_DESC.get(s, f"{s} - Subclass {s}")
        )
        snap_lc_sub["Discipline"] = snap_lc_sub["LC_Subclass"].apply(categorize_discipline)
        snap_subj = compute_subject_snapshot(circ_df)
        snap_geo, snap_region = compute_geo_snapshot(circ_df)
        ebook_df = compute_ebook_candidates(circ_df) if len(FY_COLS) >= 2 else None
        holdings_weed_df = (
            compute_holdings_weeding(circ_df, holdings_df)
            if holdings_df is not None else None
        )
    # Placeholders — the trend tables aren't used in snapshot mode
    lc_class_df = lc_sub_df = subj_df = None
    geo_df = region_df = None
    weeding_df = None
else:
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
    snap_lc_class = snap_lc_sub = snap_subj = None
    snap_geo = snap_region = None

# ----- Tabs -----------------------------------------------------------------
tab_labels = [
    "🏠 Overview",
    "LC Class",
    "LC Subclass",
    "Subject Headings",
    "Subjects × LC",
    "Geographic",
    "Weeding",
    "E-Book Candidates",
]
if holdings_weed_df is not None:
    tab_labels.append("Holdings Weeding")
tab_labels.append("📘 Guide")

# Keyed by label rather than position — the Holdings tab is conditional and
# positional indices break every time a tab is inserted.
tabs = dict(zip(tab_labels, st.tabs(tab_labels)))

# --- Overview ---------------------------------------------------------------
with tabs["🏠 Overview"]:
    if is_snapshot:
        st.subheader(f"Snapshot — {selected_years[0]}")
        st.caption("Single fiscal year selected. Showing top categories by "
                   "raw loan count. Trend analysis is disabled for one-year "
                   "views (there's nothing to fit a line through).")

        cA, cB = st.columns(2)
        with cA:
            st.metric(
                "Top LC class",
                snap_lc_class.iloc[0]["Description"]
                if not snap_lc_class.empty else "—",
                f"{int(snap_lc_class.iloc[0]['Loans']):,} loans"
                if not snap_lc_class.empty else "",
            )
            if not snap_subj.empty:
                st.metric(
                    "Top subject heading",
                    snap_subj.iloc[0]["Subject"],
                    f"{int(snap_subj.iloc[0]['Loans']):,} loans",
                )
        with cB:
            if not snap_geo.empty:
                st.metric(
                    "Top geographic mention",
                    snap_geo.iloc[0]["Geography"],
                    f"{int(snap_geo.iloc[0]['Loans']):,} loans",
                )
            if not snap_region.empty:
                st.metric(
                    "Top region",
                    snap_region.iloc[0]["Region"],
                    f"{int(snap_region.iloc[0]['Loans']):,} loans",
                )

        st.markdown("---")
        st.markdown(f"**Loans by region — {selected_years[0]}**")
        if not snap_region.empty:
            st.plotly_chart(
                make_snapshot_chart(snap_region, "Region", "Loans",
                                    f"Loans by Region — {selected_years[0]}"),
                use_container_width=True,
            )
    else:
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
                                     25, 25, _first_fy(), min_baseline_subj),
            "Subject", "Subject Heading",
        )
        if not geo_df.empty:
            _top_movers_summary(
                top_risers_and_decliners(geo_df, "Cumulative % Change (trend)",
                                         25, 25, _first_fy(), min_baseline_geo),
                "Geography", "Geography",
            )

        st.markdown("---")
        st.markdown("**Region-level roll-up** (continent buckets across the "
                    "selected fiscal years)")
        if not region_df.empty:
            st.plotly_chart(
                make_shift_chart(
                    region_df, "Region", "Cumulative % Change (trend)",
                    "Geographic Mentions by Region — Cumulative Trend",
                ),
                use_container_width=True,
            )

# --- LC Class ---------------------------------------------------------------
with tabs["LC Class"]:
    if is_snapshot:
        st.subheader(f"LC Class — {selected_years[0]} Snapshot")
        disc_choice = st.selectbox(
            "Filter by discipline",
            ["All disciplines"] + list(DISCIPLINE_ORDER),
            key="lc_class_disc_snap",
        )
        d = snap_lc_class if disc_choice == "All disciplines" \
            else snap_lc_class[snap_lc_class["Discipline"] == disc_choice]
        st.plotly_chart(
            make_snapshot_chart(d, "Description", "Loans",
                                f"LC Classes by Loans — {selected_years[0]}"),
            use_container_width=True,
        )
        st.dataframe(d, use_container_width=True, hide_index=True)
        download_button_for_df(
            snap_lc_class, "⬇ Download snapshot CSV",
            f"lc_class_snapshot_{selected_years[0]}.csv",
            key="dl_lc_class_snap",
        )
    else:
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
                               f"lc_class_circulation_shifts_{script.fy_window_slug()}.csv",
                               key="dl_lc_class")

# --- LC Subclass ------------------------------------------------------------
with tabs["LC Subclass"]:
    if is_snapshot:
        st.subheader(f"LC Subclass — {selected_years[0]} Snapshot")
        disc_choice = st.selectbox(
            "Filter by discipline",
            ["All disciplines"] + list(DISCIPLINE_ORDER),
            key="lc_sub_disc_snap",
        )
        d = snap_lc_sub if disc_choice == "All disciplines" \
            else snap_lc_sub[snap_lc_sub["Discipline"] == disc_choice]
        st.plotly_chart(
            make_snapshot_chart(d, "Description", "Loans",
                                f"Top LC Subclasses — {selected_years[0]}",
                                top_n=25),
            use_container_width=True,
        )
        st.dataframe(d, use_container_width=True, hide_index=True)
        download_button_for_df(
            snap_lc_sub, "⬇ Download snapshot CSV",
            f"lc_subclass_snapshot_{selected_years[0]}.csv",
            key="dl_lc_sub_snap",
        )
    else:
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
            baseline_col=_first_fy(), baseline_min=min_baseline_sub,
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
                               f"lc_subclass_circulation_shifts_{script.fy_window_slug()}.csv",
                               key="dl_lc_sub")

# --- Subject Headings -------------------------------------------------------
with tabs["Subject Headings"]:
    if is_snapshot:
        st.subheader(f"Subject Headings — {selected_years[0]} Snapshot")
        st.caption("Full LC heading strings ranked by loans this fiscal year.")
        disc_choice = st.selectbox(
            "Filter by discipline",
            ["All disciplines"] + list(DISCIPLINE_ORDER),
            key="subj_disc_snap",
        )
        d = snap_subj if disc_choice == "All disciplines" \
            else snap_subj[snap_subj["Discipline"] == disc_choice]
        st.plotly_chart(
            make_snapshot_chart(d, "Subject", "Loans",
                                f"Top Subject Headings — {selected_years[0]}",
                                top_n=25),
            use_container_width=True,
        )
        search_term = st.text_input(
            "🔍 Search subject headings",
            placeholder="e.g. 'African American', 'Louisiana', 'artificial'",
            key="subj_search_snap",
        )
        display = d
        if search_term:
            display = display[display["Subject"].str.contains(
                search_term, case=False, na=False)]
        st.dataframe(display, use_container_width=True, hide_index=True)
        download_button_for_df(
            snap_subj, "⬇ Download snapshot CSV",
            f"subject_snapshot_{selected_years[0]}.csv",
            key="dl_subj_snap",
        )
    else:
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
            baseline_col=_first_fy(), baseline_min=min_baseline_subj,
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
                               f"subject_term_shifts_{script.fy_window_slug()}.csv",
                               key="dl_subj")

# --- Subjects × LC ----------------------------------------------------------
# Two questions, both phrased as "subject terms across LC":
#   1. What are the popular terms *within* each class/subclass?
#   2. Which terms show up *across* many of them (interdisciplinary pull)?
# The level toggle runs the identical analysis at class or subclass depth.
with tabs["Subjects × LC"]:
    st.subheader("Subject Terms Across LC Classes and Subclasses")
    st.caption(
        f"Loans totalled across the selected fiscal years "
        f"({_fy_label() or 'selected FYs'}). No trend fitting here — this is a "
        "'where does subject use sit' view, so it reads the same in snapshot "
        "and trend mode."
    )

    c_level, c_gran, c_topn = st.columns([1.1, 1.4, 1])
    with c_level:
        lc_level = st.radio(
            "LC level",
            list(LC_LEVELS),
            horizontal=False,
            key="subjlc_level",
            help="Class is the 21-bucket view (H, P, Q). Subclass is the "
                 "working view for collection decisions (HQ, PS, QA) but "
                 "splits the same loans across far more buckets.",
        )
    with c_gran:
        granularity = st.radio(
            "Subject granularity",
            ["Full headings", "Terms"],
            horizontal=False,
            key="subjlc_gran",
            help="Full headings keeps 'Politics and government--France' whole. "
                 "Terms splits on '--' so 'France' is counted once wherever it "
                 "appears — better for spotting terms shared between buckets.",
        )
    with c_topn:
        top_n_subj = st.number_input(
            "Terms to show", min_value=5, max_value=60, value=20, step=5,
            key="subjlc_topn",
        )

    spec = LC_LEVELS[lc_level]
    lc_col = spec["col"]
    noun, noun_plural = spec["noun"], spec["noun_plural"]
    count_label = f"LC {noun_plural.capitalize()}"
    pct_label = f"% in Top {noun.capitalize()}"

    with st.spinner(f"Cross-tabulating subjects against LC {noun_plural}…"):
        pairs_df = compute_subject_by_lc(circ_df, granularity, lc_level)

    unclassed = int((circ_df[lc_col].isna() | (circ_df[lc_col] == "")).sum())
    if unclassed:
        st.caption(
            f"⚠ {unclassed:,} row(s) had no parseable LC {noun} and are "
            "excluded from this tab. They still count in the other tabs."
        )

    if pairs_df.empty:
        st.info("No subject/LC pairs to show for the current selection.")
    else:
        breadth_df = compute_subject_breadth(pairs_df, lc_level)

        st.markdown(f"#### Popular terms within a {noun}")
        bucket_options = (
            pairs_df.groupby([lc_col, "LC Description"])["Loans"]
            .sum().reset_index().sort_values("Loans", ascending=False)
        )
        bucket_labels = {
            f"{row[lc_col]} — {row['LC Description']}": row[lc_col]
            for _, row in bucket_options.iterrows()
        }
        pick = st.selectbox(
            lc_level,
            [f"All {noun_plural} (side by side)"] + list(bucket_labels),
            key="subjlc_bucket",
        )

        if pick.startswith("All "):
            top_overall = (
                pairs_df.groupby("Subject")["Loans"].sum()
                .sort_values(ascending=False).head(int(top_n_subj)).index.tolist()
            )
            # A subclass matrix can run to 100+ columns; cap it so the heatmap
            # stays readable and let the table carry the long tail.
            heat_src = pairs_df
            n_buckets = pairs_df[lc_col].nunique()
            max_cols = 30
            if n_buckets > max_cols:
                keep = (
                    pairs_df[pairs_df["Subject"].isin(top_overall)]
                    .groupby(lc_col)["Loans"].sum()
                    .nlargest(max_cols).index
                )
                heat_src = pairs_df[pairs_df[lc_col].isin(keep)]
                st.caption(
                    f"Showing the {max_cols} busiest {noun_plural} of "
                    f"{n_buckets:,}. The table below has all of them."
                )
            st.plotly_chart(
                make_subject_lc_heatmap(
                    heat_src, top_overall,
                    f"Top {len(top_overall)} Subject "
                    f"{'Terms' if granularity == 'Terms' else 'Headings'} "
                    f"by {lc_level}",
                    level=lc_level,
                ),
                use_container_width=True,
                key="chart_subjlc_heatmap",
            )
            st.caption(
                "Darker cell = more loans. A row shaded in one column belongs "
                f"to that {noun}; a row shaded across several is being pulled "
                "from more than one part of the collection."
            )
        else:
            code = bucket_labels[pick]
            d_bucket = pairs_df[pairs_df[lc_col] == code].head(int(top_n_subj))
            st.plotly_chart(
                make_snapshot_chart(
                    d_bucket, "Subject", "Loans",
                    f"Top Subjects — {pick}",
                    top_n=int(top_n_subj),
                ),
                use_container_width=True,
                key="chart_subjlc_single_bucket",
            )

        pair_search = st.text_input(
            f"🔍 Search the subject/{noun} table",
            placeholder="e.g. 'Louisiana', 'women', 'public health'",
            key="subjlc_search",
        )
        pair_display = pairs_df if pick.startswith("All ") \
            else pairs_df[pairs_df[lc_col] == bucket_labels[pick]]
        if pair_search:
            pair_display = pair_display[
                pair_display["Subject"].str.contains(pair_search, case=False, na=False)
            ]
        st.dataframe(
            pair_display.sort_values("Loans", ascending=False),
            use_container_width=True, hide_index=True,
        )
        download_button_for_df(
            pairs_df, f"⬇ Download subject-by-LC-{noun} CSV",
            f"subject_by_lc_{noun}_{script.fy_window_slug()}.csv",
            key="dl_subj_lc",
        )

        st.divider()
        st.markdown(f"#### Terms that cross the most {noun_plural}")
        st.caption(
            f"Ranked by how many distinct LC {noun_plural} a term appears in. "
            f"'{pct_label}' is the concentration check — a high number means "
            "the breadth is a long tail, a low number means the term is "
            "genuinely split across the collection. Useful for spotting "
            "interdisciplinary demand a single-bucket view would hide."
        )
        # Subclasses fragment the same loans across more buckets, so the
        # thresholds that make sense at class level are too strict here.
        default_min_buckets = 3 if lc_level == "LC Class" else 5
        c_minc, c_minl = st.columns(2)
        with c_minc:
            min_buckets = st.number_input(
                f"Minimum LC {noun_plural}", min_value=1, max_value=50,
                value=default_min_buckets, step=1,
                key=f"subjlc_minbuckets_{lc_col}",
            )
        with c_minl:
            min_loans = st.number_input(
                "Minimum total loans", min_value=0, value=10, step=5,
                key="subjlc_minloans",
            )
        cross = breadth_df[
            (breadth_df[count_label] >= min_buckets)
            & (breadth_df["Loans"] >= min_loans)
        ]
        if cross.empty:
            st.info("No terms clear those thresholds. Try lowering them, or "
                    "switch granularity to Terms — full headings rarely repeat "
                    "across buckets.")
        else:
            st.plotly_chart(
                make_snapshot_chart(
                    cross.head(int(top_n_subj)), "Subject", count_label,
                    f"Widest-Spanning Subject "
                    f"{'Terms' if granularity == 'Terms' else 'Headings'} "
                    f"(count of LC {noun_plural})",
                    top_n=int(top_n_subj),
                ),
                use_container_width=True,
                key="chart_subjlc_breadth",
            )
            st.dataframe(cross, use_container_width=True, hide_index=True)
        download_button_for_df(
            breadth_df, "⬇ Download subject breadth CSV",
            f"subject_lc_{noun}_breadth_{script.fy_window_slug()}.csv",
            key="dl_subj_breadth",
        )

# --- Geographic Trends ------------------------------------------------------
with tabs["Geographic"]:
    if is_snapshot:
        st.subheader(f"Geographic — {selected_years[0]} Snapshot")
        if snap_geo.empty:
            st.info("No geographic terms detected in this dataset.")
        else:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("**Loans by region**")
                st.plotly_chart(
                    make_snapshot_chart(snap_region, "Region", "Loans",
                                        "By Region"),
                    use_container_width=True,
                )
            with c2:
                region_options = ["All regions"] + sorted(
                    snap_geo["Region"].dropna().unique().tolist()
                )
                region_choice = st.selectbox(
                    "Filter places by region",
                    region_options, key="geo_region_snap",
                )
                d = snap_geo if region_choice == "All regions" \
                    else snap_geo[snap_geo["Region"] == region_choice]
                st.plotly_chart(
                    make_snapshot_chart(d, "Geography", "Loans",
                                        f"Top Places ({region_choice})",
                                        top_n=20),
                    use_container_width=True,
                )
            search_geo = st.text_input(
                "🔍 Search places",
                placeholder="e.g. 'Louisiana', 'New Orleans', 'France'",
                key="geo_search_snap",
            )
            display = snap_geo
            if search_geo:
                display = display[display["Geography"].str.contains(
                    search_geo, case=False, na=False)]
            st.dataframe(display, use_container_width=True, hide_index=True)
            download_button_for_df(
                snap_geo, "⬇ Download snapshot CSV",
                f"geographic_snapshot_{selected_years[0]}.csv",
                key="dl_geo_snap",
            )
    else:
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
                    baseline_col=_first_fy(), baseline_min=min_baseline_geo,
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
                    f"geographic_term_shifts_{script.fy_window_slug()}.csv",
                    key="dl_geo",
                )
            with c2:
                download_button_for_df(
                    region_df, "⬇ Download region-level shifts CSV",
                    f"geographic_region_shifts_{script.fy_window_slug()}.csv",
                    key="dl_geo_r",
                )

# --- Weeding ----------------------------------------------------------------
with tabs["Weeding"]:
    st.subheader("Weeding Candidates (Usage Decay)")
    if is_snapshot:
        st.info(
            "**Weeding needs multi-year data.** The decay signal — meaningful "
            "early use followed by silence — only makes sense when at least "
            "one 'early' and one 'recent' fiscal year are available. Select "
            "2+ fiscal years above to enable this tab."
        )
    else:
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
            f"weeding_candidates_{script.fy_window_slug()}.csv", key="dl_weed",
        )

# --- E-Book Candidates ------------------------------------------------------
with tabs["E-Book Candidates"]:
    st.subheader("E-Book Purchase Candidates")
    if is_snapshot:
        st.caption(
            "Ranked by total loans in the selected fiscal year. Rationale is "
            "based on this year's volume only; multi-year sustained/rising "
            "signals aren't available in single-year mode."
        )
    else:
        st.caption(
            f"Ranked by total loans across {_fy_label()}, with a rationale keyed "
            "to whether demand is sustained (loans in 3+ FYs) and/or rising "
            "(recent > early)."
        )
    if ebook_df is not None and not ebook_df.empty:
        st.dataframe(ebook_df, use_container_width=True, hide_index=True)
        download_button_for_df(
            ebook_df, "⬇ Download e-book candidates CSV",
            f"ebook_purchase_candidates_{script.fy_window_slug()}.csv",
            key="dl_ebook",
        )
    else:
        st.info("No e-book candidates for the current selection.")

# --- Holdings weeding (conditional) -----------------------------------------
if holdings_weed_df is not None:
    with tabs["Holdings Weeding"]:
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
            f"uncirculated_by_subclass_{script.fy_window_slug()}.csv",
            key="dl_holdings",
        )

# --- Guide ------------------------------------------------------------------
# The last tab is the decision guide, always visible. Read from DECISION_GUIDE.md
# on disk so the source of truth is a single markdown file that lives alongside
# the app.
with tabs["📘 Guide"]:
    import os
    guide_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "DECISION_GUIDE.md")
    try:
        with open(guide_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
        st.download_button(
            "⬇ Download DECISION_GUIDE.md",
            data=open(guide_path, "rb").read(),
            file_name="DECISION_GUIDE.md",
            mime="text/markdown",
            key="dl_guide",
        )
    except FileNotFoundError:
        st.info("Decision guide file (DECISION_GUIDE.md) not found in the app "
                "directory. It ships alongside the dashboard — check that "
                "both files are present.")
