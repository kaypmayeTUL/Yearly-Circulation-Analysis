"""
Library Collection Analysis: FY23-FY26
--------------------------------------
Consumes a single physical-circulation CSV (one row per Title/Call Number/Fiscal Year)
and produces:

  1. LC Class change charts (overall + one per discipline) + CSVs
  2. LC Subclass change CSV (full detail)
  3. Subject term change charts (overall + one per discipline) + CSVs
  4. Weeding candidate list (CSV)
  5. E-book purchase suggestion list (CSV)

Input columns expected:
  Title, Publication Year, Loan Fiscal Year, Location Name, Author,
  Call Number, Subjects, Loans (In House + Not In House)

Change the INPUT_PATH constant below or pass a path on the command line.
"""

import argparse
import os
import re
import string
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT = "physical_usage_23_26.csv"
DEFAULT_OUTDIR = "outputs"

# Fiscal-year columns are detected from the loaded data by configure_fy_window()
# and mutated in place. Downstream code reads FY_COLS at call time (not import
# time), so mutation is visible to importers who did `from ... import FY_COLS`.
# FIRST_FY / LAST_FY / FIRST_SHORT / LAST_SHORT are strings that get reassigned
# — anything that needs the current value must reference them through the
# module namespace (via `analysis_module.FIRST_FY`) or derive them from
# FY_COLS[0] / FY_COLS[-1] on the fly.
FY_COLS: list = []           # e.g. ["FY-2023", "FY-2024", "FY-2025", "FY-2026"]
FY_SHORT: list = []          # e.g. ["FY23", "FY24", "FY25", "FY26"]
FIRST_FY: str = ""
LAST_FY: str = ""
FIRST_SHORT: str = ""
LAST_SHORT: str = ""


def short_fy(fy_label) -> str:
    """Convert 'FY-2023' -> 'FY23'; fall back to the input if no match."""
    if fy_label is None:
        return ""
    m = re.match(r"^FY[-_ ]?(\d{4})$", str(fy_label).strip())
    if m:
        return f"FY{m.group(1)[-2:]}"
    return str(fy_label).replace(" ", "").replace("-", "").replace("_", "")


def detect_fy_window(df) -> list:
    """
    Read unique 'Loan Fiscal Year' values from the data and sort them
    chronologically by whatever 4-digit year appears in each label.
    """
    if "Loan Fiscal Year" not in df.columns:
        raise ValueError("Input must have a 'Loan Fiscal Year' column")
    unique = [
        str(v).strip() for v in df["Loan Fiscal Year"].dropna().unique()
        if str(v).strip()
    ]

    def _year_key(fy):
        m = re.search(r"(\d{4})", str(fy))
        return int(m.group(1)) if m else -1

    return sorted(set(unique), key=_year_key)


def configure_fy_window(df) -> list:
    """
    Detect the fiscal-year window from the data and populate module-level
    constants. Call once per data load. Returns the detected FY list.

    Requires at least 1 fiscal year. Callers that need trend fitting (weeding,
    e-book, subject/geographic shifts) should check `len(FY_COLS) >= 2`
    themselves — a single-year snapshot is a legitimate use case that this
    function does not need to block.
    """
    global FIRST_FY, LAST_FY, FIRST_SHORT, LAST_SHORT
    fy_cols = detect_fy_window(df)
    if len(fy_cols) < 1:
        raise ValueError(
            "No fiscal-year values detected in 'Loan Fiscal Year' column"
        )
    FY_COLS[:] = fy_cols                              # mutate in place
    FY_SHORT[:] = [short_fy(f) for f in fy_cols]      # mutate in place
    FIRST_FY = fy_cols[0]
    LAST_FY = fy_cols[-1]
    FIRST_SHORT = FY_SHORT[0]
    LAST_SHORT = FY_SHORT[-1]
    return fy_cols


def fy_window_slug() -> str:
    """Return e.g. 'fy23_fy26' for use in output file names."""
    if not FY_SHORT:
        return "fy_window"
    return f"{FY_SHORT[0].lower()}_{FY_SHORT[-1].lower()}"

# Chart styling
POS_COLOR = "#3E8E4F"   # green
NEG_COLOR = "#B84343"   # red
ZERO_COLOR = "#8A8A8A"  # gray for zero-change bars

# Discipline order (for consistent chart panel output)
DISCIPLINE_ORDER = ["Humanities", "Social & Behavioral Sciences", "Sciences", "Other/Unknown"]


# ---------------------------------------------------------------------------
# LC helpers
# ---------------------------------------------------------------------------

def get_class(call_num):
    """Single-letter LC Class (A, B, C, ...)."""
    if pd.isna(call_num):
        return None
    match = re.match(r"^([A-Z])", str(call_num).strip().upper())
    return match.group(1) if match else None


def get_subclass(call_num):
    """Full LC Subclass (B, BF, BR, KF, ...)."""
    if pd.isna(call_num):
        return None
    match = re.match(r"^([A-Z]{1,3})", str(call_num).strip().upper())
    return match.group(1) if match else None


def categorize_discipline(sub):
    """Group an LC subclass into a broad discipline bucket."""
    if sub is None or (isinstance(sub, float) and pd.isna(sub)) or not sub:
        return "Other/Unknown"
    if sub.startswith("TR"):
        return "Humanities"
    if sub.startswith("BF"):
        return "Sciences"
    first = sub[0]
    if first in {"Q", "R", "S", "T", "U", "V"}:
        return "Sciences"
    if first in {"G", "H", "J", "K", "L"}:
        return "Social & Behavioral Sciences"
    if first in {"C", "D", "E", "F", "M", "N", "P", "B"}:
        return "Humanities"
    return "Other/Unknown"


def normalize_title(title):
    """Lowercase, strip punctuation, collapse whitespace — for holdings/circulation joins."""
    if pd.isna(title):
        return ""
    t = str(title).lower().strip()
    t = t.translate(str.maketrans("", "", string.punctuation))
    return " ".join(t.split())


# Human-readable LC labels
LC_CLASS_DESC = {
    "A": "A - General Works",
    "B": "B - Philosophy, Psychology, Religion",
    "C": "C - Auxiliary Sciences of History",
    "D": "D - World History",
    "E": "E - History of the Americas (General)",
    "F": "F - History of the Americas (Local)",
    "G": "G - Geography, Anthropology, Recreation",
    "H": "H - Social Sciences",
    "J": "J - Political Science",
    "K": "K - Law",
    "L": "L - Education",
    "M": "M - Music",
    "N": "N - Fine Arts",
    "P": "P - Language and Literature",
    "Q": "Q - Science",
    "R": "R - Medicine",
    "S": "S - Agriculture",
    "T": "T - Technology",
    "U": "U - Military Science",
    "V": "V - Naval Science",
    "Z": "Z - Bibliography, Library Science",
}

LC_SUBCLASS_DESC = {
    "PS": "PS - American Literature", "NA": "NA - Architecture", "PR": "PR - English Literature",
    "PN": "PN - General Literature", "E": "E - History of America (General)", "N": "N - Visual Arts (General)",
    "PQ": "PQ - French, Italian, Spanish, Portuguese Literature", "DS": "DS - History of Asia",
    "PT": "PT - German Literature", "PA": "PA - Classical Philology", "ML": "ML - Literature of Music",
    "TR": "TR - Photography", "ND": "ND - Painting", "B": "B - Philosophy (General)",
    "F": "F - Local History of the Americas", "M": "M - Music (General)", "BL": "BL - Religions, Mythology",
    "DK": "DK - History of Russia/Soviet Union", "DA": "DA - History of Great Britain",
    "BH": "BH - Aesthetics", "BP": "BP - Islam, Bahaism, Theosophy", "CB": "CB - History of Civilization",
    "DF": "DF - History of Greece", "DG": "DG - History of Italy", "CC": "CC - Archaeology",
    "BR": "BR - Christianity", "BS": "BS - The Bible", "BX": "BX - Christian Denominations",
    "DP": "DP - History of Spain/Portugal", "PL": "PL - Languages/Literatures of Eastern Asia, Africa, Oceania",
    "BJ": "BJ - Ethics", "HQ": "HQ - Social Groups (Family, Marriage, Women)", "BF": "BF - Psychology",
    "HV": "HV - Social Pathology & Criminology", "HT": "HT - Communities, Classes, Races",
    "GN": "GN - Anthropology", "HD": "HD - Industries, Land & Labor", "JC": "JC - Political Theory",
    "HM": "HM - Sociology (General)", "LB": "LB - Theory & Practice of Education",
    "HC": "HC - Economic History & Conditions", "GE": "GE - Environmental Sciences",
    "H": "H - Social Sciences (General)", "K": "K - Law (General)", "KF": "KF - Law of the United States",
    "LC": "LC - Special Aspects of Education", "JK": "JK - Political Institutions (US)",
    "JZ": "JZ - International Relations", "HB": "HB - Economic Theory", "HF": "HF - Commerce",
    "GF": "GF - Human Ecology/Anthropogeography", "GV": "GV - Recreation, Leisure, Sports",
    "JS": "JS - Local Government", "QA": "QA - Mathematics", "QC": "QC - Physics",
    "QH": "QH - Natural History & Biology", "RC": "RC - Internal Medicine & Psychiatry",
    "Q": "Q - Science (General)", "QD": "QD - Chemistry", "QL": "QL - Zoology",
    "QP": "QP - Physiology", "SB": "SB - Plant Culture", "TX": "TX - Home Economics",
    "TK": "TK - Electrical Engineering, Electronics", "T": "T - Technology (General)",
    "S": "S - Agriculture (General)", "QK": "QK - Botany", "RA": "RA - Public Aspects of Medicine",
    "TA": "TA - Engineering (General)", "RT": "RT - Nursing", "SD": "SD - Forestry",
    "SF": "SF - Animal Culture", "R": "R - Medicine (General)", "QB": "QB - Astronomy",
    "QE": "QE - Geology", "GT": "GT - Manners and Customs (General)", "HN": "HN - Social History and Conditions",
    "D": "D - History (General)", "U": "U - Military Science", "TH": "TH - Building Construction",
    "PC": "PC - Romanic Philology/Languages", "DD": "DD - History of Germany",
    "PJ": "PJ - Oriental Philology/Literature", "PG": "PG - Slavic Philology/Languages",
}


# ---------------------------------------------------------------------------
# Subject parsing
# ---------------------------------------------------------------------------

def parse_full_headings(subject_str):
    """
    Extract full LC subject heading strings from a MARC-style Subjects field.

    Input example:
        "African Americans--Suffrage--Southern States.; Voter registration--Southern States."
    Yields:
        ["African Americans--Suffrage--Southern States",
         "Voter registration--Southern States"]

    Each `;`-separated chunk is kept whole (trailing punctuation stripped) rather
    than reduced to the head term, so a subject like 'Politics and government--
    United States' stays distinct from 'Politics and government--France' rather
    than both collapsing into 'Politics and government'.
    """
    if pd.isna(subject_str) or not str(subject_str).strip():
        return []
    headings = []
    for chunk in str(subject_str).split(";"):
        chunk = chunk.strip().rstrip(".,;: ").strip()
        if chunk:
            headings.append(chunk)
    return headings


# ---------------------------------------------------------------------------
# Geographic term detection
#
# The lists below are non-exhaustive by design — they cover the geographic
# entities that appear repeatedly in academic-library subject fields. Add more
# as gaps surface. Note that "Georgia" is intentionally excluded from countries
# because it's ambiguous with the US state; if you want the country, add
# "Georgia (Republic)" or similar disambiguation to your local list.
# ---------------------------------------------------------------------------

_US_STATE_ABBREVS = frozenset([
    "Ala.", "Alaska", "Ariz.", "Ark.", "Calif.", "Colo.", "Conn.", "D.C.",
    "Del.", "Fla.", "Ga.", "Hawaii", "Idaho", "Ill.", "Ind.", "Iowa", "Kan.",
    "Ky.", "La.", "Mass.", "Md.", "Me.", "Mich.", "Minn.", "Miss.", "Mo.",
    "Mont.", "N.C.", "N.D.", "N.H.", "N.J.", "N. Mex.", "N.Y.", "Neb.",
    "Nev.", "Ohio", "Okla.", "Oreg.", "Pa.", "R.I.", "S.C.", "S.D.", "Tenn.",
    "Tex.", "U.S.", "Utah", "Va.", "Vt.", "W. Va.", "Wash.", "Wis.", "Wyo.",
])

_ABBREV_TO_STATE = {
    "Ala.": "Alabama", "Ariz.": "Arizona", "Ark.": "Arkansas",
    "Calif.": "California", "Colo.": "Colorado", "Conn.": "Connecticut",
    "D.C.": "Washington (D.C.)", "Del.": "Delaware", "Fla.": "Florida",
    "Ga.": "Georgia", "Ill.": "Illinois", "Ind.": "Indiana",
    "Kan.": "Kansas", "Ky.": "Kentucky", "La.": "Louisiana",
    "Mass.": "Massachusetts", "Md.": "Maryland", "Me.": "Maine",
    "Mich.": "Michigan", "Minn.": "Minnesota", "Miss.": "Mississippi",
    "Mo.": "Missouri", "Mont.": "Montana", "N.C.": "North Carolina",
    "N.D.": "North Dakota", "N.H.": "New Hampshire", "N.J.": "New Jersey",
    "N. Mex.": "New Mexico", "N.Y.": "New York (State)", "Neb.": "Nebraska",
    "Nev.": "Nevada", "Okla.": "Oklahoma", "Oreg.": "Oregon",
    "Pa.": "Pennsylvania", "R.I.": "Rhode Island", "S.C.": "South Carolina",
    "S.D.": "South Dakota", "Tenn.": "Tennessee", "Tex.": "Texas",
    "U.S.": "United States", "Va.": "Virginia", "Vt.": "Vermont",
    "W. Va.": "West Virginia", "Wash.": "Washington (State)",
    "Wis.": "Wisconsin", "Wyo.": "Wyoming",
}

_KNOWN_GEOGRAPHIES = frozenset([
    # US: full state names
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York", "New York (State)",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "Washington (D.C.)", "Washington (State)", "West Virginia",
    "Wisconsin", "Wyoming",
    # US: country + regions
    "United States", "United States of America",
    "Southern States", "New England", "Middle West", "Pacific Northwest",
    "Northeastern States", "Northwestern States", "Southeastern States",
    "Southwestern States", "Great Plains", "Mountain States",
    "Atlantic States", "Appalachian Region", "Gulf States", "Great Lakes",
    "West (U.S.)", "South (U.S.)", "Northwest, Pacific",
    # US: major cities that recur in LC subject headings
    "New Orleans", "New Orleans (La.)", "New York (N.Y.)", "Chicago (Ill.)",
    "Los Angeles (Calif.)", "San Francisco (Calif.)", "Boston (Mass.)",
    "Philadelphia (Pa.)", "Atlanta (Ga.)", "Detroit (Mich.)",
    "Baltimore (Md.)", "Seattle (Wash.)", "Miami (Fla.)",
    "Houston (Tex.)", "Dallas (Tex.)", "Portland (Or.)",
    # Europe (countries)
    "France", "Germany", "Italy", "Spain", "Portugal", "United Kingdom",
    "Great Britain", "England", "Scotland", "Wales", "Ireland", "Netherlands",
    "Belgium", "Switzerland", "Austria", "Poland", "Russia", "Soviet Union",
    "Sweden", "Norway", "Denmark", "Finland", "Greece", "Turkey", "Hungary",
    "Czech Republic", "Czechoslovakia", "Romania", "Bulgaria", "Ukraine",
    "Serbia", "Croatia", "Slovenia", "Slovakia", "Iceland", "Estonia",
    "Latvia", "Lithuania", "Belarus", "Luxembourg",
    # Europe (regions)
    "Europe", "Western Europe", "Eastern Europe", "Central Europe",
    "Scandinavia", "Balkans", "Baltic States", "Mediterranean Region",
    "Iberian Peninsula",
    # Middle East
    "Israel", "Palestine", "West Bank", "Gaza Strip", "Iran", "Iraq",
    "Syria", "Lebanon", "Jordan", "Saudi Arabia", "Yemen",
    "Egypt", "Kuwait", "Qatar", "Bahrain", "Oman", "United Arab Emirates",
    "Middle East", "Persian Gulf Region",
    # Asia
    "China", "Japan", "Korea", "Korea (South)", "Korea (North)",
    "India", "Pakistan", "Bangladesh", "Vietnam", "Thailand", "Indonesia",
    "Philippines", "Malaysia", "Singapore", "Afghanistan", "Nepal",
    "Sri Lanka", "Myanmar", "Cambodia", "Laos", "Mongolia", "Taiwan",
    "Asia", "East Asia", "Southeast Asia", "South Asia", "Central Asia",
    "Kazakhstan", "Uzbekistan",
    # Africa
    "South Africa", "Nigeria", "Kenya", "Ethiopia", "Ghana", "Morocco",
    "Algeria", "Tunisia", "Libya", "Sudan", "Somalia", "Zimbabwe", "Uganda",
    "Tanzania", "Rwanda", "Mozambique", "Angola", "Cameroon", "Senegal",
    "Mali", "Namibia", "Botswana", "Zambia", "Malawi", "Ivory Coast",
    "Africa", "Sub-Saharan Africa", "Africa, Sub-Saharan",
    "West Africa", "East Africa", "North Africa", "Southern Africa",
    # Americas
    "Canada", "Mexico", "Brazil", "Argentina", "Chile", "Colombia", "Peru",
    "Venezuela", "Ecuador", "Bolivia", "Uruguay", "Paraguay", "Guatemala",
    "Honduras", "El Salvador", "Nicaragua", "Costa Rica", "Panama", "Cuba",
    "Dominican Republic", "Haiti", "Jamaica", "Puerto Rico",
    "North America", "South America", "Central America", "Latin America",
    "Caribbean Area", "West Indies",
    # Oceania
    "Australia", "New Zealand", "Fiji", "Papua New Guinea", "Oceania",
])

# Regex to catch parenthesized US state abbreviations mid-string
# e.g. "New Orleans (La.)" or "Chicago (Ill.)"
_PAREN_ABBREV_PAT = re.compile(
    r"\(([A-Z][A-Za-z]{0,3}\.(?:\s*[A-Z][A-Za-z]{0,3}\.)*)\)"
)


def _classify_geography_region(name):
    """Coarse continent-level bucket for a geographic entity."""
    us_bare_cities = {"New Orleans", "New York", "Chicago", "Los Angeles",
                      "San Francisco", "Boston", "Philadelphia", "Atlanta",
                      "Detroit", "Baltimore", "Seattle", "Miami", "Houston",
                      "Dallas", "Portland"}
    if name in us_bare_cities:
        return "United States"
    if name in {"United States", "United States of America",
                "Washington (D.C.)"} or name in _ABBREV_TO_STATE.values():
        return "United States"
    if name in {"Southern States", "New England", "Middle West",
                "Pacific Northwest", "Northeastern States",
                "Northwestern States", "Southeastern States",
                "Southwestern States", "Great Plains", "Mountain States",
                "Atlantic States", "Appalachian Region", "Gulf States",
                "Great Lakes", "West (U.S.)", "South (U.S.)"}:
        return "United States"
    if any(name.endswith(f"({abbr})") for abbr in _US_STATE_ABBREVS):
        return "United States"
    europe = {"France", "Germany", "Italy", "Spain", "Portugal",
              "United Kingdom", "Great Britain", "England", "Scotland",
              "Wales", "Ireland", "Netherlands", "Belgium", "Switzerland",
              "Austria", "Poland", "Russia", "Soviet Union", "Sweden",
              "Norway", "Denmark", "Finland", "Greece", "Turkey", "Hungary",
              "Czech Republic", "Czechoslovakia", "Romania", "Bulgaria",
              "Ukraine", "Serbia", "Croatia", "Slovenia", "Slovakia",
              "Iceland", "Estonia", "Latvia", "Lithuania", "Belarus",
              "Luxembourg", "Europe", "Western Europe", "Eastern Europe",
              "Central Europe", "Scandinavia", "Balkans", "Baltic States",
              "Mediterranean Region", "Iberian Peninsula"}
    if name in europe:
        return "Europe"
    mideast = {"Israel", "Palestine", "West Bank", "Gaza Strip", "Iran",
               "Iraq", "Syria", "Lebanon", "Jordan", "Saudi Arabia", "Yemen",
               "Egypt", "Kuwait", "Qatar", "Bahrain", "Oman",
               "United Arab Emirates", "Middle East", "Persian Gulf Region"}
    if name in mideast:
        return "Middle East / North Africa"
    asia = {"China", "Japan", "Korea", "Korea (South)", "Korea (North)",
            "India", "Pakistan", "Bangladesh", "Vietnam", "Thailand",
            "Indonesia", "Philippines", "Malaysia", "Singapore",
            "Afghanistan", "Nepal", "Sri Lanka", "Myanmar", "Cambodia",
            "Laos", "Mongolia", "Taiwan", "Asia", "East Asia",
            "Southeast Asia", "South Asia", "Central Asia", "Kazakhstan",
            "Uzbekistan"}
    if name in asia:
        return "Asia"
    africa = {"South Africa", "Nigeria", "Kenya", "Ethiopia", "Ghana",
              "Morocco", "Algeria", "Tunisia", "Libya", "Sudan", "Somalia",
              "Zimbabwe", "Uganda", "Tanzania", "Rwanda", "Mozambique",
              "Angola", "Cameroon", "Senegal", "Mali", "Namibia",
              "Botswana", "Zambia", "Malawi", "Ivory Coast", "Africa",
              "Sub-Saharan Africa", "Africa, Sub-Saharan", "West Africa",
              "East Africa", "North Africa", "Southern Africa"}
    if name in africa:
        return "Africa"
    americas = {"Canada", "Mexico", "Brazil", "Argentina", "Chile",
                "Colombia", "Peru", "Venezuela", "Ecuador", "Bolivia",
                "Uruguay", "Paraguay", "Guatemala", "Honduras", "El Salvador",
                "Nicaragua", "Costa Rica", "Panama", "Cuba",
                "Dominican Republic", "Haiti", "Jamaica", "Puerto Rico",
                "North America", "South America", "Central America",
                "Latin America", "Caribbean Area", "West Indies"}
    if name in americas:
        return "Americas (non-US)"
    oceania = {"Australia", "New Zealand", "Fiji", "Papua New Guinea",
               "Oceania"}
    if name in oceania:
        return "Oceania"
    return "Other/Unclassified"


def parse_geographic_terms(subject_str):
    """
    Extract geographic entities mentioned anywhere in a MARC-style Subjects field.

    Returns a de-duplicated list so a title mentioning 'Louisiana' in three
    different subject strings still counts toward Louisiana just once.

    Detection:
      1. Any `--`-separated part that exactly matches a known geographic entity
      2. Any parenthesized US-state abbreviation like '(La.)' or '(N.Y.)' —
         emits both the state name and the parent city string if present
      3. Head terms that are US-state-qualified cities (e.g. 'New Orleans (La.)')
    """
    if pd.isna(subject_str) or not str(subject_str).strip():
        return []

    found = set()
    for chunk in str(subject_str).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue

        # Split into parts by '--'
        for raw in chunk.split("--"):
            part = raw.strip().rstrip(".,;: ").strip()
            if not part:
                continue

            # Exact match against the known-geography set
            if part in _KNOWN_GEOGRAPHIES:
                found.add(part)

            # Parenthesized state abbreviation → emit parent state name
            for m in _PAREN_ABBREV_PAT.findall(part):
                if m in _ABBREV_TO_STATE:
                    found.add(_ABBREV_TO_STATE[m])
                    # Also keep the full '(city (State))' form
                    if part.endswith(f"({m})"):
                        found.add(part)

    return sorted(found)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _fit_trend(row_values):
    """
    Fit a linear trend y = m*x + b through the fiscal-year loan counts and
    return (slope, mean, r_squared).

    Every year contributes to the slope, so an interior dip or spike changes
    the trend even when the endpoints look identical. R² measures how well
    the straight-line fit actually describes the data — 1.0 means a clean
    monotonic line, values near 0 mean the trend is noisy and shouldn't be
    read as a real trajectory.
    """
    y = np.asarray(row_values, dtype=float)
    x = np.arange(len(y))
    if len(y) < 2 or np.all(y == y[0]):
        return 0.0, float(y.mean()) if len(y) else 0.0, float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = intercept + slope * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(y.mean()), r2


def pivot_by_year(df, group_col, value_col="Loans (In House + Not In House)"):
    """
    Sum loans per group per fiscal year, returned wide with FY columns, plus
    trend-based movement metrics computed from ALL fiscal years (not just endpoints):

      - Mean Annual Loans:              average loans across the window
      - Trend Slope (loans/yr):         regression slope through the 4 FY points
                                        (every year contributes)
      - Trend R^2:                      goodness-of-fit for the linear trend.
                                        Close to 1.0 = smooth monotonic line;
                                        close to 0 = bouncy/noisy series where
                                        the % change should not be read as a
                                        trajectory. Blank when the series is
                                        constant.
      - Cumulative % Change (trend):    trend slope projected over the full
                                        window (slope * years) expressed as a
                                        % of the FY23 baseline. Uses all 4 years
                                        via the slope, but keeps the actual FY23
                                        as denominator so the number is
                                        interpretable as a % change from
                                        baseline rather than from a fitted point.
      - Endpoint % Change: the naive two-point comparison, kept
                                        for reference so trend vs. endpoint
                                        can be compared side by side.
    """
    pv = (
        df.groupby([group_col, "Loan Fiscal Year"])[value_col]
        .sum()
        .unstack(fill_value=0)
    )
    for fy in FY_COLS:
        if fy not in pv.columns:
            pv[fy] = 0
    pv = pv[FY_COLS]

    # Vectorized linear trend fit across all rows at once.
    # For a linear regression y = m*x + b with x = [0, 1, ..., n-1] the same
    # for every row, the slope reduces to a matrix-vector product against a
    # precomputed x-centered vector — a ~50x speedup over row-wise np.polyfit
    # on 100k+ groups, which matters for the subject-heading table.
    Y = pv[FY_COLS].to_numpy(dtype=float)
    n_years = Y.shape[1]
    x = np.arange(n_years, dtype=float)
    x_mean = x.mean()
    x_centered = x - x_mean
    xx = float(np.dot(x_centered, x_centered))       # sum(x_centered^2), scalar

    y_mean = Y.mean(axis=1)                          # per-row mean
    Y_centered = Y - y_mean[:, None]
    if xx > 0:
        slope = Y_centered.dot(x_centered) / xx      # per-row slope
    else:
        slope = np.zeros(Y.shape[0])
    intercept = y_mean - slope * x_mean
    Y_pred = intercept[:, None] + slope[:, None] * x[None, :]

    ss_res = np.sum((Y - Y_pred) ** 2, axis=1)
    ss_tot = np.sum(Y_centered ** 2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)

    pv["Trend Slope (loans/yr)"] = slope
    pv["Mean Annual Loans"] = y_mean
    pv["Trend R^2"] = r2

    baseline = pv[FIRST_FY].replace(0, pd.NA)

    pv["Cumulative % Change (trend)"] = (
        (pv["Trend Slope (loans/yr)"] * (n_years - 1)) / baseline * 100
    ).replace([float("inf"), float("-inf")], pd.NA)

    pv["Endpoint % Change"] = (
        (pv[LAST_FY] - pv[FIRST_FY]) / baseline * 100
    ).replace([float("inf"), float("-inf")], pd.NA)

    pv["Absolute Change"] = pv[LAST_FY] - pv[FIRST_FY]

    pv["Trend Slope (loans/yr)"] = pv["Trend Slope (loans/yr)"].round(2)
    pv["Mean Annual Loans"] = pv["Mean Annual Loans"].round(1)
    pv["Trend R^2"] = pv["Trend R^2"].round(3)

    return pv.reset_index()


# ---------------------------------------------------------------------------
# Charting
# ---------------------------------------------------------------------------

def _bar_color(pct):
    if pd.isna(pct):
        return ZERO_COLOR
    if pct > 0:
        return POS_COLOR
    if pct < 0:
        return NEG_COLOR
    return ZERO_COLOR


def plot_change_chart(df, label_col, pct_col, title, outpath,
                      min_baseline_col=None, min_baseline=0,
                      top_n=None):
    """
    Horizontal bar chart of % change per category, matching the reference style.

    - df: long-form frame with a label column and a % change column
    - label_col: name of the column whose values become the y-axis labels
    - pct_col: name of the column holding the % change
    - min_baseline_col + min_baseline: optionally require baseline (e.g. FY-2023 loans >= N)
    - top_n: if set, keep only the N most extreme (by absolute % change) rows
    """
    plot_df = df.copy()

    if min_baseline_col is not None:
        plot_df = plot_df[plot_df[min_baseline_col] >= min_baseline]

    plot_df = plot_df.dropna(subset=[pct_col])
    if plot_df.empty:
        print(f"  [skip] no rows to chart for '{title}'")
        return

    if top_n is not None and len(plot_df) > top_n:
        plot_df = plot_df.reindex(
            plot_df[pct_col].abs().sort_values(ascending=False).index
        ).head(top_n)

    _render_bar_chart(plot_df, label_col, pct_col, title, outpath)


def plot_risers_and_decliners(df, label_col, pct_col, title, outpath,
                              min_baseline_col=None, min_baseline=0,
                              top_up=12, top_down=12):
    """
    Like plot_change_chart, but guarantees BOTH directions are visible: the top N
    positive movers and the top N negative movers are combined and charted together.
    """
    plot_df = df.copy()
    if min_baseline_col is not None:
        plot_df = plot_df[plot_df[min_baseline_col] >= min_baseline]
    plot_df = plot_df.dropna(subset=[pct_col])
    # Column may be object dtype if pd.NA got mixed in; coerce for nlargest/nsmallest
    plot_df[pct_col] = pd.to_numeric(plot_df[pct_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[pct_col])

    risers = plot_df[plot_df[pct_col] > 0].nlargest(top_up, pct_col)
    decliners = plot_df[plot_df[pct_col] < 0].nsmallest(top_down, pct_col)

    combined = pd.concat([risers, decliners]).drop_duplicates(subset=[label_col])
    if combined.empty:
        print(f"  [skip] no rows to chart for '{title}'")
        return

    _render_bar_chart(combined, label_col, pct_col, title, outpath)


def _render_bar_chart(plot_df, label_col, pct_col, title, outpath):
    plot_df = plot_df.sort_values(by=pct_col, ascending=False)

    height = max(5, 0.42 * len(plot_df) + 2)
    fig, ax = plt.subplots(figsize=(12, height))

    colors = [_bar_color(v) for v in plot_df[pct_col]]
    bars = ax.barh(plot_df[label_col], plot_df[pct_col], color=colors, height=0.7,
                   edgecolor="white", linewidth=0.5)

    ax.axvline(0, color="#333333", linestyle="--", linewidth=1)

    max_abs = plot_df[pct_col].abs().max()
    pad = max(max_abs * 0.02, 0.5)
    for bar, val in zip(bars, plot_df[pct_col]):
        x = bar.get_width()
        label = f"{val:+.1f}%"
        if x >= 0:
            ax.text(x + pad, bar.get_y() + bar.get_height() / 2, label,
                    va="center", ha="left", fontsize=9.5)
        else:
            ax.text(x - pad, bar.get_y() + bar.get_height() / 2, label,
                    va="center", ha="right", fontsize=9.5)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=16)
    ax.set_xlabel(
        f"Cumulative % Change (trend fit across {FIRST_FY}\u2013{LAST_FY})",
        fontsize=11, labelpad=8,
    )
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    limit = max_abs * 1.25 if max_abs > 0 else 10
    ax.set_xlim(-limit, limit)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> {outpath}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_data(path):
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    required = {
        "Title", "Loan Fiscal Year", "Author", "Call Number",
        "Subjects", "Loans (In House + Not In House)",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    df["LC_Class"] = df["Call Number"].apply(get_class)
    df["LC_Subclass"] = df["Call Number"].apply(get_subclass)
    df["Discipline"] = df["LC_Subclass"].apply(categorize_discipline)
    # Detect fiscal-year window from the data and populate module-level
    # FY_COLS / FIRST_FY / LAST_FY / FY_SHORT so every downstream function
    # keys off the actual data instead of hardcoded years.
    fy_cols = configure_fy_window(df)
    print(f"  Detected fiscal-year window: {fy_cols}")
    if len(fy_cols) < 2:
        raise ValueError(
            f"Standalone script needs at least 2 fiscal years for trend "
            f"analysis; found {fy_cols}. For single-year snapshot use the "
            f"Streamlit dashboard (streamlit run analysis_dashboard.py)."
        )
    return df


def run_lc_analysis(df, outdir):
    print("\n== LC change analysis ==")

    # ---- LC Class (single letter) ---------------------------------------
    class_df = pivot_by_year(df, "LC_Class")
    class_df["Description"] = class_df["LC_Class"].map(
        lambda c: LC_CLASS_DESC.get(c, f"{c} - Class {c}")
    )
    class_df["Discipline"] = class_df["LC_Class"].apply(categorize_discipline)
    class_df = class_df[
        ["Discipline", "LC_Class", "Description", *FY_COLS,
         "Mean Annual Loans", "Trend Slope (loans/yr)", "Trend R^2",
         "Cumulative % Change (trend)", "Endpoint % Change",
         "Absolute Change"]
    ]
    class_csv = os.path.join(outdir, f"lc_class_circulation_shifts_{fy_window_slug()}.csv")
    class_df.to_csv(class_csv, index=False)
    print(f"  -> {class_csv}")

    # Overall LC Class chart (matches the reference PNG style)
    plot_change_chart(
        class_df, "Description", "Cumulative % Change (trend)",
        f"Shift in Physical Book Usage by LC Class\n"
        f"(Cumulative Trend Across {FIRST_FY}\u2013{LAST_FY})",
        os.path.join(outdir, "lc_class_shifts_overall.png"),
    )

    # One LC Class chart per discipline
    for disc in DISCIPLINE_ORDER:
        subset = class_df[class_df["Discipline"] == disc]
        if subset.empty:
            continue
        safe = disc.lower().replace(" & ", "_and_").replace(" ", "_").replace("/", "_")
        plot_change_chart(
            subset, "Description", "Cumulative % Change (trend)",
            f"LC Class Usage Shift \u2014 {disc}\n"
            f"(Cumulative Trend Across {FIRST_FY}\u2013{LAST_FY})",
            os.path.join(outdir, f"lc_class_shifts_{safe}.png"),
        )

    # ---- LC Subclass (detail) -------------------------------------------
    sub_df = pivot_by_year(df, "LC_Subclass")
    sub_df["Description"] = sub_df["LC_Subclass"].map(
        lambda s: LC_SUBCLASS_DESC.get(s, f"{s} - Subclass {s}")
    )
    sub_df["Discipline"] = sub_df["LC_Subclass"].apply(categorize_discipline)
    sub_df = sub_df[
        ["Discipline", "LC_Subclass", "Description", *FY_COLS,
         "Mean Annual Loans", "Trend Slope (loans/yr)", "Trend R^2",
         "Cumulative % Change (trend)", "Endpoint % Change",
         "Absolute Change"]
    ].sort_values(["Discipline", "LC_Subclass"])
    sub_csv = os.path.join(outdir, f"lc_subclass_circulation_shifts_{fy_window_slug()}.csv")
    sub_df.to_csv(sub_csv, index=False)
    print(f"  -> {sub_csv}")

    # ---- LC Subclass "biggest movers" charts ----------------------------
    plot_change_chart(
        sub_df, "Description", "Cumulative % Change (trend)",
        f"LC Subclasses with the Greatest Cumulative Movement \u2014 Overall\n"
        f"(top 15 by |trend % change| | min 50 loans in {FIRST_FY})",
        os.path.join(outdir, "lc_subclass_top_movers_overall.png"),
        min_baseline_col=FIRST_FY, min_baseline=50, top_n=15,
    )

    for disc in DISCIPLINE_ORDER:
        subset = sub_df[sub_df["Discipline"] == disc]
        if subset.empty:
            continue
        baseline_here = 25 if disc in {"Other/Unknown"} else 30
        safe = disc.lower().replace(" & ", "_and_").replace(" ", "_").replace("/", "_")
        plot_change_chart(
            subset, "Description", "Cumulative % Change (trend)",
            f"LC Subclasses with the Greatest Cumulative Movement \u2014 {disc}\n"
            f"(top 15 by |trend % change| | min {baseline_here} loans in {FIRST_FY})",
            os.path.join(outdir, f"lc_subclass_top_movers_{safe}.png"),
            min_baseline_col=FIRST_FY, min_baseline=baseline_here, top_n=15,
        )

    return class_df, sub_df


def run_subject_analysis(df, outdir, top_up=12, top_down=12,
                         min_baseline_loans=25):
    """
    Explode Subjects into FULL heading strings (not just head terms), aggregate
    loans per heading per FY, then chart both the top risers and top decliners
    overall and per discipline.

    Full headings preserve subdivision detail — 'Politics and government--
    United States' stays distinct from 'Politics and government--France' rather
    than both collapsing into 'Politics and government', which avoids the
    over-attribution problem where common head terms get credited across
    thousands of unrelated books.
    """
    print("\n== Subject term change analysis (full headings) ==")

    df = df.copy()
    df["FullHeadings"] = df["Subjects"].apply(parse_full_headings)
    exploded = df.explode("FullHeadings").rename(
        columns={"FullHeadings": "Subject"}
    )
    exploded = exploded[exploded["Subject"].notna() & (exploded["Subject"] != "")]

    subj_df = pivot_by_year(exploded, "Subject")

    disc_totals = (
        exploded.groupby(["Subject", "Discipline"])["Loans (In House + Not In House)"]
        .sum()
        .reset_index()
    )
    dominant = (
        disc_totals.sort_values("Loans (In House + Not In House)", ascending=False)
        .drop_duplicates("Subject")
        [["Subject", "Discipline"]]
    )
    subj_df = subj_df.merge(dominant, on="Subject", how="left")
    subj_df = subj_df[
        ["Discipline", "Subject", *FY_COLS,
         "Mean Annual Loans", "Trend Slope (loans/yr)", "Trend R^2",
         "Cumulative % Change (trend)", "Endpoint % Change",
         "Absolute Change"]
    ]

    subj_csv = os.path.join(outdir, f"subject_term_shifts_{fy_window_slug()}.csv")
    subj_df.sort_values("Trend Slope (loans/yr)",
                        ascending=False).to_csv(subj_csv, index=False)
    print(f"  -> {subj_csv}")

    # Overall risers + decliners
    plot_risers_and_decliners(
        subj_df, "Subject", "Cumulative % Change (trend)",
        f"Subject Headings \u2014 Biggest Cumulative Risers & Decliners (Overall)\n"
        f"(top {top_up} up + top {top_down} down | trend across all FYs | "
        f"min {min_baseline_loans} loans in {FIRST_FY})",
        os.path.join(outdir, "subject_shifts_overall.png"),
        min_baseline_col=FIRST_FY, min_baseline=min_baseline_loans,
        top_up=top_up, top_down=top_down,
    )

    # Per-discipline risers + decliners
    for disc in DISCIPLINE_ORDER:
        subset = subj_df[subj_df["Discipline"] == disc]
        if subset.empty:
            continue
        baseline_here = max(10, int(min_baseline_loans * 0.4))
        safe = disc.lower().replace(" & ", "_and_").replace(" ", "_").replace("/", "_")
        plot_risers_and_decliners(
            subset, "Subject", "Cumulative % Change (trend)",
            f"Subject Headings \u2014 Biggest Cumulative Risers & Decliners \u2014 {disc}\n"
            f"(top {top_up} up + top {top_down} down | trend across all FYs | "
            f"min {baseline_here} loans in {FIRST_FY})",
            os.path.join(outdir, f"subject_shifts_{safe}.png"),
            min_baseline_col=FIRST_FY, min_baseline=baseline_here,
            top_up=top_up, top_down=top_down,
        )

    return subj_df


def run_geographic_analysis(df, outdir, top_up=12, top_down=12,
                            min_baseline_loans=20):
    """
    Extract geographic mentions from every subject field, aggregate loans per
    geographic entity per FY, and chart the movement.

    Produces:
      - geographic_term_shifts_fy23_fy26.csv (per-place detail)
      - geographic_region_shifts_fy23_fy26.csv (continent-level roll-up)
      - geographic_shifts_overall.png (top risers + decliners across all places)
      - geographic_shifts_us_states.png (US states only, if enough movement)
      - geographic_shifts_regions.png (continent-level roll-up chart)
    """
    print("\n== Geographic trends analysis ==")

    df = df.copy()
    df["Geographies"] = df["Subjects"].apply(parse_geographic_terms)
    exploded = df.explode("Geographies").rename(
        columns={"Geographies": "Geography"}
    )
    exploded = exploded[exploded["Geography"].notna() & (exploded["Geography"] != "")]

    if exploded.empty:
        print("  [skip] no geographic terms detected")
        return None

    # Attach continent-level bucket for each geography
    exploded["Region"] = exploded["Geography"].apply(_classify_geography_region)

    # -- Per-place shift table -------------------------------------------
    geo_df = pivot_by_year(exploded, "Geography")
    place_regions = (
        exploded[["Geography", "Region"]].drop_duplicates()
        .set_index("Geography")["Region"]
    )
    geo_df["Region"] = geo_df["Geography"].map(place_regions)
    geo_df = geo_df[
        ["Region", "Geography", *FY_COLS,
         "Mean Annual Loans", "Trend Slope (loans/yr)", "Trend R^2",
         "Cumulative % Change (trend)", "Endpoint % Change",
         "Absolute Change"]
    ]
    geo_csv = os.path.join(outdir, f"geographic_term_shifts_{fy_window_slug()}.csv")
    geo_df.sort_values("Trend Slope (loans/yr)",
                       ascending=False).to_csv(geo_csv, index=False)
    print(f"  -> {geo_csv}")

    # -- Continent-level roll-up ----------------------------------------
    region_df = pivot_by_year(exploded, "Region")
    region_df = region_df[
        ["Region", *FY_COLS,
         "Mean Annual Loans", "Trend Slope (loans/yr)", "Trend R^2",
         "Cumulative % Change (trend)", "Endpoint % Change",
         "Absolute Change"]
    ]
    region_csv = os.path.join(outdir, f"geographic_region_shifts_{fy_window_slug()}.csv")
    region_df.sort_values("Cumulative % Change (trend)",
                          ascending=False).to_csv(region_csv, index=False)
    print(f"  -> {region_csv}")

    # -- Charts ---------------------------------------------------------
    plot_risers_and_decliners(
        geo_df, "Geography", "Cumulative % Change (trend)",
        f"Geographic Mentions \u2014 Biggest Cumulative Risers & Decliners (Overall)\n"
        f"(top {top_up} up + top {top_down} down | trend across all FYs | "
        f"min {min_baseline_loans} loans in {FIRST_FY})",
        os.path.join(outdir, "geographic_shifts_overall.png"),
        min_baseline_col=FIRST_FY, min_baseline=min_baseline_loans,
        top_up=top_up, top_down=top_down,
    )

    # US states only (helpful for regional/state-collection conversations)
    us_only = geo_df[geo_df["Region"] == "United States"]
    if not us_only.empty:
        plot_risers_and_decliners(
            us_only, "Geography", "Cumulative % Change (trend)",
            f"Geographic Mentions \u2014 US States/Regions Only\n"
            f"(top {top_up} up + top {top_down} down | trend across all FYs | "
            f"min {max(10, min_baseline_loans // 2)} loans in {FIRST_FY})",
            os.path.join(outdir, "geographic_shifts_us.png"),
            min_baseline_col=FIRST_FY,
            min_baseline=max(10, min_baseline_loans // 2),
            top_up=top_up, top_down=top_down,
        )

    # Continent-level chart — shows all regions since there are only ~7
    plot_change_chart(
        region_df, "Region", "Cumulative % Change (trend)",
        f"Geographic Mentions by Region \u2014 Cumulative Trend\n"
        f"(all regions | across {FIRST_FY}\u2013{LAST_FY})",
        os.path.join(outdir, "geographic_shifts_regions.png"),
    )

    return geo_df


def run_weeding_candidates(df, outdir,
                           recent_years=None,
                           early_years=None,
                           strong_early_threshold=10,
                           medium_early_threshold=5,
                           weak_early_threshold=3,
                           protect_recent_pub_years=5):
    """
    Without a full holdings file, weeding signals are derived from usage decay:
    titles that had loans in the early window but zero loans in the recent window
    are candidates for review.

    early_years / recent_years default to the first and second halves of the
    detected FY_COLS. For a 4-year window this gives (first 2 FYs, last 2 FYs);
    for odd-numbered windows the later half gets the extra year.

    Two refinements over a flat threshold:

      1. Tiered by early demand.  'Strong' = had substantial early use, then went
         silent. 'Medium' and 'Weak' scale down. Liaisons can start with the
         strong list and only dig further if needed.

      2. Publication-year gate.  Titles published within the last N years are
         excluded even if usage looks like decay — new items need shelf time
         before their circulation history is meaningful.
    """
    print("\n== Weeding candidates ==")

    if early_years is None or recent_years is None:
        if len(FY_COLS) < 2:
            raise ValueError("Weeding needs at least 2 fiscal years in FY_COLS")
        mid = len(FY_COLS) // 2
        early_years = tuple(FY_COLS[:mid])
        recent_years = tuple(FY_COLS[mid:])
        print(f"  Early window: {list(early_years)}")
        print(f"  Recent window: {list(recent_years)}")

    # Aggregate title-level usage per FY
    title_yr = (
        df.groupby(["Title", "Author", "Publication Year", "Call Number",
                    "LC_Subclass", "Discipline", "Loan Fiscal Year"])
          ["Loans (In House + Not In House)"].sum()
          .unstack(fill_value=0)
    )
    for fy in FY_COLS:
        if fy not in title_yr.columns:
            title_yr[fy] = 0
    title_yr["Early Loans"] = title_yr[list(early_years)].sum(axis=1)
    title_yr["Recent Loans"] = title_yr[list(recent_years)].sum(axis=1)
    title_yr["Total Loans"] = title_yr[FY_COLS].sum(axis=1)
    title_yr = title_yr.reset_index()

    # Publication-year gate: parse loosely, exclude anything newer than the cutoff
    pub_year_num = pd.to_numeric(title_yr["Publication Year"], errors="coerce")
    from datetime import date
    cutoff_year = date.today().year - protect_recent_pub_years
    protected_mask = pub_year_num >= cutoff_year

    # Base decay signal
    decay = (title_yr["Early Loans"] >= weak_early_threshold) & (title_yr["Recent Loans"] == 0)

    # Assign tier: Strong > Medium > Weak (mutually exclusive)
    def assign_tier(row):
        if row["Recent Loans"] != 0:
            return None
        if row["Early Loans"] >= strong_early_threshold:
            return "Strong"
        if row["Early Loans"] >= medium_early_threshold:
            return "Medium"
        if row["Early Loans"] >= weak_early_threshold:
            return "Weak"
        return None

    title_yr["Weeding Tier"] = title_yr.apply(assign_tier, axis=1)
    title_yr.loc[protected_mask, "Weeding Tier"] = None  # protected new pubs

    candidates = title_yr[title_yr["Weeding Tier"].notna()].copy()

    tier_order = pd.CategoricalDtype(["Strong", "Medium", "Weak"], ordered=True)
    candidates["Weeding Tier"] = candidates["Weeding Tier"].astype(tier_order)
    candidates = candidates.sort_values(
        ["Weeding Tier", "Early Loans"], ascending=[True, False]
    )
    candidates = candidates[
        ["Weeding Tier", "Discipline", "LC_Subclass", "Title", "Author",
         "Publication Year", "Call Number",
         *FY_COLS, "Early Loans", "Recent Loans", "Total Loans"]
    ]

    out = os.path.join(outdir, f"weeding_candidates_{fy_window_slug()}.csv")
    candidates.to_csv(out, index=False)

    tier_counts = candidates["Weeding Tier"].value_counts().reindex(
        ["Strong", "Medium", "Weak"], fill_value=0
    )
    protected_count = int(protected_mask.sum())
    print(f"  -> {out}")
    print(f"     Strong: {tier_counts['Strong']}  "
          f"Medium: {tier_counts['Medium']}  "
          f"Weak: {tier_counts['Weak']}  "
          f"(pub-year protected: {protected_count} titles pub >= {cutoff_year})")

    # Subclass roll-up with per-tier counts
    if not candidates.empty:
        subclass_roll = (
            candidates.groupby(["Discipline", "LC_Subclass", "Weeding Tier"],
                               observed=True)
            .size().unstack(fill_value=0)
        )
        for tier in ("Strong", "Medium", "Weak"):
            if tier not in subclass_roll.columns:
                subclass_roll[tier] = 0
        subclass_roll = subclass_roll[["Strong", "Medium", "Weak"]]
        subclass_roll["Total"] = subclass_roll.sum(axis=1)
        subclass_roll = (
            subclass_roll.reset_index()
            .sort_values(["Strong", "Total"], ascending=[False, False])
        )
        roll_out = os.path.join(outdir, "weeding_candidates_by_subclass.csv")
        subclass_roll.to_csv(roll_out, index=False)
        print(f"  -> {roll_out}")

    return candidates


def run_ebook_candidates(df, outdir, top_n=25):
    """
    E-book targets: titles with high total demand across the window, prioritized when
    demand is *sustained* (loans in 3+ fiscal years) or *rising* (recent > early).
    """
    print("\n== E-book purchase candidates ==")

    title_yr = (
        df.groupby(["Title", "Author", "Call Number", "LC_Subclass",
                    "Discipline", "Loan Fiscal Year"])
          ["Loans (In House + Not In House)"].sum()
          .unstack(fill_value=0)
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
            return ("Sustained, rising demand — priority e-book candidate; "
                    "digital license unlocks multi-user simultaneous access.")
        if sustained:
            return ("Consistent multi-year demand — strong e-book candidate to "
                    "reduce shelf wear and support remote access.")
        if rising:
            return ("Recent demand spike — e-book license would meet current "
                    "course/research pressure quickly.")
        return ("High historical volume — evaluate whether digital access would "
                "better serve current use.")

    ranked = title_yr.sort_values("Total Loans", ascending=False).head(top_n).reset_index()
    ranked["Strategic Case for E-Book Acquisition"] = ranked.apply(rationale, axis=1)
    ranked = ranked[
        ["Title", "Author", "Call Number", "Discipline", *FY_COLS,
         "Early Loans", "Recent Loans", "Active Years", "Total Loans",
         "Strategic Case for E-Book Acquisition"]
    ]
    out = os.path.join(outdir, f"ebook_purchase_candidates_{fy_window_slug()}.csv")
    ranked.to_csv(out, index=False)
    print(f"  -> {out}  ({len(ranked)} titles)")
    return ranked


def run_holdings_weeding(df_circ, holdings_path, outdir):
    """
    Classic weeding analysis (from the original script) — requires a separate
    holdings file. Matches holdings to circulation on normalized title, then
    flags LC subclasses with the highest share of items that never circulated
    across the whole window.

    Expected holdings columns: 'Title (Normalized)' and 'Call Number'.
    """
    print("\n== Weeding analysis (holdings + circulation) ==")

    try:
        df_holdings = pd.read_csv(holdings_path)
    except FileNotFoundError:
        print(f"  [skip] holdings file not found: {holdings_path}")
        return None

    required = {"Call Number"}
    title_col = "Title (Normalized)" if "Title (Normalized)" in df_holdings.columns else "Title"
    required.add(title_col)
    missing = required - set(df_holdings.columns)
    if missing:
        print(f"  [skip] holdings file missing columns: {missing}")
        return None

    df_holdings["LC_Subclass"] = df_holdings["Call Number"].apply(get_subclass)
    df_holdings["Discipline"] = df_holdings["LC_Subclass"].apply(categorize_discipline)
    df_holdings["title_key"] = df_holdings[title_col].apply(normalize_title)

    # Sum loans across the whole window per title
    circ_loans = df_circ.copy()
    circ_loans["title_key"] = circ_loans["Title"].apply(normalize_title)
    per_title = (
        circ_loans.groupby("title_key")["Loans (In House + Not In House)"]
        .sum()
        .reset_index()
    )

    merged = pd.merge(df_holdings, per_title, on="title_key", how="left")
    merged["Loans (In House + Not In House)"] = (
        merged["Loans (In House + Not In House)"].fillna(0)
    )

    # Subclass-level uncirculated stats
    stats = merged.groupby(["Discipline", "LC_Subclass"]).agg(
        Total_Holdings=("title_key", "count"),
        Uncirculated_Holdings=(
            "Loans (In House + Not In House)", lambda x: (x == 0).sum()
        ),
    ).reset_index()
    stats["Percent_Uncirculated"] = (
        stats["Uncirculated_Holdings"] / stats["Total_Holdings"] * 100
    )
    stats["Description"] = stats["LC_Subclass"].map(
        lambda s: LC_SUBCLASS_DESC.get(s, f"{s} - Subclass {s}")
    )

    top = (
        stats[stats["Total_Holdings"] >= 150]
        .sort_values("Percent_Uncirculated", ascending=False)
        .head(15)
        .copy()
    )
    export = top[
        ["Discipline", "LC_Subclass", "Description",
         "Total_Holdings", "Uncirculated_Holdings", "Percent_Uncirculated"]
    ].copy()
    export["Percent_Uncirculated"] = export["Percent_Uncirculated"].round(2)
    out_csv = os.path.join(outdir, f"potential_weeding_areas_{fy_window_slug()}.csv")
    export.to_csv(out_csv, index=False)
    print(f"  -> {out_csv}")

    # Chart
    if not top.empty:
        fig, ax = plt.subplots(figsize=(11, 8))
        colors = [NEG_COLOR] * len(top)
        bars = ax.barh(top["Description"], top["Percent_Uncirculated"],
                       color=colors, height=0.7)
        ax.axvline(50, color="gray", linestyle=":", alpha=0.6)
        ax.axvline(70, color="#333333", linestyle="--", alpha=0.8,
                   label="High-Priority Weeding (70% Uncirculated)")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 1.0, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}%", va="center", ha="left",
                    fontsize=9.5, fontweight="semibold")

        ax.set_title(
            "Potential Weeding Areas: LC Subclasses with the Highest Rates of\n"
            f"Uncirculated Holdings (Min. 150 Holdings | 0 Loans "
            f"{FIRST_SHORT}\u2013{LAST_SHORT})",
            fontsize=13, fontweight="bold", pad=20,
        )
        ax.set_xlabel(
            f"% of Holdings with Zero Recorded Loans ({len(FY_COLS)}-Year Window)",
            fontsize=11, labelpad=10,
        )
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.legend(loc="lower right", framealpha=0.9)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        chart_out = os.path.join(outdir, f"weeding_analysis_{fy_window_slug()}.png")
        plt.savefig(chart_out, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"  -> {chart_out}")

    # Also emit the full subclass-level stats for review
    full_out = os.path.join(outdir, f"uncirculated_by_subclass_{fy_window_slug()}.csv")
    stats.sort_values(
        ["Discipline", "Percent_Uncirculated"], ascending=[True, False]
    ).to_csv(full_out, index=False)
    print(f"  -> {full_out}")

    return export


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT,
                        help=f"Path to the physical-usage CSV (default: {DEFAULT_INPUT})")
    parser.add_argument("--holdings", "-H", default=None,
                        help="Optional path to a holdings CSV with columns "
                             "'Title (Normalized)' (or 'Title') and 'Call Number'. "
                             "When provided, enables the classic uncirculated-holdings "
                             "weeding analysis in addition to the usage-decay weeding.")
    parser.add_argument("--outdir", "-o", default=DEFAULT_OUTDIR,
                        help=f"Directory for outputs (default: {DEFAULT_OUTDIR})")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    try:
        df = load_data(args.input)
    except FileNotFoundError:
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    run_lc_analysis(df, args.outdir)
    run_subject_analysis(df, args.outdir)
    run_geographic_analysis(df, args.outdir)
    run_weeding_candidates(df, args.outdir)
    if args.holdings:
        run_holdings_weeding(df, args.holdings, args.outdir)
    else:
        print("\n[info] No --holdings file supplied; skipping uncirculated-holdings "
              "weeding analysis. Pass --holdings PATH to enable it.")
    run_ebook_candidates(df, args.outdir)

    print(f"\nDone. All outputs written to: {args.outdir}/")


if __name__ == "__main__":
    main()
