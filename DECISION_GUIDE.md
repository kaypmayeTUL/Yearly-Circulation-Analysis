# Using the Collection Trend Analysis for Decisions

A short guide to reading what the tool produces and turning it into
weeding, purchasing, and collection-strategy calls.

---

## Two modes: aggregate vs. individual year

The fiscal-year multi-select at the top of the dashboard is the most
important control on the page.

- **Aggregate (2+ years selected)** — the tool fits a linear trend
  across every year you pick. Use this for anything long-term: annual
  reviews, five-year subscription decisions, weeding priorities,
  strategic collection statements. This is the default and the mode
  most decisions should live in.
- **Individual year (1 year selected)** — switches to a snapshot view.
  Use this to spot-check a single year, understand what a particular
  spike or dip looked like, brief a new liaison on where use is
  currently concentrated, or sanity-check the trend view.

A common workflow: run the aggregate view first to find a signal
(e.g., "M – Music is trending down 30%"), then flip to snapshot on the
most recent year to see what use in Music actually looks like today.

**Rule of thumb:** decisions with multi-year consequences want the
aggregate view; the snapshot is for context and communication.

---

## Reading the trend metric

Every "% change" in aggregate mode is a **cumulative trend fit** — a
line drawn through all the year totals — expressed as a percentage of
the first-year baseline. Two columns matter:

- **Cumulative % Change (trend)** — the direction and rough
  magnitude of the movement across the whole window.
- **Trend R²** — how well the straight line actually fits. Values
  close to 1.0 mean the years genuinely trend in a smooth line;
  values near 0 mean the years bounce around and the "trend" is
  fitting noise.

**The confidence rule:** trust the direction when R² ≥ 0.5. Below
that, treat the % change as directional guidance rather than a
finding. A subject at "+180% trend, R² 0.05" almost always means one
big year, not a trajectory — flip to snapshot mode to see the actual
year-by-year values before acting on it.

Endpoint % Change is also in every table, next to the trend. Use it
as a sanity check: when trend and endpoint agree, confidence is
high; when they disagree meaningfully, the interior years are doing
the work, and it's worth looking at the raw FY columns to see what
happened.

---

## Working through each tab

### LC Class

**What it tells you:** which broad areas of the collection are
growing or shrinking in use.

**When to use it:** to open a liaison meeting. "K – Law is up 26%,
R – Medicine is down 64%" is a five-second story that leads directly
into "what should we do about it?"

**How to act on it:**
- A large decline with high R² is a real signal — worth a
  conversation about e-resource migration, curriculum shift, or
  whether the print collection is still serving the discipline.
- A large increase is a signal that the discipline may need
  more selection attention.
- Discipline-level views (Humanities / Social Sciences / Sciences)
  are useful for showing subject librarians their own scope.

### LC Subclass

**What it tells you:** the same story at the resolution where
selection actually happens (PS – American Literature, QA – Mathematics,
KF – US Law).

**When to use it:** when the LC Class signal is interesting enough to
zoom in. If R (Medicine) is down at the class level, the subclass
view tells you whether that's RA (Public Health), RC (Internal
Medicine), or RT (Nursing) — three different weeding, purchasing,
and licensing conversations.

**How to act on it:**
- Subclasses in steep decline with high R² are strong candidates for
  weeding review and package-cancellation conversations.
- Steep risers deserve budget attention — is selection keeping up?
- Cross-reference with the Weeding tab for specific titles.

### Subject Headings

**What it tells you:** what your users are actually researching, at
the resolution of full LC subject strings. "Politics and government
--United States" stays distinct from "Politics and government--France"
so you're not muddling two different trends.

**When to use it:** for course-alignment conversations with liaisons,
new-program planning, and identifying emerging areas the selection
policy might not have caught up with.

**How to act on it:**
- Risers with high R² and a meaningful baseline (25+ loans in the
  first year) are candidates for selection focus.
- Decliners are worth an interpretive conversation: is this a course
  that stopped running, a faculty retirement, a discipline moving
  online, or a real drop in interest?
- The search box is the fastest way to check a specific topic before
  a curriculum-committee meeting.

### Geographic

**What it tells you:** which places and regions your users are
studying, and how that's shifting.

**When to use it:** for area-studies conversations, diversity and
inclusion audits of collection focus, and connecting collection use
to institutional strategic priorities (regional partnerships,
research center focus areas).

**How to act on it:**
- Region-level chart for the high-altitude story: which world
  regions are your users engaging with?
- Place-level for specific investigations — Louisiana holding steady
  vs. New Orleans (bare) declining might just be a cataloging split
  worth resolving, not a real decline.
- Steep rises in a specific region often correlate with a new faculty
  hire, new program, or shifting current events.

### Weeding Candidates

**What it tells you:** titles with meaningful early use that went
silent in the recent window, tiered by how substantial the early use
was.

**How to act on it:**
- **Start with the Strong tier.** These are titles that had real
  demand and stopped — the conversation is "what changed?" (course
  ended, faculty left, format migration) more than "should we
  weed?"
- **Medium tier** for periodic review — good to scan discipline-by-
  discipline once a year.
- **Weak tier** is a scanning list, not an action list. Volume there
  says more about how big your collection is than about specific
  weeding priorities.
- **The publication-year gate** protects recent items (default: last
  5 years). If a Strong-tier title is a recent publication that
  hasn't circulated, dig in — it might be a cataloging or shelving
  issue rather than lack of interest.
- Always cross-reference with course reserves, faculty publications,
  and ILL patterns before weeding anything.

### E-Book Candidates

**What it tells you:** high-use print titles where a digital license
might improve access.

**How to act on it:**
- The rationale column tells you which kind of case each title
  makes: sustained + rising (highest priority), sustained (steady
  demand for multi-user access), rising (recent pressure), or
  historical volume (worth investigating format).
- Check availability and licensing terms; not everything is
  available as an academic-license e-book.
- Look for course-reserve overlap — that's where multi-user
  simultaneous access has the biggest impact.

### Holdings Weeding

**What it tells you (when a holdings file is uploaded):** LC
subclasses with the highest share of items that never circulated
across the whole window.

**How to act on it:**
- 70%+ uncirculated is a real concentration worth investigating.
- Use it as the entry point for a physical shelf-shift conversation,
  not as a "weed everything here" recommendation. High uncirculated
  rates in a small collection can be intentional (reference,
  archival, uniquely local material).

---

## The annual rhythm

**Once the tool becomes annual, here's a working cadence:**

**When new fiscal-year data arrives:**
1. Run the aggregate view over the full available window. Capture
   the highest-R² risers and decliners at the class and subclass
   levels as a starting brief.
2. Flip to snapshot on the most recent year. Confirm the biggest
   movers make sense in context.
3. Export the CSVs into the shared folder for liaisons to browse.
4. Schedule liaison conversations on the strongest signals before
   proposing action.

**Before renewal-decision windows:**
1. Aggregate over the last 3–4 years to see sustained trajectories.
2. Flag subclasses with high-R² sustained decline for e-package
   review.
3. Cross-reference with vendor cost-per-use where you have it.

**Before weeding projects:**
1. Aggregate over all available years.
2. Filter Weeding tab to Strong tier + target discipline.
3. Cross-reference with holdings if uploaded.
4. Walk the list with the subject liaison before any physical
   review.

**Before purchasing conversations:**
1. Aggregate to identify high-R² risers.
2. E-Book Candidates for high-demand format migration.
3. Subject Headings for emerging topics that may need new
   selection focus.

---

## What the tool doesn't tell you

Worth naming, because it saves arguments later:

- **Reasons.** The tool shows what changed, not why. A subject
  declining could mean the discipline moved online, a course was
  cancelled, a faculty member retired, or the topic genuinely
  cooled. That's a conversation with the liaison, not a data
  question.
- **Impact.** A high-loan title isn't necessarily high-value; a
  low-loan title isn't necessarily low-value. Circulation is one
  signal among several.
- **Format substitution.** A print subclass declining while its
  e-book equivalent surges is format migration, not disinterest.
  The tool sees print only.
- **Comparators.** The tool tells you what your collection looks
  like; it doesn't tell you whether that matches peer institutions
  or national norms.

Use the tool to find the signals worth talking about, then bring
the conversation to the people who can interpret them.
