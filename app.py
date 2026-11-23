"""
Box Office vs. Buzz — Group 1 Python Dashboard (MIS 480C)
Question: Do the movies critics and audiences love actually make money?
          ...and what should we watch tonight?
Data: IMDB Movie Dataset (1,000 popular movies, 2006–2016), Kaggle.

Run locally:  python app.py   ->  open http://127.0.0.1:8051
Deploy:       Render start command
              gunicorn app:server --workers 2 --threads 8 --timeout 60

Posters, trailers and "where to watch":
  * Works with NO setup: posters come from IMDb's public search feed, trailers from a
    YouTube search, and "where to watch" shows one-click search links (JustWatch, Netflix...).
  * Optional, better: set the environment variable TMDB_API_KEY (free key from
    themoviedb.org -> Settings -> API). Then posters and official trailers come from TMDB
    and "where to watch" lists the real streaming services for your region
    (WATCH_REGION, default "US"). On Render: Environment -> Add Environment Variable.
  * Lookups are cached in media_cache.json next to this file, so each movie is only
    looked up once.
"""
import json
import os
import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from html import escape
from pathlib import Path
from urllib.parse import quote, quote_plus
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
from flask import Response, abort, redirect

# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
df = pd.read_csv(HERE / "imdb_movie_dataset.csv")
df = df.rename(columns={"Runtime (Minutes)": "Runtime", "Revenue (Millions)": "Revenue"})
df["Title"] = df["Title"].str.strip()
df["Director"] = df["Director"].str.strip()
df["GenreList"] = df["Genre"].str.split(",")
df["ActorList"] = df["Actors"].str.split(",").apply(lambda xs: [a.strip() for a in xs])
df.index = df["Rank"].values  # look a movie up with df.loc[rank]

ALL_GENRES = sorted(df["GenreList"].explode().unique())
YEAR_MIN, YEAR_MAX = int(df["Year"].min()), int(df["Year"].max())
RUN_MIN, RUN_MAX = 60, 195          # slider ends (data: 66–191 min)
VOTE_MIN, VOTE_MAX = 1.5, 6.3       # log10 of votes (data: 61 – 1.79M)

_actor_counts = df["ActorList"].explode().value_counts()
ACTOR_OPTIONS = [{"label": f"{a}  ({n})", "value": a} for a, n in sorted(_actor_counts.items())]
_dir_counts = df["Director"].value_counts()
DIRECTOR_OPTIONS = [{"label": f"{d}  ({n})", "value": d} for d, n in sorted(_dir_counts.items())]
MOVIE_OPTIONS = [{"label": f"{r.Title} ({r.Year})", "value": int(r.Rank)}
                 for r in df.sort_values("Votes", ascending=False).itertuples()]

SORTS = {
    "popular": ("Most popular (IMDb votes)", "Votes", False),
    "rating": ("Highest IMDb rating", "Rating", False),
    "metascore": ("Highest critic score", "Metascore", False),
    "revenue": ("Biggest box office", "Revenue", False),
    "newest": ("Newest first", "Year", False),
    "oldest": ("Oldest first", "Year", True),
    "shortest": ("Shortest runtime", "Runtime", True),
    "longest": ("Longest runtime", "Runtime", False),
    "title": ("Title A–Z", "Title", True),
}

METRICS = {
    "Rating": {"label": "Audience score (IMDb rating, 0–10)", "short": "IMDb rating"},
    "Metascore": {"label": "Critic score (Metascore, 0–100)", "short": "Metascore"},
}

DEFAULTS = {
    "search": "", "genre-dd": [], "actor-dd": [], "director-dd": [],
    "year-slider": [YEAR_MIN, YEAR_MAX], "rating-slider": [1, 10],
    "runtime-slider": [RUN_MIN, RUN_MAX], "votes-slider": [VOTE_MIN, VOTE_MAX],
    "sort-dd": "popular",
}
FILTER_IDS = ["search", "genre-dd", "actor-dd", "director-dd",
              "year-slider", "rating-slider", "runtime-slider", "votes-slider"]
FILTER_INPUTS = [Input(i, "value") for i in FILTER_IDS]
FILTER_STATES = [State(i, "value") for i in FILTER_IDS]

# ---------------------------------------------------------------------------
# 2. DESIGN SYSTEM  (change colors here and the whole dashboard follows)
# ---------------------------------------------------------------------------
BG = "#000000"        # pure black, like a theater
CARD = "#0b0b0e"      # card / chart background
CARD_2 = "#131318"
BORDER = "#22222b"
TEXT = "#f3efe6"      # warm "screen" white
MUTED = "#8e8c96"
GOLD = "#f5c518"      # marquee gold
RED = "#d7263d"       # velvet-curtain red
GOLD_SCALE = [[0, "#2a2108"], [0.5, "#b8901a"], [1, "#ffe07a"]]
FONT = "Inter, 'Segoe UI', Arial, sans-serif"
DISPLAY = "'Bebas Neue', Impact, 'Arial Narrow', sans-serif"


def style_fig(fig, height=380):
    """Apply one consistent look to every chart."""
    fig.update_layout(
        height=height, paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family=FONT, color=TEXT, size=13),
        margin=dict(l=20, r=20, t=10, b=20),
        hoverlabel=dict(bgcolor=CARD_2, bordercolor=GOLD, font=dict(family=FONT, color=TEXT)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    return fig


def empty_fig(message="No movies match these filters", height=380):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=MUTED, size=15))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_fig(fig, height)


# ---------------------------------------------------------------------------
# 3. HELPERS
# ---------------------------------------------------------------------------
def filter_movies(search, genres, actors, directors, years, rating, runtime, votes):
    """Shared filter used by every view in the dashboard."""
    years = years or DEFAULTS["year-slider"]
    rating = rating or DEFAULTS["rating-slider"]
    runtime = runtime or DEFAULTS["runtime-slider"]
    votes = votes or DEFAULTS["votes-slider"]
    d = df[(df["Year"] >= years[0]) & (df["Year"] <= years[1])
           & (df["Rating"] >= rating[0]) & (df["Rating"] <= rating[1])]
    d = d[d["Runtime"] >= runtime[0]]
    if runtime[1] < RUN_MAX:                      # top of the slider = "no limit"
        d = d[d["Runtime"] <= runtime[1]]
    if votes[0] > VOTE_MIN:
        d = d[d["Votes"] >= 10 ** votes[0]]
    if votes[1] < VOTE_MAX:
        d = d[d["Votes"] <= 10 ** votes[1]]
    if search:
        d = d[d["Title"].str.contains(search.strip(), case=False, regex=False)]
    if genres:
        d = d[d["GenreList"].apply(lambda gl: any(g in genres for g in gl))]
    if actors:
        d = d[d["ActorList"].apply(lambda al: any(a in actors for a in al))]
    if directors:
        d = d[d["Director"].isin(directors)]
    return d


def sort_movies(d, sort_key):
    _, col, asc = SORTS.get(sort_key or "popular", SORTS["popular"])
    return d.sort_values([col, "Votes"], ascending=[asc, False], na_position="last")


def exploded(d, genres):
    """One row per movie-genre pair (a movie can have up to 3 genres)."""
    e = d.explode("GenreList").rename(columns={"GenreList": "G"})
    if genres:
        e = e[e["G"].isin(genres)]
    return e


def money(x):
    if x is None or pd.isna(x):
        return "n/a"
    return f"${x / 1000:,.1f}B" if x >= 1000 else f"${x:,.0f}M"


def hm(minutes):
    minutes = int(minutes)
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def short_num(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K" if n >= 1e3 else f"{n:.0f}"


def similar_movies(rank, n=6):
    """Recommend films that share genres (plus a nudge for same director/cast and similar rating)."""
    m = df.loc[rank]
    gset, aset = set(m["GenreList"]), set(m["ActorList"])
    others = df[df["Rank"] != rank]
    genre_sim = others["GenreList"].apply(lambda gl: len(gset & set(gl)) / len(gset | set(gl)))
    votes_norm = (np.log10(others["Votes"]) - 2) / 4.3            # 0..1 popularity
    bonus = (0.30 * (others["Director"] == m["Director"])
             + 0.12 * others["ActorList"].apply(lambda al: len(aset & set(al)))
             + 0.40 * (others["Rating"] - 6) / 3                    # favor well-rated films
             + 0.10 * votes_norm
             - 0.10 * (others["Rating"] - m["Rating"]).abs() / 9)
    score = genre_sim + bonus
    out = others.assign(Score=score, GenreSim=genre_sim)
    out = out[out["GenreSim"] > 0].sort_values(["Score", "Votes"], ascending=False).head(n)
    return out


# ---------------------------------------------------------------------------
# 4. POSTERS, TRAILERS, WHERE TO WATCH  (looked up on demand, then cached)
# ---------------------------------------------------------------------------
MEDIA_FILE = HERE / "media_cache.json"
TMDB_KEY = os.environ.get("TMDB_API_KEY", "").strip()
WATCH_REGION = os.environ.get("WATCH_REGION", "US").strip().upper()
_lock = threading.Lock()
_unsaved = 0
try:
    _media = json.loads(MEDIA_FILE.read_text())
except Exception:
    _media = {}


def _get(url, timeout=6):
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Cookie": "CONSENT=YES+1",
    })
    with urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def _remember(rank, force_save=False, **fields):
    global _unsaved
    with _lock:
        _media.setdefault(str(rank), {}).update(fields)
        _unsaved += 1
        if force_save or _unsaved >= 10:
            try:
                tmp = MEDIA_FILE.with_suffix(".tmp")
                tmp.write_text(json.dumps(_media))
                tmp.replace(MEDIA_FILE)
                _unsaved = 0
            except OSError:
                pass  # read-only disk: keep the in-memory cache only


def _norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _imdb_suggest(title, year):
    """IMDb's public search-suggestion feed -> (poster URL, IMDb id)."""
    slug = _norm(title).replace(" ", "_")
    if not slug:
        return None, None
    data = json.loads(_get(f"https://v2.sg.media-imdb.com/suggestion/{slug[0]}/{quote(slug)}.json"))
    best, best_score = None, -1
    for c in data.get("d", []):
        if not str(c.get("id", "")).startswith("tt"):
            continue
        s = 3 if _norm(c.get("l", "")) == _norm(title) else 0
        y = c.get("y")
        if y:
            s += 3 if y == year else 2 if abs(y - year) <= 1 else 0
        s += 1 if c.get("qid") in ("movie", "tvMovie", "video") else 0
        if s > best_score:
            best, best_score = c, s
    if not best or best_score < 3:
        return None, None
    img = (best.get("i") or {}).get("imageUrl")
    if img:
        img = re.sub(r"\._V1_.*?\.(jpg|png)$", r"._V1_SX400.\1", img)
    return img, best["id"]


@lru_cache(maxsize=2048)
def _tmdb(rank):
    """Full TMDB record (with videos + watch providers). Only used when TMDB_API_KEY is set."""
    if not TMDB_KEY:
        return None
    m = df.loc[rank]
    tid = _media.get(str(rank), {}).get("tmdb")
    if not tid:
        base = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_KEY}&query={quote_plus(m['Title'])}"
        res = json.loads(_get(f"{base}&year={m['Year']}")).get("results") or \
            json.loads(_get(base)).get("results") or []
        if not res:
            return None
        tid = res[0]["id"]
        _remember(rank, tmdb=tid)
    return json.loads(_get(f"https://api.themoviedb.org/3/movie/{tid}?api_key={TMDB_KEY}"
                           "&append_to_response=videos,watch/providers"))


def poster_url(rank):
    e = _media.get(str(rank), {})
    if "poster" in e:
        return e["poster"]
    m = df.loc[rank]
    url = imdb = None
    try:
        tm = _tmdb(rank)
        if tm and tm.get("poster_path"):
            url, imdb = "https://image.tmdb.org/t/p/w500" + tm["poster_path"], tm.get("imdb_id")
        if not url:
            url, imdb2 = _imdb_suggest(m["Title"], int(m["Year"]))
            imdb = imdb or imdb2
    except Exception:
        return None          # network problem: don't cache, try again next time
    _remember(rank, poster=url, imdb=imdb)
    return url


def trailer_id(rank):
    e = _media.get(str(rank), {})
    if e.get("trailer"):
        return e["trailer"]
    m = df.loc[rank]
    vid = None
    try:
        tm = _tmdb(rank)
        if tm:
            vids = [v for v in (tm.get("videos") or {}).get("results", [])
                    if v.get("site") == "YouTube" and v.get("type") == "Trailer"]
            vids.sort(key=lambda v: (not v.get("official"), v.get("published_at", "")))
            vid = vids[0]["key"] if vids else None
        if not vid:
            page = _get("https://www.youtube.com/results?search_query="
                        + quote_plus(f"{m['Title']} {m['Year']} official trailer"))
            hit = re.search(r'"videoId":"([\w-]{11})"', page)
            vid = hit.group(1) if hit else None
    except Exception:
        return None
    if vid:
        _remember(rank, force_save=True, trailer=vid)
    return vid


@lru_cache(maxsize=1024)
def providers(rank):
    """Streaming / rent / buy services for WATCH_REGION (TMDB, data by JustWatch)."""
    try:
        tm = _tmdb(rank)
    except Exception:
        return None
    if not tm:
        return None
    return ((tm.get("watch/providers") or {}).get("results") or {}).get(WATCH_REGION)


def youtube_search_url(m):
    return "https://www.youtube.com/results?search_query=" + quote_plus(f"{m['Title']} {m['Year']} official trailer")


def placeholder_svg(rank):
    m = df.loc[rank]
    words, lines, cur = m["Title"].split(), [], ""
    for w in words:
        if len(cur) + len(w) > 12 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    lines = lines[:5]
    y0 = 225 - 22 * (len(lines) - 1)
    text = "".join(f'<text x="150" y="{y0 + i * 44}" text-anchor="middle" font-family="Impact, Arial Narrow, Arial, sans-serif" '
                   f'font-weight="bold" font-size="32" fill="{GOLD}"'
                   + (' textLength="216" lengthAdjust="spacingAndGlyphs"' if len(l) > 10 else '')
                   + f'>{escape(l.upper())}</text>' for i, l in enumerate(lines))
    holes = "".join(f'<rect x="8" y="{y}" width="12" height="16" rx="3" fill="#1d1d24"/>'
                    f'<rect x="280" y="{y}" width="12" height="16" rx="3" fill="#1d1d24"/>' for y in range(14, 450, 34))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 450" width="300" height="450">'
            f'<rect width="300" height="450" fill="#09090c"/>{holes}'
            f'<rect x="30" y="20" width="240" height="410" fill="#111117" stroke="#2a2a33"/>{text}'
            f'<text x="150" y="400" text-anchor="middle" font-family="Inter, Arial" font-size="16" fill="{MUTED}">'
            f'{m["Year"]}</text></svg>')


def _prefetch_posters():
    todo = [int(r) for r in df["Rank"] if "poster" not in _media.get(str(int(r)), {})]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(poster_url, todo))
    _remember(0, force_save=True, note="cache")


if os.environ.get("PREFETCH_MEDIA", "1") != "0":
    threading.Thread(target=_prefetch_posters, daemon=True).start()


# ---------------------------------------------------------------------------
# 5. APP + LAYOUT
# ---------------------------------------------------------------------------
app = Dash(
    __name__,
    title="Box Office vs. Buzz",
    external_stylesheets=["https://fonts.googleapis.com/css2?family=Bebas+Neue&"
                          "family=Inter:wght@400;500;600;700;800&display=swap"],
)
server = app.server  # Render needs this line


@server.route("/poster/<int:rank>")
def poster_route(rank):
    if rank not in df.index:
        abort(404)
    url = poster_url(rank)
    if url:
        resp = redirect(url, 302)
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp
    return Response(placeholder_svg(rank), mimetype="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=300"})


@server.route("/trailer/<int:rank>")
def trailer_route(rank):
    if rank not in df.index:
        abort(404)
    vid = trailer_id(rank)
    if vid:
        return redirect(f"https://www.youtube-nocookie.com/embed/{vid}?autoplay=1&rel=0&modestbranding=1", 302)
    m = df.loc[rank]
    return Response(
        f"""<html><body style="margin:0;background:#000;color:{TEXT};font-family:Inter,Arial;display:flex;
        align-items:center;justify-content:center;height:100vh;text-align:center">
        <div><div style="font-size:42px">🎞️</div><p>Couldn't load the trailer here.</p>
        <a href="{escape(youtube_search_url(m))}" target="_blank" rel="noopener"
           style="color:#000;background:{GOLD};padding:10px 18px;border-radius:999px;text-decoration:none;
           font-weight:700">▶ Watch on YouTube</a></div></body></html>""", mimetype="text/html")


GRAPH_CONFIG = {"displayModeBar": False}


def ban(title, value_id, sub_id):
    return html.Div(className="ban", children=[
        html.Div(title, className="ban-title"),
        html.Div(id=value_id, className="ban-value"),
        html.Div(id=sub_id, className="ban-sub"),
    ])


def card(title, subtitle, body, extra_class="", head_extra=None):
    return html.Div(className=f"card {extra_class}", children=[
        html.Div(className="card-head", children=[
            html.Div([html.H3(title), html.P(subtitle)]),
            head_extra,
        ]),
        body,
    ])


def section_title(number, text, anchor):
    return html.H2([html.Span(number, className="reel-no"), text], id=anchor, className="section-title")


def filter_group(label, control, readout_id=None):
    return html.Div(className="fgroup", children=[
        html.Div(className="flabel", children=[
            html.Label(label), html.Span(id=readout_id, className="readout") if readout_id else None,
        ]),
        control,
    ])


sidebar = html.Aside(className="sidebar", children=[
    html.Div("🎞  FILTERS", className="side-title"),
    filter_group("Search title", dcc.Input(id="search", type="search", value="", debounce=True,
                                           placeholder="e.g. Inception", className="text-in")),
    filter_group("Sort by", dcc.Dropdown(id="sort-dd", value="popular", clearable=False, className="dd",
                                         options=[{"label": v[0], "value": k} for k, v in SORTS.items()])),
    filter_group("Genres", dcc.Dropdown(id="genre-dd", options=ALL_GENRES, value=[], multi=True,
                                        placeholder="All genres", className="dd")),
    filter_group("Favorite actors", dcc.Dropdown(id="actor-dd", options=ACTOR_OPTIONS, value=[], multi=True,
                                                 placeholder="Type a name…", className="dd")),
    filter_group("Directors", dcc.Dropdown(id="director-dd", options=DIRECTOR_OPTIONS, value=[], multi=True,
                                           placeholder="Any director", className="dd")),
    filter_group("Release years", dcc.RangeSlider(
        id="year-slider", min=YEAR_MIN, max=YEAR_MAX, step=1, value=[YEAR_MIN, YEAR_MAX],
        allow_direct_input=False, marks={y: str(y)[2:] if y not in (YEAR_MIN, YEAR_MAX) else str(y)
                                         for y in range(YEAR_MIN, YEAR_MAX + 1, 2)}), "year-readout"),
    filter_group("IMDb user rating", dcc.RangeSlider(
        id="rating-slider", min=1, max=10, step=0.1, value=[1, 10], allow_direct_input=False,
        marks={i: str(i) for i in range(1, 11)}), "rating-readout"),
    filter_group("Runtime", dcc.RangeSlider(
        id="runtime-slider", min=RUN_MIN, max=RUN_MAX, step=5, value=[RUN_MIN, RUN_MAX],
        allow_direct_input=False,
        marks={60: "1h", 90: "1h30", 120: "2h", 150: "2h30", 180: "3h"}), "runtime-readout"),
    filter_group("Number of IMDb votes", dcc.RangeSlider(
        id="votes-slider", min=VOTE_MIN, max=VOTE_MAX, step=0.05, value=[VOTE_MIN, VOTE_MAX],
        allow_direct_input=False, marks={2: "100", 3: "1K", 4: "10K", 5: "100K", 6: "1M"}), "votes-readout"),
    html.Button("↺  Reset filters", id="reset-btn", n_clicks=0, className="btn outline full"),
    html.Div(id="match-count", className="match-count"),
])

drawer = html.Div(children=[
    html.Div(id="drawer-backdrop", className="backdrop", n_clicks=0),
    html.Aside(id="drawer", className="drawer", children=[
        html.Button("✕", id="drawer-close", className="drawer-close", n_clicks=0, title="Close"),
        html.Div(id="drawer-hero"),
        html.Div(className="drawer-actions", children=[
            html.Button("▶  Watch trailer", id="trailer-btn", className="btn red", n_clicks=0),
            html.Button("＋ Add to picks", id="add-pick-btn", className="btn gold", n_clicks=0),
            html.Button("⇄  Compare", id="compare-btn", className="btn outline", n_clicks=0),
        ]),
        html.Div(id="trailer-wrap", className="trailer-wrap", style={"display": "none"}, children=[
            html.Iframe(id="trailer-frame", src="about:blank",
                        allow="autoplay; encrypted-media; picture-in-picture; fullscreen"),
        ]),
        html.Div(id="drawer-plot"),
        html.Div(id="drawer-watch"),
        html.Div(id="drawer-similar"),
    ]),
])

app.layout = html.Div(className="page", children=[
    dcc.Store(id="selected"),
    dcc.Store(id="picks-store", storage_type="local", data=[]),

    # ---- Marquee header ----
    html.Header(className="marquee", children=[
        html.Div(className="filmstrip"),
        html.Div(className="marquee-inner", children=[
            html.Div(children=[
                html.Div("★  NOW SHOWING  ·  GROUP 1  ·  MIS 480C  ★", className="eyebrow"),
                html.H1(["BOX OFFICE ", html.Span("vs.", className="accent"), " BUZZ"]),
                html.P("1,000 popular films from 2006–2016. Find tonight's movie, then see whether "
                       "the films critics and audiences love actually make money.", className="lede"),
            ]),
            html.Div(className="header-actions", children=[
                html.A(["🎟  My Picks ", html.Span(id="picks-badge", className="badge")],
                       href="#picks", className="btn outline"),
                html.A("🎰  Movie night roulette", href="#roulette", className="btn gold"),
            ]),
        ]),
        html.Div(className="filmstrip"),
    ]),
    html.Nav(className="nav", children=[
        html.A("Discover", href="#discover"), html.A("Roulette & Picks", href="#roulette"),
        html.A("Popularity vs Rating", href="#popularity"), html.A("Compare", href="#compare"),
        html.A("Box office insights", href="#insights"),
    ]),

    html.Div(className="layout", children=[
        sidebar,
        html.Main(className="main", children=[

            # ---- 1. Discover ----
            section_title("01", "Discover", "discover"),
            html.Div(className="bans", children=[
                ban("Movies matching", "ban-movies", "ban-movies-sub"),
                ban("Avg IMDb rating", "ban-rating", "ban-rating-sub"),
                ban("Avg runtime", "ban-run", "ban-run-sub"),
                ban("Total box office", "ban-rev", "ban-rev-sub"),
                ban("Avg critic score", "ban-meta", "ban-meta-sub"),
            ]),
            card("Now showing", "Hover a poster for the plot · click it for the trailer, where to watch "
                 "and similar films.",
                 html.Div([
                     html.Div(id="poster-grid", className="poster-grid"),
                     html.Div(className="more-row", children=[
                         html.Button("Show more movies", id="more-btn", n_clicks=0, className="btn outline")]),
                 ])),

            # ---- 2. Roulette + picks ----
            section_title("02", "Roulette & My Picks", "roulette"),
            html.Div(className="row two", children=[
                card("Movie night roulette 🎰", "Can't decide? Spin for a random movie that matches your filters.",
                     html.Div(className="slot", children=[
                         html.Div(className="slot-cabinet", children=[
                             html.Div(className="slot-window", children=[
                                 html.Div(id="reel", className="reel", children=[
                                     html.Div("PRESS SPIN", className="reel-item")]),
                             ]),
                             html.Div(className="payline"),
                         ]),
                         html.Button("SPIN", id="spin-btn", n_clicks=0, className="spin-btn"),
                         html.Div(id="roulette-result"),
                     ])),
                html.Div(id="picks", children=card(
                    "My Picks 🎟", "Your watchlist — saved in this browser.",
                    html.Div([
                        html.Div(id="picks-summary", className="picks-summary"),
                        html.Div(id="picks-list", className="picks-list"),
                        html.Button("Clear all picks", id="clear-picks", n_clicks=0, className="btn ghost small"),
                    ]))),
            ]),

            # ---- 3. Popularity vs rating ----
            section_title("03", "Popularity vs Rating", "popularity"),
            card("Is the crowd's favorite also the best?",
                 "Each dot is a movie: how many people rated it (popularity) vs. how high they rated it. "
                 "Hover for the poster, click to open. Red rings = your picks.",
                 html.Div([dcc.Graph(id="pop-scatter", config=GRAPH_CONFIG, clear_on_unhover=True),
                           dcc.Tooltip(id="pop-tip", background_color=CARD_2, border_color=GOLD,
                                       direction="right")])),

            # ---- 4. Compare ----
            section_title("04", "Compare two movies", "compare"),
            card("Head to head", "Pick any two films to see them side by side. The winner of each row glows gold.",
                 html.Div([
                     html.Div(className="compare-pickers", children=[
                         dcc.Dropdown(id="compare-a", options=MOVIE_OPTIONS, clearable=False, className="dd",
                                      value=int(df.loc[df["Title"] == "The Dark Knight", "Rank"].iloc[0])),
                         html.Div("VS", className="vs"),
                         dcc.Dropdown(id="compare-b", options=MOVIE_OPTIONS, clearable=False, className="dd",
                                      value=int(df.loc[df["Title"] == "Inception", "Rank"].iloc[0])),
                     ]),
                     html.Div(id="compare-body"),
                 ])),

            # ---- 5. Insights (the original analysis) ----
            section_title("05", "Box office insights", "insights"),
            html.Div(className="insight-bar", children=[
                html.Div(id="insight-callout", className="callout"),
                html.Div(className="metric-toggle", children=[
                    html.Label("Measure buzz by"),
                    dcc.RadioItems(id="metric", className="toggle", inline=True,
                                   options=[{"label": "Audiences", "value": "Rating"},
                                            {"label": "Critics", "value": "Metascore"}], value="Rating"),
                ]),
            ]),
            html.Div(className="row", children=[
                card("Does buzz pay off?", "Each bubble is a movie (size = IMDb votes). Hover for the poster, "
                     "click to open.",
                     html.Div([dcc.Graph(id="scatter", config=GRAPH_CONFIG, clear_on_unhover=True),
                               dcc.Tooltip(id="scatter-tip", background_color=CARD_2, border_color=GOLD,
                                           direction="right")]), "span-2"),
                card("Top 10 box office hits", "Highest grossing in your selection. Click a bar to open.",
                     dcc.Graph(id="top-bar", config=GRAPH_CONFIG)),
            ]),
            html.Div(className="row two", children=[
                card("Where the money is, by genre", "Box size = total revenue, color = average score. "
                     "Click a genre to filter the whole dashboard to it.",
                     dcc.Graph(id="treemap", config=GRAPH_CONFIG)),
                card("Is quality changing over time?", "Average score by genre and year. Brighter gold = better "
                     "reviewed; empty = no movies.",
                     dcc.Graph(id="heatmap", config=GRAPH_CONFIG)),
            ]),
            html.Footer([
                "Data: IMDB Movie Dataset (Kaggle), 1,000 most popular movies 2006–2016. Revenue in US$ millions; "
                "128 movies have no revenue reported. Movies can belong to up to 3 genres, so genre totals overlap. ",
                html.Br(),
                "Posters & trailers via IMDb / YouTube" + (" and TMDB. Streaming data by JustWatch via TMDB. "
                                                          "This product uses the TMDB API but is not endorsed or "
                                                          "certified by TMDB." if TMDB_KEY else "."),
            ]),
        ]),
    ]),
    drawer,
])

# Page-wide CSS lives here so the whole app stays in one file (easy to deploy).
CSS = """
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; background: BG; color: TEXT; font-family: FONT;
       background-image: radial-gradient(ellipse at top, #1a1206 0%, #000 55%); background-attachment: fixed; }
body:after { content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 1;
             background: radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,.65) 100%); }
a { color: GOLD; }
.page { max-width: 1520px; margin: 0 auto; padding: 22px 28px 40px; position: relative; z-index: 2; }

/* ---- marquee header ---- */
.marquee { background: #050505; border: 1px solid BORDER; border-radius: 16px; overflow: hidden;
           box-shadow: 0 0 60px rgba(245,197,24,.08); }
.filmstrip { height: 22px; background: #111;
  background-image: repeating-linear-gradient(90deg, transparent 0 10px, #000 10px 24px, transparent 24px 34px);
  background-size: 34px 12px; background-repeat: repeat-x; background-position: 0 5px; }
.marquee-inner { display: flex; align-items: center; justify-content: space-between; gap: 20px; padding: 18px 30px;
  border-top: 2px dotted GOLD; border-bottom: 2px dotted GOLD; flex-wrap: wrap; }
.eyebrow { color: GOLD; font-size: 12px; letter-spacing: .3em; font-weight: 700; opacity: .85; }
h1 { font-family: DISPLAY; font-size: 72px; font-weight: 400; margin: 4px 0 0; letter-spacing: .04em; line-height: 1;
     color: TEXT; text-shadow: 0 0 18px rgba(245,197,24,.35); }
.accent { color: RED; font-style: italic; font-family: Georgia, serif; font-size: .55em; vertical-align: middle;
          text-shadow: 0 0 14px rgba(215,38,61,.6); }
.lede { color: MUTED; margin: 6px 0 0; font-size: 15px; max-width: 640px; }
.header-actions { display: flex; gap: 10px; flex-wrap: wrap; }
.nav { display: flex; gap: 6px; margin: 14px 0 18px; flex-wrap: wrap; }
.nav a { color: MUTED; text-decoration: none; font-size: 13px; padding: 6px 12px; border-radius: 999px;
         border: 1px solid transparent; }
.nav a:hover { color: GOLD; border-color: BORDER; }

/* ---- buttons ---- */
.btn { font-family: inherit; font-weight: 700; font-size: 14px; border-radius: 999px; padding: 9px 18px;
       cursor: pointer; border: 1px solid transparent; text-decoration: none; display: inline-flex;
       align-items: center; gap: 6px; transition: transform .12s, box-shadow .12s, background .12s; }
.btn:hover { transform: translateY(-1px); }
.btn.gold { background: GOLD; color: #000; box-shadow: 0 0 18px rgba(245,197,24,.25); }
.btn.red { background: RED; color: #fff; box-shadow: 0 0 18px rgba(215,38,61,.3); }
.btn.outline { background: transparent; color: TEXT; border-color: #3a3a44; }
.btn.outline:hover { border-color: GOLD; color: GOLD; }
.btn.ghost { background: transparent; color: MUTED; border-color: transparent; }
.btn.ghost:hover { color: RED; }
.btn.small { font-size: 12px; padding: 6px 10px; }
.btn.full { width: 100%; justify-content: center; margin-top: 8px; }
.btn.in-picks { background: #1d1806; color: GOLD; border-color: GOLD; box-shadow: none; }
.badge { background: RED; color: #fff; border-radius: 999px; padding: 1px 8px; font-size: 12px; }

/* ---- layout ---- */
.layout { display: grid; grid-template-columns: 300px minmax(0, 1fr); gap: 20px; align-items: start; }
.sidebar { position: sticky; top: 14px; max-height: calc(100vh - 28px); overflow-y: auto; background: CARD;
           border: 1px solid BORDER; border-radius: 16px; padding: 18px; }
.side-title { font-family: DISPLAY; font-size: 24px; letter-spacing: .08em; color: GOLD; margin-bottom: 8px; }
.fgroup { margin-bottom: 16px; }
.flabel { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; margin-bottom: 8px; }
.flabel label { font-size: 11px; color: MUTED; text-transform: uppercase; letter-spacing: .12em; font-weight: 700; }
.readout { font-size: 12px; color: GOLD; font-weight: 600; white-space: nowrap; }
.text-in { width: 100%; background: #050507; border: 1px solid BORDER; color: TEXT; border-radius: 8px;
           padding: 9px 12px; font-family: inherit; font-size: 14px; outline: none; }
.text-in:focus { border-color: GOLD; }
.match-count { color: MUTED; font-size: 12px; text-align: center; margin-top: 10px; }
.main { min-width: 0; }
.section-title { font-family: DISPLAY; font-weight: 400; font-size: 36px; letter-spacing: .06em; margin: 26px 0 12px;
                 display: flex; align-items: center; gap: 12px; scroll-margin-top: 16px; }
.section-title:first-child { margin-top: 0; }
.reel-no { font-size: 15px; color: #000; background: GOLD; border-radius: 50%; width: 34px; height: 34px;
           display: inline-flex; align-items: center; justify-content: center; font-family: Inter; font-weight: 800; }

/* ---- Dash 4 component theme (dropdown, slider, radio) ---- */
:root {
  --Dash-Fill-Interactive-Strong: GOLD; --Dash-Fill-Interactive-Weak: #2a2108;
  --Dash-Fill-Inverse-Strong: #050507; --Dash-Fill-Primary-Hover: #1d1806; --Dash-Fill-Primary-Active: #2a2108;
  --Dash-Text-Strong: TEXT; --Dash-Text-Primary: TEXT; --Dash-Text-Weak: MUTED;
  --Dash-Stroke-Strong: BORDER; --Dash-Stroke-Weak: BORDER; --Dash-Fill-Disabled: BORDER;
}
.dash-dropdown, .dash-dropdown-content { background: #050507 !important; color: TEXT !important;
                                         border-color: BORDER !important; }
.dash-dropdown-placeholder { color: MUTED !important; }
.dash-dropdown-search { background: transparent; color: TEXT; }
.dash-slider-mark { color: MUTED !important; font-size: 10px; }
.dash-slider-track { background: BORDER !important; }
.dash-slider-range { background: GOLD !important; }
.dash-slider-thumb { background: GOLD !important; border-color: GOLD !important; }
.toggle label { display: inline-flex; align-items: center; gap: 6px; margin-right: 14px; font-size: 14px;
                color: TEXT; cursor: pointer; }
.toggle input { accent-color: GOLD; }

/* ---- cards & BANs ---- */
.bans { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 14px; }
.ban { background: CARD; border: 1px solid BORDER; border-radius: 14px; padding: 14px 16px; position: relative;
       overflow: hidden; }
.ban:before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: GOLD; }
.ban-title { color: MUTED; font-size: 11px; text-transform: uppercase; letter-spacing: .12em; font-weight: 700; }
.ban-value { font-family: DISPLAY; font-size: 42px; margin: 4px 0 0; letter-spacing: .03em; line-height: 1.05; }
.ban-sub { color: MUTED; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-bottom: 14px; }
.row.two { grid-template-columns: 1fr 1fr; }
.span-2 { grid-column: span 2; }
.card { background: CARD; border: 1px solid BORDER; border-radius: 16px; padding: 16px 16px 10px; min-width: 0; }
.card-head { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
.card-head h3 { margin: 0; font-size: 18px; font-weight: 800; }
.card-head p { margin: 4px 0 10px; color: MUTED; font-size: 13px; }

/* ---- poster grid ---- */
.poster-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(138px, 1fr)); gap: 14px; }
.poster-card { cursor: pointer; position: relative; border-radius: 10px; transition: transform .15s; }
.poster-card:hover { transform: translateY(-4px) scale(1.02); }
.poster-frame { position: relative; aspect-ratio: 2 / 3; border-radius: 10px; overflow: hidden; background: #111;
                border: 1px solid BORDER; box-shadow: 0 8px 22px rgba(0,0,0,.6); }
.poster-frame img { width: 100%; height: 100%; object-fit: cover; display: block; }
.poster-card:hover .poster-frame { border-color: GOLD; box-shadow: 0 0 0 1px GOLD, 0 0 24px rgba(245,197,24,.25); }
.rate-badge { position: absolute; top: 6px; left: 6px; background: rgba(0,0,0,.8); color: GOLD; font-weight: 800;
              font-size: 12px; padding: 2px 7px; border-radius: 6px; border: 1px solid rgba(245,197,24,.4); }
.pick-flag { position: absolute; top: 6px; right: 6px; background: RED; color: #fff; font-size: 11px;
             padding: 2px 6px; border-radius: 6px; font-weight: 700; }
.plot-peek { position: absolute; inset: 0; background: linear-gradient(to top, rgba(0,0,0,.95) 30%, rgba(0,0,0,.55));
             color: TEXT; font-size: 12px; line-height: 1.4; padding: 12px; display: flex; flex-direction: column;
             justify-content: flex-end; opacity: 0; transition: opacity .18s; }
.plot-peek b { color: GOLD; font-size: 11px; letter-spacing: .1em; margin-bottom: 4px; }
.poster-card:hover .plot-peek { opacity: 1; }
.pc-title { font-weight: 700; font-size: 13px; margin-top: 7px; line-height: 1.25; }
.pc-meta { color: MUTED; font-size: 12px; margin-top: 2px; }
.more-row { text-align: center; padding: 16px 0 8px; }
.empty { color: MUTED; text-align: center; padding: 40px 10px; font-size: 14px; grid-column: 1 / -1; }

/* ---- slot machine ---- */
.slot { display: flex; flex-direction: column; align-items: center; gap: 14px; padding: 4px 0 12px; }
.slot-cabinet { width: 100%; max-width: 480px; position: relative; padding: 16px; border-radius: 18px;
  background: linear-gradient(180deg, #5c0d17, #2a050a); border: 6px dotted GOLD;
  box-shadow: 0 0 30px rgba(215,38,61,.35), inset 0 0 20px rgba(0,0,0,.6); }
.slot-window { height: 64px; overflow: hidden; background: #000; border-radius: 10px; position: relative;
  box-shadow: inset 0 12px 14px -6px rgba(0,0,0,.9), inset 0 -12px 14px -6px rgba(0,0,0,.9); border: 1px solid #000; }
.payline { position: absolute; left: 8px; right: 8px; top: 50%; height: 2px; background: rgba(215,38,61,.5); }
.reel-item { height: 64px; display: flex; align-items: center; justify-content: center; padding: 0 14px;
             font-family: DISPLAY; font-size: 30px; letter-spacing: .04em; color: GOLD; text-align: center;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.reel.spinA { animation: spinA 2.8s cubic-bezier(.15,.75,.2,1) forwards; }
.reel.spinB { animation: spinB 2.8s cubic-bezier(.15,.75,.2,1) forwards; }
@keyframes spinA { from { transform: translateY(0); } to { transform: translateY(-1856px); } }
@keyframes spinB { from { transform: translateY(0); } to { transform: translateY(-1856px); } }
.spin-btn { width: 96px; height: 96px; border-radius: 50%; border: 4px solid GOLD; background: radial-gradient(circle at 35% 30%, #ff5a6e, RED 55%, #7a0f1d);
  color: #fff; font-family: DISPLAY; font-size: 30px; letter-spacing: .08em; cursor: pointer;
  box-shadow: 0 6px 0 #5c0d17, 0 0 30px rgba(215,38,61,.45); transition: transform .08s; }
.spin-btn:active { transform: translateY(4px); box-shadow: 0 2px 0 #5c0d17; }
.roulette-result { display: flex; gap: 14px; align-items: center; background: CARD_2; border: 1px solid GOLD;
  border-radius: 12px; padding: 10px; width: 100%; max-width: 480px; }
.roulette-result img { width: 70px; aspect-ratio: 2/3; object-fit: cover; border-radius: 6px; }
.roulette-result .rr-title { font-weight: 800; font-size: 16px; }
.roulette-result .rr-meta { color: MUTED; font-size: 12px; margin: 2px 0 8px; }
.fadeA { animation: fadeA .5s ease 2.7s both; } .fadeB { animation: fadeB .5s ease 2.7s both; }
@keyframes fadeA { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes fadeB { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

/* ---- picks ---- */
.picks-summary { font-family: DISPLAY; font-size: 28px; letter-spacing: .04em; color: GOLD; margin-bottom: 10px; }
.picks-summary span { font-family: Inter; font-size: 13px; color: MUTED; letter-spacing: 0; margin-left: 6px; }
.picks-list { display: flex; flex-direction: column; gap: 8px; max-height: 360px; overflow-y: auto; margin-bottom: 8px; }
.pick-row { display: flex; align-items: center; gap: 10px; background: CARD_2; border: 1px solid BORDER;
            border-radius: 10px; padding: 6px 8px; }
.pick-open { display: flex; gap: 10px; align-items: center; flex: 1; cursor: pointer; min-width: 0; }
.pick-open:hover .pr-title { color: GOLD; }
.pick-row img { width: 36px; aspect-ratio: 2/3; object-fit: cover; border-radius: 4px; }
.pr-title { font-weight: 700; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pr-meta { color: MUTED; font-size: 12px; }
.x-btn { background: transparent; border: 1px solid BORDER; color: MUTED; border-radius: 8px; cursor: pointer;
         width: 30px; height: 30px; font-size: 14px; }
.x-btn:hover { color: RED; border-color: RED; }

/* ---- tooltip ---- */
.tip { display: flex; gap: 12px; width: 330px; white-space: normal; font-family: FONT; color: TEXT; }
.tip img { width: 96px; aspect-ratio: 2/3; object-fit: cover; border-radius: 6px; flex-shrink: 0; background: #111; }
.tip-title { font-weight: 800; font-size: 15px; line-height: 1.2; }
.tip-meta { color: MUTED; font-size: 12px; margin: 3px 0 6px; }
.tip-stats { font-size: 13px; margin-bottom: 6px; } .tip-stats b { color: GOLD; }
.tip-plot { font-size: 12px; line-height: 1.4; color: #cfcac0; margin: 0; }

/* ---- compare ---- */
.compare-pickers { display: grid; grid-template-columns: 1fr auto 1fr; gap: 14px; align-items: center; margin-bottom: 16px; }
.vs { font-family: DISPLAY; font-size: 34px; color: RED; text-shadow: 0 0 12px rgba(215,38,61,.5); }
.cmp-grid { display: grid; grid-template-columns: 1fr 150px 1fr; gap: 0; }
.cmp-head { display: flex; flex-direction: column; align-items: center; text-align: center; gap: 8px; padding-bottom: 14px; cursor: pointer; }
.cmp-head img { width: 150px; aspect-ratio: 2/3; object-fit: cover; border-radius: 10px; border: 1px solid BORDER;
                box-shadow: 0 10px 30px rgba(0,0,0,.7); }
.cmp-head:hover img { border-color: GOLD; }
.cmp-title { font-family: DISPLAY; font-size: 28px; letter-spacing: .04em; line-height: 1; }
.cmp-cell { padding: 10px 12px; border-top: 1px solid BORDER; text-align: center; font-size: 15px; }
.cmp-label { padding: 10px 6px; border-top: 1px solid BORDER; text-align: center; color: MUTED; font-size: 11px;
             text-transform: uppercase; letter-spacing: .1em; font-weight: 700; display: flex; align-items: center;
             justify-content: center; }
.cmp-cell.win { color: GOLD; font-weight: 800; text-shadow: 0 0 10px rgba(245,197,24,.35); }
.cmp-cell.win:after { content: " ★"; font-size: 12px; }
.verdict { margin-top: 12px; background: CARD_2; border-left: 3px solid GOLD; padding: 10px 14px; border-radius: 8px;
           font-size: 14px; color: #d8d3c8; }

/* ---- insights ---- */
.insight-bar { display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap;
               background: CARD; border: 1px solid BORDER; border-radius: 14px; padding: 12px 18px; margin-bottom: 14px; }
.callout { font-size: 15px; } .callout b { color: GOLD; font-size: 20px; font-family: DISPLAY; letter-spacing: .05em; }
.metric-toggle { display: flex; align-items: center; gap: 12px; }
.metric-toggle > label { font-size: 11px; color: MUTED; text-transform: uppercase; letter-spacing: .12em; font-weight: 700; }

/* ---- drawer (movie detail) ---- */
.backdrop { position: fixed; inset: 0; background: rgba(0,0,0,.72); backdrop-filter: blur(3px); z-index: 50;
            opacity: 0; pointer-events: none; transition: opacity .2s; }
.backdrop.open { opacity: 1; pointer-events: auto; }
.drawer { position: fixed; top: 0; right: 0; height: 100vh; width: min(600px, 100%); background: #060608;
          border-left: 1px solid #2a2a33; z-index: 60; overflow-y: auto; transform: translateX(105%);
          transition: transform .28s cubic-bezier(.2,.8,.2,1); padding: 0 0 30px; box-shadow: -20px 0 60px rgba(0,0,0,.8); }
.drawer.open { transform: translateX(0); }
.drawer-close { position: absolute; top: 14px; right: 14px; z-index: 3; width: 36px; height: 36px; border-radius: 50%;
                border: 1px solid #3a3a44; background: rgba(0,0,0,.6); color: TEXT; font-size: 16px; cursor: pointer; }
.drawer-close:hover { border-color: GOLD; color: GOLD; }
.hero { position: relative; padding: 26px 24px 16px; overflow: hidden; }
.hero-bg { position: absolute; inset: -40px; background-size: cover; background-position: center; filter: blur(28px) brightness(.35);
           transform: scale(1.1); }
.hero-inner { position: relative; display: grid; grid-template-columns: 170px 1fr; gap: 20px; align-items: end; }
.hero-poster { width: 170px; aspect-ratio: 2/3; object-fit: cover; border-radius: 10px; box-shadow: 0 14px 40px rgba(0,0,0,.9);
               border: 1px solid rgba(255,255,255,.1); }
.chip { display: inline-block; border: 1px solid GOLD; color: GOLD; border-radius: 999px; padding: 2px 10px;
        font-size: 11px; margin: 0 6px 6px 0; font-weight: 600; }
.mc-title { font-family: DISPLAY; font-size: 44px; letter-spacing: .03em; line-height: .95; margin: 4px 0 6px; }
.mc-meta { color: #bdb8ae; font-size: 13px; }
.mc-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; padding: 0 24px; margin: 6px 0 14px; }
.mc-stat { background: CARD_2; border: 1px solid BORDER; border-radius: 10px; padding: 8px 10px; }
.mc-stat b { display: block; font-size: 18px; color: TEXT; }
.mc-stat span { color: MUTED; font-size: 11px; }
.drawer-actions { display: flex; gap: 8px; padding: 0 24px 14px; flex-wrap: wrap; }
.trailer-wrap { margin: 0 24px 16px; aspect-ratio: 16 / 9; border-radius: 12px; overflow: hidden; border: 1px solid #2a2a33; background: #000; }
.trailer-wrap iframe { width: 100%; height: 100%; border: 0; }
.dsec { padding: 0 24px; margin-bottom: 18px; }
.dsec h4 { font-family: DISPLAY; font-weight: 400; font-size: 22px; letter-spacing: .06em; color: GOLD; margin: 0 0 8px; }
.mc-desc { color: #d8d3c8; font-size: 15px; line-height: 1.55; margin: 0 0 8px; }
.watch-links { display: flex; flex-wrap: wrap; gap: 8px; }
.watch-link { display: inline-flex; align-items: center; gap: 8px; background: CARD_2; border: 1px solid BORDER; color: TEXT;
              text-decoration: none; border-radius: 999px; padding: 6px 12px 6px 6px; font-size: 13px; font-weight: 600; }
.watch-link:hover { border-color: GOLD; color: GOLD; }
.watch-link img { width: 26px; height: 26px; border-radius: 6px; }
.watch-link .dot { width: 26px; height: 26px; border-radius: 6px; display: inline-flex; align-items: center;
                   justify-content: center; font-weight: 800; font-size: 13px; color: #fff; }
.watch-kind { color: MUTED; font-size: 11px; text-transform: uppercase; letter-spacing: .1em; margin: 8px 0 6px; font-weight: 700; }
.note { color: MUTED; font-size: 12px; margin-top: 8px; }
.similar-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.sim-match { color: GOLD; font-size: 11px; font-weight: 700; }

footer { color: MUTED; font-size: 12px; margin-top: 22px; text-align: center; line-height: 1.6; }

@media (max-width: 1100px) {
  .layout { grid-template-columns: 1fr; }
  .sidebar { position: static; max-height: none; }
  .bans { grid-template-columns: repeat(2, 1fr); }
  .row, .row.two { grid-template-columns: 1fr; }
  .span-2 { grid-column: span 1; }
  h1 { font-size: 52px; }
}
@media (max-width: 600px) {
  .hero-inner { grid-template-columns: 110px 1fr; } .hero-poster { width: 110px; }
  .mc-stats { grid-template-columns: repeat(2, 1fr); }
  .cmp-grid { grid-template-columns: 1fr 90px 1fr; } .cmp-head img { width: 100px; }
}
"""
for _k, _v in {"CARD_2": CARD_2, "CARD": CARD, "BORDER": BORDER, "TEXT": TEXT, "MUTED": MUTED, "GOLD": GOLD,
               "RED": RED, "DISPLAY": DISPLAY, "FONT": FONT, "BG": BG}.items():
    CSS = re.sub(rf"\b{_k}\b", _v, CSS)

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>""" + CSS + """</style>
</head>
<body>{%app_entry%}{%config%}{%scripts%}{%renderer%}</body>
</html>"""


# ---------------------------------------------------------------------------
# 6. SMALL VIEW BUILDERS
# ---------------------------------------------------------------------------
def poster_card(m, src, picks=(), extra=None):
    rank = int(m["Rank"])
    return html.Div(id={"type": "open-movie", "src": src, "rank": rank}, n_clicks=0, className="poster-card",
                    title=m["Title"], children=[
        html.Div(className="poster-frame", children=[
            html.Img(src=f"/poster/{rank}", alt=m["Title"]),
            html.Span(f"★ {m['Rating']:.1f}", className="rate-badge"),
            html.Span("PICK", className="pick-flag") if rank in picks else None,
            html.Div(className="plot-peek", children=[html.B("PLOT"), m["Description"]]),
        ]),
        html.Div(m["Title"], className="pc-title"),
        html.Div(extra or f"{m['Year']} · {hm(m['Runtime'])}", className="pc-meta"),
    ])


def tooltip_body(rank):
    m = df.loc[rank]
    desc = m["Description"] if len(m["Description"]) < 190 else m["Description"][:187].rsplit(" ", 1)[0] + "…"
    return html.Div(className="tip", children=[
        html.Img(src=f"/poster/{rank}"),
        html.Div([
            html.Div(m["Title"], className="tip-title"),
            html.Div(f"{m['Year']} · {hm(m['Runtime'])} · {m['Genre'].replace(',', ', ')}", className="tip-meta"),
            html.Div([html.B(f"★ {m['Rating']}"), " IMDb  ·  ", html.B(f"{m['Votes']:,}"), " votes"],
                     className="tip-stats"),
            html.P(desc, className="tip-plot"),
        ]),
    ])


def watch_section(rank):
    m = df.loc[rank]
    t = quote_plus(m["Title"])
    imdb = _media.get(str(rank), {}).get("imdb")
    p = providers(rank) if TMDB_KEY else None
    children = [html.H4("Where to watch")]
    if p and any(p.get(k) for k in ("flatrate", "free", "ads", "rent", "buy")):
        for key, label in (("flatrate", "Stream"), ("free", "Free"), ("ads", "Free with ads"),
                           ("rent", "Rent"), ("buy", "Buy")):
            if p.get(key):
                children.append(html.Div(label, className="watch-kind"))
                children.append(html.Div(className="watch-links", children=[
                    html.A([html.Img(src="https://image.tmdb.org/t/p/w92" + s["logo_path"]), s["provider_name"]],
                           href=p.get("link"), target="_blank", rel="noopener", className="watch-link")
                    for s in p[key][:8]]))
        children.append(html.Div(f"Availability in {WATCH_REGION} · data by JustWatch via TMDB.", className="note"))
        return html.Div(children, className="dsec")

    services = [
        ("JustWatch (shows every service)", "JW", "#f5c518", f"https://www.justwatch.com/us/search?q={t}"),
        ("Netflix", "N", "#e50914", f"https://www.netflix.com/search?q={t}"),
        ("Prime Video", "P", "#1399ff", f"https://www.amazon.com/s?k={t}&i=instant-video"),
        ("Apple TV", "", "#555", f"https://tv.apple.com/search?term={t}"),
        ("YouTube Movies", "▶", "#ff0000", f"https://www.youtube.com/results?search_query={t}+{m['Year']}+full+movie"),
        ("IMDb", "i", "#f5c518", f"https://www.imdb.com/title/{imdb}/" if imdb else f"https://www.imdb.com/find/?q={t}"),
    ]
    if TMDB_KEY:
        children.append(html.Div(f"No streaming listings found for {WATCH_REGION} — try a search:", className="note"))
    children.append(html.Div(className="watch-links", children=[
        html.A([html.Span(icon, className="dot", style={"background": color,
                                                         "color": "#000" if color == "#f5c518" else "#fff"}), name],
               href=url, target="_blank", rel="noopener", className="watch-link")
        for name, icon, color, url in services]))
    if not TMDB_KEY:
        children.append(html.Div("Opens a search on each service. JustWatch lists exactly where it's streaming, "
                                 "renting or for sale.", className="note"))
    return html.Div(children, className="dsec")


# ---------------------------------------------------------------------------
# 7. CALLBACKS (interactivity)
# ---------------------------------------------------------------------------

# 7a. Readouts next to each slider
@app.callback(
    Output("year-readout", "children"), Output("rating-readout", "children"),
    Output("runtime-readout", "children"), Output("votes-readout", "children"),
    Input("year-slider", "value"), Input("rating-slider", "value"),
    Input("runtime-slider", "value"), Input("votes-slider", "value"),
)
def readouts(years, rating, runtime, votes):
    run_hi = "no limit" if runtime[1] >= RUN_MAX else hm(runtime[1])
    run_txt = f"under {run_hi}" if runtime[0] <= RUN_MIN and runtime[1] < RUN_MAX else \
        f"{hm(runtime[0])} – {run_hi}" if runtime[0] > RUN_MIN or runtime[1] < RUN_MAX else "any length"
    lo = "any" if votes[0] <= VOTE_MIN else short_num(10 ** votes[0])
    hi = "any" if votes[1] >= VOTE_MAX else short_num(10 ** votes[1])
    vote_txt = "any amount" if lo == hi == "any" else f"{lo} – {hi}" if "any" not in (lo, hi) else \
        f"at least {lo}" if hi == "any" else f"at most {hi}"
    return f"{years[0]}–{years[1]}", f"{rating[0]:.1f} – {rating[1]:.1f}", run_txt, vote_txt


# 7b. Reset button puts every filter back to its default
@app.callback(
    [Output(i, "value") for i in DEFAULTS] + [Output("more-btn", "n_clicks")],
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(_):
    return list(DEFAULTS.values()) + [0]


# 7c. Clicking a genre in the treemap filters the whole dashboard to it
@app.callback(
    Output("genre-dd", "value", allow_duplicate=True),
    Output("treemap", "clickData"),
    Input("treemap", "clickData"),
    prevent_initial_call=True,
)
def treemap_to_genre(click):
    if not click:
        return no_update, no_update
    label = click["points"][0].get("label")
    return ([] if label == "All genres" else [label]), None


# 7d. Discover: KPIs + poster grid
@app.callback(
    Output("ban-movies", "children"), Output("ban-movies-sub", "children"),
    Output("ban-rating", "children"), Output("ban-rating-sub", "children"),
    Output("ban-run", "children"), Output("ban-run-sub", "children"),
    Output("ban-rev", "children"), Output("ban-rev-sub", "children"),
    Output("ban-meta", "children"), Output("ban-meta-sub", "children"),
    Output("poster-grid", "children"), Output("more-btn", "style"), Output("match-count", "children"),
    *FILTER_INPUTS, Input("sort-dd", "value"), Input("more-btn", "n_clicks"), Input("picks-store", "data"),
)
def update_discover(*args):
    *filters, sort_key, more, picks = args
    d = sort_movies(filter_movies(*filters), sort_key)
    picks = set(picks or [])
    n = len(d)
    count_txt = f"{n:,} of 1,000 movies match"
    if d.empty:
        return ("0", "of 1,000", "–", "", "–", "", "–", "", "–", "",
                html.Div("🎬 No movies match these filters. Try widening them or hit Reset.", className="empty"),
                {"display": "none"}, count_txt)
    limit = 24 * (1 + (more or 0))
    col = SORTS.get(sort_key or "popular", SORTS["popular"])[1]

    def extra(m):
        if col == "Revenue":
            return f"{m['Year']} · {money(m['Revenue'])}"
        if col == "Metascore":
            return f"{m['Year']} · Metascore {'n/a' if pd.isna(m['Metascore']) else int(m['Metascore'])}"
        if col == "Votes":
            return f"{m['Year']} · {short_num(m['Votes'])} votes"
        return None

    grid = [poster_card(m, "grid", picks, extra(m)) for _, m in d.head(limit).iterrows()]
    top = d.loc[d["Revenue"].idxmax()] if d["Revenue"].notna().any() else None
    return (
        f"{n:,}", "of 1,000 films",
        f"{d['Rating'].mean():.1f}", "out of 10",
        hm(round(d["Runtime"].mean())), f"{hm(d['Runtime'].sum())} back-to-back" if n < 100 else
        f"{d['Runtime'].sum() / 60 / 24:,.0f} days back-to-back",
        money(d["Revenue"].sum()), f"#1: {top['Title']}" if top is not None else "",
        "–" if d["Metascore"].isna().all() else f"{d['Metascore'].mean():.0f}", "Metascore out of 100",
        grid, {"display": "inline-flex" if n > limit else "none"}, count_txt,
    )


# 7e. Popularity vs rating scatter
@app.callback(Output("pop-scatter", "figure"), *FILTER_INPUTS, Input("picks-store", "data"))
def update_popularity(*args):
    *filters, picks = args
    d = filter_movies(*filters)
    if d.empty:
        return empty_fig(height=480)
    picks = set(picks or [])
    is_pick = d["Rank"].isin(picks)
    fig = go.Figure(go.Scattergl(
        x=d["Votes"], y=d["Rating"], mode="markers", customdata=d[["Rank"]].values.tolist(),
        hoverinfo="none", hovertemplate=None,
        marker=dict(size=np.where(is_pick, 13, 9), color=d["Metascore"].fillna(d["Metascore"].mean()),
                    colorscale=GOLD_SCALE, opacity=0.85, showscale=True,
                    colorbar=dict(title="Metascore", thickness=10),
                    line=dict(color=np.where(is_pick, RED, BG), width=np.where(is_pick, 3, 0.5))),
    ))
    mx, my = d["Votes"].median(), d["Rating"].median()
    fig.add_vline(x=mx, line=dict(color=MUTED, dash="dot", width=1))
    fig.add_hline(y=my, line=dict(color=MUTED, dash="dot", width=1))
    for x, y, txt, xa, ya in [(0.99, 0.99, "Crowd favorites", "right", "top"),
                              (0.01, 0.99, "Hidden gems", "left", "top"),
                              (0.99, 0.01, "Overhyped", "right", "bottom"),
                              (0.01, 0.01, "Forgettable", "left", "bottom")]:
        fig.add_annotation(xref="paper", yref="paper", x=x, y=y, text=txt, showarrow=False,
                           xanchor=xa, yanchor=ya, font=dict(color=RED, size=13, family=DISPLAY))
    style_fig(fig, height=480)
    fig.update_layout(xaxis_title="Popularity: number of IMDb votes (log scale)", yaxis_title="IMDb user rating",
                      showlegend=False)
    fig.update_xaxes(type="log", tickvals=[100, 1_000, 10_000, 100_000, 1_000_000],
                     ticktext=["100", "1K", "10K", "100K", "1M"], minor=dict(showgrid=False))
    return fig


# 7f. Hover tooltips with posters (both scatter plots)
def _tooltip(hover):
    if not hover or not hover.get("points") or "customdata" not in hover["points"][0]:
      return False, no_update, no_update
    pt = hover["points"][0]
    return True, pt["bbox"], tooltip_body(int(pt["customdata"][0]))


@app.callback(Output("pop-tip", "show"), Output("pop-tip", "bbox"), Output("pop-tip", "children"),
              Input("pop-scatter", "hoverData"))
def pop_tooltip(hover):
    return _tooltip(hover)


@app.callback(Output("scatter-tip", "show"), Output("scatter-tip", "bbox"), Output("scatter-tip", "children"),
              Input("scatter", "hoverData"))
def scatter_tooltip(hover):
    return _tooltip(hover)


# 7g. Box office insights (the original analysis)
@app.callback(
    Output("scatter", "figure"), Output("top-bar", "figure"), Output("treemap", "figure"),
    Output("heatmap", "figure"), Output("insight-callout", "children"),
    *FILTER_INPUTS, Input("metric", "value"),
)
def update_insights(*args):
    *filters, metric = args
    d = filter_movies(*filters)
    genres = filters[1]
    if d.empty:
        return empty_fig(), empty_fig(), empty_fig(), empty_fig(), "No movies match these filters."
    m_label, m_short = METRICS[metric]["label"], METRICS[metric]["short"]
    with_rev = d.dropna(subset=["Revenue", metric])

    corr = with_rev[metric].corr(with_rev["Revenue"]) if len(with_rev) > 2 else np.nan
    strength = ("no clear" if np.isnan(corr) or abs(corr) < 0.1 else
                "a weak" if abs(corr) < 0.3 else "a moderate" if abs(corr) < 0.5 else "a strong")
    callout = ["Buzz ↔ money link:  ", html.B("–" if np.isnan(corr) else f"r = {corr:.2f}"),
               f"  — {strength} link between {m_short} and box office across {len(with_rev):,} movies."]

    # --- Scatter with median lines = 4 quadrants ---
    scat = go.Figure()
    if not with_rev.empty:
        scat.add_trace(go.Scatter(
            x=with_rev[metric], y=with_rev["Revenue"], mode="markers",
            customdata=with_rev[["Rank"]].values.tolist(), hoverinfo="none", hovertemplate=None,
            marker=dict(size=with_rev["Votes"], sizemode="area", sizeref=2 * df["Votes"].max() / 45 ** 2,
                        sizemin=4, color=GOLD, opacity=0.55, line=dict(color=BG, width=1)),
        ))
        scat.add_vline(x=with_rev[metric].median(), line=dict(color=MUTED, dash="dot", width=1))
        scat.add_hline(y=with_rev["Revenue"].median(), line=dict(color=MUTED, dash="dot", width=1))
        for x, y, txt, xa, ya in [(1, 1, "Loved & lucrative", "right", "top"),
                                  (0, 1, "Panned but profitable", "left", "top"),
                                  (1, 0, "Hidden gems", "right", "bottom"),
                                  (0, 0, "Flops", "left", "bottom")]:
            scat.add_annotation(xref="paper", yref="paper", x=x, y=y, text=txt, showarrow=False,
                                xanchor=xa, yanchor=ya, font=dict(color=RED, size=13, family=DISPLAY))
    style_fig(scat, height=440)
    scat.update_layout(xaxis_title=m_label, yaxis_title="Box office revenue (US$ M)", showlegend=False)

    # --- Top 10 bar ---
    t10 = d.dropna(subset=["Revenue"]).nlargest(10, "Revenue").iloc[::-1]
    if t10.empty:
        bar = empty_fig("No revenue data for these movies", 440)
    else:
        bar = go.Figure(go.Bar(
            x=t10["Revenue"], y=[t if len(t) <= 26 else t[:25] + "…" for t in t10["Title"]], orientation="h",
            customdata=np.stack([t10["Rank"], t10[metric].fillna(0)], axis=1).tolist(),
            marker=dict(color=t10[metric], colorscale=GOLD_SCALE, line=dict(width=0)),
            text=[money(v) for v in t10["Revenue"]], textposition="inside", insidetextanchor="end",
            textfont=dict(color=BG, family=FONT),
            hovertemplate="<b>%{y}</b><br>$%{x:,.0f}M<br>" + m_short + ": %{customdata[1]}<extra></extra>",
        ))
        style_fig(bar, height=440)
        bar.update_layout(xaxis_title="US$ millions", bargap=0.25)
        bar.update_yaxes(showgrid=False, automargin=True)

    # --- Treemap ---
    e = exploded(d, genres)
    g = (e.groupby("G").agg(Revenue=("Revenue", "sum"), Score=(metric, "mean"), Movies=("Title", "count"))
         .reset_index())
    g = g[g["Revenue"] > 0]
    if g.empty:
        tree = empty_fig("No revenue data for these movies", 420)
    else:
        g["RevText"] = g["Revenue"].apply(money)
        tree = px.treemap(g, path=[px.Constant("All genres"), "G"], values="Revenue", color="Score",
                          color_continuous_scale=GOLD_SCALE, custom_data=["Movies", "Score", "RevText"])
        tree.update_traces(
            hovertemplate="<b>%{label}</b><br>Box office: %{customdata[2]}<br>"
                          "Movies: %{customdata[0]}<br>Avg " + m_short + ": %{customdata[1]:.1f}<extra></extra>",
            texttemplate="<b>%{label}</b><br>%{customdata[2]}",
            marker=dict(line=dict(color=BG, width=2)), root_color=CARD,
        )
        tree.update_layout(coloraxis_colorbar=dict(title=m_short, thickness=10))
        style_fig(tree, height=420)
        tree.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    # --- Heatmap ---
    h = e.pivot_table(index="G", columns="Year", values=metric, aggfunc="mean")
    if h.empty:
        heat = empty_fig(height=420)
    else:
        h = h.loc[h.mean(axis=1).sort_values().index]
        heat = go.Figure(go.Heatmap(
            z=h.values, x=h.columns.astype(str), y=h.index, colorscale=GOLD_SCALE,
            hovertemplate="<b>%{y}</b>, %{x}<br>Avg " + m_short + ": %{z:.1f}<extra></extra>",
            colorbar=dict(title=m_short, thickness=10), xgap=3, ygap=3,
        ))
        style_fig(heat, height=max(420, 21 * len(h) + 60))
        heat.update_xaxes(showgrid=False, type="category")
        heat.update_yaxes(showgrid=False)
    return scat, bar, tree, heat, callout


# 7h. Anything that opens a movie -> "selected" (and the drawer)
@app.callback(
    Output("selected", "data"),
    Output("pop-scatter", "clickData"), Output("scatter", "clickData"), Output("top-bar", "clickData"),
    Input("pop-scatter", "clickData"), Input("scatter", "clickData"), Input("top-bar", "clickData"),
    Input({"type": "open-movie", "src": ALL, "rank": ALL}, "n_clicks"),
    Input("drawer-close", "n_clicks"), Input("drawer-backdrop", "n_clicks"), Input("compare-btn", "n_clicks"),
    prevent_initial_call=True,
)
def select_movie(c_pop, c_scat, c_bar, _opens, _close, _backdrop, _cmp):
    trig = ctx.triggered_id
    reset = (None, None, None)
    if trig in ("pop-scatter", "scatter", "top-bar"):
        click = {"pop-scatter": c_pop, "scatter": c_scat, "top-bar": c_bar}[trig]
        if not click:
            return (no_update,) * 4
        return (int(click["points"][0]["customdata"][0]), *reset)
    if isinstance(trig, dict):
        if not ctx.triggered[0]["value"]:      # a button just appeared on screen; not a real click
            return (no_update,) * 4
        return (trig["rank"], *reset)
    if trig in ("drawer-close", "drawer-backdrop", "compare-btn"):
        return (None, no_update, no_update, no_update)
    return (no_update,) * 4


# 7i. Fill the drawer for the selected movie
@app.callback(
    Output("drawer", "className"), Output("drawer-backdrop", "className"),
    Output("drawer-hero", "children"), Output("drawer-plot", "children"), Output("drawer-similar", "children"),
    Input("selected", "data"),
)
def render_drawer(rank):
    if rank is None:
        return "drawer", "backdrop", no_update, no_update, no_update
    m = df.loc[rank]
    meta = "n/a" if pd.isna(m["Metascore"]) else f"{m['Metascore']:.0f}"
    hero = [
        html.Div(className="hero", children=[
            html.Div(className="hero-bg", style={"backgroundImage": f"url(/poster/{rank})"}),
            html.Div(className="hero-inner", children=[
                html.Img(src=f"/poster/{rank}", className="hero-poster", alt=m["Title"]),
                html.Div([
                    html.Div([html.Span(g, className="chip") for g in m["GenreList"]]),
                    html.Div(m["Title"], className="mc-title"),
                    html.Div(f"{m['Year']} · {hm(m['Runtime'])} · Directed by {m['Director']}", className="mc-meta"),
                ]),
            ]),
        ]),
        html.Div(className="mc-stats", children=[
            html.Div([html.B(f"★ {m['Rating']}"), html.Span("IMDb rating")], className="mc-stat"),
            html.Div([html.B(meta), html.Span("Metascore")], className="mc-stat"),
            html.Div([html.B(short_num(m["Votes"])), html.Span("IMDb votes")], className="mc-stat"),
            html.Div([html.B(money(m["Revenue"])), html.Span("Box office")], className="mc-stat"),
        ]),
    ]
    plot = html.Div(className="dsec", children=[
        html.H4("The plot"),
        html.P(m["Description"], className="mc-desc"),
        html.Div(f"Starring {m['Actors']}", className="mc-meta"),
    ])
    sim = similar_movies(rank)
    similar = html.Div(className="dsec", children=[
        html.H4("Similar films"),
        html.Div(className="similar-grid", children=[
            poster_card(s, "similar", extra=html.Span(
                "Shares " + ", ".join(g for g in s["GenreList"] if g in m["GenreList"])
                + (" · same director" if s["Director"] == m["Director"] else ""), className="sim-match"))
            for _, s in sim.iterrows()]),
    ])
    return "drawer open", "backdrop open", hero, plot, similar


# 7j. Where to watch (separate so a slow lookup never delays the drawer)
@app.callback(Output("drawer-watch", "children"), Input("selected", "data"))
def render_watch(rank):
    if rank is None:
        return no_update
    return watch_section(rank)


# 7k. Trailer: plays inside the drawer; stops when you open another movie or close
@app.callback(
    Output("trailer-frame", "src"), Output("trailer-wrap", "style"), Output("trailer-btn", "children"),
    Input("trailer-btn", "n_clicks"), Input("selected", "data"),
    State("trailer-frame", "src"),
    prevent_initial_call=True,
)
def play_trailer(_n, rank, current):
    hidden = ("about:blank", {"display": "none"}, "▶  Watch trailer")
    if ctx.triggered_id == "selected" or rank is None:
        return hidden
    if current and current != "about:blank":      # second click hides it again
        return hidden
    return f"/trailer/{rank}", {"display": "block"}, "■  Hide trailer"


# 7l. My Picks (watchlist): add / remove / clear
@app.callback(
    Output("picks-store", "data"),
    Input("add-pick-btn", "n_clicks"), Input({"type": "remove-pick", "rank": ALL}, "n_clicks"),
    Input("clear-picks", "n_clicks"),
    State("selected", "data"), State("picks-store", "data"),
    prevent_initial_call=True,
)
def edit_picks(_add, _remove, _clear, rank, picks):
    picks = list(picks or [])
    trig = ctx.triggered_id
    if trig == "add-pick-btn" and rank is not None:
        return [p for p in picks if p != rank] if rank in picks else picks + [rank]
    if isinstance(trig, dict) and ctx.triggered[0]["value"]:
        return [p for p in picks if p != trig["rank"]]
    if trig == "clear-picks" and ctx.triggered[0]["value"]:
        return []
    return no_update


@app.callback(
    Output("picks-summary", "children"), Output("picks-list", "children"), Output("picks-badge", "children"),
    Input("picks-store", "data"),
)
def render_picks(picks):
    picks = [p for p in (picks or []) if p in df.index]
    mins = int(df.loc[picks, "Runtime"].sum()) if picks else 0
    summary = [f"{len(picks)} movie{'s' if len(picks) != 1 else ''} · {mins:,} mins",
               html.Span(f"of viewing in total{f'  ({hm(mins)})' if mins >= 60 else ''}")]
    if not picks:
        rows = html.Div("Open any movie and press “＋ Add to picks” to start your list.", className="note")
    else:
        rows = [html.Div(className="pick-row", children=[
            html.Div(id={"type": "open-movie", "src": "picks", "rank": int(p)}, n_clicks=0, className="pick-open",
                     children=[html.Img(src=f"/poster/{p}"),
                               html.Div([html.Div(df.loc[p, "Title"], className="pr-title"),
                                         html.Div(f"{df.loc[p, 'Year']} · {hm(df.loc[p, 'Runtime'])} · "
                                                  f"★ {df.loc[p, 'Rating']}", className="pr-meta")],
                                        style={"minWidth": 0})]),
            html.Button("✕", id={"type": "remove-pick", "rank": int(p)}, n_clicks=0, className="x-btn",
                        title="Remove from picks"),
        ]) for p in picks]
    return summary, rows, str(len(picks))


@app.callback(Output("add-pick-btn", "children"), Output("add-pick-btn", "className"),
              Input("picks-store", "data"), Input("selected", "data"))
def pick_button(picks, rank):
    if rank is not None and rank in (picks or []):
        return "✓ In my picks", "btn in-picks"
    return "＋ Add to picks", "btn gold"


# 7m. Movie night roulette
@app.callback(
    Output("reel", "children"), Output("reel", "className"), Output("roulette-result", "children"),
    Input("spin-btn", "n_clicks"), *FILTER_STATES,
    prevent_initial_call=True,
)
def spin(n, *filters):
    d = filter_movies(*filters)
    if d.empty:
        return [html.Div("NO MATCHES", className="reel-item")], "reel", \
            html.Div("No movies match your filters — loosen them and spin again.", className="note")
    pick = d.sample(1).iloc[0]
    titles = list(d["Title"].sample(29, replace=len(d) < 29)) if len(d) > 1 else [pick["Title"]] * 29
    reel = [html.Div(t, className="reel-item") for t in titles + [pick["Title"]]]
    phase = "A" if n % 2 else "B"
    rank = int(pick["Rank"])
    result = html.Div(className=f"roulette-result fade{phase}", children=[
        html.Img(src=f"/poster/{rank}"),
        html.Div([
            html.Div("🎉 Tonight's movie", className="watch-kind", style={"margin": 0}),
            html.Div(pick["Title"], className="rr-title"),
            html.Div(f"{pick['Year']} · {hm(pick['Runtime'])} · ★ {pick['Rating']} · {pick['Genre'].replace(',', ', ')}",
                     className="rr-meta"),
            html.Button("🎬  Open movie", id={"type": "open-movie", "src": "roulette", "rank": rank},
                        n_clicks=0, className="btn gold small"),
        ]),
    ])
    return reel, f"reel spin{phase}", result


# 7n. "Compare" button in the drawer sends the movie to the comparison
@app.callback(
    Output("compare-a", "value"), Output("compare-b", "value"),
    Input("compare-btn", "n_clicks"), State("selected", "data"), State("compare-a", "value"),
    prevent_initial_call=True,
)
def send_to_compare(_n, rank, current_a):
    if rank is None or rank == current_a:
        return no_update, no_update
    return rank, current_a


app.clientside_callback(
    """function(n) { if (n) { setTimeout(function () {
           var el = document.getElementById('compare'); if (el) el.scrollIntoView({behavior: 'smooth'}); }, 250); }
       return window.dash_clientside.no_update; }""",
    Output("compare-btn", "title"), Input("compare-btn", "n_clicks"), prevent_initial_call=True,
)


# 7o. Side-by-side comparison
@app.callback(Output("compare-body", "children"), Input("compare-a", "value"), Input("compare-b", "value"))
def render_compare(a, b):
    if a is None or b is None:
        return html.Div("Pick two movies above.", className="note")
    A, B = df.loc[a], df.loc[b]
    shared = set(A["GenreList"]) & set(B["GenreList"])

    def num_row(label, va, vb, fmt, higher_wins=True):
        wa = wb = False
        if not (pd.isna(va) or pd.isna(vb)) and va != vb:
            wa = (va > vb) == higher_wins
            wb = not wa
        return [html.Div(fmt(va), className="cmp-cell" + (" win" if wa else "")),
                html.Div(label, className="cmp-label"),
                html.Div(fmt(vb), className="cmp-cell" + (" win" if wb else ""))]

    def txt_row(label, va, vb):
        return [html.Div(va, className="cmp-cell"), html.Div(label, className="cmp-label"),
                html.Div(vb, className="cmp-cell")]

    def head(m):
        return html.Div(id={"type": "open-movie", "src": "compare", "rank": int(m["Rank"])}, n_clicks=0,
                        className="cmp-head", children=[
            html.Img(src=f"/poster/{int(m['Rank'])}"),
            html.Div(m["Title"], className="cmp-title"),
            html.Div([html.Span(g, className="chip", style={"opacity": 1 if g in shared else .45})
                      for g in m["GenreList"]]),
        ])

    fmt_meta = lambda v: "n/a" if pd.isna(v) else f"{v:.0f}"
    rows = (num_row("IMDb rating", A["Rating"], B["Rating"], lambda v: f"★ {v}")
            + num_row("Metascore", A["Metascore"], B["Metascore"], fmt_meta)
            + num_row("IMDb votes", A["Votes"], B["Votes"], lambda v: f"{v:,}")
            + num_row("Box office", A["Revenue"], B["Revenue"], money)
            + num_row("Runtime", A["Runtime"], B["Runtime"], hm, higher_wins=False)
            + txt_row("Year", str(A["Year"]), str(B["Year"]))
            + txt_row("Director", A["Director"], B["Director"]))

    def better(col, higher=True):
        va, vb = A[col], B[col]
        if pd.isna(va) or pd.isna(vb) or va == vb:
            return None
        return A["Title"] if (va > vb) == higher else B["Title"]

    bits = []
    if better("Rating"):
        bits.append(f"Audiences prefer {better('Rating')}")
    if better("Metascore"):
        bits.append(f"critics prefer {better('Metascore')}")
    if better("Revenue"):
        bits.append(f"{better('Revenue')} made more money")
    if better("Runtime", higher=False):
        bits.append(f"{better('Runtime', higher=False)} is the quicker watch")
    verdict = ("; ".join(bits) + ".") if bits else "Dead heat."
    if shared:
        verdict += f"  Both are {', '.join(sorted(shared))}."
    return html.Div([
        html.Div(className="cmp-grid", children=[head(A), html.Div(), head(B), *rows]),
        html.Div(["🎬  ", verdict[0].upper() + verdict[1:]], className="verdict"),
    ])


if __name__ == "__main__":
    app.run(debug=True, port=8051)

# Completed
# Edited