"""
Box Office vs. Buzz — Group 1 Python Dashboard
Question: Do the movies critics and audiences love actually make money?
Data: IMDB Movie Dataset (1,000 popular movies, 2006–2016), Kaggle.

Run locally:  python app.py   ->  open http://127.0.0.1:8050
Deploy:       Render start command  gunicorn app:server
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "imdb_movie_dataset.csv"
df = pd.read_csv(DATA_FILE)
df = df.rename(columns={
    "Runtime (Minutes)": "Runtime",
    "Revenue (Millions)": "Revenue",
})
df["GenreList"] = df["Genre"].str.split(",")
ALL_GENRES = sorted(df["GenreList"].explode().unique())
YEAR_MIN, YEAR_MAX = int(df["Year"].min()), int(df["Year"].max())

# The two "buzz" measures the user can switch between
METRICS = {
    "Rating": {"label": "Audience score (IMDB rating, 0–10)", "short": "IMDB rating"},
    "Metascore": {"label": "Critic score (Metascore, 0–100)", "short": "Metascore"},
}

# ---------------------------------------------------------------------------
# 2. DESIGN SYSTEM  (change colors here and the whole dashboard follows)
# ---------------------------------------------------------------------------
BG = "#0e1016"        # page background
CARD = "#171a23"      # card background
BORDER = "#262a36"
TEXT = "#ece9e4"
MUTED = "#8d91a3"
GOLD = "#f2b134"      # main accent
TEAL = "#3fb8af"      # secondary accent
GOLD_SCALE = [[0, "#3a2f1a"], [0.5, "#b9842a"], [1, "#ffd27a"]]
FONT = "Inter, 'Segoe UI', Arial, sans-serif"


def style_fig(fig, height=380):
    """Apply one consistent look to every chart."""
    fig.update_layout(
        height=height,
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(family=FONT, color=TEXT, size=13),
        margin=dict(l=20, r=20, t=10, b=20),
        hoverlabel=dict(bgcolor="#222634", bordercolor=GOLD, font=dict(family=FONT, color=TEXT)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    return fig


def empty_fig(message="No movies match these filters"):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=MUTED, size=15))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_fig(fig)


# ---------------------------------------------------------------------------
# 3. HELPERS
# ---------------------------------------------------------------------------
def filter_movies(genres, years, focus=None):
    """Shared filter used by every chart: genre dropdown + year slider (+ clicked genre)."""
    d = df[(df["Year"] >= years[0]) & (df["Year"] <= years[1])]
    if genres:
        d = d[d["GenreList"].apply(lambda gl: any(g in genres for g in gl))]
    if focus:
        d = d[d["GenreList"].apply(lambda gl: focus in gl)]
    return d


def exploded(d, genres):
    """One row per movie-genre pair (a movie can have up to 3 genres)."""
    e = d.explode("GenreList").rename(columns={"GenreList": "G"})
    if genres:
        e = e[e["G"].isin(genres)]
    return e


def money(x):
    return f"${x / 1000:,.1f}B" if x >= 1000 else f"${x:,.0f}M"


def ban(title, value_id, sub_id):
    return html.Div(className="ban", children=[
        html.Div(title, className="ban-title"),
        html.Div(id=value_id, className="ban-value"),
        html.Div(id=sub_id, className="ban-sub"),
    ])


def card(title, subtitle, body, extra_class=""):
    return html.Div(className=f"card {extra_class}", children=[
        html.Div(className="card-head", children=[
            html.H3(title), html.P(subtitle),
        ]),
        body,
    ])


# ---------------------------------------------------------------------------
# 4. APP + LAYOUT
# ---------------------------------------------------------------------------
app = Dash(
    __name__,
    title="Box Office vs. Buzz",
    external_stylesheets=["https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap"],
)
server = app.server  # Render needs this line

GRAPH_CONFIG = {"displayModeBar": False}

app.layout = html.Div(className="page", children=[
    dcc.Store(id="focus-genre"),

    # ---- Header ----
    html.Header(className="header", children=[
        html.Div(children=[
            html.Div("🎬  GROUP 1 · MIS 480C", className="eyebrow"),
            html.H1(["Box Office ", html.Span("vs.", className="accent"), " Buzz"]),
            html.P("Do the movies critics and audiences love actually make money?  "
                   "1,000 popular films, 2006–2016.", className="lede"),
        ]),
    ]),

    # ---- Controls (shared by all charts) ----
    html.Div(className="controls", children=[
        html.Div(className="control", children=[
            html.Label("Genres"),
            dcc.Dropdown(id="genre-dd", options=ALL_GENRES, value=[], multi=True,
                         placeholder="All genres", className="dd"),
        ]),
        html.Div(className="control wide", children=[
            html.Label("Release years"),
            dcc.RangeSlider(id="year-slider", min=YEAR_MIN, max=YEAR_MAX, step=1,
                            value=[YEAR_MIN, YEAR_MAX], allow_direct_input=False,
                            marks={y: str(y) for y in range(YEAR_MIN, YEAR_MAX + 1)}),
        ]),
        html.Div(className="control", children=[
            html.Label("Measure buzz by"),
            dcc.RadioItems(id="metric", className="toggle", inline=True,
                           options=[{"label": "Audiences", "value": "Rating"},
                                    {"label": "Critics", "value": "Metascore"}],
                           value="Rating"),
        ]),
    ]),

    html.Div(id="focus-banner", className="focus-banner"),

    # ---- BANs ----
    html.Div(className="bans", children=[
        ban("Movies", "ban-movies", "ban-movies-sub"),
        ban("Total box office", "ban-rev", "ban-rev-sub"),
        ban("Avg audience score", "ban-rating", "ban-rating-sub"),
        ban("Avg critic score", "ban-meta", "ban-meta-sub"),
        ban("Buzz ↔ money link", "ban-corr", "ban-corr-sub"),
    ]),

    # ---- Row 1: scatter + movie detail ----
    html.Div(className="row", children=[
        card("Does buzz pay off?",
             "Each bubble is a movie. Size = number of IMDB votes. Click a bubble to see the movie.",
             dcc.Graph(id="scatter", config=GRAPH_CONFIG), "span-2"),
        card("Movie spotlight", "Click any bubble or bar.",
             html.Div(id="movie-card", className="movie-card")),
    ]),

    # ---- Row 2: treemap + top 10 bar ----
    html.Div(className="row two", children=[
        card("Where the money is, by genre",
             "Box size = total revenue, color = average score. Click a genre to focus the "
             "whole dashboard on it; click it again to reset.",
             dcc.Graph(id="treemap", config=GRAPH_CONFIG)),
        card("Top 10 box office hits",
             "Highest-grossing movies in the current selection, colored by score.",
             dcc.Graph(id="top-bar", config=GRAPH_CONFIG)),
    ]),

    # ---- Row 3: heatmap ----
    html.Div(className="row", children=[
        card("Is quality changing over time?",
             "Average score by genre and release year. Brighter gold = better reviewed; empty = no movies.",
             dcc.Graph(id="heatmap", config=GRAPH_CONFIG), "span-3"),
    ]),

    html.Footer("Data: IMDB Movie Dataset (Kaggle), 1,000 most popular movies 2006–2016. "
                "Revenue in US$ millions; 128 movies have no revenue reported. "
                "Movies can belong to up to 3 genres, so genre totals overlap."),
])

# Page-wide CSS lives here so the whole app stays in one file (easy to deploy).
app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: """ + BG + """; color: """ + TEXT + """; font-family: """ + FONT + """; }
  .page { max-width: 1400px; margin: 0 auto; padding: 28px 32px 40px; }
  .header { padding: 8px 0 22px; border-bottom: 1px solid """ + BORDER + """; margin-bottom: 20px; }
  .eyebrow { color: """ + MUTED + """; font-size: 12px; letter-spacing: .18em; font-weight: 600; }
  h1 { font-size: 44px; font-weight: 800; margin: 6px 0 4px; letter-spacing: -0.02em; }
  .accent { color: """ + GOLD + """; font-style: italic; }
  .lede { color: """ + MUTED + """; margin: 0; font-size: 16px; }
  .controls { display: grid; grid-template-columns: 1.2fr 2fr 1fr; gap: 18px; background: """ + CARD + """;
              border: 1px solid """ + BORDER + """; border-radius: 14px; padding: 16px 20px; margin-bottom: 14px; }
  .control label { display: block; font-size: 12px; color: """ + MUTED + """; text-transform: uppercase;
                   letter-spacing: .1em; margin-bottom: 8px; font-weight: 600; }
  .toggle label { display: inline-flex; align-items: center; gap: 6px; margin-right: 14px; text-transform: none;
                  letter-spacing: 0; font-size: 14px; color: """ + TEXT + """; cursor: pointer; }
  .toggle input { accent-color: """ + GOLD + """; }
  /* Dash 4 component theme (dropdown, slider, radio) */
  :root {
    --Dash-Fill-Interactive-Strong: """ + GOLD + """;
    --Dash-Fill-Interactive-Weak: #2a2414;
    --Dash-Fill-Inverse-Strong: #10131a;
    --Dash-Fill-Primary-Hover: #2a2414;
    --Dash-Fill-Primary-Active: #3a2f1a;
    --Dash-Text-Strong: """ + TEXT + """;
    --Dash-Text-Primary: """ + TEXT + """;
    --Dash-Text-Weak: """ + MUTED + """;
    --Dash-Stroke-Strong: """ + BORDER + """;
    --Dash-Stroke-Weak: """ + BORDER + """;
    --Dash-Fill-Disabled: """ + BORDER + """;
  }
  .dash-dropdown, .dash-dropdown-content { background: #10131a !important; color: """ + TEXT + """ !important;
                                           border-color: """ + BORDER + """ !important; }
  .dash-dropdown-placeholder { color: """ + MUTED + """ !important; }
  .dash-dropdown-search { background: transparent; color: """ + TEXT + """; }
  .dash-slider-mark { color: """ + MUTED + """ !important; font-size: 11px; }
  .dash-slider-track { background: """ + BORDER + """ !important; }
  .dash-slider-range { background: """ + GOLD + """ !important; }
  .dash-slider-thumb { background: """ + GOLD + """ !important; border-color: """ + GOLD + """ !important; }
  .focus-banner:empty { display: none; }
  .focus-banner { background: #2a2414; border: 1px solid """ + GOLD + """; color: """ + GOLD + """; border-radius: 10px;
                  padding: 10px 16px; margin-bottom: 14px; font-weight: 600; font-size: 14px; }
  .focus-banner button { margin-left: 12px; background: transparent; color: """ + GOLD + """; border: 1px solid """ + GOLD + """;
                         border-radius: 6px; padding: 3px 10px; cursor: pointer; font-family: inherit; }
  .bans { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 14px; }
  .ban { background: """ + CARD + """; border: 1px solid """ + BORDER + """; border-radius: 14px; padding: 16px 18px;
         border-top: 3px solid """ + GOLD + """; }
  .ban-title { color: """ + MUTED + """; font-size: 12px; text-transform: uppercase; letter-spacing: .1em; font-weight: 600; }
  .ban-value { font-size: 34px; font-weight: 800; margin: 6px 0 2px; letter-spacing: -0.02em; }
  .ban-sub { color: """ + MUTED + """; font-size: 13px; }
  .row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 14px; }
  .row.two { grid-template-columns: 1fr 1fr; }
  .span-2 { grid-column: span 2; }
  .span-3 { grid-column: span 3; }
  .card { background: """ + CARD + """; border: 1px solid """ + BORDER + """; border-radius: 14px; padding: 16px 16px 6px; min-width: 0; }
  .card-head h3 { margin: 0; font-size: 17px; font-weight: 700; }
  .card-head p { margin: 4px 0 6px; color: """ + MUTED + """; font-size: 13px; }
  .movie-card { padding: 8px 4px 16px; }
  .mc-title { font-size: 24px; font-weight: 800; margin: 6px 0 2px; }
  .mc-meta { color: """ + MUTED + """; font-size: 13px; margin-bottom: 10px; }
  .chip { display: inline-block; border: 1px solid """ + GOLD + """; color: """ + GOLD + """; border-radius: 999px;
          padding: 2px 10px; font-size: 12px; margin: 0 6px 6px 0; }
  .mc-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 12px 0; }
  .mc-stat { background: #10131a; border-radius: 10px; padding: 10px 12px; }
  .mc-stat b { display: block; font-size: 20px; color: """ + TEXT + """; }
  .mc-stat span { color: """ + MUTED + """; font-size: 12px; }
  .mc-desc { color: #c9c6c0; font-size: 14px; line-height: 1.5; }
  .mc-empty { color: """ + MUTED + """; font-size: 14px; padding: 40px 10px; text-align: center; line-height: 1.6; }
  footer { color: """ + MUTED + """; font-size: 12px; margin-top: 18px; text-align: center; }
  @media (max-width: 1000px) {
    .controls, .bans, .row { grid-template-columns: 1fr; }
    .row.two { grid-template-columns: 1fr; }
    .span-2, .span-3 { grid-column: span 1; }
  }
</style>
</head>
<body>{%app_entry%}{%config%}{%scripts%}{%renderer%}</body>
</html>"""


# ---------------------------------------------------------------------------
# 5. CALLBACKS (interactivity)
# ---------------------------------------------------------------------------

# 5a. Clicking a genre in the treemap sets the "focus genre" (click again / reset to clear).
@app.callback(
    Output("focus-genre", "data"),
    Input("treemap", "clickData"),
    Input({"type": "clear-focus", "id": ALL}, "n_clicks"),
    Input("genre-dd", "value"),
    Input("year-slider", "value"),
    State("focus-genre", "data"),
    prevent_initial_call=True,
)
def set_focus(click, _clear, _genres, _years, current):
    trigger = ctx.triggered_id
    if trigger == "treemap" and click:
        label = click["points"][0].get("label")
        if label == "All genres" or label == current:
            return None           # clicking the same genre (or the root) resets
        return label
    if isinstance(trigger, dict):  # the "clear" button (ignore it simply appearing)
        return None if ctx.triggered[0]["value"] else no_update
    return current if trigger in ("genre-dd", "year-slider") else no_update


# 5b. Shared filters (dropdown + slider + toggle) -> treemap and heatmap
@app.callback(
    Output("treemap", "figure"),
    Output("heatmap", "figure"),
    Input("genre-dd", "value"),
    Input("year-slider", "value"),
    Input("metric", "value"),
    Input("focus-genre", "data"),
)
def update_genre_views(genres, years, metric, focus):
    d = filter_movies(genres, years)
    if d.empty:
        return empty_fig(), empty_fig()
    e = exploded(d, genres)
    m = METRICS[metric]["short"]

    # --- Treemap (advanced chart) ---
    g = (e.groupby("G")
           .agg(Revenue=("Revenue", "sum"), Score=(metric, "mean"), Movies=("Title", "count"))
           .reset_index())
    g = g[g["Revenue"] > 0]
    g["RevText"] = g["Revenue"].apply(money)
    tree = px.treemap(
        g, path=[px.Constant("All genres"), "G"], values="Revenue", color="Score",
        color_continuous_scale=GOLD_SCALE, custom_data=["Movies", "Score", "RevText"],
    )
    tree.update_traces(
        hovertemplate="<b>%{label}</b><br>Box office: %{customdata[2]}<br>"
                      "Movies: %{customdata[0]}<br>Avg " + m + ": %{customdata[1]:.1f}<extra></extra>",
        texttemplate="<b>%{label}</b><br>%{customdata[2]}",
        marker=dict(line=dict(color=BG, width=2)),
        root_color=CARD,
    )
    if focus:  # outline the focused genre
        widths = [6 if lbl == focus else 2 for lbl in tree.data[0].labels]
        colors = [TEAL if lbl == focus else BG for lbl in tree.data[0].labels]
        tree.update_traces(marker=dict(line=dict(color=colors, width=widths)))
    tree.update_layout(coloraxis_colorbar=dict(title=m, thickness=10))
    style_fig(tree, height=400)
    tree.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    # --- Heatmap (advanced chart) ---
    h = e.pivot_table(index="G", columns="Year", values=metric, aggfunc="mean")
    h = h.loc[h.mean(axis=1).sort_values().index]  # best genres on top
    heat = go.Figure(go.Heatmap(
        z=h.values, x=h.columns.astype(str), y=h.index, colorscale=GOLD_SCALE,
        hovertemplate="<b>%{y}</b>, %{x}<br>Avg " + m + ": %{z:.1f}<extra></extra>",
        colorbar=dict(title=m, thickness=10), xgap=3, ygap=3,
    ))
    style_fig(heat, height=max(300, 21 * len(h) + 60))
    heat.update_xaxes(showgrid=False, type="category")
    heat.update_yaxes(showgrid=False)
    return tree, heat


# 5c. Shared filters + focus genre -> BANs, scatter, top-10 bar, focus banner
@app.callback(
    Output("ban-movies", "children"), Output("ban-movies-sub", "children"),
    Output("ban-rev", "children"), Output("ban-rev-sub", "children"),
    Output("ban-rating", "children"), Output("ban-rating-sub", "children"),
    Output("ban-meta", "children"), Output("ban-meta-sub", "children"),
    Output("ban-corr", "children"), Output("ban-corr-sub", "children"),
    Output("scatter", "figure"),
    Output("top-bar", "figure"),
    Output("focus-banner", "children"),
    Input("genre-dd", "value"),
    Input("year-slider", "value"),
    Input("metric", "value"),
    Input("focus-genre", "data"),
)
def update_movie_views(genres, years, metric, focus):
    d = filter_movies(genres, years, focus)
    banner = [] if not focus else [
        f"🎯 Focused on {focus} (from the treemap)",
        html.Button("✕ Clear", id={"type": "clear-focus", "id": "btn"}),
    ]
    if d.empty:
        blanks = ["–"] * 10
        return (*blanks, empty_fig(), empty_fig(), banner)

    m_label, m_short = METRICS[metric]["label"], METRICS[metric]["short"]
    with_rev = d.dropna(subset=["Revenue", metric])

    # --- BANs ---
    top = d.loc[d["Revenue"].idxmax()] if d["Revenue"].notna().any() else None
    corr = with_rev[metric].corr(with_rev["Revenue"]) if len(with_rev) > 2 else np.nan
    strength = ("no clear" if np.isnan(corr) or abs(corr) < 0.1 else
                "a weak" if abs(corr) < 0.3 else "a moderate" if abs(corr) < 0.5 else "a strong")
    bans = (
        f"{len(d):,}", f"{years[0]}–{years[1]}",
        money(d["Revenue"].sum()), f"#1: {top['Title']}" if top is not None else "",
        f"{d['Rating'].mean():.1f}", "IMDB rating out of 10",
        f"{d['Metascore'].mean():.0f}", "Metascore out of 100",
        "–" if np.isnan(corr) else f"r = {corr:.2f}", f"{strength} link ({m_short} vs. revenue)",
    )

    # --- Scatter (basic chart) with median lines = 4 quadrants ---
    scat = go.Figure()
    if not with_rev.empty:
        scat.add_trace(go.Scatter(
            x=with_rev[metric], y=with_rev["Revenue"], mode="markers",
            customdata=np.stack([with_rev["Rank"], with_rev["Title"], with_rev["Year"]], axis=1),
            marker=dict(
                size=with_rev["Votes"], sizemode="area", sizeref=2 * df["Votes"].max() / 45 ** 2,
                sizemin=4, color=GOLD, opacity=0.55, line=dict(color=BG, width=1),
            ),
            hovertemplate="<b>%{customdata[1]}</b> (%{customdata[2]})<br>" + m_short +
                          ": %{x}<br>Revenue: $%{y:,.0f}M<extra></extra>",
        ))
        mx, my = with_rev[metric].median(), with_rev["Revenue"].median()
        scat.add_vline(x=mx, line=dict(color=MUTED, dash="dot", width=1))
        scat.add_hline(y=my, line=dict(color=MUTED, dash="dot", width=1))
        for x, y, txt, xa, ya in [(1, 1, "Loved & lucrative", "right", "top"),
                                  (0, 1, "Panned but profitable", "left", "top"),
                                  (1, 0, "Hidden gems", "right", "bottom"),
                                  (0, 0, "Flops", "left", "bottom")]:
            scat.add_annotation(xref="paper", yref="paper", x=x, y=y, text=txt, showarrow=False,
                                xanchor=xa, yanchor=ya, font=dict(color=TEAL, size=12))
    style_fig(scat, height=440)
    scat.update_layout(xaxis_title=m_label, yaxis_title="Box office revenue (US$ M)", showlegend=False)

    # --- Top 10 bar (basic chart) ---
    t10 = d.dropna(subset=["Revenue"]).nlargest(10, "Revenue").iloc[::-1]
    bar = go.Figure(go.Bar(
        x=t10["Revenue"], y=t10["Title"], orientation="h",
        customdata=np.stack([t10["Rank"], t10[metric].fillna(0)], axis=1),
        marker=dict(color=t10[metric], colorscale=GOLD_SCALE, line=dict(width=0)),
        text=[money(v) for v in t10["Revenue"]], textposition="inside", insidetextanchor="end",
        textfont=dict(color=BG, family=FONT),
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f}M<br>" + m_short +
                      ": %{customdata[1]}<extra></extra>",
    ))
    style_fig(bar, height=400)
    bar.update_layout(xaxis_title="US$ millions", bargap=0.25)
    bar.update_yaxes(showgrid=False, automargin=True)
    return (*bans, scat, bar, banner)


# 5d. Click a bubble or a bar -> movie spotlight card
@app.callback(
    Output("movie-card", "children"),
    Input("scatter", "clickData"),
    Input("top-bar", "clickData"),
)
def show_movie(scatter_click, bar_click):
    click = bar_click if ctx.triggered_id == "top-bar" else scatter_click
    if not click:
        return html.Div(className="mc-empty", children=[
            html.Div("🍿", style={"fontSize": "40px"}),
            "Click any bubble in the scatter plot or any bar in the Top 10 chart "
            "to see that movie's details here.",
        ])
    rank = int(click["points"][0]["customdata"][0])
    mv = df[df["Rank"] == rank].iloc[0]
    rev = "n/a" if pd.isna(mv["Revenue"]) else money(mv["Revenue"])
    meta = "n/a" if pd.isna(mv["Metascore"]) else f"{mv['Metascore']:.0f}"
    return [
        html.Div([html.Span(g, className="chip") for g in mv["GenreList"]]),
        html.Div(mv["Title"], className="mc-title"),
        html.Div(f"{mv['Year']} · {mv['Runtime']} min · Directed by {mv['Director']}", className="mc-meta"),
        html.Div(className="mc-stats", children=[
            html.Div([html.B(f"⭐ {mv['Rating']}"), html.Span("IMDB rating")], className="mc-stat"),
            html.Div([html.B(meta), html.Span("Metascore")], className="mc-stat"),
            html.Div([html.B(rev), html.Span("Box office")], className="mc-stat"),
            html.Div([html.B(f"{mv['Votes']:,}"), html.Span("IMDB votes")], className="mc-stat"),
        ]),
        html.P(mv["Description"], className="mc-desc"),
        html.P(f"Starring {mv['Actors']}", className="mc-meta"),
    ]


if __name__ == "__main__":
    app.run(debug=True)

    #Can you see this
    # Change 2
    # Shaboingboing
    # change 4
