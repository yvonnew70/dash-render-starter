from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, ctx


# One row = one order-product line. Order ID is absent: do not count orders.
DATA_PATH = Path(__file__).parent / "superstore_orders.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["Order Date"])
required = ["Row ID", "Order Date", "Region", "Category", "Sales", "Profit"]
assert df[required].notna().all().all(), "Missing classroom data values"
assert df["Row ID"].is_unique, "Expected one unique Row ID per order line"
assert len(df) == 51290, "Use the supplied, unchanged classroom CSV"
df = df.assign(Year=df["Order Date"].dt.year)  # original CSV is never modified
REGIONS = sorted(df["Region"].unique())
YEARS = sorted(int(y) for y in df["Year"].unique())
CATEGORIES = sorted(df["Category"].unique())

def empty_figure(title):
    fig = go.Figure()
    fig.add_annotation(text="No matching rows. Choose a region/category.", showarrow=False)
    fig.update_layout(title=title, template="plotly_white")
    return fig


ASSETS_DIR = Path(__file__).parent / "assets"
app = Dash(__name__, assets_folder=str(ASSETS_DIR))
server = app.server  # deployment target for gunicorn app:server
app.title = "Superstore Explorer"


# TODO: choose a business question, title, colors, and layout for your audience.
app.layout = html.Main([
    html.H1("Superstore Explorer"),
    html.P("Question: How do category sales and monthly trends vary by region and year?"),
    html.Div([
        html.Div([
            html.Label("Regions (clear all = no rows)", htmlFor="regions"),
            dcc.Dropdown(id="regions", options=REGIONS, value=REGIONS,
                         multi=True, closeOnSelect=False),
        ], className="control"),
        html.Div([
            html.Label("Year", htmlFor="year"),
            dcc.Slider(id="year", min=min(YEARS), max=max(YEARS), step=1,
                       marks={y: str(y) for y in YEARS}, value=max(YEARS),
                       updatemode="mouseup"),
        ], className="control"),
    ], className="controls"),
    html.Button("Reset category selection", id="reset", n_clicks=0),
    html.P("Click a category bar to filter the monthly chart and summary. "
           "Changing a control also resets the category selection."),
    html.Div(id="summary", className="summary", role="status"),
    html.Div([
        dcc.Graph(id="category-bar"),
        dcc.Graph(id="monthly-line"),
    ], className="charts"),
    html.Footer("Global Superstore | 2011–2014 | one row = one order line"),
], className="page")


# A control supplies a value. pandas performs the actual filtering.
def filter_rows(regions, year):
    mask = df["Region"].isin(regions or []) & df["Year"].eq(year)
    return df.loc[mask].copy()

@app.callback(Output("category-bar", "figure"),
              Input("regions", "value"), Input("year", "value"))
def update_overview(regions, year):
    dff = filter_rows(regions, year)
    if dff.empty:
        return empty_figure("Sales by category")
    totals = dff.groupby("Category", as_index=False)["Sales"].sum()
    return px.bar(totals, x="Category", y="Sales", template="plotly_white",
                  title=f"Sales by category — {year}")

@app.callback(Output("monthly-line", "figure"), Output("summary", "children"),
              Input("regions", "value"), Input("year", "value"),
              Input("category-bar", "clickData"), Input("reset", "n_clicks"))
def update_detail(regions, year, click_data, reset_clicks):
    dff = filter_rows(regions, year)
    # Initial load, Reset, and any filter change show all categories.
    category = None
    if ctx.triggered_id == "category-bar" and click_data:
        category = click_data["points"][0]["x"]
        dff = dff.loc[dff["Category"].eq(category)]
    scope = category or "All categories"
    summary = (f"Detail: {scope} | Year: {year} | Sales: ${dff['Sales'].sum():,.0f} | "
               f"Profit: ${dff['Profit'].sum():,.0f} | Order lines: {len(dff):,}")
    if dff.empty:
        return empty_figure("Monthly sales"), summary
    monthly = dff.groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"].sum().reset_index()
    fig = px.line(monthly, x="Order Date", y="Sales", markers=True,
                  title=f"Monthly sales — {scope}", template="plotly_white")
    return fig, summary


if __name__ == "__main__":
    app.run(debug=False, port=8050)

