# -*- coding: utf-8 -*-
"""
Created on 2025-08-06

Dash app that maps DAC polygons and school points for NYC. DAC
polygons are styled by DAC_Desig and Pop_Cnt. School points show
only the Name in hover. Points can be symbolized by DAC County,
DAC designation, or any selected DAC numeric field. A DAC subset
filter lets you show only Designated / Not Designated schools.

Rules for polygons (unchanged):
- DAC_Desig == "Designated as DAC" -> purple, 50% opacity
- DAC_Desig == "Not Designated as DAC" -> white, 50% opacity
- Pop_Cnt == 0 -> fully transparent (excluded from draw)

@author: André
"""
# --------------------------- Imports & Timing ------------------------------ #
import time  # measure startup time
t0 = time.time()  # start timer

# Dash pieces
from dash import Dash, html, dcc  # app and UI elements
from dash import Input, Output, State  # callback primitives
from dash import Patch  # diff-only updates for speed

# Plotly for figures
import plotly.graph_objects as go  # low-level API
import plotly.express as px  # color palettes

# Geo stack
import geopandas as gpd  # read shapefiles
import pandas as pd  # tabular ops
import numpy as np  # arrays
import json  # to make proper GeoJSON
from pathlib import Path


# --------------------------- User Variables -------------------------------- #
# Paths to layers
BASE_DIR = Path(__file__).resolve().parent
SCHOOLS_PTS_PATH = BASE_DIR / "data" / "SchoolPoints_APS_2024_08_28.geojson"
DAC_POLY_PATH   = BASE_DIR / "data" / "DAC_NYC.geojson"

# Map style that needs no token
MAP_STYLE = "open-street-map"

# Default marker size
POINT_SIZE = 8

# Max categories to split for point coloring
MAX_CATS = 12

# NYC viewport override
NYC_CENTER = {"lon": -74.0060, "lat": 40.7128}
NYC_ZOOM = 10

# DAC numeric fields usable for point coloring
DAC_NUM_FIELDS = [
    'Pop_Cnt', 'HH_Cnt', 'Rank_State', 'Rank_NYC', 'Rank_ROS', 'Comb_Sc',
    'Burden_Pct', 'Vulner_Pct', 'Burden_Sc', 'Vulner_Sc', 'Benzene', 'PM25',
    'Traff_Trk', 'Traff_Veh', 'Waste_H2O', 'Vacancy', 'Ind_LU', 'Landfills',
    'Oil_Stor', 'Waste_Com', 'Pwr_Gen', 'RMP_Sites', 'Rem_Sites',
    'Scrap_Met', 'Ag_LU', 'Coast_Fld', 'Days_90_D', 'Drv_Health',
    'In_Flood', 'Low_Veg', 'Asian_Pct', 'Black_Pct', 'Redline', 'Lat_Pct',
    'Eng_Prof', 'Native_Pct', 'LMI_80_AMI', 'LMI_Fed', 'No_College',
    'HH_Single', 'Unemploymt', 'Asthma', 'COPD', 'HH_Disab', 'Birth_Wt',
    'MI_Rates', 'Health_Ins', 'Age_Ovr_65', 'Prem_Death', 'Internet',
    'Energy_Aff', 'Homes_1960', 'Mobile', 'Rent_Inc', 'Rent_Pct'
]

# --------------------------- Data Loading ---------------------------------- #
# Read polygon shapefile
gdf_poly = gpd.read_file(DAC_POLY_PATH)  # load polygons

# Read point shapefile
gdf_pts = gpd.read_file(SCHOOLS_PTS_PATH)  # load points

# Reproject to WGS84 if needed
if gdf_poly.crs is None or gdf_poly.crs.to_epsg() != 4326:
    gdf_poly = gdf_poly.to_crs(epsg=4326)  # reproject polys

if gdf_pts.crs is None or gdf_pts.crs.to_epsg() != 4326:
    gdf_pts = gdf_pts.to_crs(epsg=4326)  # reproject points

# Ensure points are true points (fallback to centroids)
if not all(gdf_pts.geometry.geom_type == "Point"):
    gdf_pts = gdf_pts.copy()  # avoid view warnings
    gdf_pts["geometry"] = gdf_pts.geometry.centroid  # centroids

# Add lon/lat columns for Scattermap
gdf_pts["lon"] = gdf_pts.geometry.x  # longitudes
gdf_pts["lat"] = gdf_pts.geometry.y  # latitudes

# Spatially join DAC attributes onto points
join_cols = ["DAC_Desig", "County"] + DAC_NUM_FIELDS  # fields
poly_attrs = gdf_poly[join_cols].copy()  # subset attrs
pts_join = gpd.sjoin(  # spatial join
    gdf_pts, gpd.GeoDataFrame(gdf_poly[join_cols], geometry=gdf_poly.geometry),
    how="left", predicate="within"
).drop(columns=["index_right"])  # drop sjoin index

# Keep a clean DataFrame for hover/categorization (exclude geometry)
pts_df = pd.DataFrame(pts_join.drop(columns="geometry"))  # copy
# Ensure id is a stable string for click selection -> customdata
pts_df = pts_df.reset_index(drop=True)
pts_df["id"] = pts_df["id"].astype(str)

# Aggregated hover text for coincident points (round to group near-identical coords)
PREC = 5  # ~1 meter at NYC lat
pts_df["_lonr"] = pts_df["lon"].round(PREC)
pts_df["_latr"] = pts_df["lat"].round(PREC)
agg = (
    pts_df.groupby(["_lonr", "_latr"])["Name"]
    .apply(lambda s: "<br>".join(sorted({str(x) for x in s if pd.notna(x)})))
    .rename("hover_names")
    .reset_index()
)
pts_df = pts_df.merge(agg, on=["_lonr", "_latr"], how="left")



# Ensure DAC numeric fields are numeric for coloring
for f in DAC_NUM_FIELDS:  # loop fields
    if f in pts_df.columns:  # exists
        pts_df[f] = pd.to_numeric(pts_df[f], errors="coerce")  # to num

# --------------------------- Helpers --------------------------------------- #
def candidate_color_fields(df: pd.DataFrame) -> list:
    """
    Pick fields to color points by. Prefer compact categories, else
    numerics. (Not used by the new UI; kept for reference.)
    """
    ignore = {"lon", "lat"}  # helper fields to skip
    cols = [c for c in df.columns if c not in ignore]  # fields
    cats = []  # categorical picks
    for c in cols:  # scan fields
        if df[c].dtype == "object" or str(df[c].dtype).startswith("cat"):
            nun = df[c].nunique(dropna=True)  # unique count
            if 1 < nun <= MAX_CATS:  # compact
                cats.append(c)  # add candidate
    nums = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]  # num
    return cats + nums  # cats first, then nums


def make_poly_traces(
    gdf: gpd.GeoDataFrame, visible: bool
) -> list[go.Choroplethmap]:
    """
    Build polygon traces for DAC styling rules. Pop_Cnt == 0 is
    excluded to make those fully transparent.
    """
    # Exclude Pop_Cnt == 0 (transparent by spec)
    g = gdf.copy()  # safe copy
    g["Pop_Cnt"] = pd.to_numeric(g["Pop_Cnt"], errors="coerce")  # nums
    g = g[g["Pop_Cnt"].fillna(0) > 0]  # keep > 0 only

    # Split by DAC_Desig
    dac = g[g["DAC_Desig"] == "Designated as DAC"]  # DAC polys
    nodac = g[g["DAC_Desig"] != "Designated as DAC"]  # not DAC

    # Map visible flag to Plotly expected value
    vis_flag = visible if visible else "legendonly"  # hide mode

    traces = []  # collect traces

    # DAC trace: purple with 50% opacity
    if len(dac) > 0:
        gj_dac = json.loads(dac.to_json())  # GeoJSON dict
        z_dac = np.ones(len(dac), dtype=float)  # uniform z
        tr_dac = go.Choroplethmap(
            geojson=gj_dac,  # geojson data
            locations=dac.index.astype(str),  # ids as strings
            featureidkey="id",  # link by "id"
            z=z_dac,  # scalar field
            colorscale=[[0.0, "black"], [1.0, "black"]],  # fill
            marker_opacity=0.5,  # 50% opacity
            marker_line_width=0.5,  # thin edges
            marker_line_color="white",  # white edges
            showscale=False,  # no colorbar
            visible=vis_flag,  # visibility
            name="DAC: Designated",  # legend name
            meta={"id": "POLY_DAC"},  # stable id
        )
        traces.append(tr_dac)  # add

    # Non-DAC trace: white with 50% opacity
    if len(nodac) > 0:
        gj_nd = json.loads(nodac.to_json())  # GeoJSON dict
        z_nd = np.ones(len(nodac), dtype=float)  # uniform z
        tr_nd = go.Choroplethmap(
            geojson=gj_nd,  # geojson data
            locations=nodac.index.astype(str),  # ids
            featureidkey="id",  # link by "id"
            z=z_nd,  # scalar field
            colorscale=[[0.0, "white"], [1.0, "white"]],  # fill
            marker_opacity=0.5,  # 50% opacity
            marker_line_width=0.5,  # thin edges
            marker_line_color="gray",  # subtle border
            showscale=False,  # no colorbar
            visible=vis_flag,  # visibility
            name="DAC: Not designated",  # legend name
            meta={"id": "POLY_NODAC"},  # stable id
        )
        traces.append(tr_nd)  # add

    return traces  # return both layers


def _hover_name(series_name: pd.Series) -> pd.Series:
    """
    Return hover text. If 'hover_names' was precomputed in pts_df,
    use it aligned to the incoming Series index; else fall back to Name.
    """
    if series_name is None:
        return pd.Series([], dtype=str)

    try:
        # align to the same index the trace subset uses
        if "hover_names" in pts_df.columns:
            out = pts_df.loc[series_name.index, "hover_names"].astype(str)
            # if any missing, fall back to the original names
            if out.isna().any() and series_name is not None:
                out = out.fillna(series_name.astype(str))
            return out
    except Exception:
        pass

    # Fallback: original behavior
    return series_name.fillna("").astype(str)


# --------------------------- Histogram Setup -------------------------------- #

def build_distribution_figure(points_mode: str, dac_subset: list[str]) -> go.Figure:
    """
    Build a small histogram for the right panel that summarizes the
    distribution of schools by DAC vs not DAC under the current mode.

    - ALL: x = DAC_Desig (Yes/No), simple two bars.
    - DAC_DESIG: x = County, color = DAC_Desig (grouped by county).
    - Numeric field: bin values into quartiles; x = quartile (Q1..Q4),
      color = DAC_Desig (Yes/No).
    """
    df = pts_df.copy()

    # Filter by the currently selected DAC subset so the chart matches the map
    if isinstance(dac_subset, (list, tuple, set)) and len(dac_subset) > 0:
        df = df[df["DAC_Desig"].isin(dac_subset)]

    # Normalize labels
    df["DAC_Desig"] = df["DAC_Desig"].fillna("Unknown")
    order_dac = [ND_FLAG, DAC_FLAG]
    if "Unknown" in df["DAC_Desig"].unique():
        order_dac.append("Unknown")
    color_map = {DAC_FLAG: "purple", ND_FLAG: "orange", "Unknown": "gray"}


    if points_mode == "ALL":
        fig = px.histogram(
            df, x="DAC_Desig", color="DAC_Desig",
            category_orders={"DAC_Desig": order_dac}, barmode="group",
            color_discrete_map=color_map,
        )

        fig.update_xaxes(title="")

    elif points_mode == "DAC_DESIG":
        df["County_"] = df["County"].fillna("Unknown").astype(str)
        county_order = sorted(df["County_"].unique().tolist())
        fig = px.histogram(
            df, x="County_", color="DAC_Desig",
            category_orders={"DAC_Desig": order_dac, "County_": county_order},
            barmode="group",
            color_discrete_map=color_map,
        )

        fig.update_xaxes(title="County")

    elif points_mode in DAC_NUM_FIELDS and points_mode in df.columns:
        vals = pd.to_numeric(df[points_mode], errors="coerce")
        qlabels = ["Q1", "Q2", "Q3", "Q4"]
        try:
            df["quartile"] = pd.qcut(vals, 4, labels=qlabels, duplicates="drop")
        except Exception:
            df["quartile"] = pd.Series([np.nan] * len(df))
        q_order = [q for q in qlabels if q in df["quartile"].dropna().unique().tolist()]
        fig = px.histogram(
            df.dropna(subset=["quartile"]),
            x="quartile", color="DAC_Desig",
            category_orders={"DAC_Desig": order_dac, "quartile": q_order},
            barmode="group",
            color_discrete_map=color_map,
        )

        fig.update_xaxes(title=f"{points_mode} (quartiles)")

    else:
        fig = px.histogram(
            df, x="DAC_Desig", color="DAC_Desig",
            category_orders={"DAC_Desig": order_dac}, barmode="group",
            color_discrete_map=color_map,
        )
        fig.update_xaxes(title="")


    fig.update_yaxes(title="Count")
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


# --------------------------- Viewport Setup -------------------------------- #
# Compute initial viewport (then override to NYC)
if len(gdf_poly) > 0:  # have polys
    center, zoom = NYC_CENTER, NYC_ZOOM  # force NYC
else:  # fallback to points
    center, zoom = NYC_CENTER, NYC_ZOOM  # force NYC

# --------------------------- Initial Figure -------------------------------- #
# Build initial figure and preload all traces we may show/hide.
fig = go.Figure()  # empty

# 1) Polygons (precomputed once; we only toggle visibility)
poly_traces = make_poly_traces(gdf_poly, visible=True)
for tr in poly_traces:
    fig.add_trace(tr)

# 2) Points: prepare masks and categories
DAC_FLAG = "Designated as DAC"
ND_FLAG = "Not Designated as DAC"
pts_all = pts_df
pts_dac = pts_df[pts_df["DAC_Desig"] == DAC_FLAG]
pts_nd = pts_df[pts_df["DAC_Desig"] != DAC_FLAG]
counties = sorted([c for c in pts_df["County"].dropna().unique()])

# Color palette per county (stable mapping)
pal = px.colors.qualitative.Plotly
county_color = {c: pal[i % len(pal)] for i, c in enumerate(counties)}

# Friendly helpers
def _county_label(c):
    return "Unknown" if pd.isna(c) else str(c)

# 2a) ALL mode: per-county, split by DAC subset for filtering
for c in counties:
    # ND subset of this county (legend hidden; avoid duplicates)
    sub = pts_nd[pts_nd["County"] == c]
    fig.add_trace(go.Scattermap(
        lon=sub["lon"], lat=sub["lat"], mode="markers",
        marker={"size": POINT_SIZE, "color": county_color[c]},
        name=f"Schools — County: {_county_label(c)}",
        hoverinfo="text",
        hovertext=_hover_name(sub.get("Name")), visible=True,
        legendgroup="ALL", showlegend=False,
        meta={"id": f"PT_ALL_ND:{c}"},
        customdata=sub[["id"]].to_numpy(),
    ))
    # DAC subset of this county (legend shown)
    sub = pts_dac[pts_dac["County"] == c]
    fig.add_trace(go.Scattermap(
        lon=sub["lon"], lat=sub["lat"], mode="markers",
        marker={"size": POINT_SIZE, "color": county_color[c]},
        name=f"Schools — County: {_county_label(c)}",
        hoverinfo="text",
        hovertext=_hover_name(sub.get("Name")), visible=True,
        legendgroup="ALL", showlegend=True,
        meta={"id": f"PT_ALL_DAC:{c}"},
        customdata=sub[["id"]].to_numpy(),
    ))
# Handle County NaN as a single class (gray)
if pts_nd["County"].isna().any():
    sub = pts_nd[pts_nd["County"].isna()]
    fig.add_trace(go.Scattermap(
        lon=sub["lon"], lat=sub["lat"], mode="markers",
        marker={"size": POINT_SIZE, "color": "gray"},
        name="Schools — County: Unknown", hoverinfo="text",
        hovertext=_hover_name(sub.get("Name")), visible=True,
        legendgroup="ALL", showlegend=False,
        meta={"id": "PT_ALL_ND:NaN"},
        customdata=sub[["id"]].to_numpy(),
    ))
if pts_dac["County"].isna().any():
    sub = pts_dac[pts_dac["County"].isna()]
    fig.add_trace(go.Scattermap(
        lon=sub["lon"], lat=sub["lat"], mode="markers",
        marker={"size": POINT_SIZE, "color": "gray"},
        name="Schools — County: Unknown", hoverinfo="text",
        hovertext=_hover_name(sub.get("Name")), visible=True,
        legendgroup="ALL", showlegend=True,
        meta={"id": "PT_ALL_DAC:NaN"},
        customdata=sub[["id"]].to_numpy(),
    ))

# 2b) DAC_DESIG mode: white ND + per-county DAC
# ND (white) — legend shows "Schools — Not DAC"
fig.add_trace(go.Scattermap(
    lon=pts_nd["lon"], lat=pts_nd["lat"], mode="markers",
    marker={"size": POINT_SIZE, "color": "LightGray"},
    name="Schools — Not DAC", hoverinfo="text",
    hovertext=_hover_name(pts_nd.get("Name")), visible=False,
    legendgroup="DAC_DESIG", showlegend=True,
    meta={"id": "PT_DESIG_ND"},
    customdata=pts_nd[["id"]].to_numpy(),
))
# DAC per county — legend "Schools — County: X"
for c in counties:
    sub = pts_dac[pts_dac["County"] == c]
    fig.add_trace(go.Scattermap(
        lon=sub["lon"], lat=sub["lat"], mode="markers",
        marker={"size": POINT_SIZE, "color": county_color[c]},
        name=f"Schools — County: {_county_label(c)}",
        hoverinfo="text",
        hovertext=_hover_name(sub.get("Name")), visible=False,
        legendgroup="DAC_DESIG", showlegend=True,
        meta={"id": f"PT_DESIG_DAC:{c}"},
        customdata=sub[["id"]].to_numpy(),
    ))
if pts_dac["County"].isna().any():
    sub = pts_dac[pts_dac["County"].isna()]
    fig.add_trace(go.Scattermap(
        lon=sub["lon"], lat=sub["lat"], mode="markers",
        marker={"size": POINT_SIZE, "color": "gray"},
        name="Schools — County: Unknown", hoverinfo="text",
        hovertext=_hover_name(sub.get("Name")), visible=False,
        legendgroup="DAC_DESIG", showlegend=True,
        meta={"id": "PT_DESIG_DAC:NaN"},
        customdata=sub[["id"]].to_numpy(),
    ))

# 2c) NUMERIC mode: two traces share one coloraxis + colorbar
# Pick a default numeric field
default_num = next((f for f in DAC_NUM_FIELDS if f in pts_df), "Pop_Cnt")
vals_dac = pd.to_numeric(pts_dac.get(default_num), errors="coerce")
vals_nd = pd.to_numeric(pts_nd.get(default_num), errors="coerce")

fig.add_trace(go.Scattermap(
    lon=pts_nd["lon"], lat=pts_nd["lat"], mode="markers",
    marker={"size": POINT_SIZE, "color": vals_nd, "coloraxis": "coloraxis"},
    name="Schools", hoverinfo="text",
    hovertext=_hover_name(pts_nd.get("Name")), visible=False,
    legendgroup="NUMERIC", showlegend=False,
    meta={"id": "PT_NUM_ND"},
    customdata=pts_nd[["id"]].to_numpy(),
))
fig.add_trace(go.Scattermap(
    lon=pts_dac["lon"], lat=pts_dac["lat"], mode="markers",
    marker={"size": POINT_SIZE, "color": vals_dac, "coloraxis": "coloraxis"},
    name="Schools", hoverinfo="text",
    hovertext=_hover_name(pts_dac.get("Name")), visible=False,
    legendgroup="NUMERIC", showlegend=False,
    meta={"id": "PT_NUM_DAC"},
    customdata=pts_dac[["id"]].to_numpy(),
))

# Map and layout (add coloraxis for numeric mode)
fig.update_layout(
    map={"style": MAP_STYLE, "center": center, "zoom": zoom},
    coloraxis={"colorscale": "Viridis",
               "colorbar": {"title": default_num}},
    uirevision="keep-view",
    clickmode="event",
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
    legend={
        "orientation": "h",
        "yanchor": "bottom",
        "y": 0.01,
        "xanchor": "center",
        "x": 0.5,
        "traceorder": "normal",
    },
)


# Build an id->index map once for fast toggling in callbacks
ID_IDX = {getattr(tr, "meta", {}).get("id"): i for i, tr in
          enumerate(fig.data)}

# --------------------------- App Layout ------------------------------------ #
# Create Dash app
app = Dash(__name__)  # app
server = app.server   # <-- required for Render/Gunicorn

# Build options for points mode dropdown
points_mode_options = (
    [{"label": "All (color by County)", "value": "ALL"},
     {"label": "DAC Designation", "value": "DAC_DESIG"}] +
    [{"label": f, "value": f} for f in DAC_NUM_FIELDS]
)

# Page layout per Dash tutorial style
app.layout = html.Div(
    children=[
        html.H3("Schools & DAC Areas Map", style={"margin": "8px 0 4px 8px"}),
        html.Div(
            style={
                "display": "flex", "gap": "12px", "flexWrap": "wrap",
                "alignItems": "center", "margin": "0 8px 8px 8px",
            },
            children=[
                dcc.Checklist(
                    id="layer-toggle",  # toggle layers
                    options=[
                        {"label": "Show DAC areas", "value": "poly"},
                        {"label": "Show schools", "value": "pts"},
                    ],
                    value=["poly", "pts"],  # both on
                    inputStyle={"marginRight": "6px"},  # spacing
                    labelStyle={"marginRight": "12px"},  # spacing
                    persistence=True,  # remember choice
                ),
                dcc.Checklist(
                    id="dac-subset",  # DAC subset filter
                    options=[
                        {"label": "Schools in DAC",
                         "value": "Designated as DAC"},
                        {"label": "Schools not in DAC",
                         "value": "Not Designated as DAC"},
                    ],
                    value=["Designated as DAC", "Not Designated as DAC"],
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"marginRight": "12px"},
                    persistence=True,
                ),
                dcc.Dropdown(
                    id="points-mode",  # symbology mode
                    options=points_mode_options,
                    value="ALL",  # default mode
                    placeholder="Points symbology…",  # hint
                    style={"minWidth": "260px"},  # width
                    persistence=True,  # remember choice
                ),
            ],
        ),
        # Map + Right Panel container
        html.Div(
            style={"display": "flex", "gap": "12px", "alignItems": "stretch",
                   "margin": "0 8px 8px 8px"},
            children=[
                dcc.Graph(
                    id="map-graph",  # target graph
                    figure=fig,  # initial fig
                    style={"height": "80vh", "flex": "1 1 auto"},  # size
                    config={"displayModeBar": True},  # toolbar
                    animate=False,  # faster updates, no transitions
                ),
                html.Div(
                    id="details-panel",
                    style={
                        "width": "340px", "flex": "0 0 340px",
                        "border": "1px solid #ddd", "borderRadius": "8px",
                        "padding": "10px", "height": "80vh", "overflow": "auto",
                        "background": "#fafafa",
                    },
                    children=[
                        html.H4("Selection"),
                        html.Div("Click a school point to see DAC polygon attributes."),
                    ],
                ),
            ],
        ),
        html.Div(
            "Hover shows the school Name. Points are symbolized by the "
            "selected DAC-based mode.",
            style={"fontSize": "12px", "margin": "4px 8px 8px 8px"},
        ),
    ]
)

# --------------------------- Callbacks ------------------------------------- #
@app.callback(
    Output("map-graph", "figure"),
    Input("layer-toggle", "value"),
    Input("points-mode", "value"),
    Input("dac-subset", "value"),
    Input("map-graph", "relayoutData"),
    State("map-graph", "figure"),
)
def update_map(layers, points_mode, dac_subset, relayout, cur_fig):
    """
    Patch only what's needed: visibility, numeric colors, legend, view.
    """
    # Preserve current view (MapLibre + legacy keys)
    cur_center = center.copy()
    cur_zoom = zoom
    if isinstance(relayout, dict):
        ctr_ml = relayout.get("map.center")
        zm_ml = relayout.get("map.zoom")
        ctr_mb = relayout.get("mapbox.center")
        zm_mb = relayout.get("mapbox.zoom")
        if isinstance(ctr_ml, dict):
            cur_center = {"lon": ctr_ml.get("lon", cur_center["lon"]),
                          "lat": ctr_ml.get("lat", cur_center["lat"])}
        elif isinstance(ctr_mb, dict):
            cur_center = {"lon": ctr_mb.get("lon", cur_center["lon"]),
                          "lat": ctr_mb.get("lat", cur_center["lat"])}
        if isinstance(zm_ml, (int, float)):
            cur_zoom = zm_ml
        elif isinstance(zm_mb, (int, float)):
            cur_zoom = zm_mb

    # Patch object
    patch = Patch()

    # 1) Polygons visibility (no re-adding)
    show_poly = "poly" in layers
    for pid in ("POLY_DAC", "POLY_NODAC"):
        idx = ID_IDX.get(pid)
        if idx is not None:
            patch["data"][idx]["visible"] = show_poly

    # 2) Points visibility groups
    show_pts = "pts" in layers
    # Hide all points first
    for pid, idx in ID_IDX.items():
        if isinstance(pid, str) and pid.startswith("PT_"):
            patch["data"][idx]["visible"] = False

    # Legend title text
    legend_title = "Schools — County"

    if show_pts:
        if points_mode == "ALL":
            # Show ND/DAC per selected subset
            want_nd = ND_FLAG in dac_subset
            want_dac = DAC_FLAG in dac_subset
            # Counties incl. NaN alias
            names = [*counties]
            if "PT_ALL_ND:NaN" in ID_IDX:
                names += ["NaN"]
            for c in names:
                if want_nd:
                    idx = ID_IDX.get(f"PT_ALL_ND:{c}")
                    if idx is not None:
                        patch["data"][idx]["visible"] = True
                if want_dac:
                    idx = ID_IDX.get(f"PT_ALL_DAC:{c}")
                    if idx is not None:
                        patch["data"][idx]["visible"] = True
            legend_title = "Schools — County"

        elif points_mode == "DAC_DESIG":
            # ND (white) if requested
            if ND_FLAG in dac_subset:
                idx = ID_IDX.get("PT_DESIG_ND")
                if idx is not None:
                    patch["data"][idx]["visible"] = True
            # DAC per county if requested
            if DAC_FLAG in dac_subset:
                for c in counties:
                    idx = ID_IDX.get(f"PT_DESIG_DAC:{c}")
                    if idx is not None:
                        patch["data"][idx]["visible"] = True
                idx = ID_IDX.get("PT_DESIG_DAC:NaN")
                if idx is not None:
                    patch["data"][idx]["visible"] = True
            legend_title = "Schools — DAC Designation"

        elif points_mode in DAC_NUM_FIELDS:
            # Update numeric color arrays only (fast)
            patch["layout"]["coloraxis"]["colorbar"]["title"] = points_mode
            # Recompute values (no coords resend)
            vals_nd = pd.to_numeric(
                pts_nd.get(points_mode), errors="coerce"
            )
            vals_d = pd.to_numeric(
                pts_dac.get(points_mode), errors="coerce"
            )
            idx_nd = ID_IDX.get("PT_NUM_ND")
            idx_d = ID_IDX.get("PT_NUM_DAC")
            if idx_nd is not None:
                patch["data"][idx_nd]["marker"]["color"] = vals_nd
                patch["data"][idx_nd]["visible"] = ND_FLAG in dac_subset
                patch["data"][idx_nd]["showlegend"] = False
                patch["data"][idx_nd]["name"] = "Schools"
            if idx_d is not None:
                patch["data"][idx_d]["marker"]["color"] = vals_d
                patch["data"][idx_d]["visible"] = DAC_FLAG in dac_subset
                patch["data"][idx_d]["showlegend"] = False
                patch["data"][idx_d]["name"] = "Schools"
            legend_title = f"Schools — {points_mode}"

        else:
            # Fallback: keep ALL mode visible
            for c in counties:
                for pre in ("PT_ALL_ND", "PT_ALL_DAC"):
                    idx = ID_IDX.get(f"{pre}:{c}")
                    if idx is not None:
                        patch["data"][idx]["visible"] = True
            legend_title = "Schools — County"

    # 3) Preserve current view
    patch["layout"]["map"]["center"] = cur_center
    patch["layout"]["map"]["zoom"] = cur_zoom
    patch["layout"]["uirevision"] = "keep-view"
    # 4) Update legend title for clarity
    patch["layout"]["legend"]["title"]["text"] = legend_title

    return patch


@app.callback(
    Output("details-panel", "children"),
    Input("map-graph", "clickData"),
    Input("points-mode", "value"),
    Input("dac-subset", "value"),
)

def show_polygon_attributes(click, points_mode, dac_subset):
    """
    Fill the right-hand panel with DAC polygon attributes for the clicked
    school point, and include a histogram summarizing the distribution
    under the current mode/subset.
    """
    # Build the histogram first (always shown)
    dist_fig = build_distribution_figure(points_mode, dac_subset)

    # No click -> show only instructions + histogram
    if not click or "points" not in click or not click["points"]:
        return [
            html.H4("Selection"),
            html.Div("Click a school point to see DAC polygon attributes."),
            html.Hr(),
            html.H4("Distribution"),
            dcc.Graph(figure=dist_fig, style={"height": "260px"}),
        ]

    cd = click["points"][0].get("customdata")
    if cd is None:
        return [
            html.H4("Distribution"),
            dcc.Graph(figure=dist_fig, style={"height": "260px"}),
            html.Hr(),
            html.H4("Selection"),
            html.Div("No selection details available."),
        ]

    # accept scalar or 1-element array; compare as string to be safe
    key = str(cd if not isinstance(cd, (list, np.ndarray)) else np.ravel(cd)[0])

    row = pts_df.loc[pts_df["id"].astype(str) == key]
    if row.empty:
        return [
            html.H4("Distribution"),
            dcc.Graph(figure=dist_fig, style={"height": "260px"}),
            html.Hr(),
            html.H4("Selection"),
            html.Div("No selection details available."),
        ]


    rec = row.iloc[0]

    # Use the polygon-related columns we joined (present in pts_df)
    cols = [c for c in join_cols if c in rec.index]
    # Build a small HTML table
    table_rows = []
    for c in cols:
        val = rec[c]
        # Format floats to 2 decimals; show blanks for NaN
        if isinstance(val, (float, np.floating)):
            val_str = "" if pd.isna(val) else f"{float(val):.2f}"
        else:
            val_str = "" if pd.isna(val) else str(val)

        table_rows.append(
            html.Tr([
                html.Th(str(c), style={"textAlign": "left", "paddingRight": "8px"}),
                html.Td(val_str)
            ])
        )


    # Build a bullet list of ALL school names at this location
    names_list = []
    try:
        if "_lonr" in rec.index and "_latr" in rec.index:
            sel = pts_df[(pts_df["_lonr"] == rec["_lonr"]) & (pts_df["_latr"] == rec["_latr"])]
            names_list = sorted({str(x) for x in sel["Name"].dropna()})
        elif "lon" in rec.index and "lat" in rec.index and "_lonr" in pts_df.columns and "_latr" in pts_df.columns:
            lonr = round(float(rec["lon"]), 5)
            latr = round(float(rec["lat"]), 5)
            sel = pts_df[(pts_df["_lonr"] == lonr) & (pts_df["_latr"] == latr)]
            names_list = sorted({str(x) for x in sel["Name"].dropna()})
    except Exception:
        names_list = []

    # Fallback if nothing aggregated found
    if not names_list:
        fallback_name = str(rec.get("Name")) if "Name" in rec.index and pd.notna(rec.get("Name")) else "Selected school"
        names_section = [
            html.H4("Selection"),
            html.Div(fallback_name, style={"marginBottom": "6px", "fontWeight": "bold"}),
        ]
    else:
        names_section = [
            html.H4("Selection"),
            html.Div("Schools at this location:", style={"fontSize": "12px", "color": "#555"}),
            html.Ul([html.Li(n) for n in names_list], style={"marginTop": "4px", "marginBottom": "8px"}),
        ]

    return [
        html.H4("Distribution"),
        dcc.Graph(figure=dist_fig, style={"height": "260px"}),
        html.Hr(),
        *names_section,
        html.Div("DAC polygon attributes:", style={"fontSize": "12px", "color": "#555"}),
        html.Table(table_rows, style={"fontSize": "12px", "width": "100%"}),
    ]





# --------------------------- Main ------------------------------------------ #
if __name__ == "__main__":
    # Print startup time
    dt = time.time() - t0  # seconds
    hrs = int(dt // 3600)  # hours
    mins = int((dt % 3600) // 60)  # minutes
    secs = int(dt % 60)  # seconds
    print(
        f"App initialized in {hrs}h {mins}m {secs}s. "
        f"Open http://127.0.0.1:8050"
    )  # status
    # Run the Dash server (single process: no stuck port)
    app.run(debug=True, port=8050, use_reloader=False)
    #  - debug=True: keep dev tools
    #  - port=8050: use the usual port
    #  - use_reloader=False: don't spawn a second process
