from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FMCG Smart Launch | Decision Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "FMCG_clean.csv"
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "FMCG_2022_2024.csv"
RESULTS_DIR = BASE_DIR / "results"


# ============================================================
# DESIGN SYSTEM
# ============================================================
COLORS = {
    "bg": "#F5F1EB",
    "surface": "#FFFDFC",
    "surface_alt": "#F9F5EF",
    "ink": "#2E2520",
    "muted": "#75685E",
    "line": "#E7DED4",
    "brown": "#604436",
    "brown_dark": "#3D2B22",
    "terracotta": "#A86A4B",
    "sage": "#6F7D69",
    "gold": "#B48A4E",
    "blue": "#5B7288",
    "green": "#5E7B67",
    "red": "#A05A52",
}

st.markdown(
    f"""
    <style>
        .stApp {{
            background: {COLORS['bg']};
            color: {COLORS['ink']};
        }}

        .block-container {{
            max-width: 1480px;
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }}

        [data-testid="stSidebar"] {{
            background: {COLORS['brown_dark']};
            border-right: 1px solid rgba(255,255,255,0.06);
        }}

        [data-testid="stSidebar"] * {{
            color: #F8F4EE !important;
        }}

        [data-testid="stSidebar"] .stRadio > div {{
            gap: 5px;
        }}

        [data-testid="stSidebar"] .stRadio label {{
            border-radius: 9px;
            padding: 7px 9px;
        }}

        [data-testid="stSidebar"] .stRadio label:hover {{
            background: rgba(255,255,255,0.07);
        }}

        .app-kicker {{
            color: {COLORS['terracotta']};
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}

        .app-title {{
            color: {COLORS['ink']};
            font-size: 2.05rem;
            font-weight: 850;
            line-height: 1.15;
            margin-bottom: 6px;
        }}

        .app-subtitle {{
            color: {COLORS['muted']};
            font-size: 0.98rem;
            line-height: 1.55;
            max-width: 950px;
        }}

        .hero {{
            background: linear-gradient(135deg, {COLORS['brown_dark']} 0%, {COLORS['brown']} 72%, #765240 100%);
            border-radius: 18px;
            padding: 24px 26px;
            color: #FFFDFC;
            margin: 0.2rem 0 1.15rem 0;
            box-shadow: 0 12px 30px rgba(60,42,32,0.15);
        }}

        .hero-small {{
            color: rgba(255,255,255,0.77);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 7px;
        }}

        .hero-title {{
            font-size: 1.92rem;
            font-weight: 850;
            line-height: 1.15;
            margin-bottom: 8px;
        }}

        .hero-subtitle {{
            color: rgba(255,255,255,0.86);
            font-size: 0.96rem;
            line-height: 1.55;
            max-width: 1000px;
        }}

        .page-note {{
            background: {COLORS['surface_alt']};
            border: 1px solid {COLORS['line']};
            border-radius: 11px;
            padding: 11px 14px;
            color: {COLORS['muted']};
            font-size: 0.86rem;
            line-height: 1.5;
            margin-bottom: 12px;
        }}

        .section-title {{
            color: {COLORS['ink']};
            font-size: 1.16rem;
            font-weight: 820;
            margin: 19px 0 8px 0;
        }}

        .section-subtitle {{
            color: {COLORS['muted']};
            font-size: 0.88rem;
            margin-top: -3px;
            margin-bottom: 11px;
        }}

        .chart-explanation {{
            background: {COLORS['surface_alt']};
            border: 1px solid {COLORS['line']};
            border-left: 4px solid {COLORS['terracotta']};
            border-radius: 9px;
            padding: 10px 12px;
            margin: -2px 0 12px 0;
        }}

        .chart-explanation-title {{
            color: {COLORS['brown_dark']};
            font-weight: 800;
            font-size: 0.82rem;
            margin-bottom: 3px;
        }}

        .chart-explanation-text {{
            color: {COLORS['muted']};
            font-size: 0.82rem;
            line-height: 1.5;
        }}

        .insight-box {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['line']};
            border-radius: 12px;
            padding: 14px 15px;
            margin: 8px 0 13px 0;
            box-shadow: 0 4px 12px rgba(60,42,32,0.04);
        }}

        .insight-eyebrow {{
            color: {COLORS['terracotta']};
            font-size: 0.74rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 3px;
        }}

        .insight-title {{
            color: {COLORS['ink']};
            font-weight: 820;
            font-size: 0.95rem;
            margin-bottom: 5px;
        }}

        .insight-text {{
            color: {COLORS['muted']};
            font-size: 0.87rem;
            line-height: 1.56;
        }}

        .decision-card {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['line']};
            border-radius: 14px;
            padding: 15px;
            min-height: 125px;
            box-shadow: 0 4px 12px rgba(60,42,32,0.03);
        }}

        .decision-number {{
            color: {COLORS['terracotta']};
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.08em;
        }}

        .decision-title {{
            color: {COLORS['ink']};
            font-weight: 820;
            margin: 5px 0;
        }}

        .decision-text {{
            color: {COLORS['muted']};
            font-size: 0.81rem;
            line-height: 1.47;
        }}

        .footer-note {{
            color: {COLORS['muted']};
            font-size: 0.77rem;
            margin-top: 26px;
            padding-top: 12px;
            border-top: 1px solid {COLORS['line']};
        }}

        [data-testid="stMetric"] {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['line']};
            border-radius: 13px;
            padding: 11px 14px;
            box-shadow: 0 4px 12px rgba(60,42,32,0.045);
        }}

        [data-testid="stMetricLabel"] {{
            color: {COLORS['muted']};
            font-size: 0.75rem;
            font-weight: 750;
        }}

        [data-testid="stMetricValue"] {{
            color: {COLORS['ink']};
            font-size: 1.45rem;
            font-weight: 820;
        }}

        .stButton > button, .stDownloadButton > button {{
            border-radius: 9px;
            border: 1px solid {COLORS['line']};
            background: {COLORS['surface']};
            color: {COLORS['brown_dark']};
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 10px;
        }}

        .small-muted {{
            color: {COLORS['muted']};
            font-size: 0.76rem;
            line-height: 1.45;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FORMATTING HELPERS
# ============================================================
def number(value) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:,.0f}"


def money(value) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:,.0f}"


def pct(value, decimals: int = 1) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.{decimals}f}%"


def compact(value: float) -> str:
    if pd.isna(value):
        return "—"
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:.0f}"


def section(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-small">FMCG SMART LAUNCH · DECISION DASHBOARD</div>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_note(text: str) -> None:
    st.markdown(f'<div class="page-note">{text}</div>', unsafe_allow_html=True)


def chart_explanation(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="chart-explanation">
            <div class="chart-explanation-title">{title}</div>
            <div class="chart-explanation-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_box(title: str, text: str, eyebrow: str = "Analytical readout") -> None:
    st.markdown(
        f"""
        <div class="insight-box">
            <div class="insight-eyebrow">{eyebrow}</div>
            <div class="insight-title">{title}</div>
            <div class="insight-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def decision_card(number_text: str, title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="decision-card">
            <div class="decision-number">{number_text}</div>
            <div class="decision-title">{title}</div>
            <div class="decision-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def styled_table(data: pd.DataFrame, decimals: int = 2) -> None:
    table = data.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].round(decimals)
    st.table(table)


def base_plot(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["ink"], family="Arial"),
        margin=dict(l=30, r=20, t=45, b=35),
        title=dict(font=dict(size=14, color=COLORS["ink"], family="Arial")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(bgcolor=COLORS["brown_dark"], font_color="#FFFDFC"),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=COLORS["line"],
        tickfont=dict(size=10, color=COLORS["muted"]),
        title_font=dict(size=10, color=COLORS["muted"]),
    )
    fig.update_yaxes(
        gridcolor="#EEE7E0",
        zeroline=False,
        linecolor=COLORS["line"],
        tickfont=dict(size=10, color=COLORS["muted"]),
        title_font=dict(size=10, color=COLORS["muted"]),
    )
    return fig


def render_plot(fig: go.Figure, explanation_title: str, explanation: str, height: int = 390) -> None:
    st.plotly_chart(
        base_plot(fig, height),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )
    chart_explanation(explanation_title, explanation)


# ============================================================
# DATA / MODEL FUNCTIONS
# ============================================================
@st.cache_data(show_spinner="Loading FMCG sales data...")
def load_data() -> pd.DataFrame:
    path = DATA_PATH if DATA_PATH.exists() else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            "Dataset not found. Put FMCG_clean.csv in data/processed/ "
            "or FMCG_2022_2024.csv in data/raw/."
        )

    data = pd.read_csv(path)

    # The processed notebook file may contain analysis-derived columns.
    # The dashboard rebuilds these metrics from the raw transactional fields,
    # so remove them to prevent duplicate-column collisions during merges.
    derived_columns = [
        "First_Observed_Date",
        "Last_Observed_Date",
        "Active_Days",
        "Total_Units_Sold",
        "Total_Sales_Value",
        "Avg_Units_Per_Day",
        "Avg_Price",
        "Observed_Duration_Days",
        "Introduction_Year",
        "Introduction_Month",
        "Product_Age_Days",
        "Product_Age_Group",
        "price_group",
        "log_price",
        "log_units_sold",
        "month_num",
        "year_num",
    ]
    data = data.drop(columns=derived_columns, errors="ignore")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    numeric_columns = [
        "price_unit",
        "promotion_flag",
        "delivery_days",
        "stock_available",
        "delivered_qty",
        "units_sold",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if "sales_value" not in data.columns:
        data["sales_value"] = data["price_unit"] * data["units_sold"]
    else:
        data["sales_value"] = pd.to_numeric(data["sales_value"], errors="coerce")

    data = data.dropna(subset=["date", "price_unit", "units_sold"]).copy()
    return data


@st.cache_data(show_spinner="Building lifecycle view...")
def build_lifecycle(data: pd.DataFrame) -> pd.DataFrame:
    lifecycle = (
        data.groupby(["sku", "brand", "segment", "category"], as_index=False)
        .agg(
            First_Observed_Date=("date", "min"),
            Last_Observed_Date=("date", "max"),
            Active_Days=("date", "nunique"),
            Total_Units_Sold=("units_sold", "sum"),
            Total_Sales_Value=("sales_value", "sum"),
            Avg_Units_Per_Day=("units_sold", "mean"),
            Avg_Price=("price_unit", "mean"),
        )
    )
    lifecycle["Observed_Duration_Days"] = (
        lifecycle["Last_Observed_Date"] - lifecycle["First_Observed_Date"]
    ).dt.days + 1
    lifecycle["Introduction_Year"] = lifecycle["First_Observed_Date"].dt.year
    return lifecycle


@st.cache_data(show_spinner="Building demand series...")
def build_daily_demand(data: pd.DataFrame) -> pd.DataFrame:
    daily = (
        data.groupby("date", as_index=False)
        .agg(
            units_sold=("units_sold", "sum"),
            sales_value=("sales_value", "sum"),
            avg_price=("price_unit", "mean"),
            promotion_rate=("promotion_flag", "mean"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["year"] = daily["date"].dt.year
    daily["dow_sin"] = np.sin(2 * np.pi * daily["day_of_week"] / 7)
    daily["dow_cos"] = np.cos(2 * np.pi * daily["day_of_week"] / 7)
    daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)

    for lag in [1, 7, 14, 28]:
        daily[f"lag_{lag}"] = daily["units_sold"].shift(lag)

    for window in [7, 14, 28]:
        daily[f"rolling_mean_{window}"] = (
            daily["units_sold"].shift(1).rolling(window).mean()
        )

    return daily


@st.cache_data(show_spinner="Training demand forecasting model...")
def build_forecast(data: pd.DataFrame):
    daily = build_daily_demand(data).dropna().copy()

    features = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "promotion_rate",
    ]

    split_index = int(len(daily) * 0.80)
    train_df = daily.iloc[:split_index].copy()
    test_df = daily.iloc[split_index:].copy()

    X_train = train_df[features]
    y_train = train_df["units_sold"]
    X_test = test_df[features]
    y_test = test_df["units_sold"]

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    baseline_pred = test_df["lag_1"].to_numpy()
    ml_pred = model.predict(X_test)

    metrics = {
        "baseline_mae": mean_absolute_error(y_test, baseline_pred),
        "baseline_rmse": np.sqrt(mean_squared_error(y_test, baseline_pred)),
        "baseline_mape": np.mean(
            np.abs((y_test.to_numpy() - baseline_pred) / y_test.to_numpy())
        )
        * 100,
        "ml_mae": mean_absolute_error(y_test, ml_pred),
        "ml_rmse": np.sqrt(mean_squared_error(y_test, ml_pred)),
        "ml_mape": np.mean(
            np.abs((y_test.to_numpy() - ml_pred) / y_test.to_numpy())
        )
        * 100,
        "ml_r2": r2_score(y_test, ml_pred),
        "mean_error": np.mean(y_test.to_numpy() - ml_pred),
    }

    results = test_df[["date", "units_sold"]].copy()
    results["predicted_units_sold"] = ml_pred
    results["error"] = results["units_sold"] - results["predicted_units_sold"]
    results["absolute_error"] = results["error"].abs()

    return model, features, daily, train_df, test_df, results, metrics


@st.cache_data(show_spinner=False)
def load_result_csv(filename: str) -> pd.DataFrame:
    path = RESULTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ============================================================
# LOAD
# ============================================================
try:
    df = load_data()
    lifecycle = build_lifecycle(df)
    daily_demand = build_daily_demand(df)
    forecast_model, forecast_features, forecast_daily, train_df, test_df, forecast_results, forecast_metrics = build_forecast(df)
except Exception as exc:
    st.error(str(exc))
    st.stop()


# ============================================================
# SIDEBAR — NAVIGATION + CONTEXT
# ============================================================
st.sidebar.markdown(
    """
    <div style="padding: 6px 4px 15px 4px;">
        <div style="font-size:0.72rem;font-weight:800;letter-spacing:0.1em;opacity:0.72;">PORTFOLIO ANALYTICS</div>
        <div style="font-size:1.2rem;font-weight:850;margin-top:3px;">FMCG Smart Launch</div>
        <div style="font-size:0.79rem;line-height:1.45;opacity:0.73;margin-top:5px;">Product portfolio intelligence for launch, demand planning, pricing and cannibalization monitoring.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

pages = [
    "Executive Overview",
    "Sales & Pricing",
    "Product Lifecycle",
    "Cannibalization Monitor",
    "Demand Forecast",
]

page = st.sidebar.radio(
    "Dashboard section",
    pages,
    label_visibility="visible",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Analysis context**")
st.sidebar.caption(
    f"{len(df):,} sales records · {df['sku'].nunique()} SKUs · "
    f"{df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y}"
)

# All dashboard pages use the full validated dataset so the reported
# metrics stay consistent with the notebook analysis.
filtered_df = df
filter_label = "Full portfolio"



# ============================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ============================================================
if page == "Executive Overview":
    page_header(
        "Executive Overview",
        "A management-level view of portfolio scale, demand movement, product concentration and the analytical signals used for launch and planning decisions.",
    )

    st.markdown(
        f'<div class="page-note"><strong>{filter_label}:</strong> {len(filtered_df):,} observations in scope. All metrics on this page use the full validated portfolio dataset.</div>',
        unsafe_allow_html=True,
    )

    total_units = filtered_df["units_sold"].sum()
    sales_value = filtered_df["sales_value"].sum()
    active_skus = filtered_df["sku"].nunique()
    avg_price = filtered_df["price_unit"].mean()
    promo_share = filtered_df["promotion_flag"].mean() * 100

    kpis = st.columns(5)
    kpis[0].metric("Units Sold", compact(total_units))
    kpis[1].metric("Sales Value", compact(sales_value))
    kpis[2].metric("Active SKUs", f"{active_skus:,}")
    kpis[3].metric("Average Price", f"{avg_price:,.2f}")
    kpis[4].metric("Promotion Share", pct(promo_share, 1))

    section(
        "Demand trajectory",
        "Monthly unit demand provides the cleanest portfolio-level view of trend, seasonality and demand shifts.",
    )
    trend = (
        filtered_df.assign(month=filtered_df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)["units_sold"]
        .sum()
        .rename(columns={"units_sold": "Units_Sold"})
        .sort_values("month")
        .reset_index(drop=True)
    )
    trend["Units_Sold"] = pd.to_numeric(trend["Units_Sold"], errors="coerce")
    trend = trend.dropna(subset=["Units_Sold"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend["month"],
            y=trend["Units_Sold"],
            mode="lines+markers",
            name="Units Sold",
            line=dict(color=COLORS["brown"], width=2.6),
            marker=dict(size=6),
            hovertemplate="%{x|%b %Y}<br>Units sold: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(title="Monthly units sold")
    fig.update_yaxes(title="Units sold", tickformat=",.0f", rangemode="tozero")
    fig.update_xaxes(title="")
    render_plot(
        fig,
        "What the chart tells management",
        "This is the portfolio demand baseline. Sustained movement is more decision-relevant than isolated daily spikes, so the dashboard uses monthly aggregation here.",
        360,
    )

    # SKU introduction summary used by the dashboard.
    intro = (
        lifecycle.groupby("Introduction_Year", as_index=False)
        .agg(
            New_SKUs=("sku", "nunique"),
            Avg_Observed_Duration=("Observed_Duration_Days", "mean"),
            Avg_Units_Per_Day=("Avg_Units_Per_Day", "mean"),
            Total_Units_Sold=("Total_Units_Sold", "sum"),
        )
    )
    intro["Introduction_Year"] = pd.to_numeric(intro["Introduction_Year"], errors="coerce")
    intro["New_SKUs"] = pd.to_numeric(intro["New_SKUs"], errors="coerce")
    intro = intro.dropna(subset=["Introduction_Year", "New_SKUs"]).sort_values("Introduction_Year")

    c1, c2 = st.columns(2)
    with c1:
        category = (
            filtered_df.groupby("category", as_index=False)["units_sold"]
            .sum()
            .sort_values("units_sold", ascending=False)
        )
        fig = px.bar(category, x="category", y="units_sold", title="Category contribution to volume")
        fig.update_traces(marker_color=COLORS["terracotta"])
        fig.update_yaxes(title="Units sold", tickformat=",.0f")
        fig.update_xaxes(title="")
        render_plot(
            fig,
            "Portfolio mix",
            "Category contribution shows where volume is concentrated. This helps frame whether a launch expands the portfolio or enters an already crowded demand pool.",
            330,
        )

    with c2:
        channel = (
            filtered_df.groupby("channel", as_index=False)["units_sold"]
            .sum()
            .sort_values("units_sold", ascending=False)
        )
        fig = px.bar(channel, x="channel", y="units_sold", title="Channel contribution to volume")
        fig.update_traces(marker_color=COLORS["blue"])
        fig.update_yaxes(title="Units sold", tickformat=",.0f")
        fig.update_xaxes(title="")
        render_plot(
            fig,
            "Channel context",
            "Channel mix helps separate portfolio demand from channel-specific concentration. A launch decision should consider whether the new SKU depends on the same routes to market as existing products.",
            330,
        )



# ============================================================
# PAGE 2 — SALES & PRICING
# ============================================================
elif page == "Sales & Pricing":
    page_header(
        "Sales & Pricing",
        "Explore volume concentration, promotional patterns and the controlled price-response result behind the portfolio pricing assessment.",
    )

    page_note(
        "Promotion comparison is shown for both promotion states across the full portfolio."
    )

    sales_cols = st.columns(4)
    sales_cols[0].metric("Units Sold", compact(filtered_df["units_sold"].sum()))
    sales_cols[1].metric("Sales Value", compact(filtered_df["sales_value"].sum()))
    sales_cols[2].metric("Average Unit Price", f"{filtered_df['price_unit'].mean():,.2f}")
    sales_cols[3].metric("Transactions", f"{len(filtered_df):,}")

    c1, c2 = st.columns(2)
    with c1:
        sku_perf = (
            filtered_df.groupby("sku", as_index=False)
            .agg(Units_Sold=("units_sold", "sum"), Sales_Value=("sales_value", "sum"))
            .sort_values("Units_Sold", ascending=False)
            .head(12)
            .sort_values("Units_Sold")
        )
        fig = px.bar(sku_perf, x="Units_Sold", y="sku", orientation="h", title="Top SKUs by unit volume")
        fig.update_traces(marker_color=COLORS["brown"])
        fig.update_xaxes(title="Units sold", tickformat=",.0f")
        fig.update_yaxes(title="")
        render_plot(
            fig,
            "Concentration check",
            "The chart highlights the products carrying the most unit volume within the selected commercial slice. Concentration matters when evaluating whether a new SKU enters a core demand pool.",
            400,
        )

    with c2:
        promo_scope = df.copy()

        promo = (
            promo_scope.groupby("promotion_flag", as_index=False)
            .agg(
                Avg_Units_Sold=("units_sold", "mean"),
                Total_Units_Sold=("units_sold", "sum"),
                Transactions=("sku", "size"),
            )
        )
        promo["Promotion_Status"] = promo["promotion_flag"].map({0: "No Promotion", 1: "Promotion"})
        fig = px.bar(promo, x="Promotion_Status", y="Avg_Units_Sold", title="Average units sold by promotion state")
        fig.update_traces(marker_color=COLORS["sage"])
        fig.update_yaxes(title="Average units sold", tickformat=",.1f")
        fig.update_xaxes(title="")
        render_plot(
            fig,
            "Descriptive comparison",
            "This is a descriptive comparison between promoted and non-promoted observations. A higher bar does not prove promotion caused the difference because other factors may move at the same time.",
            400,
        )

    section("Price and demand", "A descriptive view of average demand across price bands. The controlled regression below remains the statistical test used for elasticity.")

    price_scope = filtered_df.copy()
    price_scope["price_unit"] = pd.to_numeric(price_scope["price_unit"], errors="coerce")
    price_scope["units_sold"] = pd.to_numeric(price_scope["units_sold"], errors="coerce")
    price_scope = price_scope.replace([np.inf, -np.inf], np.nan).dropna(subset=["price_unit", "units_sold"])

    if len(price_scope) >= 10 and price_scope["price_unit"].nunique() >= 5:
        price_scope["Price_Band"] = pd.qcut(
            price_scope["price_unit"],
            q=5,
            duplicates="drop",
        )

        price_band = (
            price_scope.groupby("Price_Band", observed=False, as_index=False)
            .agg(
                Avg_Price=("price_unit", "mean"),
                Avg_Units_Sold=("units_sold", "mean"),
                Transactions=("sku", "size"),
            )
        )
        price_band = price_band.dropna(subset=["Avg_Price", "Avg_Units_Sold"]).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(11.5, 5.3))
        x = np.arange(len(price_band))
        bars = ax.bar(x, price_band["Avg_Units_Sold"], alpha=0.88)

        for i, (bar, avg_units) in enumerate(zip(bars, price_band["Avg_Units_Sold"])):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(price_band["Avg_Units_Sold"].max() * 0.025, 0.1),
                f"{avg_units:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        labels = [f"{v:.2f}" for v in price_band["Avg_Price"]]
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(
            "Average demand across price bands",
            loc="left",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Average unit price within band")
        ax.set_ylabel("Average units sold")
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        chart_explanation(
            "How to read this view",
            "The bars summarize observed demand across five price bands. This is descriptive evidence only; differences can reflect SKU mix, channel, promotion and timing. The controlled elasticity model below is therefore kept separate from this chart.",
        )
    else:
        st.info("Not enough price variation is available for a five-band comparison in the selected filter context.")

    section("Controlled price elasticity", "Model result from the notebook: SKU, promotion, channel, region, month and year controls included.")
    ecols = st.columns(4)
    ecols[0].metric("Elasticity", "-0.0019")
    ecols[1].metric("P-value", "0.3669")
    ecols[2].metric("R²", "0.2969")
    ecols[3].metric("Observations", "186,892")

    insight_box(
        "What this means",
        "The estimated coefficient is very close to zero and is not statistically significant at the 5% level. Within this specification, the dataset does not provide sufficient evidence of a meaningful price-demand relationship after the included controls are applied. This should be read as an observed conditional association, not a causal estimate.",
        eyebrow="Statistical interpretation",
    )

    section("Promotion detail", "Numbers behind the promotion comparison.")
    promo_table = promo[["Promotion_Status", "Transactions", "Avg_Units_Sold", "Total_Units_Sold"]].copy()
    promo_table.columns = ["Promotion Status", "Transactions", "Avg Units Sold", "Total Units Sold"]
    styled_table(promo_table, 2)


# ============================================================
# PAGE 3 — PRODUCT LIFECYCLE
# ============================================================
elif page == "Product Lifecycle":
    page_header(
        "Product Lifecycle",
        "Understand when SKUs first appear in the observed data, how long they remain active and how demand varies with product age.",
    )

    page_note(
        "Method note: first observed sales date is used as a proxy for product introduction because the dataset does not provide official launch dates."
    )

    intro = (
        lifecycle.groupby("Introduction_Year", as_index=False)
        .agg(
            New_SKUs=("sku", "nunique"),
            Avg_Observed_Duration=("Observed_Duration_Days", "mean"),
            Avg_Units_Per_Day=("Avg_Units_Per_Day", "mean"),
            Total_Units_Sold=("Total_Units_Sold", "sum"),
        )
    )

    peak_age_value = 25.45
    peak_age_label = "91–180 Days"

    kcols = st.columns(4)
    kcols[0].metric("Total SKUs", "30")
    kcols[1].metric("First Observed", lifecycle["First_Observed_Date"].min().strftime("%d %b %Y"))
    kcols[2].metric("2023 New SKUs", "10")
    kcols[3].metric("Peak Avg Units", f"{peak_age_value:.2f}")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8.8, 4.9))

        intro_plot = intro.copy()
        intro_plot["Introduction_Year"] = pd.to_numeric(
            intro_plot["Introduction_Year"], errors="coerce"
        )
        intro_plot["New_SKUs"] = pd.to_numeric(
            intro_plot["New_SKUs"], errors="coerce"
        )
        intro_plot = intro_plot.dropna(subset=["Introduction_Year", "New_SKUs"])

        ax.bar(
            intro_plot["Introduction_Year"].astype(int).astype(str),
            intro_plot["New_SKUs"],
            alpha=0.88,
        )

        for i, value in enumerate(intro_plot["New_SKUs"]):
            ax.text(
                i,
                value + max(intro_plot["New_SKUs"].max() * 0.03, 0.2),
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title("New SKU introduction by year", loc="left", fontsize=13, fontweight="bold")
        ax.set_xlabel("Introduction year")
        ax.set_ylabel("New SKUs")
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        chart_explanation(
            "Introduction timing",
            "The observed portfolio added 20 SKUs in 2022 and 10 in 2023. No SKU was first observed in 2024, so 2022–2023 is the main introduction window for the cannibalization analysis.",
        )

    with c2:
        # Use lifecycle as the single source of truth for first observed date.
        # Mapping by SKU avoids duplicate-column collisions even if an older
        # processed CSV contains lifecycle-derived fields.
        age_df = df.copy()
        first_observed_map = lifecycle.set_index("sku")["First_Observed_Date"]
        age_df["First_Observed_Date"] = age_df["sku"].map(first_observed_map)
        age_df["Product_Age_Days"] = (
            age_df["date"] - age_df["First_Observed_Date"]
        ).dt.days
        age_df["Product_Age_Group"] = pd.cut(
            age_df["Product_Age_Days"],
            bins=[-1, 30, 90, 180, 365, 730, np.inf],
            labels=["0–30", "31–90", "91–180", "181–365", "366–730", "730+"],
        )
        age_perf = (
            age_df.groupby("Product_Age_Group", observed=False, as_index=False)
            .agg(Avg_Units_Sold=("units_sold", "mean"))
        )
        fig = px.bar(age_perf, x="Product_Age_Group", y="Avg_Units_Sold", title="Average units sold by product age")
        fig.update_traces(marker_color=COLORS["sage"])
        fig.update_xaxes(title="Product age")
        fig.update_yaxes(title="Average units sold", tickformat=",.1f")
        render_plot(
            fig,
            "Lifecycle pattern",
            "Average demand rises to the 91–180 day group, then declines across older age groups. This is an observed pattern across age buckets, not evidence that age itself causes the change.",
            330,
        )

    section("Portfolio lifecycle table", "A compact product-level view used to understand the introduction timeline.")
    lifecycle_display = lifecycle[
        [
            "sku", "brand", "segment", "category", "First_Observed_Date",
            "Last_Observed_Date", "Observed_Duration_Days", "Total_Units_Sold", "Avg_Units_Per_Day"
        ]
    ].sort_values("First_Observed_Date").copy()
    lifecycle_display.columns = [
        "SKU", "Brand", "Segment", "Category", "First Observed", "Last Observed",
        "Duration (days)", "Total Units", "Avg Units / Day"
    ]
    styled_table(lifecycle_display, 2)

    insight_box(
        "Why this matters for launch analysis",
        "A new product should be evaluated relative to how long it has been observed and which existing products occupy the same segment. Lifecycle context prevents a newly introduced SKU from being compared directly with a mature SKU without considering time in market.",
    )


# ============================================================
# PAGE 4 — CANNIBALIZATION MONITOR
# ============================================================
elif page == "Cannibalization Monitor":
    page_header(
        "Cannibalization Monitor",
        "A transparent event-based monitor that compares existing same-segment SKU demand before and after a new SKU appears.",
    )

    page_note(
        "Decision rule: 90-day pre/post window. A potential indication requires comparable SKU demand to decline while broader segment demand remains stable or increases. This is an indication framework, not causal proof."
    )

    cannibal = load_result_csv("cannibalization_analysis.csv")
    if cannibal.empty:
        st.warning("cannibalization_analysis.csv was not found in results/. Run the notebook export cell first.")
    else:
        if "Final_Indication" not in cannibal.columns and "Indication" in cannibal.columns:
            cannibal["Final_Indication"] = cannibal["Indication"]

        total_events = int((lifecycle["Introduction_Year"] == 2023).sum())
        analyzed_events = len(cannibal)
        potential_events = int((cannibal["Final_Indication"] == "Potential Cannibalization Indication").sum())
        insufficient_events = int((cannibal["Final_Indication"] == "Insufficient Comparison").sum())

        kcols = st.columns(4)
        kcols[0].metric("2023 New SKU Events", total_events)
        kcols[1].metric("Analyzed Events", analyzed_events)
        kcols[2].metric("Potential Indications", potential_events)
        kcols[3].metric("Insufficient Comparison", insufficient_events)

        # Relationship view: peer vs segment change.
        plot_data = cannibal.copy()
        plot_data["Segment_Change_%"] = pd.to_numeric(plot_data["Segment_Change_%"], errors="coerce")
        plot_data["Peer_Change_%"] = pd.to_numeric(plot_data["Peer_Change_%"], errors="coerce")
        plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["Segment_Change_%", "Peer_Change_%"]
        )

        if len(plot_data):
            plot_data = plot_data.sort_values("Peer_Change_%")
            fig, ax = plt.subplots(figsize=(11.8, max(5.0, len(plot_data) * 0.62)))

            y = np.arange(len(plot_data))
            h = 0.34
            peer_bars = ax.barh(
                y - h / 2,
                plot_data["Peer_Change_%"],
                height=h,
                label="Comparable SKU",
                alpha=0.9,
            )
            segment_bars = ax.barh(
                y + h / 2,
                plot_data["Segment_Change_%"],
                height=h,
                label="Segment",
                alpha=0.65,
            )

            ax.axvline(0, linestyle="--", linewidth=1.0, alpha=0.55)
            ax.set_yticks(y)
            ax.set_yticklabels(plot_data["New_SKU"])
            ax.set_xlabel("Demand change (%)")
            ax.set_ylabel("New SKU event")
            ax.set_title(
                "Comparable SKU vs segment demand change",
                loc="left",
                fontsize=13,
                fontweight="bold",
            )
            ax.legend(frameon=False, loc="upper right")
            ax.grid(axis="x", alpha=0.18)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            max_abs = max(
                float(plot_data["Peer_Change_%"].abs().max()),
                float(plot_data["Segment_Change_%"].abs().max()),
                1.0,
            )
            ax.set_xlim(-max_abs * 1.18, max_abs * 1.18)

            for bars in [peer_bars, segment_bars]:
                for bar in bars:
                    value = bar.get_width()
                    offset = max_abs * 0.02
                    ax.text(
                        value + offset if value >= 0 else value - offset,
                        bar.get_y() + bar.get_height() / 2,
                        f"{value:.1f}%",
                        va="center",
                        ha="left" if value >= 0 else "right",
                        fontsize=8,
                    )

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            chart_explanation(
                "How to read the comparison",
                "This view compares the change in comparable existing-SKU demand with the broader segment change for each new-SKU event. A peer decline needs to be distinguished from a wider segment decline before it is treated as a potential cannibalization signal.",
            )
        else:
            st.info("No comparable cannibalization events are available under the current result set.")

        section("Event-level findings", "The table preserves the evidence used to classify each 2023 introduction event.")
        display_cols = [
            "New_SKU", "Launch_Date", "Segment", "Category", "Comparable_SKUs",
            "Peer_Change_%", "Segment_Change_%", "Final_Indication"
        ]
        display_cols = [c for c in display_cols if c in cannibal.columns]
        table = cannibal[display_cols].copy()
        table.columns = [
            "New SKU", "Introduction Date", "Segment", "Category", "Comparable SKUs",
            "Peer Change %", "Segment Change %", "Classification"
        ][:len(table.columns)]
        styled_table(table, 2)

        i1, i2 = st.columns(2)
        with i1:
            insight_box(
                "Result",
                "No analyzed event met the preliminary potential-cannibalization criterion. Two events showed small peer-demand declines, but segment demand also declined, so the portfolio evidence was not distinct enough to classify them as a potential cannibalization indication.",
                eyebrow="Portfolio interaction",
            )
        with i2:
            insight_box(
                "Method limitation",
                "Two of the ten 2023 new-SKU events did not have sufficient comparable existing SKUs within the same segment. They are kept as insufficient comparisons instead of being treated as negative evidence.",
                eyebrow="Data discipline",
            )


# ============================================================
# PAGE 5 — DEMAND FORECAST
# ============================================================
elif page == "Demand Forecast":
    page_header(
        "Demand Forecast",
        "Time-aware machine learning forecast for daily portfolio demand, benchmarked against a naive previous-day baseline.",
    )

    page_note(
        "Forecast scope: full historical dataset. The model uses chronological train/test validation with lag, rolling demand, calendar and promotion features. Sidebar commercial filters do not retrain the model."
    )

    mae_improvement = (
        (forecast_metrics["baseline_mae"] - forecast_metrics["ml_mae"])
        / forecast_metrics["baseline_mae"]
        * 100
    )
    rmse_improvement = (
        (forecast_metrics["baseline_rmse"] - forecast_metrics["ml_rmse"])
        / forecast_metrics["baseline_rmse"]
        * 100
    )

    kcols = st.columns(5)
    kcols[0].metric("MAE", f"{forecast_metrics['ml_mae']:,.2f}")
    kcols[1].metric("RMSE", f"{forecast_metrics['ml_rmse']:,.2f}")
    kcols[2].metric("MAPE", pct(forecast_metrics["ml_mape"], 2))
    kcols[3].metric("R²", f"{forecast_metrics['ml_r2']:.4f}")
    kcols[4].metric("MAE Improvement", pct(mae_improvement, 2))

    section("Model performance", "The ML model is judged against a simple previous-day baseline to establish whether the added complexity is useful.")
    comparison = pd.DataFrame({
        "Model": ["Naive Baseline", "HistGradientBoosting"],
        "MAE": [forecast_metrics["baseline_mae"], forecast_metrics["ml_mae"]],
        "RMSE": [forecast_metrics["baseline_rmse"], forecast_metrics["ml_rmse"]],
        "MAPE_%": [forecast_metrics["baseline_mape"], forecast_metrics["ml_mape"]],
    })
    styled_table(comparison, 2)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_results["date"],
            y=forecast_results["units_sold"],
            mode="lines",
            name="Actual",
            line=dict(color=COLORS["brown"], width=2.2),
        ))
        fig.add_trace(go.Scatter(
            x=forecast_results["date"],
            y=forecast_results["predicted_units_sold"],
            mode="lines",
            name="Predicted",
            line=dict(color=COLORS["terracotta"], width=2),
        ))
        fig.update_xaxes(title="Date")
        fig.update_yaxes(title="Units sold", tickformat=",.0f")
        fig.update_layout(title="Actual vs predicted demand")
        render_plot(
            fig,
            "Forecast reading",
            "The model follows the chronological test period and is evaluated on observations that were not used during training. Closer alignment means lower forecast error, but spikes can still be missed.",
            385,
        )

    with c2:
        importance = load_result_csv("forecast_feature_importance.csv")
        if importance.empty:
            importance = pd.DataFrame({
                "Feature": forecast_features,
                "Importance_MAE": 0.0,
            })
        importance = importance.sort_values("Importance_MAE", ascending=True)
        fig = px.bar(
            importance,
            x="Importance_MAE",
            y="Feature",
            orientation="h",
            title="Permutation feature importance",
        )
        fig.update_traces(marker_color=COLORS["sage"])
        fig.update_xaxes(title="Change in error after permutation")
        fig.update_yaxes(title="")
        render_plot(
            fig,
            "What drives the model",
            "Rolling demand features are the strongest contributors in the project results, with the 14-day and 7-day rolling means leading the importance view. Importance reflects predictive contribution, not causality.",
            385,
        )

    section("Forecast diagnostics", "Residual behavior helps identify systematic bias and periods where the model struggles.")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(forecast_results, x="error", nbins=35, title="Forecast error distribution")
        fig.update_traces(marker_color=COLORS["blue"])
        fig.add_vline(x=0, line_dash="dash", line_color=COLORS["brown_dark"])
        fig.update_xaxes(title="Actual − predicted units")
        fig.update_yaxes(title="Frequency")
        render_plot(
            fig,
            "Bias check",
            f"Mean forecast error is {forecast_metrics['mean_error']:,.2f} units. Because the definition is actual minus predicted, the negative average indicates a tendency to over-forecast demand.",
            330,
        )

    with c2:
        fig = px.line(forecast_results, x="date", y="absolute_error", title="Absolute error over time")
        fig.update_traces(line=dict(color=COLORS["gold"], width=2))
        fig.update_xaxes(title="Date")
        fig.update_yaxes(title="Absolute error", tickformat=",.0f")
        render_plot(
            fig,
            "Operational use",
            "Periods with larger absolute error deserve review because they indicate where a planning workflow may need additional context or model recalibration.",
            330,
        )

    insight_box(
        "Model takeaway",
        f"HistGradientBoosting reduced MAE by {mae_improvement:.2f}% and RMSE by {rmse_improvement:.2f}% versus the naive baseline, while achieving a MAPE of {forecast_metrics['ml_mape']:.2f}% on the chronological test period. This supports using the model as a planning aid, with ongoing monitoring rather than treating the forecast as a guaranteed future outcome.",
        eyebrow="Model performance",
    )

    st.download_button(
        "Download Forecast Results",
        data=forecast_results.to_csv(index=False).encode("utf-8"),
        file_name="fmcg_forecast_results.csv",
        mime="text/csv",
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    "<div class='footer-note'>FMCG Smart Launch · Analytics case study built from FMCG Daily Sales Data 2022–2024 · Designed around transparent business questions, traceable methodology and decision support.</div>",
    unsafe_allow_html=True,
)
