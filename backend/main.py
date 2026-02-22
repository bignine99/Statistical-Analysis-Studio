"""
STAT-AGENT Russell v2.5 — 통계 분석 백엔드
Skills: statsmodels + plotly + pandas/scipy
실제 통계 연산 + 인터랙티브 시각화 엔진
"""

import io
import json
import traceback
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# ─── numpy 타입 JSON 직렬화 ─────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    """numpy 타입을 Python 기본 타입으로 변환하는 JSON 인코더"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        return super().default(obj)


def safe_json_response(content: dict, status_code: int = 200) -> JSONResponse:
    """numpy 타입이 포함된 dict를 안전하게 JSON 응답으로 변환한다."""
    json_str = json.dumps(content, cls=NumpyEncoder, ensure_ascii=False)
    return JSONResponse(content=json.loads(json_str), status_code=status_code)

app = FastAPI(title="Russell Statistical Engine", version="2.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 색상 팔레트 ──────────────────────────────────────────────────
COLORS = {
    "primary": "#10B981",
    "secondary": "#3B82F6",
    "accent": "#8B5CF6",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "dark": "#0F172A",
    "muted": "#94A3B8",
    "bg": "#FFFFFF",
    "grid": "#F1F5F9",
}

CHART_PALETTE = ["#10B981", "#3B82F6", "#8B5CF6", "#F59E0B", "#EF4444",
                  "#06B6D4", "#EC4899", "#14B8A6", "#F97316", "#6366F1"]

PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, Noto Sans KR, sans-serif", size=11, color="#334155"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=50, r=30, t=50, b=50),
    hoverlabel=dict(
        bgcolor="#0F172A",
        font_size=11,
        font_family="Inter, monospace",
        font_color="white",
    ),
)


# ─── 유틸리티 ────────────────────────────────────────────────────
def safe_float(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), 6)


def fig_to_json(fig) -> dict:
    """Plotly figure를 JSON-serializable dict로 변환한다."""
    return json.loads(fig.to_json())


def series_stats(s: pd.Series) -> dict:
    return {
        "count": int(s.count()),
        "mean": safe_float(s.mean()),
        "std": safe_float(s.std()),
        "min": safe_float(s.min()),
        "q1": safe_float(s.quantile(0.25)),
        "median": safe_float(s.median()),
        "q3": safe_float(s.quantile(0.75)),
        "max": safe_float(s.max()),
        "skewness": safe_float(s.skew()),
        "kurtosis": safe_float(s.kurtosis()),
    }


# ─── Plotly 차트 생성 함수들 ─────────────────────────────────────
def create_distribution_chart(df: pd.DataFrame, numeric_cols: list) -> dict:
    """Card 1: 기술통계 분포 히스토그램 (plotly)"""
    cols = numeric_cols[:6]
    n = len(cols)
    rows = (n + 1) // 2

    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=[f"{c}" for c in cols],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for i, col in enumerate(cols):
        r, c = (i // 2) + 1, (i % 2) + 1
        data = df[col].dropna()

        fig.add_trace(go.Histogram(
            x=data, nbinsx=25,
            marker_color=CHART_PALETTE[i % len(CHART_PALETTE)],
            opacity=0.85,
            name=col,
            showlegend=False,
            hovertemplate=f"{col}: %{{x:.2f}}<br>빈도: %{{y}}<extra></extra>",
        ), row=r, col=c)

        # 평균선
        mean_val = data.mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="#EF4444",
                      line_width=1.5, row=r, col=c,
                      annotation_text=f"μ={mean_val:.1f}", annotation_font_size=9)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="변수별 분포 히스토그램", font=dict(size=14, color="#0F172A")),
        height=max(300, rows * 220),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(gridcolor="#F1F5F9", zeroline=False)

    return fig_to_json(fig)


def create_correlation_chart(dep_correlations: pd.Series, dep_var: str) -> dict:
    """Card 2: 종속변수 상관계수 랭킹 Bar Chart (plotly)"""
    top = dep_correlations.head(10)
    colors = [COLORS["primary"] if v > 0 else COLORS["danger"] for v in top.values]

    fig = go.Figure(go.Bar(
        x=top.values,
        y=top.index,
        orientation="h",
        marker_color=colors,
        text=[f"r={v:.4f}" for v in top.values],
        textposition="outside",
        textfont=dict(size=10, family="JetBrains Mono"),
        hovertemplate="%{y}: r=%{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"'{dep_var}' 상관계수 랭킹 (Pearson r)", font=dict(size=14)),
        xaxis_title="Pearson Correlation Coefficient",
        yaxis=dict(autorange="reversed"),
        height=max(300, len(top) * 45 + 100),
    )
    fig.update_xaxes(gridcolor="#F1F5F9", zeroline=True, zerolinecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#F1F5F9")

    return fig_to_json(fig)


def create_heatmap_chart(corr_matrix: pd.DataFrame) -> dict:
    """Card 2 보조: 상관관계 히트맵 (plotly)"""
    cols = corr_matrix.columns[:8].tolist()
    sub = corr_matrix.loc[cols, cols]

    fig = go.Figure(go.Heatmap(
        z=sub.values,
        x=cols,
        y=cols,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1, zmax=1,
        text=np.round(sub.values, 3),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="%{x} × %{y}: r=%{z:.4f}<extra></extra>",
        colorbar=dict(title="r", thickness=15),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="상관관계 행렬 히트맵", font=dict(size=14)),
        height=max(400, len(cols) * 55 + 100),
        xaxis=dict(tickangle=-45),
    )

    return fig_to_json(fig)


def create_regression_chart(df: pd.DataFrame, x_col: str, y_col: str, model) -> dict:
    """Card 3: 산점도 + 회귀선 (plotly)"""
    x_data = df[x_col].dropna().values
    y_data = df[y_col].dropna().values
    n_sample = min(500, len(x_data))

    if len(x_data) > n_sample:
        idx = np.random.choice(len(x_data), n_sample, replace=False)
        x_sample = x_data[idx]
        y_sample = y_data[idx]
    else:
        x_sample = x_data
        y_sample = y_data

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_sample, y=y_sample,
        mode="markers",
        marker=dict(size=6, color=COLORS["secondary"], opacity=0.5,
                    line=dict(width=0.5, color="white")),
        name="관측값",
        hovertemplate=f"{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
    ))

    # 회귀선
    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    y_line = model.params[0] + model.params[1] * x_line

    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        line=dict(color=COLORS["danger"], width=2.5),
        name=f"OLS (R²={model.rsquared:.4f})",
    ))

    # 신뢰구간 (95%)
    X_pred = sm.add_constant(x_line)
    pred = model.get_prediction(X_pred)
    ci = pred.conf_int(alpha=0.05)

    fig.add_trace(go.Scatter(
        x=np.concatenate([x_line, x_line[::-1]]),
        y=np.concatenate([ci[:, 1], ci[:, 0][::-1]]),
        fill="toself",
        fillcolor="rgba(239,68,68,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=True,
        name="95% 신뢰구간",
    ))

    # R² callout
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=f"<b>R² = {model.rsquared:.4f}</b><br>"
             f"β = {model.params[1]:.4f}<br>"
             f"p = {model.f_pvalue:.2e}",
        showarrow=False,
        font=dict(family="JetBrains Mono", size=11, color="#0F172A"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#E2E8F0",
        borderwidth=1,
        borderpad=8,
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"단순회귀: {x_col} → {y_col}", font=dict(size=14)),
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    fig.update_xaxes(gridcolor="#F1F5F9")
    fig.update_yaxes(gridcolor="#F1F5F9")

    return fig_to_json(fig)


def create_actual_vs_pred_chart(y_actual, y_pred, dep_var: str, model) -> dict:
    """Card 4: 실제값 vs 예측값 + 잔차도 (plotly)"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["실제값 vs 예측값", "잔차 분포"],
        horizontal_spacing=0.1,
    )

    n_sample = min(300, len(y_actual))
    if len(y_actual) > n_sample:
        idx = np.random.choice(len(y_actual), n_sample, replace=False)
    else:
        idx = np.arange(len(y_actual))

    y_a = np.array(y_actual)[idx]
    y_p = np.array(y_pred)[idx]

    # 실제 vs 예측
    fig.add_trace(go.Scatter(
        x=y_a, y=y_p,
        mode="markers",
        marker=dict(size=5, color=COLORS["accent"], opacity=0.6),
        name="관측점",
        hovertemplate=f"실제: %{{x:.2f}}<br>예측: %{{y:.2f}}<extra></extra>",
    ), row=1, col=1)

    # 45도 기준선
    min_val = min(y_a.min(), y_p.min())
    max_val = max(y_a.max(), y_p.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        line=dict(color=COLORS["danger"], width=2, dash="dash"),
        name="완벽 예측선",
        showlegend=True,
    ), row=1, col=1)

    # 잔차 히스토그램
    residuals = model.resid
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=30,
        marker_color=COLORS["warning"],
        opacity=0.8,
        name="잔차",
        showlegend=False,
    ), row=1, col=2)

    # 잔차 정규분포 오버레이
    res_mean = residuals.mean()
    res_std = residuals.std()
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = stats.norm.pdf(x_range, res_mean, res_std) * len(residuals) * (residuals.max() - residuals.min()) / 30

    fig.add_trace(go.Scatter(
        x=x_range, y=y_norm,
        mode="lines",
        line=dict(color=COLORS["danger"], width=2),
        name="정규분포",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"다중회귀 모델 진단 — {dep_var}", font=dict(size=14)),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    fig.update_xaxes(gridcolor="#F1F5F9", title_text="실제값", row=1, col=1)
    fig.update_yaxes(gridcolor="#F1F5F9", title_text="예측값", row=1, col=1)
    fig.update_xaxes(gridcolor="#F1F5F9", title_text="잔차", row=1, col=2)
    fig.update_yaxes(gridcolor="#F1F5F9", title_text="빈도", row=1, col=2)

    return fig_to_json(fig)


def create_coefficient_chart(coefficients: list, dep_var: str) -> dict:
    """보조: 회귀계수 시각화 (plotly)"""
    coefs = [c for c in coefficients if c["variable"] != "const"]
    if not coefs:
        return {}

    vars_sorted = sorted(coefs, key=lambda x: abs(x["coef"] or 0), reverse=True)
    names = [c["variable"] for c in vars_sorted]
    values = [c["coef"] for c in vars_sorted]
    p_vals = [c["p_value"] for c in vars_sorted]
    colors = [COLORS["primary"] if (p or 1) < 0.05 else COLORS["muted"] for p in p_vals]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=colors,
        text=[f"β={v:.4f} (p={p:.3f})" if p else f"β={v:.4f}" for v, p in zip(values, p_vals)],
        textposition="outside",
        textfont=dict(size=9, family="JetBrains Mono"),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"회귀계수 (β) — 종속변수: {dep_var}", font=dict(size=13)),
        xaxis_title="Coefficient (β)",
        height=max(250, len(coefs) * 40 + 100),
        yaxis=dict(autorange="reversed"),
    )
    fig.update_xaxes(gridcolor="#F1F5F9", zeroline=True, zerolinecolor="#CBD5E1")

    return fig_to_json(fig)


def create_categorical_impact_chart(df: pd.DataFrame, cat_col: str, dep_var: str, anova_result: dict) -> dict:
    """범주형 변수 영향도: 상위 7개 변수 Box Plot + η² 바차트"""
    if not isinstance(anova_result, list) or not anova_result:
        return {}

    # 상위 7개 변수 선택
    sorted_results = sorted(anova_result, key=lambda x: x.get("eta_squared", 0) or 0, reverse=True)
    top_vars = sorted_results[:7]
    n_vars = len(top_vars)

    # 레이아웃: 상단에 η² 바차트, 하단에 Box Plot들 (최대 4열)
    box_cols = min(4, n_vars)
    box_rows = (n_vars + box_cols - 1) // box_cols
    total_rows = 1 + box_rows  # 1행: η² 바차트, 나머지: Box Plot

    specs = [[{"colspan": box_cols}, *[None]*(box_cols - 1)]]  # 첫 행: η² bar (전체 너비)
    for _ in range(box_rows):
        specs.append([{} for _ in range(box_cols)])

    subplot_titles = ["Effect Size (η²) — 상위 변수 랭킹"]
    for v in top_vars:
        subplot_titles.append(f"{v['variable'][:25]}")
    # 빈 셀 채우기
    while len(subplot_titles) < 1 + box_rows * box_cols:
        subplot_titles.append("")

    row_heights = [0.25] + [0.75 / box_rows] * box_rows

    fig = make_subplots(
        rows=total_rows, cols=box_cols,
        subplot_titles=subplot_titles,
        specs=specs,
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
        row_heights=row_heights,
    )

    # Row 1: η² 바차트
    names = [r["variable"][:22] for r in sorted_results]
    eta_values = [r.get("eta_squared", 0) or 0 for r in sorted_results]
    sig = [r.get("p_value", 1) or 1 for r in sorted_results]
    colors = [COLORS["primary"] if p < 0.05 else COLORS["muted"] for p in sig]

    fig.add_trace(go.Bar(
        x=eta_values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"η²={v:.4f}" for v in eta_values],
        textposition="outside",
        textfont=dict(size=9, family="JetBrains Mono"),
        showlegend=False,
        hovertemplate="%{y}: η²=%{x:.4f}<extra></extra>",
    ), row=1, col=1)

    # Row 2+: 각 변수별 Box Plot
    for idx, var_info in enumerate(top_vars):
        var_name = var_info["variable"]
        r = 2 + idx // box_cols
        c = 1 + idx % box_cols

        categories = df[var_name].dropna().unique()
        for ci, cat in enumerate(sorted(categories)[:12]):
            cat_data = df[df[var_name] == cat][dep_var].dropna()
            fig.add_trace(go.Box(
                y=cat_data,
                name=str(cat)[:15],
                marker_color=CHART_PALETTE[ci % len(CHART_PALETTE)],
                boxmean="sd",
                showlegend=False,
                hovertemplate=f"{var_name}={cat}<br>{dep_var}: %{{y:.2f}}<extra></extra>",
            ), row=r, col=c)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"범주형 변수 영향도 분석 — {dep_var} (상위 {n_vars}개)", font=dict(size=14)),
        height=max(500, 250 + box_rows * 300),
    )
    fig.update_xaxes(gridcolor="#F1F5F9", row=1, col=1, title_text="η² (Effect Size)")
    fig.update_yaxes(gridcolor="#F1F5F9", row=1, col=1, autorange="reversed")

    # Box Plot 축 설정
    for idx in range(n_vars):
        r = 2 + idx // box_cols
        c = 1 + idx % box_cols
        fig.update_xaxes(gridcolor="#F1F5F9", row=r, col=c)
        fig.update_yaxes(gridcolor="#F1F5F9", row=r, col=c, title_text=dep_var if c == 1 else "")

    return fig_to_json(fig)


def create_timeseries_chart(df: pd.DataFrame, time_col: str, dep_var: str, trend_info: dict) -> dict:
    """시계열 추세 분석 차트: 원본 + 이동평균 + 추세선"""
    df_sorted = df.sort_values(time_col).copy()
    x_vals = df_sorted[time_col].values
    y_vals = df_sorted[dep_var].values

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f"{dep_var} 시계열 추세", "이동평균 vs 원본"],
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4],
    )

    # 원본 데이터
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers",
        marker=dict(color=COLORS["secondary"], size=4, opacity=0.4),
        name="원본 데이터",
        hovertemplate=f"{time_col}: %{{x}}<br>{dep_var}: %{{y:.2f}}<extra></extra>",
    ), row=1, col=1)

    # 이동평균 (window=max(5, len//20))
    window = max(5, len(df_sorted) // 20)
    y_series = pd.Series(y_vals)
    y_ma = y_series.rolling(window=window, center=True, min_periods=1).mean()

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_ma.values,
        mode="lines",
        line=dict(color=COLORS["primary"], width=3),
        name=f"이동평균 (window={window})",
    ), row=1, col=1)

    # OLS 추세선
    try:
        x_numeric = np.arange(len(x_vals)).astype(float)
        mask = ~np.isnan(y_vals.astype(float))
        X_trend = sm.add_constant(x_numeric[mask])
        trend_model = sm.OLS(y_vals[mask].astype(float), X_trend).fit()
        y_trend = trend_model.predict(sm.add_constant(x_numeric))

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_trend,
            mode="lines",
            line=dict(color=COLORS["danger"], width=2, dash="dash"),
            name=f"OLS 추세선 (β={trend_model.params[1]:.4f})",
        ), row=1, col=1)
    except Exception:
        pass

    # 하단: 이동평균 비교
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="lines",
        line=dict(color=COLORS["muted"], width=1),
        name="원본",
        opacity=0.5,
        showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_ma.values,
        mode="lines",
        line=dict(color=COLORS["primary"], width=2),
        name="이동평균",
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"시계열 분석 — {time_col} → {dep_var}", font=dict(size=14)),
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    fig.update_xaxes(gridcolor="#F1F5F9", title_text=time_col, row=1, col=1)
    fig.update_yaxes(gridcolor="#F1F5F9", title_text=dep_var, row=1, col=1)
    fig.update_xaxes(gridcolor="#F1F5F9", title_text=time_col, row=2, col=1)
    fig.update_yaxes(gridcolor="#F1F5F9", title_text=dep_var, row=2, col=1)

    return fig_to_json(fig)


def create_monthly_chart(df: pd.DataFrame, month_col: str, dep_var: str) -> dict:
    """월별 분석 차트: 월별 평균 Bar + Box Plot"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"{month_col} 별 {dep_var} 평균", f"{month_col} 별 {dep_var} 분포"],
        horizontal_spacing=0.1,
    )

    # 월별 그룹 통계
    monthly = df.groupby(month_col)[dep_var].agg(["mean", "std", "count"]).reset_index()
    monthly = monthly.sort_values(month_col)

    # Bar chart — 월별 평균
    fig.add_trace(go.Bar(
        x=monthly[month_col].astype(str),
        y=monthly["mean"],
        marker_color=CHART_PALETTE[:len(monthly)],
        text=[f"{v:.1f}" for v in monthly["mean"]],
        textposition="outside",
        textfont=dict(size=9, family="JetBrains Mono"),
        showlegend=False,
        hovertemplate=f"{month_col}: %{{x}}<br>평균 {dep_var}: %{{y:.2f}}<br>N=%{{customdata}}<extra></extra>",
        customdata=monthly["count"],
    ), row=1, col=1)

    # Box Plot — 월별 분포
    months_sorted = sorted(df[month_col].dropna().unique())
    for i, m in enumerate(months_sorted[:12]):
        m_data = df[df[month_col] == m][dep_var].dropna()
        fig.add_trace(go.Box(
            y=m_data,
            name=str(int(m)) + "월" if isinstance(m, (int, float, np.integer, np.floating)) else str(m),
            marker_color=CHART_PALETTE[i % len(CHART_PALETTE)],
            boxmean="sd",
            showlegend=False,
        ), row=1, col=2)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"월별 분석 — {month_col} → {dep_var}", font=dict(size=14)),
        height=450,
    )
    fig.update_xaxes(gridcolor="#F1F5F9", title_text=month_col, row=1, col=1)
    fig.update_yaxes(gridcolor="#F1F5F9", title_text=dep_var, row=1, col=1)
    fig.update_xaxes(gridcolor="#F1F5F9", row=1, col=2)
    fig.update_yaxes(gridcolor="#F1F5F9", title_text=dep_var, row=1, col=2)

    return fig_to_json(fig)


def create_pareto_chart(df: pd.DataFrame, cat_col: str, dep_var: str) -> dict:
    """파레토 분석 차트: 누적 기여도 Bar + Line"""
    grouped = df.groupby(cat_col)[dep_var].sum().sort_values(ascending=False)
    total = grouped.sum()
    if total == 0:
        return {}

    cumulative_pct = (grouped.cumsum() / total * 100)
    categories = [str(c)[:20] for c in grouped.index]
    values = grouped.values.tolist()
    cum_values = cumulative_pct.values.tolist()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar: 개별 합계
    n_bars = len(categories)
    colors = []
    for i, cp in enumerate(cum_values):
        if cp <= 80:
            colors.append(COLORS["primary"])
        elif i == 0 or cum_values[i-1] <= 80:
            colors.append(COLORS["warning"])
        else:
            colors.append(COLORS["muted"])

    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:,.0f}" for v in values],
        textposition="outside",
        textfont=dict(size=8, family="JetBrains Mono"),
        showlegend=False,
        hovertemplate="%{x}<br>합계: %{y:,.0f}<extra></extra>",
    ), secondary_y=False)

    # Line: 누적 %
    fig.add_trace(go.Scatter(
        x=categories,
        y=cum_values,
        mode="lines+markers",
        line=dict(color=COLORS["danger"], width=2.5),
        marker=dict(size=6, color=COLORS["danger"]),
        showlegend=False,
        hovertemplate="%{x}<br>누적: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)

    # 80% 기준선
    fig.add_hline(y=80, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text="80%", secondary_y=True)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"파레토 분석 — {cat_col} → {dep_var} (누적 기여도)", font=dict(size=14)),
        height=500,
    )
    fig.update_xaxes(gridcolor="#F1F5F9", tickangle=-45)
    fig.update_yaxes(title_text=f"{dep_var} 합계", gridcolor="#F1F5F9", secondary_y=False)
    fig.update_yaxes(title_text="누적 비율 (%)", gridcolor="#F1F5F9",
                     range=[0, 105], secondary_y=True)

    return fig_to_json(fig)


def create_crosstab_chart(contingency: pd.DataFrame, chi2: float, p_val: float,
                          cramers_v: float, var1: str, var2: str) -> dict:
    """교차분석 히트맵: 관측빈도 + 기대빈도 대비"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"관측 빈도 (Observed)", f"표준화 잔차 (Std. Residuals)"],
        horizontal_spacing=0.12,
    )

    # Observed heatmap
    row_labels = [str(r)[:18] for r in contingency.index]
    col_labels = [str(c)[:18] for c in contingency.columns]
    obs_vals = contingency.values.astype(float)

    fig.add_trace(go.Heatmap(
        z=obs_vals,
        x=col_labels,
        y=row_labels,
        colorscale="Blues",
        showscale=False,
        text=[[f"{int(v)}" for v in row] for row in obs_vals],
        texttemplate="%{text}",
        textfont=dict(size=10, family="JetBrains Mono"),
        hovertemplate=f"{var1}: %{{y}}<br>{var2}: %{{x}}<br>빈도: %{{z}}<extra></extra>",
    ), row=1, col=1)

    # Standardized residuals heatmap
    expected = stats.contingency.expected_freq(obs_vals)
    std_resid = (obs_vals - expected) / np.sqrt(expected + 1e-10)

    fig.add_trace(go.Heatmap(
        z=std_resid,
        x=col_labels,
        y=row_labels,
        colorscale="RdBu_r",
        zmid=0,
        showscale=True,
        colorbar=dict(title="Std.Res", x=1.02),
        text=[[f"{v:.1f}" for v in row] for row in std_resid],
        texttemplate="%{text}",
        textfont=dict(size=10, family="JetBrains Mono"),
        hovertemplate=f"{var1}: %{{y}}<br>{var2}: %{{x}}<br>잔차: %{{z:.2f}}<extra></extra>",
    ), row=1, col=2)

    sig_label = "유의" if p_val < 0.05 else "비유의"
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"교차분석 — {var1} × {var2} | χ²={chi2:.2f}, p={p_val:.4f}, V={cramers_v:.3f} ({sig_label})",
            font=dict(size=13),
        ),
        height=420,
    )

    return fig_to_json(fig)


def create_anomaly_chart(df: pd.DataFrame, dep_var: str, numeric_cols: list,
                         outlier_indices: set, method_results: dict) -> dict:
    """이상치 심층 분석 차트: 분포 + 변수별 이상치 비율"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"{dep_var} 이상치 분포", "변수별 이상치 비율 (%)"],
        horizontal_spacing=0.1,
    )

    dep_data = df[dep_var].dropna()
    normal_data = dep_data[~dep_data.index.isin(outlier_indices)]
    outlier_data = dep_data[dep_data.index.isin(outlier_indices)]

    # Left: 정상 vs 이상치 히스토그램
    fig.add_trace(go.Histogram(
        x=normal_data,
        name="정상",
        marker_color=COLORS["primary"],
        opacity=0.7,
        nbinsx=30,
        showlegend=True,
    ), row=1, col=1)

    if len(outlier_data) > 0:
        fig.add_trace(go.Histogram(
            x=outlier_data,
            name=f"이상치 ({len(outlier_data)}건)",
            marker_color=COLORS["danger"],
            opacity=0.8,
            nbinsx=15,
            showlegend=True,
        ), row=1, col=1)

    # Right: 변수별 이상치 비율 bar
    var_outlier_pcts = {}
    for col in numeric_cols[:15]:
        col_data = df[col].dropna()
        if len(col_data) >= 10:
            Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                out_count = ((col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)).sum()
                var_outlier_pcts[col] = float(out_count / len(col_data) * 100)

    if var_outlier_pcts:
        sorted_vars = sorted(var_outlier_pcts.items(), key=lambda x: x[1], reverse=True)
        var_names = [v[0][:22] for v in sorted_vars]
        var_pcts = [v[1] for v in sorted_vars]
        bar_colors = [COLORS["danger"] if p > 5 else COLORS["warning"] if p > 2 else COLORS["primary"] for p in var_pcts]

        fig.add_trace(go.Bar(
            y=var_names,
            x=var_pcts,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{p:.1f}%" for p in var_pcts],
            textposition="outside",
            textfont=dict(size=8, family="JetBrains Mono"),
            showlegend=False,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ), row=1, col=2)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"이상치 심층 분석 — {dep_var} ({len(outlier_indices)}건 탐지)",
                   font=dict(size=14)),
        height=450,
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.25),
    )
    fig.update_xaxes(gridcolor="#F1F5F9", title_text=dep_var, row=1, col=1)
    fig.update_yaxes(gridcolor="#F1F5F9", row=1, col=1)
    fig.update_xaxes(gridcolor="#F1F5F9", title_text="이상치 비율 (%)", row=1, col=2)
    fig.update_yaxes(gridcolor="#F1F5F9", autorange="reversed", row=1, col=2)

    return fig_to_json(fig)


# ─── API 엔드포인트 ──────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    dependent_variable: Optional[str] = Form(None),
):
    """CSV → 4-Card 분석 (statsmodels + plotly)"""
    try:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(content), encoding="cp949")

        # ── 전처리 (수치 데이터 자동 변환 로직 대폭 강화) ─────────────────────
        original_rows = len(df)
        df = df.drop_duplicates()
        
        # 1. 컬럼명 공백 제거 (예: ' 합계_금액 ' -> '합계_금액')
        df.columns = [col.strip() for col in df.columns]

        # 2. 모든 문자열 컬럼에 대해 수치 변환 시도
        for col in df.columns:
            if df[col].dtype == 'object':
                # 숫자가 섞여있는지 샘플 확인
                sample = df[col].dropna().head(20).astype(str)
                # 숫자, 콤마, 공백, 마이너스, 점 외의 문자가 없는지 확인
                if sample.str.contains(r'\d').any():
                    # 수치형 변환 시도: 숫자, 마이너스, 점만 남기고 모두 제거
                    cleaned = df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
                    # 빈 문자열이나 하이픈만 있는 경우 NaN으로 처리
                    cleaned = cleaned.replace(['', '-'], np.nan)
                    temp_numeric = pd.to_numeric(cleaned, errors='coerce')
                    
                    # 변환 결과 중 유효한 숫자가 50% 이상이면 해당 컬럼을 숫자로 확정
                    if temp_numeric.notnull().sum() > (len(df) * 0.5):
                        df[col] = temp_numeric

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # 결측치 처리 (숫자형은 중앙값, 없으면 0)
        for col in numeric_cols:
            if not df[col].empty and df[col].notnull().any():
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)

        dep_var = dependent_variable
        if dep_var and dep_var not in numeric_cols:
            dep_var = None
        if not dep_var and len(numeric_cols) >= 2:
            dep_var = numeric_cols[-1]

        if not dep_var or len(numeric_cols) < 2:
            return JSONResponse(content={
                "error": "분석을 위해 최소 2개의 수치형 컬럼이 필요합니다.",
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
            }, status_code=400)

        indep_cols = [c for c in numeric_cols if c != dep_var]

        # ═══════════════════════════════════════════════════════
        # CARD 1: 기술통계 (statsmodels + plotly)
        # ═══════════════════════════════════════════════════════
        desc_stats = {col: series_stats(df[col]) for col in numeric_cols}

        normality = {}
        for col in numeric_cols:
            col_data = df[col].dropna().values
            if len(col_data) >= 3:
                sample = col_data if len(col_data) <= 5000 else np.random.choice(col_data, 5000, replace=False)
                try:
                    stat_val, p_val = stats.shapiro(sample)
                    normality[col] = {"statistic": safe_float(stat_val), "p_value": safe_float(p_val), "is_normal": bool(p_val > 0.05)}
                except Exception:
                    normality[col] = {"statistic": None, "p_value": None, "is_normal": None}

        outlier_summary = {}
        for col in numeric_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            outlier_count = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
            outlier_summary[col] = {"count": outlier_count, "lower": safe_float(q1 - 1.5 * iqr), "upper": safe_float(q3 + 1.5 * iqr)}

        plotly_distribution = create_distribution_chart(df, numeric_cols)

        # Card 1 상세 요약 구성
        total_outliers = sum(v["count"] for v in outlier_summary.values())
        high_skew_vars = [col for col in numeric_cols if abs(desc_stats[col].get("skewness", 0) or 0) > 1]
        high_kurt_vars = [col for col in numeric_cols if abs(desc_stats[col].get("kurtosis", 0) or 0) > 3]
        normal_count = sum(1 for v in normality.values() if v.get("is_normal"))
        non_normal_count = len(normality) - normal_count

        finding_parts = [
            f"데이터셋 {len(df)}행 × {len(df.columns)}열 (수치형 {len(numeric_cols)}개, 범주형 {len(categorical_cols)}개).",
        ]
        if total_outliers > 0:
            finding_parts.append(f"IQR 기반 이상치 총 {total_outliers}건 탐지.")
        if high_skew_vars:
            finding_parts.append(f"왜도(|skew|>1) 변수: {', '.join(high_skew_vars[:5])}{'...' if len(high_skew_vars) > 5 else ''} ({len(high_skew_vars)}개).")
        if high_kurt_vars:
            finding_parts.append(f"첨도(|kurt|>3) 변수: {', '.join(high_kurt_vars[:5])}{'...' if len(high_kurt_vars) > 5 else ''} ({len(high_kurt_vars)}개).")

        stat_sig_parts = [
            f"Shapiro-Wilk 정규성 검정(α=0.05): 정규 {normal_count}개 / 비정규 {non_normal_count}개.",
            f"이상치(IQR 1.5배): {total_outliers}건.",
        ]

        card1 = {
            "id": "CARD_1",
            "title": "기술통계 및 분포 분석",
            "finding": " ".join(finding_parts),
            "statistical_significance": " ".join(stat_sig_parts),
            "interpretation": f"전체 {len(numeric_cols)}개 수치형 변수에 대해 기술통계량(평균, 표준편차, 사분위수, 왜도, 첨도), Shapiro-Wilk 정규성 검정, IQR 이상치 탐지를 수행하였다. 정규분포를 따르는 변수는 {normal_count}개이며, {non_normal_count}개 변수는 비정규 분포를 보인다.",
            "viz_type": "distribution",
            "plotly_chart": plotly_distribution,
            "descriptive_stats": desc_stats,
            "normality_test": normality,
            "outlier_summary": outlier_summary,
            "confidence": 0.95,
        }

        # ═══════════════════════════════════════════════════════
        # CARD 1B: 범주형 변수 영향도 분석 (ANOVA + η²)
        # ═══════════════════════════════════════════════════════
        card1b = None
        if categorical_cols:
            anova_results = []
            category_detail = {}
            y_data = df[dep_var].dropna().values.astype(float)
            ss_total = float(np.sum((y_data - y_data.mean())**2))

            for cat_col in categorical_cols:
                groups = []
                group_names = []
                cat_stats = {}
                for cat_val in df[cat_col].dropna().unique():
                    grp = df[df[cat_col] == cat_val][dep_var].dropna().values.astype(float)
                    if len(grp) >= 2:
                        groups.append(grp)
                        group_names.append(str(cat_val))
                        cat_stats[str(cat_val)] = {
                            "count": int(len(grp)),
                            "mean": safe_float(np.mean(grp)),
                            "std": safe_float(np.std(grp, ddof=1)),
                            "min": safe_float(np.min(grp)),
                            "max": safe_float(np.max(grp)),
                        }

                if len(groups) >= 2:
                    try:
                        f_stat, p_val = stats.f_oneway(*groups)
                        # η² = SS_between / SS_total
                        grand_mean = np.mean(np.concatenate(groups))
                        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
                        ss_total_grp = sum(np.sum((g - grand_mean)**2) for g in groups)
                        eta_sq = float(ss_between / ss_total_grp) if ss_total_grp > 0 else 0.0

                        anova_results.append({
                            "variable": cat_col,
                            "n_categories": len(groups),
                            "f_statistic": safe_float(f_stat),
                            "p_value": safe_float(p_val),
                            "eta_squared": safe_float(eta_sq),
                            "is_significant": bool(p_val < 0.05) if p_val is not None and not np.isnan(p_val) else False,
                        })
                        category_detail[cat_col] = cat_stats
                    except Exception:
                        pass

            if anova_results:
                anova_results.sort(key=lambda x: x.get("eta_squared", 0) or 0, reverse=True)
                top_cat = anova_results[0]
                sig_count = sum(1 for r in anova_results if r.get("is_significant"))

                # 가장 영향력 큰 범주형 변수로 차트 생성
                top_cat_col = top_cat["variable"]
                plotly_cat = create_categorical_impact_chart(df, top_cat_col, dep_var, anova_results)

                card1b = {
                    "id": "CARD_1B",
                    "title": "범주형 변수 영향도 분석 (ANOVA)",
                    "finding": (
                        f"{len(anova_results)}개 범주형 변수 분석 완료. "
                        f"유의한 변수: {sig_count}개 (α=0.05). "
                        f"최대 효과 크기: '{top_cat['variable']}' (η²={top_cat['eta_squared']}, F={top_cat['f_statistic']}, p={top_cat['p_value']})."
                    ),
                    "statistical_significance": (
                        "One-Way ANOVA 결과: " +
                        ", ".join(f"{r['variable']}(F={r['f_statistic']}, η²={r['eta_squared']}, p={r['p_value']})" for r in anova_results[:5])
                    ),
                    "interpretation": (
                        f"'{top_cat['variable']}' 변수가 '{dep_var}'에 가장 큰 영향력(η²={top_cat['eta_squared']})을 가진다. "
                        f"η²>0.14: 큰 효과, 0.06~0.14: 중간 효과, 0.01~0.06: 작은 효과. "
                        f"전체 {len(anova_results)}개 범주형 변수 중 {sig_count}개가 통계적으로 유의한 차이를 보인다."
                    ),
                    "viz_type": "categorical",
                    "plotly_chart": plotly_cat,
                    "anova_results": anova_results,
                    "category_detail": category_detail,
                    "confidence": 0.95,
                }

        # ═══════════════════════════════════════════════════════
        # CARD 1C: 시계열 분석 (시간 변수 자동탐지 + 추세 + 월별)
        # ═══════════════════════════════════════════════════════
        card1c = None

        # 시간 관련 컬럼 전부 수집
        year_keywords = ['년', '연도', 'year']
        month_keywords = ['월', 'month', '발주월']
        time_keywords_all = year_keywords + month_keywords + ['date', '분기', 'quarter',
                         '시간', 'time', '일', 'day', '기간', 'period', '날짜']

        time_cols_found = []  # (col_name, col_type) — type: 'year', 'month', 'time'
        for col in df.columns:
            col_lower = col.lower().strip()
            if col in numeric_cols or col in categorical_cols:
                if any(kw in col_lower for kw in year_keywords):
                    time_cols_found.append((col, 'year'))
                elif any(kw in col_lower for kw in month_keywords):
                    time_cols_found.append((col, 'month'))
                elif any(kw in col_lower for kw in time_keywords_all):
                    time_cols_found.append((col, 'time'))

        # 수치형 컬럼 중 연도 범위 패턴 탐지 (이미 찾은 컬럼 제외)
        found_names = {c[0] for c in time_cols_found}
        for col in numeric_cols:
            if col == dep_var or col in found_names:
                continue
            col_data = df[col].dropna()
            if len(col_data) > 0:
                col_min, col_max = col_data.min(), col_data.max()
                if 1900 <= col_min <= 2100 and 1900 <= col_max <= 2100 and col_data.nunique() > 3:
                    time_cols_found.append((col, 'year'))
                    break

        # 메인 시간 컬럼 선택 (year > time > month 우선)
        time_col = None
        month_col = None
        for col, ctype in time_cols_found:
            if ctype == 'year' and not time_col:
                time_col = col
            elif ctype == 'month' and not month_col:
                month_col = col
            elif ctype == 'time' and not time_col:
                time_col = col

        # 메인 시간컬럼이 없으면 month를 time_col로도 사용
        if not time_col and month_col:
            time_col = month_col

        if time_col:
            try:
                df_ts = df[[time_col, dep_var]].dropna().copy()
                if time_col not in numeric_cols:
                    try:
                        df_ts[time_col] = pd.to_numeric(df_ts[time_col], errors='coerce')
                        df_ts = df_ts.dropna()
                    except Exception:
                        pass

                if len(df_ts) >= 10:
                    df_ts = df_ts.sort_values(time_col)
                    x_num = np.arange(len(df_ts)).astype(float)
                    y_ts = df_ts[dep_var].values.astype(float)
                    mask_ts = ~np.isnan(y_ts)

                    # OLS 추세 분석
                    X_ts = sm.add_constant(x_num[mask_ts])
                    ts_model = sm.OLS(y_ts[mask_ts], X_ts).fit()

                    slope = float(ts_model.params[1])
                    trend_dir = "상승" if slope > 0 else "하락"
                    slope_p = float(ts_model.pvalues[1]) if len(ts_model.pvalues) > 1 else 1.0

                    # 기간별 통계
                    time_vals = df_ts[time_col].values
                    n_periods = min(5, len(df_ts[time_col].unique()))
                    period_stats = {}
                    period_boundaries = np.linspace(time_vals.min(), time_vals.max(), n_periods + 1)
                    for i in range(n_periods):
                        lo, hi = period_boundaries[i], period_boundaries[i+1]
                        pmask = (time_vals >= lo) & (time_vals <= hi if i == n_periods - 1 else time_vals < hi)
                        pdata = y_ts[pmask]
                        if len(pdata) > 0:
                            label = f"{lo:.0f}~{hi:.0f}"
                            period_stats[label] = {
                                "count": int(len(pdata)),
                                "mean": safe_float(np.mean(pdata)),
                                "std": safe_float(np.std(pdata, ddof=1) if len(pdata) > 1 else 0),
                                "min": safe_float(np.min(pdata)),
                                "max": safe_float(np.max(pdata)),
                            }

                    trend_info = {
                        "time_column": time_col,
                        "slope": safe_float(slope),
                        "slope_pvalue": safe_float(slope_p),
                        "r_squared": safe_float(ts_model.rsquared),
                        "direction": trend_dir,
                        "is_significant": bool(slope_p < 0.05),
                        "n_observations": int(len(df_ts)),
                        "period_range": f"{time_vals.min():.0f} ~ {time_vals.max():.0f}",
                    }

                    plotly_ts = create_timeseries_chart(df_ts, time_col, dep_var, trend_info)

                    # 변동률 계산
                    first_period_mean = y_ts[:max(1, len(y_ts)//5)].mean()
                    last_period_mean = y_ts[-max(1, len(y_ts)//5):].mean()
                    change_pct = ((last_period_mean - first_period_mean) / first_period_mean * 100) if first_period_mean != 0 else 0

                    # ── 월별 분석 (발주월 등) ──
                    plotly_monthly = None
                    monthly_stats = {}
                    monthly_analysis = None
                    if month_col and month_col != time_col:
                        # 별도 월 컬럼이 있는 경우
                        try:
                            df_month = df[[month_col, dep_var]].dropna().copy()
                            if month_col not in numeric_cols:
                                df_month[month_col] = pd.to_numeric(df_month[month_col], errors='coerce')
                                df_month = df_month.dropna()

                            if len(df_month) >= 10:
                                plotly_monthly = create_monthly_chart(df_month, month_col, dep_var)

                                # 월별 통계
                                for m_val in sorted(df_month[month_col].unique()):
                                    m_data = df_month[df_month[month_col] == m_val][dep_var].values
                                    if len(m_data) > 0:
                                        label = f"{int(m_val)}월"
                                        monthly_stats[label] = {
                                            "count": int(len(m_data)),
                                            "mean": safe_float(np.mean(m_data)),
                                            "std": safe_float(np.std(m_data, ddof=1) if len(m_data) > 1 else 0),
                                            "min": safe_float(np.min(m_data)),
                                            "max": safe_float(np.max(m_data)),
                                        }

                                # ANOVA on months
                                month_groups = [df_month[df_month[month_col] == m][dep_var].dropna().values
                                                for m in sorted(df_month[month_col].unique())
                                                if len(df_month[df_month[month_col] == m][dep_var].dropna()) >= 2]
                                if len(month_groups) >= 2:
                                    f_m, p_m = stats.f_oneway(*month_groups)
                                    grand_m = np.mean(np.concatenate(month_groups))
                                    ss_b = sum(len(g) * (np.mean(g) - grand_m)**2 for g in month_groups)
                                    ss_t = sum(np.sum((g - grand_m)**2) for g in month_groups)
                                    eta_m = float(ss_b / ss_t) if ss_t > 0 else 0
                                    monthly_analysis = {
                                        "column": month_col,
                                        "f_statistic": safe_float(f_m),
                                        "p_value": safe_float(p_m),
                                        "eta_squared": safe_float(eta_m),
                                        "is_significant": bool(p_m < 0.05) if not np.isnan(p_m) else False,
                                        "n_months": len(month_groups),
                                    }
                        except Exception as me:
                            print(f"월별 분석 오류: {me}")
                    elif time_col:
                        # time_col 자체가 월 데이터인 경우 (1~12 범위)
                        tc_data = df_ts[time_col].dropna()
                        if tc_data.min() >= 1 and tc_data.max() <= 12 and tc_data.nunique() >= 3:
                            try:
                                plotly_monthly = create_monthly_chart(df_ts, time_col, dep_var)
                                for m_val in sorted(df_ts[time_col].unique()):
                                    m_data = df_ts[df_ts[time_col] == m_val][dep_var].values
                                    if len(m_data) > 0:
                                        label = f"{int(m_val)}월"
                                        monthly_stats[label] = {
                                            "count": int(len(m_data)),
                                            "mean": safe_float(np.mean(m_data)),
                                            "std": safe_float(np.std(m_data, ddof=1) if len(m_data) > 1 else 0),
                                            "min": safe_float(np.min(m_data)),
                                            "max": safe_float(np.max(m_data)),
                                        }
                            except Exception:
                                pass

                    finding_parts = [
                        f"'{time_col}' 기준 시계열 분석 ({trend_info['period_range']}). ",
                        f"전체 {trend_dir} 추세 (기울기={slope:.4f}, p={slope_p:.4f}). ",
                        f"시작 대비 종료 구간 변동률: {change_pct:+.1f}%.",
                    ]
                    if monthly_analysis:
                        finding_parts.append(
                            f" '{month_col}' 월별 ANOVA: F={monthly_analysis['f_statistic']}, "
                            f"η²={monthly_analysis['eta_squared']}, {'유의' if monthly_analysis['is_significant'] else '비유의'}."
                        )

                    card1c = {
                        "id": "CARD_1C",
                        "title": f"시계열 추세 분석 ({time_col}{' + ' + month_col if month_col and month_col != time_col else ''} → {dep_var})",
                        "finding": "".join(finding_parts),
                        "statistical_significance": (
                            f"OLS Trend: β={slope:.6f}, p={slope_p:.4f}, R²={safe_float(ts_model.rsquared)}. "
                            f"추세 {'유의' if slope_p < 0.05 else '비유의'} (α=0.05). N={len(df_ts)}."
                            + (f" 월별 ANOVA: F={monthly_analysis['f_statistic']}, p={monthly_analysis['p_value']}." if monthly_analysis else "")
                        ),
                        "interpretation": (
                            f"'{dep_var}'는 '{time_col}' 기준으로 {trend_dir} 추세를 보이며, "
                            f"{'통계적으로 유의한 변화' if slope_p < 0.05 else '통계적으로 유의하지 않은 변화'}이다. "
                            f"변동률 {change_pct:+.1f}%는 {'실질적 영향' if abs(change_pct) > 10 else '미미한 수준'}으로 판단된다."
                            + (f" 월별({month_col}) 간 평균 차이는 {'유의' if monthly_analysis and monthly_analysis['is_significant'] else '비유의'}하다." if month_col else "")
                        ),
                        "viz_type": "timeseries",
                        "plotly_chart": plotly_ts,
                        "plotly_monthly": plotly_monthly,
                        "time_analysis": trend_info,
                        "period_stats": period_stats,
                        "monthly_stats": monthly_stats if monthly_stats else None,
                        "monthly_analysis": monthly_analysis,
                        "confidence": min(0.95, safe_float(ts_model.rsquared) + 0.5) if ts_model.rsquared else 0.6,
                    }
            except Exception as ts_err:
                print(f"시계열 분석 오류: {ts_err}")
                traceback.print_exc()


        # ═══════════════════════════════════════════════════════
        # CARD 2: 상관분석 (statsmodels + plotly)
        # ═══════════════════════════════════════════════════════
        corr_matrix = df[numeric_cols].corr()
        dep_correlations = corr_matrix[dep_var].drop(dep_var).sort_values(key=abs, ascending=False)

        plotly_correlation = create_correlation_chart(dep_correlations, dep_var)
        plotly_heatmap = create_heatmap_chart(corr_matrix)

        card2 = {
            "id": "CARD_2",
            "title": f"종속변수({dep_var}) 상관관계 분석",
            "finding": f"'{dep_var}'와 가장 높은 상관: '{dep_correlations.index[0]}' (r={safe_float(dep_correlations.iloc[0])})" if len(dep_correlations) > 0 else "",
            "statistical_significance": "Pearson r: " + ", ".join(f"{c}({safe_float(v)})" for c, v in dep_correlations.head(5).items()),
            "interpretation": f"'{dep_correlations.index[0]}'이 회귀 모델의 주요 예측 변수로 활용 가능하다." if len(dep_correlations) > 0 else "",
            "viz_type": "correlation",
            "plotly_chart": plotly_correlation,
            "plotly_heatmap": plotly_heatmap,
            "confidence": 0.95,
        }

        # ═══════════════════════════════════════════════════════
        # CARD 3: 단순회귀 OLS (statsmodels + plotly)
        # ═══════════════════════════════════════════════════════
        top_x = dep_correlations.index[0] if len(dep_correlations) > 0 else indep_cols[0]

        X_simple = sm.add_constant(df[top_x].values.astype(float))
        y = df[dep_var].values.astype(float)
        mask = ~(np.isnan(X_simple).any(axis=1) | np.isnan(y))
        model_simple = sm.OLS(y[mask], X_simple[mask]).fit()

        plotly_regression = create_regression_chart(df, top_x, dep_var, model_simple)

        simple_coefs = [
            {"variable": "const", "coef": safe_float(model_simple.params[0]),
             "std_err": safe_float(model_simple.bse[0]), "t_stat": safe_float(model_simple.tvalues[0]),
             "p_value": safe_float(model_simple.pvalues[0])},
            {"variable": top_x, "coef": safe_float(model_simple.params[1]),
             "std_err": safe_float(model_simple.bse[1]), "t_stat": safe_float(model_simple.tvalues[1]),
             "p_value": safe_float(model_simple.pvalues[1])},
        ]

        card3 = {
            "id": "CARD_3",
            "title": f"단순 회귀: {top_x} → {dep_var}",
            "finding": f"'{top_x}'이 '{dep_var}' 변동의 {safe_float(model_simple.rsquared * 100)}%를 설명 (R²={safe_float(model_simple.rsquared)}).",
            "statistical_significance": f"R²={safe_float(model_simple.rsquared)}, Adj.R²={safe_float(model_simple.rsquared_adj)}, F={safe_float(model_simple.fvalue)}, p={safe_float(model_simple.f_pvalue)}, β={safe_float(model_simple.params[1])}, N={int(model_simple.nobs)}",
            "interpretation": f"'{top_x}' 1단위↑ → '{dep_var}' {safe_float(model_simple.params[1])} 변화. 모델 {'유의' if (model_simple.f_pvalue or 1) < 0.05 else '비유의'}.",
            "viz_type": "regression",
            "plotly_chart": plotly_regression,
            "coefficients": simple_coefs,
            "model_summary": {
                "r_squared": safe_float(model_simple.rsquared),
                "adj_r_squared": safe_float(model_simple.rsquared_adj),
                "f_statistic": safe_float(model_simple.fvalue),
                "f_pvalue": safe_float(model_simple.f_pvalue),
                "aic": safe_float(model_simple.aic),
                "bic": safe_float(model_simple.bic),
                "durbin_watson": safe_float(sm.stats.durbin_watson(model_simple.resid)),
                "n_obs": int(model_simple.nobs),
            },
            "confidence": safe_float(model_simple.rsquared),
        }

        # ═══════════════════════════════════════════════════════
        # CARD 4: 다중회귀 + VIF (statsmodels + plotly)
        # ═══════════════════════════════════════════════════════
        top_vars = dep_correlations.head(min(5, len(dep_correlations))).index.tolist()
        X_multi = sm.add_constant(df[top_vars].values.astype(float))
        mask_m = ~(np.isnan(X_multi).any(axis=1) | np.isnan(y))
        model_multi = sm.OLS(y[mask_m], X_multi[mask_m]).fit()

        # VIF
        vif_data = []
        X_vif = df[top_vars].dropna().values.astype(float)
        for i, cn in enumerate(top_vars):
            try:
                vif_data.append({"variable": cn, "vif": safe_float(variance_inflation_factor(X_vif, i))})
            except Exception:
                vif_data.append({"variable": cn, "vif": None})

        # 계수
        multi_coefs = [{"variable": "const", "coef": safe_float(model_multi.params[0]),
                        "std_err": safe_float(model_multi.bse[0]), "t_stat": safe_float(model_multi.tvalues[0]),
                        "p_value": safe_float(model_multi.pvalues[0])}]
        for i, vn in enumerate(top_vars):
            multi_coefs.append({"variable": vn, "coef": safe_float(model_multi.params[i+1]),
                                "std_err": safe_float(model_multi.bse[i+1]),
                                "t_stat": safe_float(model_multi.tvalues[i+1]),
                                "p_value": safe_float(model_multi.pvalues[i+1])})

        y_pred = model_multi.predict(X_multi[mask_m])
        plotly_avp = create_actual_vs_pred_chart(y[mask_m], y_pred, dep_var, model_multi)
        plotly_coef = create_coefficient_chart(multi_coefs, dep_var)

        card4 = {
            "id": "CARD_4",
            "title": f"다중 회귀: {', '.join(top_vars[:3])}{'...' if len(top_vars)>3 else ''} → {dep_var}",
            "finding": f"{len(top_vars)}개 변수 → '{dep_var}' 변동의 {safe_float(model_multi.rsquared*100)}% 설명 (Adj.R²={safe_float(model_multi.rsquared_adj)}).",
            "statistical_significance": f"R²={safe_float(model_multi.rsquared)}, Adj.R²={safe_float(model_multi.rsquared_adj)}, F={safe_float(model_multi.fvalue)}, p={safe_float(model_multi.f_pvalue)}, AIC={safe_float(model_multi.aic)}, DW={safe_float(sm.stats.durbin_watson(model_multi.resid))}, N={int(model_multi.nobs)}",
            "interpretation": (
                f"다중회귀 모델 설명력 {'향상' if model_multi.rsquared > model_simple.rsquared else '유사'}. "
                f"모델 {'유의' if (model_multi.f_pvalue or 1)<0.05 else '비유의'}. "
                + (f"VIF max={max((v['vif'] or 0) for v in vif_data):.1f} — {'다중공선성 우려' if max((v['vif'] or 0) for v in vif_data)>10 else '허용범위'}." if vif_data else "")
            ),
            "viz_type": "actual_vs_pred",
            "plotly_chart": plotly_avp,
            "plotly_coef_chart": plotly_coef,
            "coefficients": multi_coefs,
            "vif": vif_data,
            "model_summary": {
                "r_squared": safe_float(model_multi.rsquared),
                "adj_r_squared": safe_float(model_multi.rsquared_adj),
                "f_statistic": safe_float(model_multi.fvalue),
                "f_pvalue": safe_float(model_multi.f_pvalue),
                "aic": safe_float(model_multi.aic),
                "bic": safe_float(model_multi.bic),
                "durbin_watson": safe_float(sm.stats.durbin_watson(model_multi.resid)),
                "n_obs": int(model_multi.nobs),
            },
            "confidence": safe_float(model_multi.rsquared),
        }

        # ═══════════════════════════════════════════════════════
        # CARD 5: 파레토 분석 (80/20 Rule)
        # ═══════════════════════════════════════════════════════
        card5 = None
        if categorical_cols:
            try:
                # 가장 적절한 범주형 변수 선택 (카테고리 수가 3~30 사이)
                pareto_col = None
                for cc in categorical_cols:
                    n_unique = df[cc].nunique()
                    if 3 <= n_unique <= 30:
                        pareto_col = cc
                        break
                if not pareto_col:
                    pareto_col = categorical_cols[0]

                pareto_grouped = df.groupby(pareto_col)[dep_var].agg(["sum", "count", "mean"])
                pareto_grouped = pareto_grouped.sort_values("sum", ascending=False)
                total_sum = pareto_grouped["sum"].sum()

                if total_sum > 0:
                    cumsum = pareto_grouped["sum"].cumsum()
                    cum_pct = (cumsum / total_sum * 100)

                    # 80% 도달 카테고리 수
                    n_80 = int((cum_pct <= 80).sum()) + 1
                    n_total = len(pareto_grouped)
                    pct_categories = (n_80 / n_total * 100) if n_total > 0 else 0

                    plotly_pareto = create_pareto_chart(df, pareto_col, dep_var)

                    # 파레토 상세 데이터
                    pareto_detail = []
                    for idx, (cat_name, row) in enumerate(pareto_grouped.iterrows()):
                        pareto_detail.append({
                            "rank": idx + 1,
                            "category": str(cat_name),
                            "sum": safe_float(row["sum"]),
                            "count": int(row["count"]),
                            "mean": safe_float(row["mean"]),
                            "contribution_pct": safe_float(row["sum"] / total_sum * 100),
                            "cumulative_pct": safe_float(cum_pct.iloc[idx]),
                        })

                    card5 = {
                        "id": "CARD_5",
                        "title": f"파레토 분석 — {pareto_col} → {dep_var} (80/20 Rule)",
                        "finding": (
                            f"'{pareto_col}' 기준: 상위 {n_80}개 카테고리 ({pct_categories:.0f}%)가 "
                            f"전체 '{dep_var}'의 80%를 차지한다. "
                            f"1위 '{pareto_grouped.index[0]}' (기여도 {cum_pct.iloc[0]:.1f}%)."
                        ),
                        "statistical_significance": (
                            f"총 {n_total}개 카테고리 분석. "
                            f"상위 {n_80}개가 80% 임계점. "
                            f"집중도 지수: {pct_categories:.1f}% 카테고리 → 80% 기여."
                        ),
                        "interpretation": (
                            f"파레토 원칙(80/20 Rule): '{pareto_col}' 중 소수({n_80}개)가 "
                            f"'{dep_var}'의 대부분을 결정한다. "
                            f"{'높은' if pct_categories < 30 else '보통' if pct_categories < 50 else '낮은'} 집중도를 보인다."
                        ),
                        "viz_type": "pareto",
                        "plotly_chart": plotly_pareto,
                        "pareto_detail": pareto_detail[:20],  # 상위 20개
                        "pareto_summary": {
                            "pareto_column": pareto_col,
                            "n_categories": n_total,
                            "n_80pct": n_80,
                            "concentration_pct": safe_float(pct_categories),
                            "top_category": str(pareto_grouped.index[0]),
                            "top_contribution": safe_float(cum_pct.iloc[0]),
                        },
                        "confidence": 0.9,
                    }
            except Exception as e5:
                print(f"파레토 분석 오류: {e5}")
                traceback.print_exc()

        # ═══════════════════════════════════════════════════════
        # CARD 6: 교차분석 (Chi-square + Cramér's V)
        # ═══════════════════════════════════════════════════════
        card6 = None
        if len(categorical_cols) >= 2:
            try:
                crosstab_results = []
                tested_pairs = set()

                for i, var1 in enumerate(categorical_cols):
                    for j, var2 in enumerate(categorical_cols):
                        if i >= j:
                            continue
                        pair_key = tuple(sorted([var1, var2]))
                        if pair_key in tested_pairs:
                            continue
                        tested_pairs.add(pair_key)

                        # 카테고리 수 제한 (메모리/시각화)
                        if df[var1].nunique() > 15 or df[var2].nunique() > 15:
                            continue

                        contingency = pd.crosstab(df[var1], df[var2])
                        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                            continue

                        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                        # Cramér's V
                        n = contingency.sum().sum()
                        min_dim = min(contingency.shape[0], contingency.shape[1]) - 1
                        cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 and n > 0 else 0

                        crosstab_results.append({
                            "var1": var1,
                            "var2": var2,
                            "chi2": safe_float(chi2),
                            "p_value": safe_float(p_val),
                            "dof": int(dof),
                            "cramers_v": safe_float(cramers_v),
                            "is_significant": bool(p_val < 0.05) if not np.isnan(p_val) else False,
                            "n_cells": int(contingency.shape[0] * contingency.shape[1]),
                            "contingency_shape": f"{contingency.shape[0]}×{contingency.shape[1]}",
                        })

                if crosstab_results:
                    crosstab_results.sort(key=lambda x: x.get("cramers_v", 0) or 0, reverse=True)
                    top_pair = crosstab_results[0]
                    sig_count = sum(1 for r in crosstab_results if r.get("is_significant"))

                    # 가장 강한 관계의 교차표 차트
                    ct = pd.crosstab(df[top_pair["var1"]], df[top_pair["var2"]])
                    plotly_crosstab = create_crosstab_chart(
                        ct, top_pair["chi2"], top_pair["p_value"],
                        top_pair["cramers_v"], top_pair["var1"], top_pair["var2"]
                    )

                    card6 = {
                        "id": "CARD_6",
                        "title": "교차분석 (Chi-square + Cramér's V)",
                        "finding": (
                            f"{len(crosstab_results)}개 범주 쌍 분석. "
                            f"유의한 관계: {sig_count}개. "
                            f"최강 관계: '{top_pair['var1']}' × '{top_pair['var2']}' "
                            f"(V={top_pair['cramers_v']}, χ²={top_pair['chi2']:.1f})."
                        ),
                        "statistical_significance": (
                            "Chi-square: " +
                            ", ".join(
                                f"{r['var1']}×{r['var2']}(χ²={r['chi2']:.1f}, V={r['cramers_v']:.3f}, p={r['p_value']:.4f})"
                                for r in crosstab_results[:5]
                            )
                        ),
                        "interpretation": (
                            f"Cramér's V 기준: "
                            f"'{top_pair['var1']}' × '{top_pair['var2']}' 간 "
                            f"{'강한' if top_pair['cramers_v'] >= 0.3 else '중간' if top_pair['cramers_v'] >= 0.1 else '약한'} "
                            f"관련이 있다 (V={top_pair['cramers_v']}). "
                            f"V≥0.3: 강한, 0.1~0.3: 중간, <0.1: 약한 관련."
                        ),
                        "viz_type": "crosstab",
                        "plotly_chart": plotly_crosstab,
                        "crosstab_results": crosstab_results[:10],
                        "confidence": 0.9,
                    }
            except Exception as e6:
                print(f"교차분석 오류: {e6}")
                traceback.print_exc()

        # ═══════════════════════════════════════════════════════
        # CARD 7: 이상치 심층 분석 (IQR + Z-score)
        # ═══════════════════════════════════════════════════════
        card7 = None
        try:
            dep_data = df[dep_var].dropna()
            if len(dep_data) >= 10:
                # IQR method
                Q1 = dep_data.quantile(0.25)
                Q3 = dep_data.quantile(0.75)
                IQR = Q3 - Q1
                iqr_lower = Q1 - 1.5 * IQR
                iqr_upper = Q3 + 1.5 * IQR
                iqr_outliers = set(dep_data[(dep_data < iqr_lower) | (dep_data > iqr_upper)].index)

                # Z-score method
                z_scores = np.abs(stats.zscore(dep_data.values))
                z_outliers = set(dep_data.index[z_scores > 3])

                # Combine
                all_outliers = iqr_outliers | z_outliers
                both_outliers = iqr_outliers & z_outliers

                method_results = {
                    "iqr": {
                        "count": len(iqr_outliers),
                        "lower_bound": safe_float(iqr_lower),
                        "upper_bound": safe_float(iqr_upper),
                        "pct": safe_float(len(iqr_outliers) / len(dep_data) * 100),
                    },
                    "zscore": {
                        "count": len(z_outliers),
                        "threshold": 3.0,
                        "pct": safe_float(len(z_outliers) / len(dep_data) * 100),
                    },
                    "combined": {
                        "count": len(all_outliers),
                        "both_methods": len(both_outliers),
                        "pct": safe_float(len(all_outliers) / len(dep_data) * 100),
                    },
                }

                plotly_anomaly = create_anomaly_chart(df, dep_var, numeric_cols, all_outliers, method_results)

                # 이상치 상위 10건 데이터
                outlier_records = []
                if all_outliers:
                    outlier_df = df.loc[list(all_outliers)].copy()
                    outlier_df["__outlier_value"] = outlier_df[dep_var]
                    outlier_df = outlier_df.sort_values(dep_var, ascending=False).head(10)
                    for _, row in outlier_df.iterrows():
                        record = {}
                        for col in [dep_var] + [c for c in df.columns[:5] if c != dep_var]:
                            val = row.get(col)
                            if pd.notna(val):
                                record[col] = safe_float(val) if isinstance(val, (int, float, np.integer, np.floating)) else str(val)
                        outlier_records.append(record)

                # 변수별 이상치 비율
                var_outlier_summary = {}
                for col in numeric_cols:
                    col_data = df[col].dropna()
                    if len(col_data) >= 10:
                        cq1, cq3 = col_data.quantile(0.25), col_data.quantile(0.75)
                        ciqr = cq3 - cq1
                        if ciqr > 0:
                            c_out = ((col_data < cq1 - 1.5 * ciqr) | (col_data > cq3 + 1.5 * ciqr)).sum()
                            var_outlier_summary[col] = {
                                "count": int(c_out),
                                "pct": safe_float(c_out / len(col_data) * 100),
                                "total": int(len(col_data)),
                            }

                card7 = {
                    "id": "CARD_7",
                    "title": f"이상치 심층 분석 — {dep_var}",
                    "finding": (
                        f"IQR: {method_results['iqr']['count']}건 ({method_results['iqr']['pct']:.1f}%), "
                        f"Z-score(>3σ): {method_results['zscore']['count']}건 ({method_results['zscore']['pct']:.1f}%). "
                        f"총 {method_results['combined']['count']}건 이상치 탐지 "
                        f"(양 방법 동시: {method_results['combined']['both_methods']}건)."
                    ),
                    "statistical_significance": (
                        f"IQR 범위: [{method_results['iqr']['lower_bound']:.2f}, {method_results['iqr']['upper_bound']:.2f}]. "
                        f"Z-score 임계값: ±3.0σ. "
                        f"전체 N={len(dep_data)}, 이상치 비율={method_results['combined']['pct']:.1f}%."
                    ),
                    "interpretation": (
                        f"'{dep_var}'의 이상치 비율 {method_results['combined']['pct']:.1f}%는 "
                        f"{'높은 수준(>5%)으로 데이터 검증 필요' if method_results['combined']['pct'] > 5 else '보통 수준(2~5%)' if method_results['combined']['pct'] > 2 else '정상 범위(<2%)'}. "
                        f"IQR과 Z-score 양쪽에서 동시 탐지된 {method_results['combined']['both_methods']}건은 "
                        f"강한 이상치로 우선 검토 대상이다."
                    ),
                    "viz_type": "anomaly",
                    "plotly_chart": plotly_anomaly,
                    "anomaly_methods": method_results,
                    "outlier_records": outlier_records,
                    "var_outlier_summary": var_outlier_summary,
                    "confidence": 0.85,
                }
        except Exception as e7:
            print(f"이상치 심층 분석 오류: {e7}")
            traceback.print_exc()

        # ── 응답 ────────────────────────────────────────────────
        return safe_json_response({
            "success": True,
            "file_name": file.filename,
            "preprocessing": {
                "original_rows": int(original_rows),
                "cleaned_rows": int(len(df)),
                "duplicates_removed": int(duplicates_removed),
                "missing_values_fixed": int(missing_before),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
            },
            "dependent_variable": dep_var,
            "executive_summary": {
                "headline": card3["finding"],
                "r_squared_simple": safe_float(model_simple.rsquared),
                "r_squared_multi": safe_float(model_multi.rsquared),
                "top_predictor": top_x,
                "dependent_variable": dep_var,
                "n_observations": int(len(df)),
                "model_significant": bool((model_multi.f_pvalue or 1) < 0.05),
            },
            "cards": [c for c in [card1, card1b, card1c, card2, card3, card4, card5, card6, card7] if c is not None],
        })

    except Exception as e:
        traceback.print_exc()
        return safe_json_response({"success": False, "error": str(e), "traceback": traceback.format_exc()}, status_code=500)


@app.get("/api/health")
async def health():
    return {"status": "ok", "engine": "Russell v2.5", "skills": ["statsmodels", "plotly", "scipy", "pandas"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
