
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# 1. 시뮬레이션 데이터 생성 (공정 지연 요인 분석용)
np.random.seed(42)
n_samples = 200

# 독립 변수: 인력 투입도, 자재 공급 지연(일), 기상 영향도(0-1), 설계 변경 횟수
manpower = np.random.uniform(50, 100, n_samples)
material_delay = np.random.poisson(2, n_samples)
weather_impact = np.random.beta(2, 5, n_samples)
design_changes = np.random.randint(0, 5, n_samples)

# 종속 변수: 최종 공기 지연일 (Delay_Days)
# 모델: Delay = 0.5*Material + 10*Weather + 3*Design - 0.1*Manpower + Error
delay = (0.8 * material_delay + 15 * weather_impact + 4 * design_changes - 0.05 * manpower + 10 + np.random.normal(0, 2, n_samples))
delay = np.maximum(0, delay) # 음수 지연은 없음

df = pd.DataFrame({
    'Manpower_Ratio': manpower,
    'Material_Delay_Days': material_delay,
    'Weather_Impact': weather_impact,
    'Design_Changes': design_changes,
    'Delay_Days': delay
})

# 2. Statsmodels를 이용한 다중 회귀 분석
X = df[['Manpower_Ratio', 'Material_Delay_Days', 'Weather_Impact', 'Design_Changes']]
X = sm.add_constant(X)
y = df['Delay_Days']

model = sm.OLS(y, X).fit()
summary = model.summary()

# 분석 결과 추출 (안정적인 JSON 전달을 위함)
analysis_results = {
    "r_squared": float(model.rsquared),
    "adj_r_squared": float(model.rsquared_adj),
    "p_values": model.pvalues.to_dict(),
    "coefficients": model.params.to_dict(),
    "const": float(model.params['const'])
}

# 3. Plotly를 이용한 인터랙티브 시각화 (영향도 분석)
# 계수(Coefficients) 시각화
coef_df = pd.DataFrame({
    'Factor': ['Const', 'Manpower', 'Material', 'Weather', 'Design'],
    'Impact': model.params.values
})

fig = px.bar(
    coef_df[1:], x='Factor', y='Impact',
    title="공정 관리 지연 요인별 영향도 (Regression Coefficients)",
    labels={'Impact': 'Impact Weight (Days)'},
    color='Impact',
    color_continuous_scale='RdBu_r',
    template='plotly_dark'
)

fig.update_layout(
    font_family="Noto Sans KR",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

# HTML 및 데이터 저장
fig.write_html("delay_analysis.html")
with open('analysis_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(analysis_results, f, indent=2)

print("Statsmodels Analysis Complete.")
print(f"R-squared: {model.rsquared:.4f}")
