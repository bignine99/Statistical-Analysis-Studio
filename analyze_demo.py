
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

# 1. Generate Synthetic Construction Data
np.random.seed(42)
n_samples = 100
area = np.random.uniform(100, 5000, n_samples)  # m2
base_cost = 2000000  # 2M per m2
cost = base_cost * area + np.random.normal(0, 50000000, n_samples)  # Add noise

df = pd.DataFrame({
    'Area_m2': area,
    'Construction_Cost': cost
})

# 2. Regression Analysis using statsmodels
X = sm.add_constant(df['Area_m2'])
y = df['Construction_Cost']
model = sm.OLS(y, X).fit()

print("--- Statsmodels OLS Results ---")
print(model.summary())

# 3. Interactive Visualization using plotly
fig = px.scatter(
    df, x='Area_m2', y='Construction_Cost',
    trendline="ols",
    title="Construction Cost vs Area Analysis (Statsmodels + Plotly)",
    labels={'Area_m2': 'Area (m²)', 'Construction_Cost': 'Cost (KRW)'},
    template="plotly_dark"
)

# Customize layout for premium look
fig.update_layout(
    font_family="Inter",
    title_font_size=24,
    showlegend=True
)

# Save as interactive HTML
fig.write_html("regression_analysis.html")
print("\nSuccess: Interactive visualization saved as 'regression_analysis.html'")
