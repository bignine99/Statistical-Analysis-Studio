
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

export interface Dataset {
  headers: string[];
  rows: any[];
  fileName: string;
  rowCount: number;
}

export interface InvestigationTask {
  id: string;
  title: string;
  target_columns: string[];
  methodology: string;
  status: 'pending' | 'processing' | 'completed';
}

export interface StatisticalInsight {
  id: string;
  task_title: string;
  finding: string;
  statistical_significance: string;
  interpretation: string;
  recommended_viz: string;
  confidence: number;
  viz_type: 'distribution' | 'correlation' | 'regression' | 'actual_vs_pred' | 'heatmap' | 'outlier' | 'table' | 'categorical' | 'timeseries' | 'pareto' | 'crosstab' | 'anomaly';
  viz_data: any[];
  outliers?: { column: string; row_index: number; value: any; reason: string }[];
  // Extended fields from Python backend
  descriptive_stats?: Record<string, any>;
  normality_test?: Record<string, any>;
  outlier_summary?: Record<string, any>;
  heatmap_data?: any[];
  regression_line?: { slope: number; intercept: number };
  model_summary?: Record<string, any>;
  coefficients?: { variable: string; coef: number; std_err: number; t_stat: number; p_value: number }[];
  vif?: { variable: string; vif: number }[];
  actual_vs_predicted?: { actual: number; predicted: number }[];
  residuals?: any[];
  // Categorical ANOVA results
  anova_results?: { variable: string; n_categories: number; f_statistic: number; p_value: number; eta_squared: number; is_significant: boolean }[];
  category_detail?: Record<string, Record<string, any>>;
  // Time series analysis
  time_analysis?: { time_column: string; slope: number; slope_pvalue: number; r_squared: number; direction: string; is_significant: boolean; n_observations: number; period_range: string };
  period_stats?: Record<string, any>;
  // Monthly analysis
  plotly_monthly?: any;
  monthly_stats?: Record<string, any>;
  monthly_analysis?: { column: string; f_statistic: number; p_value: number; eta_squared: number; is_significant: boolean; n_months: number };
  // Pareto analysis
  pareto_detail?: { rank: number; category: string; sum: number; count: number; mean: number; contribution_pct: number; cumulative_pct: number }[];
  pareto_summary?: { pareto_column: string; n_categories: number; n_80pct: number; concentration_pct: number; top_category: string; top_contribution: number };
  // Crosstab / Chi-square
  crosstab_results?: { var1: string; var2: string; chi2: number; p_value: number; dof: number; cramers_v: number; is_significant: boolean; n_cells: number; contingency_shape: string }[];
  // Anomaly detection
  anomaly_methods?: Record<string, any>;
  outlier_records?: Record<string, any>[];
  var_outlier_summary?: Record<string, any>;
}

export interface PreprocessingSummary {
  missing_values_fixed: number;
  outliers_detected: number;
  duplicates_removed: number;
}

export interface MergedProfile {
  customer_id: string;
  name: string;
  email: string;
  phone: string;
  current_tier: string;
  latest_sentiment: string;
  identified_intent: string;
  confidence_score: number;
  updates_applied: string[];
}

export interface AgentState {
  dataset: Dataset | null;
  tasks: InvestigationTask[];
  insights: StatisticalInsight[];
  logs: string[];
  isExploring: boolean;
  dependentVariable?: string;
}
