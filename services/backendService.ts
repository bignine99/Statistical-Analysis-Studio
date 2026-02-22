/**
 * Russell Statistical Engine v2.5
 * Python 백엔드(statsmodels)를 호출하여 실제 통계 분석을 수행하는 서비스
 */

import { Dataset, StatisticalInsight } from "../types";

const BACKEND_URL = "/stat/api";

/**
 * CSV 데이터를 Python 백엔드로 전송하여 실제 통계 분석을 수행한다.
 * statsmodels OLS, scipy, pandas를 사용한 정밀 연산 결과를 반환한다.
 */
export async function analyzeWithBackend(
    file: File,
    dependentVariable?: string
): Promise<{
    success: boolean;
    preprocessing: any;
    executive_summary: any;
    cards: any[];
    dependent_variable: string;
    error?: string;
}> {
    const formData = new FormData();
    formData.append("file", file);
    if (dependentVariable) {
        formData.append("dependent_variable", dependentVariable);
    }

    const response = await fetch(`${BACKEND_URL}/api/analyze`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `서버 오류: ${response.status}`);
    }

    return response.json();
}

/**
 * 백엔드 카드 결과를 프론트엔드 StatisticalInsight 형식으로 변환한다.
 */
export function cardToInsight(card: any): StatisticalInsight & { plotly_chart?: any; plotly_heatmap?: any; plotly_coef_chart?: any } {
    return {
        id: card.id || Math.random().toString(36).substring(7),
        task_title: card.title,
        finding: card.finding,
        statistical_significance: card.statistical_significance,
        interpretation: card.interpretation,
        recommended_viz: card.title,
        confidence: card.confidence || 0.95,
        viz_type: card.viz_type,
        viz_data: card.viz_data || [],
        // 확장 데이터
        descriptive_stats: card.descriptive_stats,
        normality_test: card.normality_test,
        outlier_summary: card.outlier_summary,
        heatmap_data: card.heatmap_data,
        regression_line: card.regression_line,
        model_summary: card.model_summary,
        coefficients: card.coefficients,
        vif: card.vif,
        actual_vs_predicted: card.actual_vs_predicted,
        residuals: card.residuals,
        // Plotly 차트 JSON (plotly skill)
        plotly_chart: card.plotly_chart,
        plotly_heatmap: card.plotly_heatmap,
        plotly_coef_chart: card.plotly_coef_chart,
        // Categorical ANOVA
        anova_results: card.anova_results,
        category_detail: card.category_detail,
        // Time Series
        time_analysis: card.time_analysis,
        period_stats: card.period_stats,
        // Monthly
        plotly_monthly: card.plotly_monthly,
        monthly_stats: card.monthly_stats,
        monthly_analysis: card.monthly_analysis,
        // Pareto
        pareto_detail: card.pareto_detail,
        pareto_summary: card.pareto_summary,
        // Crosstab
        crosstab_results: card.crosstab_results,
        // Anomaly
        anomaly_methods: card.anomaly_methods,
        outlier_records: card.outlier_records,
        var_outlier_summary: card.var_outlier_summary,
    };
}

/**
 * 백엔드 헬스체크
 */
export async function checkBackendHealth(): Promise<boolean> {
    try {
        const res = await fetch(`${BACKEND_URL}/api/health`, {
            method: "GET",
            signal: AbortSignal.timeout(3000),
        });
        return res.ok;
    } catch {
        return false;
    }
}
