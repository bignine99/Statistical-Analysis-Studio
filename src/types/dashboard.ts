
export interface DelayFactor {
    factor: string;
    impact: number;
    pValue: number;
}

export interface AnalysisSummary {
    rSquared: number;
    adjRSquared: number;
    const: number;
    factors: DelayFactor[];
}

export interface DashboardState {
    analysis: AnalysisSummary | null;
    isLoading: boolean;
    error: string | null;
}
