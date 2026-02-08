
import React from 'react';
import { AnalysisSummary } from '../../types/dashboard';
import { Activity, Clock, AlertTriangle, CheckCircle2 } from 'lucide-react';

interface DelayDashboardProps {
    data: AnalysisSummary;
}

export function ConstructionDelayDashboard({ data }: DelayDashboardProps) {
    return (
        <div className="w-full aspect-[16/9] bg-[#F8FAFC] text-[#0F172A] p-8 font-['Noto_Sans_KR',sans-serif] flex flex-col gap-6 overflow-hidden border border-slate-200 shadow-sm">
            {/* Header */}
            <div className="flex justify-between items-end border-b border-slate-200 pb-4">
                <div>
                    <h1 className="text-2xl font-bold tracking-tight text-slate-900">공정 지연 요인 분석 대시보드</h1>
                    <p className="text-slate-500 text-sm mt-1">Statsmodels OLS Regression & Plotly Analytics</p>
                </div>
                <div className="flex gap-4">
                    <div className="px-4 py-2 bg-white border border-slate-200 rounded text-center">
                        <span className="block text-[10px] text-slate-400 font-bold uppercase tracking-wider">Model R²</span>
                        <span className="text-lg font-bold text-slate-900">{data.rSquared.toFixed(4)}</span>
                    </div>
                    <div className="px-4 py-2 bg-white border border-slate-200 rounded text-center">
                        <span className="block text-[10px] text-slate-400 font-bold uppercase tracking-wider">Analysis Status</span>
                        <span className="text-sm font-bold text-emerald-600 flex items-center gap-1">
                            <CheckCircle2 size={12} /> VERIFIED
                        </span>
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 grid grid-cols-12 gap-6 overflow-hidden">
                {/* Left Column: Stats & Table */}
                <div className="col-span-4 flex flex-col gap-4">
                    <div className="bg-white p-5 rounded-lg border border-slate-200 shadow-sm flex-1 overflow-y-auto">
                        <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                            <Activity size={14} className="text-blue-500" /> 핵심 영향 요인 (Impact Factors)
                        </h2>
                        <div className="space-y-4">
                            {data.factors.map((f, i) => (
                                <div key={i} className="flex flex-col gap-1 border-b border-slate-50 pb-2">
                                    <div className="flex justify-between items-center">
                                        <span className="text-sm font-bold text-slate-700">{f.factor}</span>
                                        <span className={`text-xs font-mono font-bold ${f.pValue < 0.05 ? 'text-emerald-600' : 'text-slate-400'}`}>
                                            p={f.pValue.toFixed(3)}
                                        </span>
                                    </div>
                                    <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden">
                                        <div
                                            className="bg-blue-500 h-full transition-all duration-1000"
                                            style={{ width: `${Math.min(Math.abs(f.impact) * 20, 100)}%` }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="bg-[#0F172A] p-5 rounded-lg text-white">
                        <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3">Russell AI Summary</h2>
                        <p className="text-xs leading-relaxed text-slate-300 italic">
                            "본 모델의 수정 결정계수는 {data.adjRSquared.toFixed(3)}이며, 기상 요인이 공기 지연에 가장 유의미한 양(+)의 상관관계(p={data.factors.find(f => f.factor.includes('Weather'))?.pValue.toFixed(3)})를 보이고 있습니다."
                        </p>
                    </div>
                </div>

                {/* Right Column: Visualization Placeholder (Plotly) */}
                <div className="col-span-8 bg-white rounded-lg border border-slate-200 shadow-sm relative overflow-hidden flex flex-col">
                    <div className="px-5 py-3 border-b border-slate-50 flex justify-between items-center">
                        <span className="text-xs font-bold text-slate-800">지연 영향도 시각화 (Interactive Chart)</span>
                        <div className="flex gap-2">
                            <div className="w-2 h-2 rounded-full bg-slate-200"></div>
                            <div className="w-2 h-2 rounded-full bg-slate-200"></div>
                        </div>
                    </div>
                    <div className="flex-1 bg-slate-50/50 flex items-center justify-center">
                        {/* 이 부분은 실제 Plotly 차트가 렌더링되거나 iframe/이미지가 들어갈 자리입니다. */}
                        <div className="text-center">
                            <div className="animate-pulse bg-slate-200 w-64 h-32 rounded-lg mb-4 mx-auto" />
                            <p className="text-[11px] text-slate-400 font-mono tracking-widest uppercase">Awaiting Plotly Stream...</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Footer */}
            <div className="mt-auto pt-4 border-t border-slate-100 flex justify-between text-[10px] font-bold text-slate-400 tracking-widest uppercase">
                <div className="flex items-center gap-4">
                    <span>Engine: STAT-AGENT Russell v2.5</span>
                    <span>Mode: Enterprise Professional</span>
                </div>
                <div className="flex items-center gap-4">
                    <span className="flex items-center gap-1"><Clock size={10} /> {new Date().toLocaleTimeString()}</span>
                    <span>LOC: C:/PROJECT/ANALYSIS</span>
                </div>
            </div>
        </div>
    );
}
