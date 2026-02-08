
import React, { useState } from 'react';
import { ConstructionDelayDashboard } from './ConstructionDelayDashboard';
import mockData from '../../mock_analysis.json';
import { LayoutDashboard, FileSpreadsheet, RotateCcw, Share2 } from 'lucide-react';

export default function DashboardPage() {
    const [isRefreshing, setIsRefreshing] = useState(false);

    const handleRefresh = () => {
        setIsRefreshing(true);
        setTimeout(() => setIsRefreshing(false), 1500);
    };

    return (
        <div className="min-h-screen bg-[#F1F5F9] p-10 flex flex-col items-center justify-center">
            {/* Control Bar */}
            <div className="w-full max-w-[1200px] mb-6 flex justify-between items-center text-slate-600">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-slate-900 rounded-lg text-emerald-400">
                        <LayoutDashboard size={20} />
                    </div>
                    <div>
                        <h3 className="font-bold text-slate-900 text-lg tracking-tight">Enterprise Analytics Monitor</h3>
                        <p className="text-xs font-medium text-slate-400 uppercase tracking-widest">Construction Delay Intelligence System</p>
                    </div>
                </div>

                <div className="flex gap-2">
                    <button className="flex items-center gap-2 bg-white border border-slate-200 px-4 py-2 rounded-lg text-xs font-bold hover:bg-slate-50 transition-all">
                        <FileSpreadsheet size={14} /> EXPORT_DATA
                    </button>
                    <button onClick={handleRefresh} className="flex items-center gap-2 bg-white border border-slate-200 px-4 py-2 rounded-lg text-xs font-bold hover:bg-slate-50 transition-all">
                        <RotateCcw size={14} className={isRefreshing ? 'animate-spin' : ''} /> RE-RUN_OLS
                    </button>
                    <button className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg text-xs font-bold hover:bg-blue-700 transition-all shadow-md">
                        <Share2 size={14} /> PUBLISH_REPORT
                    </button>
                </div>
            </div>

            {/* Main Dashboard Wrapper */}
            <div className="w-full max-w-[1200px] bg-white rounded-2xl shadow-2xl overflow-hidden border border-slate-200">
                <ConstructionDelayDashboard data={mockData} />
            </div>

            {/* Analytics Log (Simulated) */}
            <div className="w-full max-w-[1200px] mt-6 bg-[#0F172A] rounded-xl p-4 font-mono text-[10px] text-slate-400 border border-slate-800">
                <div className="flex gap-4">
                    <span className="text-emerald-500 font-bold">[SYSTEM]</span>
                    <span>Statsmodels OLS engine synchronized successfully.</span>
                </div>
                <div className="flex gap-4 mt-1">
                    <span className="text-blue-400 font-bold">[ENGINE]</span>
                    <span>Regression analysis for "Delay_Days" completed with significance level α=0.05.</span>
                </div>
            </div>
        </div>
    );
}
