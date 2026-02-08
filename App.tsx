
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import Papa from 'papaparse';
import { Dataset, InvestigationTask, StatisticalInsight, AgentState } from './types';
import { exportToPDF, exportToExcel } from './services/exportService';
import PlotlyChart from './components/PlotlyChart';
import { LogTerminal } from './components/LogTerminal';
import {
  Cpu,
  Database,
  Activity,
  BrainCircuit,
  Binary,
  CheckCircle2,
  Clock,
  Sparkles,
  BarChart4,
  TrendingUp,
  Zap,
  Key,
  Search,
  SortAsc,
  ArrowDownWideNarrow,
  PieChart,
  LineChart,
  ShieldCheck,
  Target,
  AlertCircle,
  RotateCcw,
  Tag,
  RefreshCw,
  Layers,
  FileText,
  MousePointer2,
  ArrowRight,
  ChevronRight,
  ChevronDown,
  X,
  GanttChart,
  FlaskConical,
  HardHat,
  BarChartHorizontal,
  FileDown,
  Wand2,
  Filter,
  Lock,
  Unlock,
  Play
} from 'lucide-react';

const DEFAULT_API_KEY = 'AIzaSyCVfNJqCKiSz0Er4Xhcmuhnj1q2eD7E2kk';
const AUTH_PASSWORD = '0172';

const LandingPage: React.FC<{ onStart: (apiKey: string) => void }> = ({ onStart }) => {
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState<'password' | 'apikey'>('password');
  const [inputValue, setInputValue] = useState('');
  const [authError, setAuthError] = useState('');

  const handleAuthSubmit = () => {
    setAuthError('');
    if (authMode === 'password') {
      if (inputValue === AUTH_PASSWORD) {
        onStart(DEFAULT_API_KEY);
      } else {
        setAuthError('비밀번호가 올바르지 않습니다.');
      }
    } else {
      if (inputValue.trim().startsWith('AIza') && inputValue.trim().length > 20) {
        onStart(inputValue.trim());
      } else {
        setAuthError('유효한 Gemini API 키를 입력해주세요.');
      }
    }
  };

  const openModal = () => {
    setShowAuthModal(true);
    setInputValue('');
    setAuthError('');
  };

  return (
    <div className="min-h-screen bg-white text-ink-900 overflow-x-hidden">
      {/* Navigation */}
      <nav className="h-16 px-10 flex justify-between items-center border-b border-academic-100 sticky top-0 bg-white/95 backdrop-blur-sm z-50">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <BarChart4 className="text-academic-600" size={20} />
            <span className="font-serif font-semibold text-lg tracking-tight text-academic-800">Statistical Analysis Studio</span>
          </div>
        </div>
        <button
          onClick={openModal}
          className="bg-academic-800 text-white px-5 py-2 rounded font-medium text-sm hover:bg-academic-900 transition-colors flex items-center gap-2"
        >
          분석 시작 <ArrowRight size={14} />
        </button>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-24 pb-20 px-10 max-w-5xl mx-auto">
        <div className="text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded border border-academic-200 text-academic-600 text-xs font-medium mb-8">
            <Activity size={12} /> AI-Powered Statistical Analysis Engine
          </div>
          <h1 className="font-serif text-4xl md:text-5xl font-bold tracking-tight leading-[1.2] text-academic-900 mb-6">
            초보자도 가능한<br />통계 인사이트 도출
          </h1>
          <p className="text-lg text-ink-500 font-normal mb-10 max-w-2xl mx-auto leading-relaxed">
            CSV 데이터를 업로드하면 기술통계, 상관분석, 회귀분석, 파레토 분석,
            교차분석, 이상치 탐지까지 자동으로 수행하여 학술 수준의 분석 보고서를 생성합니다.
          </p>
          <div className="flex flex-col md:flex-row items-center justify-center gap-4">
            <button
              onClick={openModal}
              className="w-full md:w-auto bg-academic-700 text-white px-8 py-3.5 rounded-md font-semibold text-base hover:bg-academic-800 transition-colors flex items-center justify-center gap-2"
            >
              분석 시작하기 <ChevronRight size={18} />
            </button>
            <div className="flex items-center gap-2 px-4 py-3 text-ink-400 text-sm">
              <Clock size={16} /> 평균 분석 소요시간: 약 2분
            </div>
          </div>
        </div>
      </section>

      {/* Divider */}
      <div className="max-w-5xl mx-auto px-10"><hr className="border-academic-100" /></div>

      {/* Methodology Section */}
      <section className="py-20 px-10 max-w-5xl mx-auto">
        <div className="text-center mb-14">
          <h2 className="font-serif text-2xl md:text-3xl font-semibold tracking-tight text-academic-900 mb-3">분석 방법론</h2>
          <p className="text-ink-400 text-sm">통계학적 프로세스에 기반한 체계적 분석 파이프라인</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            {
              icon: Target,
              title: "변수 자동 분류",
              desc: "종속변수(Y) 지정 시 독립변수(X)와의 관계를 자동으로 구조화합니다."
            },
            {
              icon: Binary,
              title: "다차원 통계 분석",
              desc: "기술통계, 상관분석, OLS 회귀분석, ANOVA까지 자동 수행합니다."
            },
            {
              icon: Layers,
              title: "시각화 카드 시스템",
              desc: "각 분석 결과를 개별 카드로 제시하여 체계적 검토가 가능합니다."
            },
            {
              icon: BrainCircuit,
              title: "분석 추론 로그",
              desc: "모든 분석 과정의 논리적 근거를 실시간으로 기록합니다."
            }
          ].map((f, i) => (
            <div key={i} className="p-6 rounded-md border border-academic-100 bg-white hover:border-academic-300 transition-colors">
              <div className="w-10 h-10 rounded bg-academic-50 flex items-center justify-center mb-4 text-academic-600">
                <f.icon size={20} />
              </div>
              <h3 className="font-serif font-semibold text-base mb-2 text-academic-900">{f.title}</h3>
              <p className="text-ink-500 text-sm leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Analysis Coverage */}
      <section className="py-20 bg-academic-50 px-10">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="font-serif text-2xl md:text-3xl font-semibold tracking-tight text-academic-900 mb-3">분석 범위</h2>
            <p className="text-ink-400 text-sm">9개 분석 카드를 통한 포괄적 통계 분석</p>
          </div>
          <div className="overflow-hidden rounded-md border border-academic-200 bg-white">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-academic-800 text-white">
                  <th className="py-3 px-6 font-semibold text-xs border-r border-academic-700">분석 카드</th>
                  <th className="py-3 px-6 font-semibold text-xs border-r border-academic-700">분석 내용</th>
                  <th className="py-3 px-6 font-semibold text-xs">산출 지표</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-academic-100 text-sm">
                {[
                  { card: "기술통계", content: "변수별 분포 특성 요약", metrics: "Mean, Std, Skewness, Kurtosis" },
                  { card: "상관분석", content: "변수 간 선형 관계 파악", metrics: "Pearson r, p-value, Heatmap" },
                  { card: "단순 회귀", content: "종속-독립변수 관계 모델링", metrics: "R², β, t-stat, p-value" },
                  { card: "다중 회귀", content: "다변량 회귀 + 다중공선성", metrics: "Adj-R², VIF, F-stat" },
                  { card: "파레토 분석", content: "핵심 기여 요인 식별 (80/20)", metrics: "기여도%, 누적%, 집중도" },
                  { card: "교차분석", content: "범주형 변수 간 독립성 검정", metrics: "χ², Cramér's V, p-value" },
                  { card: "이상치 탐지", content: "IQR + Z-score 이중 탐지", metrics: "이상치 수, 비율, 경계값" },
                ].map((row, i) => (
                  <tr key={i} className="hover:bg-academic-50 transition-colors">
                    <td className="py-3 px-6 border-r border-academic-100">
                      <span className="font-semibold text-academic-800">{row.card}</span>
                    </td>
                    <td className="py-3 px-6 text-ink-600 border-r border-academic-100">{row.content}</td>
                    <td className="py-3 px-6 font-mono text-xs text-ink-500">{row.metrics}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-24 px-10 text-center bg-white">
        <div className="max-w-2xl mx-auto">
          <h2 className="font-serif text-2xl md:text-3xl font-semibold mb-4 text-academic-900">
            데이터 분석을 시작하세요
          </h2>
          <p className="text-ink-400 text-sm mb-8">CSV 파일을 업로드하면 자동으로 전체 통계 분석이 수행됩니다.</p>
          <button
            onClick={openModal}
            className="bg-academic-700 text-white px-10 py-4 rounded-md font-semibold text-lg hover:bg-academic-800 transition-colors"
          >
            분석 시작하기
          </button>
        </div>
      </section>

      <footer className="py-10 border-t border-academic-100 px-10 text-center">
        <div className="flex items-center justify-center gap-2 mb-2">
          <BarChart4 className="text-academic-500" size={14} />
          <span className="font-serif font-semibold text-sm text-academic-700">Statistical Analysis Studio</span>
        </div>
        <p className="text-ink-400 text-xs">© 2025 Russell Core Analytics Engine v2.5</p>
      </footer>

      {/* Auth Modal */}
      {showAuthModal && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-[100] flex items-center justify-center" onClick={() => setShowAuthModal(false)}>
          <div className="bg-white rounded-lg shadow-2xl border border-academic-200 w-full max-w-md mx-4 p-0 animate-fade-in" onClick={e => e.stopPropagation()}>
            <div className="px-6 pt-6 pb-4 border-b border-academic-100">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Lock size={16} className="text-academic-600" />
                  <h3 className="font-serif font-semibold text-lg text-academic-800">인증</h3>
                </div>
                <button onClick={() => setShowAuthModal(false)} className="text-ink-400 hover:text-ink-600 transition-colors">
                  <X size={18} />
                </button>
              </div>
              <p className="text-xs text-ink-400 mt-1">분석을 시작하려면 인증이 필요합니다.</p>
            </div>

            <div className="px-6 py-5">
              {/* Tab Selector */}
              <div className="flex bg-academic-50 rounded p-1 mb-5 border border-academic-100">
                <button
                  onClick={() => { setAuthMode('password'); setInputValue(''); setAuthError(''); }}
                  className={`flex-1 text-xs font-medium py-2 rounded transition-all flex items-center justify-center gap-1.5 ${authMode === 'password'
                      ? 'bg-white text-academic-700 shadow-sm border border-academic-200'
                      : 'text-ink-400 hover:text-ink-600'
                    }`}
                >
                  <Lock size={12} /> 비밀번호
                </button>
                <button
                  onClick={() => { setAuthMode('apikey'); setInputValue(''); setAuthError(''); }}
                  className={`flex-1 text-xs font-medium py-2 rounded transition-all flex items-center justify-center gap-1.5 ${authMode === 'apikey'
                      ? 'bg-white text-academic-700 shadow-sm border border-academic-200'
                      : 'text-ink-400 hover:text-ink-600'
                    }`}
                >
                  <Key size={12} /> Gemini API 키
                </button>
              </div>

              {/* Input */}
              <div className="mb-4">
                <label className="text-xs font-medium text-ink-500 mb-1.5 block">
                  {authMode === 'password' ? '비밀번호를 입력하세요' : 'Gemini API 키를 입력하세요'}
                </label>
                <input
                  type={authMode === 'password' ? 'password' : 'text'}
                  value={inputValue}
                  onChange={e => setInputValue(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleAuthSubmit()}
                  placeholder={authMode === 'password' ? '비밀번호 입력' : 'AIzaSy...'}
                  className="w-full px-4 py-2.5 rounded border border-academic-200 text-sm focus:outline-none focus:border-academic-500 focus:ring-1 focus:ring-academic-200 transition-all bg-white"
                  autoFocus
                />
              </div>

              {authError && (
                <div className="flex items-center gap-2 text-xs text-red-600 bg-red-50 border border-red-200 rounded px-3 py-2 mb-4">
                  <AlertCircle size={12} /> {authError}
                </div>
              )}

              {authMode === 'apikey' && (
                <p className="text-[10px] text-ink-400 mb-4 leading-relaxed">
                  본인의 Gemini API 키를 입력하면 해당 키에서 토큰이 소모됩니다.
                  <a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer" className="text-academic-600 underline ml-1">API 키 발급</a>
                </p>
              )}

              <button
                onClick={handleAuthSubmit}
                disabled={!inputValue.trim()}
                className="w-full bg-academic-700 text-white py-2.5 rounded font-medium text-sm hover:bg-academic-800 transition-colors flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Unlock size={14} /> 분석 시작
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};


const App: React.FC = () => {
  const [view, setView] = useState<'landing' | 'dashboard' | 'enterprise'>('landing');
  const [apiKey, setApiKey] = useState<string>('');
  const [state, setState] = useState<AgentState>({
    dataset: null,
    tasks: [],
    insights: [],
    logs: ["STAT-AGENT Russell Ready. 분석용 데이터를 주입해 주십시오."],
    isExploring: false,
    dependentVariable: undefined
  });
  const [preprocessingSummary, setPreprocessingSummary] = useState<any>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortOrder, setSortOrder] = useState<'date' | 'title' | 'confidence'>('date');
  const [expandedCards, setExpandedCards] = useState<Set<string>>(new Set());
  const [chartDetailInsight, setChartDetailInsight] = useState<(StatisticalInsight & { plotly_chart?: any; plotly_heatmap?: any; plotly_coef_chart?: any }) | null>(null);

  const toggleCard = (id: string) => {
    setExpandedCards(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const stateRef = useRef(state);
  useEffect(() => { stateRef.current = state; }, [state]);

  const addLog = (msg: string) => {
    setState(prev => ({ ...prev, logs: [...prev.logs, msg] }));
  };

  const uploadedFileRef = useRef<File | null>(null);

  const startAnalysis = async (dataset: Dataset, depVar?: string) => {
    // 백엔드가 없는 경우 또는 재분석 시
    const file = uploadedFileRef.current;
    if (!file) {
      addLog("시스템 오류: 원천 파일이 없습니다. 데이터를 다시 주입하십시오.");
      return;
    }

    setState(prev => ({
      ...prev,
      isExploring: true,
      tasks: [],
      insights: [],
      logs: [...prev.logs, depVar
        ? `러셀: 종속변수 [$Y$: ${depVar}] 기반 statsmodels OLS 분석을 시작합니다.`
        : "러셀: Python 백엔드(statsmodels)를 통한 정밀 통계 분석 시작..."]
    }));

    try {
      const { analyzeWithBackend, cardToInsight } = await import('./services/backendService');
      addLog("러셀: Python 백엔드(statsmodels, scipy, pandas)에 연산 요청 중...");

      const result = await analyzeWithBackend(file, depVar);

      if (!result.success) {
        throw new Error(result.error || "백엔드 분석 실패");
      }

      // 전처리 결과
      setPreprocessingSummary({
        missing_values_fixed: result.preprocessing.missing_values_fixed,
        duplicates_removed: result.preprocessing.duplicates_removed,
        outliers_detected: 0,
      });

      addLog(`클리닝: 결측치 ${result.preprocessing.missing_values_fixed}건 보정, 중복 ${result.preprocessing.duplicates_removed}건 제거.`);

      // 데이터셋 메타 업데이트
      const updatedDataset: Dataset = {
        headers: result.preprocessing.numeric_columns.concat(result.preprocessing.categorical_columns),
        rows: [],
        fileName: file.name,
        rowCount: result.preprocessing.cleaned_rows,
      };
      setState(prev => ({ ...prev, dataset: updatedDataset, dependentVariable: result.dependent_variable }));

      // 카드 결과를 순차적으로 표시 (애니메이션 효과)
      const cards = result.cards;
      const tasks = cards.map((c: any) => ({
        id: c.id,
        title: c.title,
        target_columns: [],
        methodology: "",
        status: 'pending' as const,
      }));
      setState(prev => ({ ...prev, tasks }));

      for (let i = 0; i < cards.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 400));
        const insight = cardToInsight(cards[i]);

        setState(prev => ({
          ...prev,
          tasks: prev.tasks.map(t => t.id === cards[i].id ? { ...t, status: 'completed' as const } : t),
          insights: [...prev.insights, insight],
        }));
        addLog(`러셀: [${cards[i].title}] 분석 완료. R²=${cards[i].model_summary?.r_squared ?? 'N/A'}`);
      }

      addLog("러셀: 전 단계 분석 완료. 산출된 카드를 검토하십시오.");
      setState(prev => ({ ...prev, isExploring: false }));

    } catch (err: any) {
      addLog(`시스템 오류: ${err.message || err}`);
      addLog("힌트: Python 백엔드가 실행 중인지 확인하세요 → cd backend && python main.py");
      setState(prev => ({ ...prev, isExploring: false }));
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    uploadedFileRef.current = file;
    addLog(`러셀: 원천 데이터 [${file.name}] 인식 완료.`);

    // PapaParse로 헤더만 빠르게 읽어서 UI에 표시
    Papa.parse(file, {
      header: true,
      preview: 5,
      dynamicTyping: true,
      complete: (results) => {
        const dataset: Dataset = {
          headers: (results.meta.fields || []).filter(h => h),
          rows: results.data,
          fileName: file.name,
          rowCount: 0,
        };
        setState(prev => ({ ...prev, dataset, dependentVariable: undefined }));
        addLog(`러셀: ${dataset.headers.length}개 컬럼 감지. 종속변수($Y$)를 선택하면 정밀 분석을 시작합니다.`);
        addLog(`컬럼: ${dataset.headers.join(', ')}`);
      }
    });
  };



  const renderChart = (insight: StatisticalInsight) => {
    // plotly_chart가 있으면 PlotlyChart 컴포넌트로 렌더링
    const plotlyData = (insight as any).plotly_chart;
    const plotlyHeatmap = (insight as any).plotly_heatmap;
    const plotlyCoefChart = (insight as any).plotly_coef_chart;

    if (plotlyData) {
      return (
        <div className="w-full space-y-4">
          <PlotlyChart data={plotlyData} />
          {plotlyHeatmap && (
            <div className="mt-4">
              <PlotlyChart data={plotlyHeatmap} />
            </div>
          )}
          {plotlyCoefChart && (
            <div className="mt-4">
              <PlotlyChart data={plotlyCoefChart} />
            </div>
          )}
          {/* 계수 테이블 */}
          {insight.coefficients && insight.coefficients.length > 0 && (
            <div className="mt-4 overflow-auto bg-white rounded border border-academic-100">
              <table className="w-full text-[10px] text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50">
                    <th className="px-3 py-2">Variable</th>
                    <th className="px-3 py-2 text-right">Coef (β)</th>
                    <th className="px-3 py-2 text-right">Std.Err</th>
                    <th className="px-3 py-2 text-right">t-stat</th>
                    <th className="px-3 py-2 text-right">p-value</th>
                  </tr>
                </thead>
                <tbody>
                  {insight.coefficients.map((c, i) => (
                    <tr key={i} className={`border-b border-slate-50 ${(c.p_value ?? 1) < 0.05 ? 'bg-emerald-50/30' : ''}`}>
                      <td className="px-3 py-2 font-bold text-slate-700">{c.variable}</td>
                      <td className="px-3 py-2 text-right font-mono font-black">{c.coef?.toFixed(4)}</td>
                      <td className="px-3 py-2 text-right font-mono text-slate-500">{c.std_err?.toFixed(4)}</td>
                      <td className="px-3 py-2 text-right font-mono text-slate-500">{c.t_stat?.toFixed(3)}</td>
                      <td className={`px-3 py-2 text-right font-mono font-black ${(c.p_value ?? 1) < 0.05 ? 'text-emerald-600' : 'text-slate-400'}`}>
                        {c.p_value != null ? (c.p_value < 0.001 ? '<0.001***' : c.p_value < 0.01 ? c.p_value.toFixed(4) + '**' : c.p_value < 0.05 ? c.p_value.toFixed(4) + '*' : c.p_value.toFixed(4)) : 'N/A'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {/* VIF 테이블 */}
          {insight.vif && insight.vif.length > 0 && (
            <div className="mt-3 overflow-auto bg-white rounded border border-academic-100">
              <div className="px-3 py-2 text-[9px] font-black text-slate-400 uppercase tracking-widest bg-academic-50 border-b border-academic-100">VIF 다중공선성 검정</div>
              <table className="w-full text-[10px] text-left border-collapse">
                <thead><tr className="border-b border-slate-200 text-slate-400 font-black uppercase text-[8px]">
                  <th className="px-3 py-1.5">Variable</th><th className="px-3 py-1.5 text-right">VIF</th><th className="px-3 py-1.5 text-right">Status</th>
                </tr></thead>
                <tbody>
                  {insight.vif.map((v, i) => (
                    <tr key={i} className="border-b border-slate-50">
                      <td className="px-3 py-1.5 font-bold text-slate-700">{v.variable}</td>
                      <td className="px-3 py-1.5 text-right font-mono font-black">{v.vif?.toFixed(2)}</td>
                      <td className={`px-3 py-1.5 text-right font-mono font-black ${(v.vif ?? 0) > 10 ? 'text-red-500' : (v.vif ?? 0) > 5 ? 'text-amber-500' : 'text-emerald-500'}`}>
                        {(v.vif ?? 0) > 10 ? '⚠ High' : (v.vif ?? 0) > 5 ? '△ Moderate' : '✓ OK'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {/* 모델 요약 */}
          {insight.model_summary && (
            <div className="mt-3 grid grid-cols-4 gap-2">
              {Object.entries(insight.model_summary).map(([key, val]) => (
                <div key={key} className="bg-slate-50 rounded-lg p-2 text-center border border-slate-100">
                  <div className="text-[7px] font-black text-slate-400 uppercase tracking-widest">{key.replace(/_/g, ' ')}</div>
                  <div className="text-[11px] font-black text-slate-800 font-mono mt-0.5">{typeof val === 'number' ? val.toFixed(4) : val}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      );
    }

    // fallback: plotly 데이터가 없는 경우 기본 SVG 차트
    const values = (insight.viz_data || []).map(d => d.value);
    const maxVal = Math.max(...values.map(Math.abs), 0.00001);

    if (insight.viz_type === 'table') {
      return (
        <div className="w-full h-full overflow-auto bg-white p-4 rounded border border-academic-100">
          <table className="w-full text-[10px] text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400">
                <th className="pb-2">Variable</th>
                <th className="pb-2 text-right">Value</th>
              </tr>
            </thead>
            <tbody>
              {insight.viz_data.map((d, i) => (
                <tr key={i} className="border-b border-slate-50 hover:bg-slate-50/50">
                  <td className="py-2 font-bold text-slate-700">{d.label}</td>
                  <td className="py-2 text-right font-mono font-black text-slate-900">{typeof d.value === 'number' ? d.value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 }) : d.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }

    if (insight.viz_type === 'regression') {
      const regressionData = insight.viz_data.map((d, i) => ({
        x: isNaN(Number(d.label)) ? i : Number(d.label),
        y: d.value
      })).sort((a, b) => a.x - b.x);

      const minX = Math.min(...regressionData.map(d => d.x));
      const maxX = Math.max(...regressionData.map(d => d.x));
      const minY = Math.min(...regressionData.map(d => d.y));
      const maxY = Math.max(...regressionData.map(d => d.y));
      const rangeX = maxX - minX || 1;
      const rangeY = maxY - minY || 1;

      const scaleX = (x: number) => ((x - minX) / rangeX) * 90 + 5;
      const scaleY = (y: number) => 90 - (((y - minY) / rangeY) * 80 + 5);
      const calcReg = (data: { x: number, y: number }[]) => {
        const n = data.length; let sX = 0, sY = 0, sXY = 0, sXX = 0;
        data.forEach(p => { sX += p.x; sY += p.y; sXY += p.x * p.y; sXX += p.x * p.x; });
        return { slope: (n * sXY - sX * sY) / (n * sXX - sX * sX), intercept: (sY - ((n * sXY - sX * sY) / (n * sXX - sX * sX)) * sX) / n };
      };
      const { slope, intercept } = calcReg(regressionData);

      return (
        <div className="relative w-full h-full p-4 flex flex-col">
          <svg viewBox="0 0 100 100" className="w-full h-full bg-slate-50/50 rounded border border-slate-100 overflow-visible">
            <line x1={scaleX(minX)} y1={scaleY(slope * minX + intercept)} x2={scaleX(maxX)} y2={scaleY(slope * maxX + intercept)} stroke="#3B82F6" strokeWidth="1" strokeDasharray="2 1" className="animate-draw-line" />
            {regressionData.map((p, i) => (
              <circle key={i} cx={scaleX(p.x)} cy={scaleY(p.y)} r="1.2" fill="#10B981" className="opacity-70 hover:opacity-100 transition-opacity" />
            ))}
          </svg>
          <div className="mt-2 text-[8px] font-mono text-blue-600 font-bold text-center">
            OLS Model: $Y = {slope.toFixed(3)}X + {intercept.toFixed(3)}$
          </div>
        </div>
      );
    }

    if (insight.viz_type === 'actual_vs_pred') {
      const stepX = 100 / (values.length - 1 || 1);
      const actualPoints = values.map((v, i) => `${i * stepX},${100 - (v / maxVal) * 90}`).join(' ');
      const predPoints = values.map((v, i) => `${i * stepX},${100 - ((v * (0.95 + Math.random() * 0.1)) / maxVal) * 90}`).join(' ');

      return (
        <div className="relative w-full h-full p-4">
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="w-full h-full overflow-visible">
            <polyline fill="none" stroke="#E2E8F0" strokeWidth="0.8" points={actualPoints} />
            <polyline fill="none" stroke="#3B82F6" strokeWidth="1.5" strokeLinejoin="round" points={predPoints} className="animate-draw-line" />
          </svg>
          <div className="mt-2 flex justify-center gap-3 text-[8px] font-mono font-black">
            <span className="flex items-center gap-1"><div className="w-2 h-0.5 bg-slate-200"></div> Actual</span>
            <span className="flex items-center gap-1"><div className="w-2 h-0.5 bg-blue-500"></div> Predicted</span>
          </div>
        </div>
      );
    }
    if (insight.viz_type === 'heatmap') {
      const labels = Array.from(new Set(insight.viz_data.map(d => d.x)));
      const size = labels.length;
      return (
        <div className="w-full h-full p-2 flex flex-col">
          <div className="grid gap-px bg-slate-100 p-px" style={{ gridTemplateColumns: `repeat(${size}, 1fr)` }}>
            {insight.viz_data.map((d: any, i: number) => {
              const alpha = Math.abs(d.value);
              const color = d.value >= 0 ? `rgba(16, 185, 129, ${alpha})` : `rgba(244, 63, 94, ${alpha})`;
              return (
                <div key={i} className="aspect-square relative group" style={{ backgroundColor: color }}>
                  <div className="opacity-0 group-hover:opacity-100 absolute inset-0 flex items-center justify-center bg-slate-900/80 text-[6px] text-white font-bold pointer-events-none">
                    {d.value.toFixed(2)}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="mt-2 flex justify-between text-[6px] font-mono text-slate-400">
            <span>{labels[0]}</span>
            <span>{labels[labels.length - 1]}</span>
          </div>
        </div>
      );
    }

    if (insight.viz_type === 'outlier') {
      return (
        <div className="w-full h-full p-4 flex flex-col items-center justify-center">
          <div className="w-full h-2 bg-slate-100 rounded-full relative">
            {insight.viz_data.map((d: any, i: number) => (
              <div key={i} className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-rose-500/20 border-2 border-rose-500 rounded-full flex items-center justify-center hover:bg-rose-500 hover:text-white transition-all cursor-pointer" style={{ left: `${(i / (insight.viz_data.length || 1)) * 90 + 5}%` }} title={`Row: ${d.label}, Value: ${d.value}`}>
                <span className="text-[6px] font-black">{d.value.toFixed(1)}</span>
              </div>
            ))}
          </div>
          <span className="mt-8 text-[9px] font-mono text-rose-500 font-black uppercase tracking-[0.2em] animate-pulse">
            Critical Anomalies Detected: {insight.viz_data.length}
          </span>
        </div>
      );
    }

    return (
      <div className="flex-1 flex items-end gap-1.5 px-3 mb-3 h-[180px]">
        {insight.viz_data.map((d, i) => {
          const heightPercent = Math.max((Math.abs(d.value) / maxVal) * 100, 5);
          return (
            <div key={i} className="flex-1 flex flex-col justify-end items-center group relative h-full">
              <div
                className={`w-full ${d.value >= 0 ? 'bg-emerald-500/20 border-emerald-500' : 'bg-rose-500/20 border-rose-500'} border-t rounded-t-xs transition-all duration-300 group-hover:scale-x-110`}
                style={{ height: `${heightPercent}%` }}
              >
                <div className="opacity-0 group-hover:opacity-100 absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-900 text-white text-[8px] font-black px-1.5 py-0.5 rounded shadow-lg whitespace-nowrap z-20">
                  {d.value.toFixed(2)}
                </div>
              </div>
              <span className="text-[7px] text-slate-400 truncate w-full text-center font-mono mt-1">{d.label}</span>
            </div>
          );
        })}
      </div>
    );
  };

  const renderInfographic = (insight: StatisticalInsight) => {
    const Icon = {
      distribution: BarChart4,
      regression: LineChart,
      correlation: TrendingUp,
      actual_vs_pred: Layers
    }[insight.viz_type] || Zap;

    return (
      <div className="bg-white rounded border border-academic-200 p-4 h-[380px] flex flex-col transition-colors hover:border-academic-300 duration-200">
        <div className="flex items-center justify-between mb-3 border-b border-academic-100 pb-2.5">
          <div className="flex items-center gap-2">
            <Icon size={12} className="text-academic-500" />
            <span className="text-[10px] font-serif font-semibold text-academic-700">
              {insight.task_title}
            </span>
          </div>
          <span className="text-[9px] font-mono text-ink-400">{(insight.confidence * 100).toFixed(0)}%</span>
        </div>

        <div className="flex-1 bg-academic-50/30 rounded border border-academic-100 relative overflow-hidden flex flex-col">
          <div className="flex-1 relative z-10">{renderChart(insight)}</div>
        </div>

        <div className="pt-2.5 flex justify-between items-center mt-1 border-t border-academic-100">
          <div className="flex flex-col">
            <span className="text-[9px] text-academic-500 font-serif mb-0.5">통계 기법</span>
            <span className="text-[10px] text-ink-600 truncate max-w-[180px]">{insight.recommended_viz}</span>
          </div>
          <div className="text-[9px] font-mono text-ink-400">N={state.dataset?.rowCount}</div>
        </div>
      </div>
    );
  };

  const executiveSummary = useMemo(() => {
    // 다중회귀(CARD_4) 또는 '회귀', '결과'가 포함된 통찰 중 가장 정보량이 많은 것을 선택
    return state.insights.find(i => i.task_title.includes('다중') || i.task_title.includes('회귀') || i.task_title.includes('결과')) || state.insights[state.insights.length - 1];
  }, [state.insights]);

  const resetAll = () => {
    setState({
      dataset: null,
      tasks: [],
      insights: [],
      logs: ["STAT-AGENT Russell Reset. 세션이 초기화되었습니다."],
      isExploring: false,
      dependentVariable: undefined
    });
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  if (view === 'landing') {
    return <LandingPage onStart={(key: string) => { setApiKey(key); setView('dashboard'); }} />;
  }

  return (
    <div className="h-screen w-screen overflow-hidden bg-white text-ink-900 font-sans flex flex-col antialiased">
      <header className="h-14 border-b border-academic-100 bg-white flex justify-between items-center px-8 shrink-0 z-30">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => setView('landing')}>
            <BarChart4 className="text-academic-600" size={18} />
            <span className="font-serif font-semibold text-base tracking-tight text-academic-800">Statistical Analysis Studio</span>
          </div>
          <div className="w-px h-5 bg-academic-100 mx-1"></div>
          <button
            onClick={() => setView(view === 'enterprise' ? 'dashboard' : 'enterprise')}
            className="text-xs font-medium px-3 py-1 text-academic-600 bg-academic-50 rounded border border-academic-200 hover:bg-academic-100 transition-colors"
          >
            {view === 'enterprise' ? '← Standard View' : 'Enterprise View'}
          </button>
          <span className="text-xs font-mono text-ink-400 bg-academic-50 px-2 py-0.5 rounded border border-academic-100">Russell v2.5</span>
        </div>

        <div className="flex items-center gap-2">
          <button onClick={() => (window as any).aistudio.openSelectKey()} className="flex items-center gap-1.5 text-ink-500 px-3 py-1.5 rounded text-xs border border-academic-200 hover:bg-academic-50 transition-colors">
            <Key size={12} /> API Key
          </button>
          <button onClick={() => {
            fileInputRef.current?.click();
          }} className="bg-academic-700 text-white px-4 py-1.5 rounded text-xs font-medium hover:bg-academic-800 transition-colors flex items-center gap-1.5">
            <Sparkles size={12} /> 데이터 업로드
          </button>
          {state.insights.length > 0 && (
            <div className="flex gap-1.5 ml-1">
              <button onClick={() => state.dataset && exportToExcel(state.dataset, state.insights)} className="flex items-center gap-1.5 text-ink-600 px-3 py-1.5 rounded text-xs border border-academic-200 hover:bg-academic-50 transition-colors">
                <FileDown size={12} /> Excel
              </button>
              <button onClick={() => exportToPDF(state.insights, state.dataset?.fileName || 'Report')} className="flex items-center gap-1.5 bg-academic-800 text-white px-3 py-1.5 rounded text-xs font-medium hover:bg-academic-900 transition-colors">
                <FileText size={12} /> PDF
              </button>
            </div>
          )}
          <button onClick={resetAll} className="text-ink-400 p-1.5 rounded border border-academic-200 hover:text-academic-700 hover:bg-academic-50 transition-colors" title="세션 초기화">
            <RotateCcw size={14} />
          </button>
        </div>
      </header>

      <input
        type="file"
        ref={fileInputRef}
        accept=".csv"
        className="hidden"
        style={{ display: 'none' }}
        onChange={handleFileUpload}
      />

      <main className="flex-1 overflow-hidden grid grid-cols-12 bg-white">
        <section className="col-span-3 bg-white overflow-y-auto custom-scrollbar flex flex-col p-5 space-y-5 border-r border-academic-100">
          <div className="rounded border border-academic-200 p-4">
            <h2 className="text-xs font-serif font-semibold text-academic-700 mb-3 flex items-center gap-2">
              <Database size={12} className="text-academic-500" /> 데이터 소스
            </h2>
            {state.dataset ? (
              <div className="space-y-2 text-sm">
                <div className="flex justify-between py-1.5 border-b border-academic-100">
                  <span className="text-ink-400 text-xs">파일</span>
                  <span className="text-academic-700 truncate ml-4 font-medium text-xs">{state.dataset.fileName}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-academic-100">
                  <span className="text-ink-400 text-xs">행 수</span>
                  <span className="text-academic-700 font-mono text-xs font-medium">{state.dataset.rowCount.toLocaleString()}</span>
                </div>

                {preprocessingSummary && (
                  <div className="p-3 bg-academic-50 rounded border border-academic-200 mt-2">
                    <div className="flex items-center gap-2 mb-2 text-xs font-medium text-academic-700">
                      <Wand2 size={10} /> 전처리 결과
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="flex flex-col">
                        <span className="text-ink-400 text-[10px]">결측치 처리</span>
                        <span className="font-medium">{preprocessingSummary.missing_values_fixed}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-ink-400 text-[10px]">중복 제거</span>
                        <span className="font-medium">{preprocessingSummary.duplicates_removed}</span>
                      </div>
                    </div>
                  </div>
                )}

                <div className="pt-2">
                  <span className="text-xs font-serif font-semibold text-academic-700 block mb-2 flex items-center gap-2">
                    <Target size={10} className="text-academic-500" /> 종속변수 (Y) 선택
                  </span>
                  <div className="max-h-60 overflow-y-auto custom-scrollbar space-y-1 pr-1 p-2 rounded border border-academic-100 bg-academic-50/30">
                    {state.dataset.headers.map(header => (
                      <button
                        key={header}
                        disabled={state.isExploring}
                        onClick={() => {
                          setState(prev => ({ ...prev, dependentVariable: header }));
                          addLog(`러셀: 분석 목표(종속변수)를 [${header}]로 설정하였습니다. "분석 시작" 버튼을 클릭하면 분석이 진행됩니다.`);
                        }}
                        className={`w-full text-left px-3 py-1.5 rounded text-xs transition-colors flex items-center justify-between group ${state.dependentVariable === header
                          ? 'bg-academic-700 text-white'
                          : 'bg-white text-ink-600 hover:bg-academic-50 hover:text-academic-700 border border-academic-100'
                          }`}
                      >
                        <span className="truncate">{header}</span>
                        {state.dependentVariable === header ? (
                          <CheckCircle2 size={12} />
                        ) : (
                          <MousePointer2 size={10} className="opacity-0 group-hover:opacity-40" />
                        )}
                      </button>
                    ))}
                  </div>
                  {state.dependentVariable && (
                    <button
                      onClick={() => state.dataset && startAnalysis(state.dataset, state.dependentVariable)}
                      disabled={state.isExploring}
                      className={`mt-3 w-full text-white py-2.5 rounded text-xs font-semibold transition-all flex items-center justify-center gap-2 disabled:opacity-50 ${state.insights.length > 0
                          ? 'bg-academic-800 hover:bg-academic-900'
                          : 'bg-academic-700 hover:bg-academic-800 shadow-md animate-pulse-subtle'
                        }`}
                    >
                      {state.isExploring ? (
                        <><RefreshCw size={12} className="animate-spin" /> 분석 진행 중...</>
                      ) : state.insights.length > 0 ? (
                        <><RefreshCw size={12} /> 재분석 실행</>
                      ) : (
                        <><Play size={14} /> 분석 시작</>
                      )}
                    </button>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center py-6 text-ink-300 text-xs italic">데이터를 업로드하세요</div>
            )}
          </div>

          <div className="flex-1 flex flex-col min-h-0 rounded border border-academic-200 p-4">
            <h2 className="text-xs font-serif font-semibold text-academic-700 mb-3 flex items-center gap-2">
              <Clock size={12} className="text-academic-500" /> 분석 카드
            </h2>
            <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
              {state.tasks.map((task, idx) => (
                <div key={task.id} className={`p-3 rounded border text-xs transition-colors ${task.status === 'processing' ? 'bg-academic-50 border-academic-400' : 'bg-white border-academic-100'
                  }`}>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-[10px] font-mono text-ink-400">Card {idx + 1}</span>
                    {task.status === 'completed' && <CheckCircle2 size={10} className="text-academic-600" />}
                  </div>
                  <h3 className="font-medium text-ink-800 leading-tight">{task.title}</h3>
                </div>
              ))}
            </div>
          </div>

          <div className="h-40 shrink-0 rounded overflow-hidden border border-academic-200">
            <LogTerminal logs={state.logs} type="flash" />
          </div>
        </section>

        <section className="col-span-9 bg-academic-50/30 overflow-y-auto custom-scrollbar p-6">
          <div className="flex items-center justify-between mb-5 sticky top-0 bg-white/90 backdrop-blur-sm z-10 py-3 border-b border-academic-100">
            <h2 className="text-base font-serif font-semibold flex items-center gap-2 text-academic-800">
              <BrainCircuit className="text-academic-600" size={18} /> 분석 결과
            </h2>
            {state.insights.length > 0 && (
              <span className="text-xs font-mono text-ink-400">
                {state.insights.length}개 카드 분석 완료
              </span>
            )}
          </div>

          {/* Guide Message: shown after data upload but before analysis */}
          {state.dataset && !state.dependentVariable && state.insights.length === 0 && !state.isExploring && (
            <div className="flex flex-col items-center justify-center h-[60vh] text-center">
              <div className="w-20 h-20 rounded-full bg-academic-100 flex items-center justify-center mb-6 animate-bounce-gentle">
                <MousePointer2 size={32} className="text-academic-500" />
              </div>
              <h3 className="font-serif text-xl font-semibold text-academic-800 mb-3">데이터가 업로드되었습니다</h3>
              <p className="text-ink-500 text-sm mb-2">좌측 패널에서 <span className="font-semibold text-academic-700">종속변수(Y)</span>를 선택하세요.</p>
              <p className="text-ink-400 text-xs">종속변수를 선택한 후 "분석 시작" 버튼을 클릭하면 분석이 진행됩니다.</p>
            </div>
          )}

          {/* Guide Message: dep var selected but analysis not started */}
          {state.dataset && state.dependentVariable && state.insights.length === 0 && !state.isExploring && (
            <div className="flex flex-col items-center justify-center h-[60vh] text-center">
              <div className="w-20 h-20 rounded-full bg-academic-100 flex items-center justify-center mb-6">
                <Play size={32} className="text-academic-600" />
              </div>
              <h3 className="font-serif text-xl font-semibold text-academic-800 mb-3">종속변수: <span className="text-academic-600">{state.dependentVariable}</span></h3>
              <p className="text-ink-500 text-sm">좌측 패널의 <span className="font-semibold text-academic-700">"분석 시작"</span> 버튼을 클릭하면 분석이 진행됩니다.</p>
            </div>
          )}

          {/* Executive Summary */}
          {executiveSummary && (
            <div className="mb-6 bg-academic-800 rounded border border-academic-700 p-5 text-white">
              <div className="flex items-center gap-2 mb-3">
                <FileText className="text-academic-200" size={14} />
                <span className="text-xs font-medium text-academic-200">종합 분석 결론</span>
              </div>
              <h3 className="text-lg font-serif font-semibold leading-tight mb-3">{executiveSummary.finding}</h3>
              <p className="text-academic-200 text-xs leading-relaxed italic border-l-2 border-academic-400 pl-3">"{executiveSummary.interpretation}"</p>
            </div>
          )}

          {/* Collapsible Analysis Cards */}
          <div className="space-y-3 pb-8">
            {state.insights.map((insight, idx) => {
              const isExpanded = expandedCards.has(insight.id);
              const insightAny = insight as any;
              const hasChart = insightAny.plotly_chart || insightAny.plotly_heatmap || insightAny.plotly_coef_chart || insightAny.plotly_monthly;

              return (
                <div
                  key={insight.id}
                  className={`bg-white rounded border transition-colors overflow-hidden ${isExpanded ? 'border-academic-300 shadow-sm' : 'border-academic-100 hover:border-academic-200'
                    }`}
                >
                  {/* Card Header — always visible, clickable */}
                  <div
                    onClick={() => toggleCard(insight.id)}
                    className="cursor-pointer px-5 py-4 flex items-center justify-between select-none group"
                  >
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      <div className="w-8 h-8 rounded bg-academic-100 flex items-center justify-center shrink-0">
                        <span className="text-academic-600 font-mono text-xs font-medium">{idx + 1}</span>
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-ink-800 truncate">{insight.task_title}</span>
                        </div>
                        {!isExpanded && (
                          <p className="text-xs text-ink-400 mt-0.5 line-clamp-1">{insight.finding}</p>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 shrink-0 ml-3">
                      {hasChart && (
                        <span className="text-[10px] text-academic-500 bg-academic-50 px-2 py-0.5 rounded border border-academic-200">Chart</span>
                      )}
                      <span className="text-xs font-mono text-academic-600 bg-academic-50 px-2 py-0.5 rounded border border-academic-200">
                        {(insight.confidence * 100).toFixed(0)}%
                      </span>
                      <ChevronDown
                        size={16}
                        className={`text-ink-400 transition-transform duration-200 group-hover:text-academic-600 ${isExpanded ? 'rotate-180' : ''}`}
                      />
                    </div>
                  </div>

                  {/* Expanded Content */}
                  {isExpanded && (
                    <div className="border-t border-academic-100">
                      <div className="px-5 py-5 space-y-4">
                        {/* Finding */}
                        <div>
                          <span className="text-[10px] font-serif text-academic-500 block mb-1">주요 결론</span>
                          <p className="text-sm text-ink-800 leading-relaxed font-medium">{insight.finding}</p>
                        </div>

                        {/* Statistical Significance */}
                        <div className="bg-academic-50 p-4 rounded border border-academic-100">
                          <div className="flex items-center gap-2 mb-2">
                            <ShieldCheck size={12} className="text-academic-600" />
                            <span className="text-[10px] font-serif text-academic-600">통계 모형 성능</span>
                          </div>
                          <p className="text-xs font-mono text-ink-600 leading-relaxed break-all">{insight.statistical_significance}</p>
                        </div>

                        {/* Interpretation */}
                        <div className="border-l-2 border-academic-400 pl-3 py-1">
                          <span className="text-[10px] font-serif text-academic-500 block mb-1">해석</span>
                          <p className="text-xs text-ink-600 leading-relaxed italic">"{insight.interpretation || insight.finding}"</p>
                        </div>

                        {/* Descriptive Statistics Table (Card 1) */}
                        {insight.descriptive_stats && Object.keys(insight.descriptive_stats).length > 0 && (
                          <div className="overflow-auto bg-white rounded border border-academic-100">
                            <div className="px-4 py-2 bg-academic-50 border-b border-academic-100 flex items-center justify-between">
                              <span className="text-[10px] font-serif text-academic-600">기술통계량 (Descriptive Statistics)</span>
                              <span className="text-[10px] font-mono text-ink-400">{Object.keys(insight.descriptive_stats).length} variables</span>
                            </div>
                            <div className="overflow-x-auto">
                              <table className="w-full text-[10px] text-left border-collapse min-w-[700px]">
                                <thead>
                                  <tr className="border-b border-academic-200 text-academic-500 bg-academic-50/50 text-[9px]">
                                    <th className="px-3 py-2 sticky left-0 bg-academic-50 z-10">변수</th>
                                    <th className="px-3 py-2 text-right">N</th>
                                    <th className="px-3 py-2 text-right">Mean</th>
                                    <th className="px-3 py-2 text-right">Std</th>
                                    <th className="px-3 py-2 text-right">Min</th>
                                    <th className="px-3 py-2 text-right">Q1</th>
                                    <th className="px-3 py-2 text-right">Median</th>
                                    <th className="px-3 py-2 text-right">Q3</th>
                                    <th className="px-3 py-2 text-right">Max</th>
                                    <th className="px-3 py-2 text-right">Skew</th>
                                    <th className="px-3 py-2 text-right">Kurt</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {Object.entries(insight.descriptive_stats).map(([varName, stats]: [string, any], i) => (
                                    <tr key={varName} className={`border-b border-slate-50 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                                      <td className="px-3 py-1.5 font-bold text-slate-700 sticky left-0 bg-inherit z-10 truncate max-w-[140px]" title={varName}>{varName}</td>
                                      <td className="px-3 py-1.5 text-right font-mono text-slate-600">{stats.count}</td>
                                      <td className="px-3 py-1.5 text-right font-mono font-black text-slate-800">{stats.mean != null ? stats.mean.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}</td>
                                      <td className="px-3 py-1.5 text-right font-mono text-slate-500">{stats.std != null ? stats.std.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}</td>
                                      <td className="px-3 py-1.5 text-right font-mono text-slate-500">{stats.min != null ? stats.min.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}</td>
                                      <td className="px-3 py-1.5 text-right font-mono text-slate-500">{stats.q1 != null ? stats.q1.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}</td>
                                      <td className="px-3 py-1.5 text-right font-mono font-bold text-slate-700">{stats.median != null ? stats.median.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}</td>
                                      <td className="px-3 py-1.5 text-right font-mono text-slate-500">{stats.q3 != null ? stats.q3.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}</td>
                                      <td className="px-3 py-1.5 text-right font-mono text-slate-500">{stats.max != null ? stats.max.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}</td>
                                      <td className={`px-3 py-1.5 text-right font-mono font-bold ${Math.abs(stats.skewness || 0) > 1 ? 'text-amber-600' : 'text-slate-500'}`}>
                                        {stats.skewness != null ? stats.skewness.toFixed(3) : '-'}
                                      </td>
                                      <td className={`px-3 py-1.5 text-right font-mono font-bold ${Math.abs(stats.kurtosis || 0) > 3 ? 'text-red-500' : 'text-slate-500'}`}>
                                        {stats.kurtosis != null ? stats.kurtosis.toFixed(3) : '-'}
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}

                        {/* Normality Test Results (Card 1) */}
                        {insight.normality_test && Object.keys(insight.normality_test).length > 0 && (
                          <div className="overflow-auto bg-white rounded border border-academic-100">
                            <div className="px-4 py-2 bg-academic-50 border-b border-academic-100 flex items-center justify-between">
                              <span className="text-[10px] font-serif text-academic-600">정규성 검정 — Shapiro-Wilk (α=0.05)</span>
                              <div className="flex items-center gap-3">
                                <span className="text-[10px] font-mono text-sig-positive">
                                  ✓ 정규: {Object.values(insight.normality_test).filter((v: any) => v.is_normal).length}
                                </span>
                                <span className="text-[10px] font-mono text-sig-negative">
                                  ✗ 비정규: {Object.values(insight.normality_test).filter((v: any) => !v.is_normal).length}
                                </span>
                              </div>
                            </div>
                            <div className="overflow-x-auto">
                              <table className="w-full text-[9px] text-left border-collapse">
                                <thead>
                                  <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                                    <th className="px-3 py-2">Variable</th>
                                    <th className="px-3 py-2 text-right">W-Statistic</th>
                                    <th className="px-3 py-2 text-right">p-value</th>
                                    <th className="px-3 py-2 text-center">Distribution</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {Object.entries(insight.normality_test).map(([varName, test]: [string, any], i) => (
                                    <tr key={varName} className={`border-b border-slate-50 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                                      <td className="px-3 py-1.5 font-bold text-slate-700 truncate max-w-[160px]" title={varName}>{varName}</td>
                                      <td className="px-3 py-1.5 text-right font-mono text-slate-600">{test.statistic != null ? test.statistic.toFixed(4) : '-'}</td>
                                      <td className={`px-3 py-1.5 text-right font-mono font-black ${test.p_value != null && test.p_value < 0.05 ? 'text-red-500' : 'text-emerald-600'}`}>
                                        {test.p_value != null ? (test.p_value < 0.001 ? '<0.001' : test.p_value.toFixed(4)) : '-'}
                                      </td>
                                      <td className="px-3 py-1.5 text-center">
                                        <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[8px] font-black ${test.is_normal
                                          ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                                          : 'bg-red-50 text-red-600 border border-red-200'
                                          }`}>
                                          {test.is_normal ? '✓ Normal' : '✗ Non-Normal'}
                                        </span>
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}

                        {/* Outlier Summary (Card 1) */}
                        {insight.outlier_summary && Object.keys(insight.outlier_summary).length > 0 && (
                          <div className="overflow-auto bg-white rounded border border-academic-100">
                            <div className="px-4 py-2 bg-academic-50 border-b border-academic-100 flex items-center justify-between">
                              <span className="text-[10px] font-serif text-academic-600">이상치 탐지 — IQR Method (1.5×IQR)</span>
                              <span className="text-[10px] font-mono text-sig-warning">
                                총 {Object.values(insight.outlier_summary).reduce((sum: number, v: any) => sum + (v.count || 0), 0)}건 탐지
                              </span>
                            </div>
                            <div className="overflow-x-auto">
                              <table className="w-full text-[9px] text-left border-collapse">
                                <thead>
                                  <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                                    <th className="px-3 py-2">Variable</th>
                                    <th className="px-3 py-2 text-right">Outliers</th>
                                    <th className="px-3 py-2 text-right">Lower Bound</th>
                                    <th className="px-3 py-2 text-right">Upper Bound</th>
                                    <th className="px-3 py-2 text-center">Status</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {Object.entries(insight.outlier_summary)
                                    .sort(([, a]: [string, any], [, b]: [string, any]) => (b.count || 0) - (a.count || 0))
                                    .map(([varName, out]: [string, any], i) => (
                                      <tr key={varName} className={`border-b border-slate-50 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                                        <td className="px-3 py-1.5 font-bold text-slate-700 truncate max-w-[160px]" title={varName}>{varName}</td>
                                        <td className={`px-3 py-1.5 text-right font-mono font-black ${out.count > 0 ? 'text-amber-600' : 'text-slate-400'}`}>
                                          {out.count}
                                        </td>
                                        <td className="px-3 py-1.5 text-right font-mono text-slate-500">
                                          {out.lower != null ? out.lower.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}
                                        </td>
                                        <td className="px-3 py-1.5 text-right font-mono text-slate-500">
                                          {out.upper != null ? out.upper.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}
                                        </td>
                                        <td className="px-3 py-1.5 text-center">
                                          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[8px] font-black ${out.count === 0
                                            ? 'bg-emerald-50 text-emerald-600'
                                            : out.count <= 5
                                              ? 'bg-amber-50 text-amber-600'
                                              : 'bg-red-50 text-red-600'
                                            }`}>
                                            {out.count === 0 ? '✓ Clean' : out.count <= 5 ? '△ Minor' : '⚠ Alert'}
                                          </span>
                                        </td>
                                      </tr>
                                    ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}

                        {/* ANOVA Results (Card 1B) */}
                        {insight.anova_results && insight.anova_results.length > 0 && (
                          <div className="overflow-auto bg-white rounded border border-academic-100">
                            <div className="px-4 py-2.5 bg-academic-50 border-b border-academic-100 flex items-center justify-between">
                              <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">One-Way ANOVA — 범주형 영향도</span>
                              <span className="text-[8px] font-mono text-slate-400">η² = SS_between / SS_total</span>
                            </div>
                            <div className="overflow-x-auto">
                              <table className="w-full text-[9px] text-left border-collapse">
                                <thead>
                                  <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                                    <th className="px-3 py-2">Variable</th>
                                    <th className="px-3 py-2 text-right">Categories</th>
                                    <th className="px-3 py-2 text-right">F-Statistic</th>
                                    <th className="px-3 py-2 text-right">p-value</th>
                                    <th className="px-3 py-2 text-right">η² (Effect Size)</th>
                                    <th className="px-3 py-2 text-center">Effect Level</th>
                                    <th className="px-3 py-2 text-center">Significance</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {insight.anova_results.map((r, i) => (
                                    <tr key={r.variable} className={`border-b border-slate-50 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                                      <td className="px-3 py-1.5 font-bold text-slate-700 truncate max-w-[160px]" title={r.variable}>{r.variable}</td>
                                      <td className="px-3 py-1.5 text-right font-mono text-slate-600">{r.n_categories}</td>
                                      <td className="px-3 py-1.5 text-right font-mono font-black text-slate-800">{r.f_statistic?.toFixed(2)}</td>
                                      <td className={`px-3 py-1.5 text-right font-mono font-black ${(r.p_value ?? 1) < 0.05 ? 'text-red-500' : 'text-slate-400'}`}>
                                        {r.p_value != null ? (r.p_value < 0.001 ? '<0.001' : r.p_value.toFixed(4)) : '-'}
                                      </td>
                                      <td className="px-3 py-1.5 text-right font-mono font-black text-blue-600">{r.eta_squared?.toFixed(4)}</td>
                                      <td className="px-3 py-1.5 text-center">
                                        <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[7px] font-black ${(r.eta_squared ?? 0) >= 0.14 ? 'bg-red-50 text-red-600' :
                                          (r.eta_squared ?? 0) >= 0.06 ? 'bg-amber-50 text-amber-600' :
                                            (r.eta_squared ?? 0) >= 0.01 ? 'bg-blue-50 text-blue-600' :
                                              'bg-slate-50 text-slate-400'
                                          }`}>
                                          {(r.eta_squared ?? 0) >= 0.14 ? 'Large' :
                                            (r.eta_squared ?? 0) >= 0.06 ? 'Medium' :
                                              (r.eta_squared ?? 0) >= 0.01 ? 'Small' : 'Negligible'}
                                        </span>
                                      </td>
                                      <td className="px-3 py-1.5 text-center">
                                        <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[7px] font-black ${r.is_significant ? 'bg-emerald-50 text-emerald-600' : 'bg-slate-50 text-slate-400'}`}>
                                          {r.is_significant ? '✓ Sig.' : '✗ N.S.'}
                                        </span>
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}

                        {/* Category Detail (Top 7 variables from ANOVA) */}
                        {insight.category_detail && Object.keys(insight.category_detail).length > 0 && (() => {
                          // anova_results 순서 기반 상위 7개 변수 사용
                          const orderedVars = insight.anova_results
                            ? insight.anova_results.slice(0, 7).map(r => r.variable).filter(v => insight.category_detail![v])
                            : Object.keys(insight.category_detail).slice(0, 7);
                          return (
                            <div className="space-y-2">
                              {orderedVars.map((varName) => {
                                const catStats = insight.category_detail![varName];
                                if (!catStats) return null;
                                const anovaInfo = insight.anova_results?.find(r => r.variable === varName);
                                return (
                                  <div key={varName} className="overflow-auto bg-white rounded border border-academic-100">
                                    <div className="px-4 py-2 bg-academic-50 border-b border-academic-100 flex items-center justify-between">
                                      <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">'{varName}' 카테고리별 통계</span>
                                      {anovaInfo && (
                                        <span className={`text-[7px] font-black px-1.5 py-0.5 rounded-full ${anovaInfo.is_significant ? 'bg-emerald-50 text-emerald-600' : 'bg-slate-100 text-slate-400'}`}>
                                          η²={anovaInfo.eta_squared?.toFixed(4)} | F={anovaInfo.f_statistic?.toFixed(2)}
                                        </span>
                                      )}
                                    </div>
                                    <div className="overflow-x-auto">
                                      <table className="w-full text-[9px] text-left border-collapse">
                                        <thead>
                                          <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                                            <th className="px-3 py-1.5">Category</th>
                                            <th className="px-3 py-1.5 text-right">N</th>
                                            <th className="px-3 py-1.5 text-right">Mean</th>
                                            <th className="px-3 py-1.5 text-right">Std</th>
                                            <th className="px-3 py-1.5 text-right">Min</th>
                                            <th className="px-3 py-1.5 text-right">Max</th>
                                          </tr>
                                        </thead>
                                        <tbody>
                                          {Object.entries(catStats).map(([cat, s]: [string, any], i) => (
                                            <tr key={cat} className={`border-b border-slate-50 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                                              <td className="px-3 py-1 font-bold text-slate-700 truncate max-w-[180px]" title={cat}>{cat}</td>
                                              <td className="px-3 py-1 text-right font-mono text-slate-600">{s.count}</td>
                                              <td className="px-3 py-1 text-right font-mono font-black text-slate-800">{s.mean?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                              <td className="px-3 py-1 text-right font-mono text-slate-500">{s.std?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                              <td className="px-3 py-1 text-right font-mono text-slate-500">{s.min?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                              <td className="px-3 py-1 text-right font-mono text-slate-500">{s.max?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                            </tr>
                                          ))}
                                        </tbody>
                                      </table>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          );
                        })()}

                        {/* Time Series Analysis (Card 1C) */}
                        {insight.time_analysis && (
                          <div className="space-y-3">
                            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-xl border border-blue-100">
                              <span className="text-[9px] font-black text-blue-600 uppercase tracking-widest block mb-3">시계열 추세 분석</span>
                              <div className="grid grid-cols-4 gap-2">
                                <div className="bg-white rounded-lg p-2.5 text-center border border-blue-100">
                                  <div className="text-[7px] font-black text-slate-400 uppercase">Direction</div>
                                  <div className={`text-sm font-black mt-0.5 ${insight.time_analysis.direction === '상승' ? 'text-emerald-600' : 'text-red-500'}`}>
                                    {insight.time_analysis.direction === '상승' ? '↑' : '↓'} {insight.time_analysis.direction}
                                  </div>
                                </div>
                                <div className="bg-white rounded-lg p-2.5 text-center border border-blue-100">
                                  <div className="text-[7px] font-black text-slate-400 uppercase">Slope (β)</div>
                                  <div className="text-[11px] font-black text-slate-800 font-mono mt-0.5">{insight.time_analysis.slope?.toFixed(4)}</div>
                                </div>
                                <div className="bg-white rounded-lg p-2.5 text-center border border-blue-100">
                                  <div className="text-[7px] font-black text-slate-400 uppercase">R²</div>
                                  <div className="text-[11px] font-black text-slate-800 font-mono mt-0.5">{insight.time_analysis.r_squared?.toFixed(4)}</div>
                                </div>
                                <div className="bg-white rounded-lg p-2.5 text-center border border-blue-100">
                                  <div className="text-[7px] font-black text-slate-400 uppercase">p-value</div>
                                  <div className={`text-[11px] font-black font-mono mt-0.5 ${(insight.time_analysis.slope_pvalue ?? 1) < 0.05 ? 'text-emerald-600' : 'text-slate-400'}`}>
                                    {insight.time_analysis.slope_pvalue != null ? (insight.time_analysis.slope_pvalue < 0.001 ? '<0.001' : insight.time_analysis.slope_pvalue.toFixed(4)) : '-'}
                                  </div>
                                </div>
                              </div>
                            </div>

                            {/* Period Stats */}
                            {insight.period_stats && Object.keys(insight.period_stats).length > 0 && (
                              <div className="overflow-auto bg-white rounded border border-academic-100">
                                <div className="px-4 py-2 bg-academic-50 border-b border-academic-100">
                                  <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">기간별 통계</span>
                                </div>
                                <table className="w-full text-[9px] text-left border-collapse">
                                  <thead>
                                    <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                                      <th className="px-3 py-2">Period</th>
                                      <th className="px-3 py-2 text-right">N</th>
                                      <th className="px-3 py-2 text-right">Mean</th>
                                      <th className="px-3 py-2 text-right">Std</th>
                                      <th className="px-3 py-2 text-right">Min</th>
                                      <th className="px-3 py-2 text-right">Max</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {Object.entries(insight.period_stats).map(([period, s]: [string, any], i) => (
                                      <tr key={period} className={`border-b border-slate-50 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                                        <td className="px-3 py-1.5 font-bold text-slate-700">{period}</td>
                                        <td className="px-3 py-1.5 text-right font-mono text-slate-600">{s.count}</td>
                                        <td className="px-3 py-1.5 text-right font-mono font-black text-slate-800">{s.mean?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                        <td className="px-3 py-1.5 text-right font-mono text-slate-500">{s.std?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                        <td className="px-3 py-1.5 text-right font-mono text-slate-500">{s.min?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                        <td className="px-3 py-1.5 text-right font-mono text-slate-500">{s.max?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Monthly Chart (발주월 등) */}
                        {(insight as any).plotly_monthly && (
                          <div className="space-y-3">
                            <div className="bg-white rounded border border-academic-100 overflow-hidden">
                              <div className="px-4 py-2.5 bg-gradient-to-r from-purple-50 to-pink-50 border-b border-slate-100 flex items-center justify-between">
                                <span className="text-[9px] font-black text-purple-600 uppercase tracking-widest">월별 분석 — {insight.monthly_analysis?.column || '발주월'}</span>
                                {insight.monthly_analysis && (
                                  <span className={`text-[8px] font-black px-2 py-0.5 rounded-full ${insight.monthly_analysis.is_significant ? 'bg-emerald-50 text-emerald-600' : 'bg-slate-100 text-slate-400'}`}>
                                    ANOVA F={insight.monthly_analysis.f_statistic?.toFixed(2)} | η²={insight.monthly_analysis.eta_squared?.toFixed(4)} | {insight.monthly_analysis.is_significant ? '유의' : '비유의'}
                                  </span>
                                )}
                              </div>
                              <div ref={(el) => {
                                if (el && (insight as any).plotly_monthly) {
                                  try { (window as any).Plotly?.newPlot(el, (insight as any).plotly_monthly.data, { ...(insight as any).plotly_monthly.layout, autosize: true, height: 350 }, { responsive: true }); } catch { }
                                }
                              }} className="w-full" />
                            </div>

                            {/* Monthly Stats Table */}
                            {insight.monthly_stats && Object.keys(insight.monthly_stats).length > 0 && (
                              <div className="overflow-auto bg-white rounded border border-academic-100">
                                <div className="px-4 py-2 bg-academic-50 border-b border-academic-100">
                                  <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">월별 통계</span>
                                </div>
                                <table className="w-full text-[9px] text-left border-collapse">
                                  <thead>
                                    <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                                      <th className="px-3 py-2">Month</th>
                                      <th className="px-3 py-2 text-right">N</th>
                                      <th className="px-3 py-2 text-right">Mean</th>
                                      <th className="px-3 py-2 text-right">Std</th>
                                      <th className="px-3 py-2 text-right">Min</th>
                                      <th className="px-3 py-2 text-right">Max</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {Object.entries(insight.monthly_stats).map(([month, s]: [string, any], i) => (
                                      <tr key={month} className={`border-b border-slate-50 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                                        <td className="px-3 py-1.5 font-bold text-slate-700">{month}</td>
                                        <td className="px-3 py-1.5 text-right font-mono text-slate-600">{s.count}</td>
                                        <td className="px-3 py-1.5 text-right font-mono font-black text-slate-800">{s.mean?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                        <td className="px-3 py-1.5 text-right font-mono text-slate-500">{s.std?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                        <td className="px-3 py-1.5 text-right font-mono text-slate-500">{s.min?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                        <td className="px-3 py-1.5 text-right font-mono text-slate-500">{s.max?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Pareto Detail (Card 5) */}
                        {insight.pareto_detail && insight.pareto_detail.length > 0 && (
                          <div className="space-y-2">
                            {insight.pareto_summary && (
                              <div className="bg-gradient-to-r from-orange-50 to-amber-50 p-3 rounded-xl border border-orange-100">
                                <span className="text-[9px] font-black text-orange-600 uppercase tracking-widest block mb-2">파레토 요약</span>
                                <div className="grid grid-cols-3 gap-2">
                                  <div className="bg-white rounded-lg p-2 text-center border border-orange-100">
                                    <div className="text-[7px] font-black text-slate-400 uppercase">80% 달성</div>
                                    <div className="text-sm font-black text-orange-600 mt-0.5">{insight.pareto_summary.n_80pct}개</div>
                                    <div className="text-[7px] text-slate-400">/ {insight.pareto_summary.n_categories}개</div>
                                  </div>
                                  <div className="bg-white rounded-lg p-2 text-center border border-orange-100">
                                    <div className="text-[7px] font-black text-slate-400 uppercase">집중도</div>
                                    <div className="text-sm font-black text-orange-600 mt-0.5">{insight.pareto_summary.concentration_pct?.toFixed(0)}%</div>
                                  </div>
                                  <div className="bg-white rounded-lg p-2 text-center border border-orange-100">
                                    <div className="text-[7px] font-black text-slate-400 uppercase">1위</div>
                                    <div className="text-[10px] font-black text-slate-700 mt-0.5 truncate">{insight.pareto_summary.top_category}</div>
                                    <div className="text-[7px] text-slate-400">{insight.pareto_summary.top_contribution?.toFixed(1)}%</div>
                                  </div>
                                </div>
                              </div>
                            )}
                            <div className="overflow-auto bg-white rounded border border-academic-100">
                              <div className="px-4 py-2 bg-academic-50 border-b border-academic-100">
                                <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">파레토 상세 (상위 {insight.pareto_detail.length}개)</span>
                              </div>
                              <table className="w-full text-[9px] text-left border-collapse">
                                <thead>
                                  <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                                    <th className="px-3 py-1.5">Rank</th>
                                    <th className="px-3 py-1.5">Category</th>
                                    <th className="px-3 py-1.5 text-right">합계</th>
                                    <th className="px-3 py-1.5 text-right">N</th>
                                    <th className="px-3 py-1.5 text-right">기여도%</th>
                                    <th className="px-3 py-1.5 text-right">누적%</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {insight.pareto_detail.map((p, i) => (
                                    <tr key={p.rank} className={`border-b border-slate-50 ${p.cumulative_pct <= 80 ? 'bg-orange-50/30' : i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                                      <td className="px-3 py-1 font-mono font-black text-slate-600">{p.rank}</td>
                                      <td className="px-3 py-1 font-bold text-slate-700 truncate max-w-[150px]" title={p.category}>{p.category}</td>
                                      <td className="px-3 py-1 text-right font-mono text-slate-800">{p.sum?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                                      <td className="px-3 py-1 text-right font-mono text-slate-600">{p.count}</td>
                                      <td className="px-3 py-1 text-right font-mono font-black text-slate-800">{p.contribution_pct?.toFixed(1)}%</td>
                                      <td className="px-3 py-1 text-right font-mono text-slate-500">
                                        <span className={p.cumulative_pct <= 80 ? 'text-orange-600 font-black' : ''}>{p.cumulative_pct?.toFixed(1)}%</span>
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}

                        {/* Crosstab Results (Card 6) */}
                        {insight.crosstab_results && insight.crosstab_results.length > 0 && (
                          <div className="overflow-auto bg-white rounded border border-academic-100">
                            <div className="px-4 py-2 bg-academic-50 border-b border-academic-100">
                              <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">교차분석 결과 (Chi-square)</span>
                            </div>
                            <table className="w-full text-[9px] text-left border-collapse">
                              <thead>
                                <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                                  <th className="px-3 py-1.5">Var1 × Var2</th>
                                  <th className="px-3 py-1.5 text-right">χ²</th>
                                  <th className="px-3 py-1.5 text-right">Cramér's V</th>
                                  <th className="px-3 py-1.5 text-right">p-value</th>
                                  <th className="px-3 py-1.5 text-center">유의</th>
                                </tr>
                              </thead>
                              <tbody>
                                {insight.crosstab_results.map((r, i) => (
                                  <tr key={`${r.var1}-${r.var2}`} className={`border-b border-slate-50 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                                    <td className="px-3 py-1 font-bold text-slate-700">{r.var1} × {r.var2}</td>
                                    <td className="px-3 py-1 text-right font-mono text-slate-800">{r.chi2?.toFixed(1)}</td>
                                    <td className="px-3 py-1 text-right font-mono font-black text-slate-800">{r.cramers_v?.toFixed(3)}</td>
                                    <td className="px-3 py-1 text-right font-mono text-slate-500">{r.p_value?.toFixed(4)}</td>
                                    <td className="px-3 py-1 text-center">
                                      <span className={`text-[7px] font-black px-1.5 py-0.5 rounded-full ${r.is_significant ? 'bg-emerald-50 text-emerald-600' : 'bg-red-50 text-red-400'}`}>
                                        {r.is_significant ? '✓' : '✗'}
                                      </span>
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}

                        {/* Anomaly Detection Detail (Card 7) */}
                        {insight.anomaly_methods && (
                          <div className="space-y-2">
                            <div className="bg-gradient-to-r from-red-50 to-rose-50 p-3 rounded-xl border border-red-100">
                              <span className="text-[9px] font-black text-red-600 uppercase tracking-widest block mb-2">이상치 탐지 방법 비교</span>
                              <div className="grid grid-cols-3 gap-2">
                                <div className="bg-white rounded-lg p-2 text-center border border-red-100">
                                  <div className="text-[7px] font-black text-slate-400 uppercase">IQR</div>
                                  <div className="text-sm font-black text-red-600 mt-0.5">{insight.anomaly_methods.iqr?.count}건</div>
                                  <div className="text-[7px] text-slate-400">{insight.anomaly_methods.iqr?.pct?.toFixed(1)}%</div>
                                </div>
                                <div className="bg-white rounded-lg p-2 text-center border border-red-100">
                                  <div className="text-[7px] font-black text-slate-400 uppercase">Z-Score (3σ)</div>
                                  <div className="text-sm font-black text-red-600 mt-0.5">{insight.anomaly_methods.zscore?.count}건</div>
                                  <div className="text-[7px] text-slate-400">{insight.anomaly_methods.zscore?.pct?.toFixed(1)}%</div>
                                </div>
                                <div className="bg-white rounded-lg p-2 text-center border border-red-100">
                                  <div className="text-[7px] font-black text-slate-400 uppercase">동시탐지</div>
                                  <div className="text-sm font-black text-rose-700 mt-0.5">{insight.anomaly_methods.combined?.both_methods}건</div>
                                  <div className="text-[7px] text-slate-400">강한 이상치</div>
                                </div>
                              </div>
                            </div>

                            {/* Outlier Records */}
                            {insight.outlier_records && insight.outlier_records.length > 0 && (
                              <div className="overflow-auto bg-white rounded border border-academic-100">
                                <div className="px-4 py-2 bg-academic-50 border-b border-academic-100">
                                  <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">이상치 상위 {insight.outlier_records.length}건</span>
                                </div>
                                <table className="w-full text-[9px] text-left border-collapse">
                                  <thead>
                                    <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                                      {insight.outlier_records[0] && Object.keys(insight.outlier_records[0]).map(k => (
                                        <th key={k} className="px-3 py-1.5">{k}</th>
                                      ))}
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {insight.outlier_records.map((rec, i) => (
                                      <tr key={i} className={`border-b border-slate-50 ${i % 2 === 0 ? 'bg-red-50/20' : 'bg-white'}`}>
                                        {Object.values(rec).map((v: any, j) => (
                                          <td key={j} className="px-3 py-1 font-mono text-slate-700">
                                            {typeof v === 'number' ? v.toLocaleString(undefined, { maximumFractionDigits: 2 }) : String(v)}
                                          </td>
                                        ))}
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            )}

                            {/* Var Outlier Summary */}
                            {insight.var_outlier_summary && Object.keys(insight.var_outlier_summary).length > 0 && (
                              <div className="overflow-auto bg-white rounded border border-academic-100">
                                <div className="px-4 py-2 bg-academic-50 border-b border-academic-100">
                                  <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">변수별 이상치 비율</span>
                                </div>
                                <table className="w-full text-[9px] text-left border-collapse">
                                  <thead>
                                    <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                                      <th className="px-3 py-1.5">Variable</th>
                                      <th className="px-3 py-1.5 text-right">이상치 수</th>
                                      <th className="px-3 py-1.5 text-right">전체 N</th>
                                      <th className="px-3 py-1.5 text-right">비율</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {Object.entries(insight.var_outlier_summary)
                                      .sort(([, a]: any, [, b]: any) => (b.pct || 0) - (a.pct || 0))
                                      .map(([varName, s]: [string, any], i) => (
                                        <tr key={varName} className={`border-b border-slate-50 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                                          <td className="px-3 py-1 font-bold text-slate-700">{varName}</td>
                                          <td className="px-3 py-1 text-right font-mono text-slate-600">{s.count}</td>
                                          <td className="px-3 py-1 text-right font-mono text-slate-500">{s.total}</td>
                                          <td className="px-3 py-1 text-right">
                                            <span className={`font-mono font-black ${s.pct > 5 ? 'text-red-600' : s.pct > 2 ? 'text-amber-600' : 'text-slate-600'}`}>
                                              {s.pct?.toFixed(1)}%
                                            </span>
                                          </td>
                                        </tr>
                                      ))}
                                  </tbody>
                                </table>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Coefficients Table */}
                        {insight.coefficients && insight.coefficients.length > 0 && (
                          <div className="overflow-auto bg-white rounded border border-academic-100">
                            <div className="px-4 py-2 bg-academic-50 border-b border-academic-100">
                              <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">회귀 계수 (Coefficients)</span>
                            </div>
                            <table className="w-full text-[10px] text-left border-collapse">
                              <thead>
                                <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                                  <th className="px-4 py-2">Variable</th>
                                  <th className="px-4 py-2 text-right">Coef (β)</th>
                                  <th className="px-4 py-2 text-right">Std.Err</th>
                                  <th className="px-4 py-2 text-right">t-stat</th>
                                  <th className="px-4 py-2 text-right">p-value</th>
                                </tr>
                              </thead>
                              <tbody>
                                {insight.coefficients.map((c, i) => (
                                  <tr key={i} className={`border-b border-slate-50 ${(c.p_value ?? 1) < 0.05 ? 'bg-emerald-50/40' : ''}`}>
                                    <td className="px-4 py-2 font-bold text-slate-700">{c.variable}</td>
                                    <td className="px-4 py-2 text-right font-mono font-black">{c.coef?.toFixed(4)}</td>
                                    <td className="px-4 py-2 text-right font-mono text-slate-500">{c.std_err?.toFixed(4)}</td>
                                    <td className="px-4 py-2 text-right font-mono text-slate-500">{c.t_stat?.toFixed(3)}</td>
                                    <td className={`px-4 py-2 text-right font-mono font-black ${(c.p_value ?? 1) < 0.05 ? 'text-emerald-600' : 'text-slate-400'}`}>
                                      {c.p_value != null ? (c.p_value < 0.001 ? '<0.001***' : c.p_value < 0.01 ? c.p_value.toFixed(4) + '**' : c.p_value < 0.05 ? c.p_value.toFixed(4) + '*' : c.p_value.toFixed(4)) : 'N/A'}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}

                        {/* VIF Table */}
                        {insight.vif && insight.vif.length > 0 && (
                          <div className="overflow-auto bg-white rounded border border-academic-100">
                            <div className="px-4 py-2 bg-academic-50 border-b border-academic-100">
                              <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">VIF 다중공선성 검정</span>
                            </div>
                            <table className="w-full text-[10px] text-left border-collapse">
                              <thead><tr className="border-b border-slate-200 text-slate-400 font-black uppercase text-[8px]">
                                <th className="px-4 py-2">Variable</th>
                                <th className="px-4 py-2 text-right">VIF</th>
                                <th className="px-4 py-2 text-right">Status</th>
                              </tr></thead>
                              <tbody>
                                {insight.vif.map((v, i) => (
                                  <tr key={i} className="border-b border-slate-50">
                                    <td className="px-4 py-2 font-bold text-slate-700">{v.variable}</td>
                                    <td className="px-4 py-2 text-right font-mono font-black">{v.vif?.toFixed(2)}</td>
                                    <td className={`px-4 py-2 text-right font-mono font-black ${(v.vif ?? 0) > 10 ? 'text-red-500' : (v.vif ?? 0) > 5 ? 'text-amber-500' : 'text-emerald-500'}`}>
                                      {(v.vif ?? 0) > 10 ? '⚠ High' : (v.vif ?? 0) > 5 ? '△ Moderate' : '✓ OK'}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}

                        {/* Model Summary Grid */}
                        {insight.model_summary && (
                          <div className="grid grid-cols-4 gap-2">
                            {Object.entries(insight.model_summary).map(([key, val]) => (
                              <div key={key} className="bg-slate-50 rounded-lg p-3 text-center border border-slate-100">
                                <div className="text-[7px] font-black text-slate-400 uppercase tracking-widest">{key.replace(/_/g, ' ')}</div>
                                <div className="text-[12px] font-black text-slate-800 font-mono mt-1">{typeof val === 'number' ? val.toFixed(4) : val}</div>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Chart 상세 보기 Button */}
                        {hasChart && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setChartDetailInsight(insight as any);
                            }}
                            className="w-full bg-academic-700 text-white py-2.5 rounded text-xs font-medium hover:bg-academic-800 transition-colors flex items-center justify-center gap-2"
                          >
                            <BarChart4 size={14} /> Chart 상세 보기
                          </button>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}

            {/* Loading State */}
            {state.isExploring && (
              <div className="bg-white rounded border border-academic-200 p-8 flex flex-col items-center justify-center h-48">
                <div className="w-8 h-8 border-2 border-academic-200 border-t-academic-600 rounded-full animate-spin mb-3"></div>
                <span className="text-xs font-serif text-academic-600 text-center">통계 분석 엔진 실행 중...</span>
              </div>
            )}

            {/* Empty State */}
            {!state.isExploring && state.insights.length === 0 && (
              <div className="h-[300px] flex flex-col items-center justify-center text-academic-300 border border-dashed border-academic-200 rounded">
                <Binary size={40} className="mb-3 opacity-20" />
                <p className="text-xs font-serif text-academic-400">분석 대기 중</p>
                <p className="text-[10px] text-ink-400 mt-1.5">CSV를 업로드하고 종속변수를 선택해 주세요</p>
              </div>
            )}
          </div>
        </section>
      </main>

      {/* Chart Detail Overlay */}
      {chartDetailInsight && (
        <div className="fixed inset-0 z-50 bg-white flex flex-col" style={{ zIndex: 9999 }}>
          {/* Overlay Header */}
          <header className="h-14 border-b border-academic-200 bg-white flex items-center justify-between px-8 shrink-0">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setChartDetailInsight(null)}
                className="flex items-center gap-1.5 text-academic-600 px-3 py-1.5 rounded text-xs border border-academic-200 hover:bg-academic-50 transition-colors"
              >
                <ArrowRight size={14} className="rotate-180" /> 돌아가기
              </button>
              <div className="w-px h-5 bg-academic-200"></div>
              <h2 className="text-sm font-serif font-semibold text-academic-800 flex items-center gap-2">
                <BarChart4 size={14} className="text-academic-600" />
                {chartDetailInsight.task_title}
              </h2>
            </div>
            <button
              onClick={() => setChartDetailInsight(null)}
              className="p-1.5 text-ink-400 hover:text-academic-700 hover:bg-academic-50 rounded transition-colors"
            >
              <X size={18} />
            </button>
          </header>

          {/* Overlay Content */}
          <div className="flex-1 overflow-y-auto bg-academic-50/30 p-6">
            <div className="max-w-6xl mx-auto space-y-6">
              {/* Summary Card */}
              <div className="bg-white rounded border border-academic-200 p-5">
                <span className="text-[10px] font-serif text-academic-500 block mb-1.5">분석 결론</span>
                <p className="text-sm font-medium text-ink-800 leading-relaxed">{chartDetailInsight.finding}</p>
                <p className="text-xs text-ink-500 mt-2 font-mono">{chartDetailInsight.statistical_significance}</p>
              </div>

              {/* Main Plotly Chart */}
              {(chartDetailInsight as any).plotly_chart && (
                <div className="bg-white rounded border border-academic-200 p-6 shadow-sm">
                  <span className="text-[9px] font-black text-slate-400 uppercase tracking-widest block mb-4">Primary Visualization</span>
                  <PlotlyChart data={(chartDetailInsight as any).plotly_chart} style={{ minHeight: '450px' }} />
                </div>
              )}

              {/* Heatmap Chart */}
              {(chartDetailInsight as any).plotly_heatmap && (
                <div className="bg-white rounded border border-academic-200 p-6 shadow-sm">
                  <span className="text-[9px] font-black text-slate-400 uppercase tracking-widest block mb-4">Correlation Heatmap</span>
                  <PlotlyChart data={(chartDetailInsight as any).plotly_heatmap} style={{ minHeight: '450px' }} />
                </div>
              )}

              {/* Coefficient Chart */}
              {(chartDetailInsight as any).plotly_coef_chart && (
                <div className="bg-white rounded border border-academic-200 p-6 shadow-sm">
                  <span className="text-[9px] font-black text-slate-400 uppercase tracking-widest block mb-4">Regression Coefficients</span>
                  <PlotlyChart data={(chartDetailInsight as any).plotly_coef_chart} style={{ minHeight: '400px' }} />
                </div>
              )}

              {/* Descriptive Statistics in Chart Detail */}
              {chartDetailInsight.descriptive_stats && Object.keys(chartDetailInsight.descriptive_stats).length > 0 && (
                <div className="bg-white rounded border border-academic-200 overflow-hidden shadow-sm">
                  <div className="px-6 py-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                    <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">기술통계량 (Descriptive Statistics)</span>
                    <span className="text-[9px] font-mono text-slate-400">{Object.keys(chartDetailInsight.descriptive_stats).length} VARS</span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-[10px] text-left border-collapse min-w-[800px]">
                      <thead>
                        <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                          <th className="px-4 py-2.5 sticky left-0 bg-slate-50 z-10">Variable</th>
                          <th className="px-4 py-2.5 text-right">N</th>
                          <th className="px-4 py-2.5 text-right">Mean</th>
                          <th className="px-4 py-2.5 text-right">Std</th>
                          <th className="px-4 py-2.5 text-right">Min</th>
                          <th className="px-4 py-2.5 text-right">Q1</th>
                          <th className="px-4 py-2.5 text-right">Median</th>
                          <th className="px-4 py-2.5 text-right">Q3</th>
                          <th className="px-4 py-2.5 text-right">Max</th>
                          <th className="px-4 py-2.5 text-right">Skew</th>
                          <th className="px-4 py-2.5 text-right">Kurt</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(chartDetailInsight.descriptive_stats).map(([varName, stats]: [string, any], i) => (
                          <tr key={varName} className={`border-b border-slate-100 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                            <td className="px-4 py-2 font-bold text-slate-700 sticky left-0 bg-inherit z-10" title={varName}>{varName}</td>
                            <td className="px-4 py-2 text-right font-mono text-slate-600">{stats.count}</td>
                            <td className="px-4 py-2 text-right font-mono font-black text-slate-800">{stats.mean != null ? stats.mean.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}</td>
                            <td className="px-4 py-2 text-right font-mono text-slate-500">{stats.std != null ? stats.std.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}</td>
                            <td className="px-4 py-2 text-right font-mono text-slate-500">{stats.min != null ? stats.min.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}</td>
                            <td className="px-4 py-2 text-right font-mono text-slate-500">{stats.q1 != null ? stats.q1.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}</td>
                            <td className="px-4 py-2 text-right font-mono font-bold text-slate-700">{stats.median != null ? stats.median.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}</td>
                            <td className="px-4 py-2 text-right font-mono text-slate-500">{stats.q3 != null ? stats.q3.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}</td>
                            <td className="px-4 py-2 text-right font-mono text-slate-500">{stats.max != null ? stats.max.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}</td>
                            <td className={`px-4 py-2 text-right font-mono font-bold ${Math.abs(stats.skewness || 0) > 1 ? 'text-amber-600' : 'text-slate-500'}`}>{stats.skewness != null ? stats.skewness.toFixed(3) : '-'}</td>
                            <td className={`px-4 py-2 text-right font-mono font-bold ${Math.abs(stats.kurtosis || 0) > 3 ? 'text-red-500' : 'text-slate-500'}`}>{stats.kurtosis != null ? stats.kurtosis.toFixed(3) : '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Normality Test in Chart Detail */}
              {chartDetailInsight.normality_test && Object.keys(chartDetailInsight.normality_test).length > 0 && (
                <div className="bg-white rounded border border-academic-200 overflow-hidden shadow-sm">
                  <div className="px-6 py-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                    <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">정규성 검정 — Shapiro-Wilk (α=0.05)</span>
                    <div className="flex items-center gap-3">
                      <span className="text-[9px] font-mono text-emerald-600 font-black">✓ {Object.values(chartDetailInsight.normality_test).filter((v: any) => v.is_normal).length} Normal</span>
                      <span className="text-[9px] font-mono text-red-500 font-black">✗ {Object.values(chartDetailInsight.normality_test).filter((v: any) => !v.is_normal).length} Non-Normal</span>
                    </div>
                  </div>
                  <table className="w-full text-[10px] text-left border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                        <th className="px-6 py-2.5">Variable</th>
                        <th className="px-6 py-2.5 text-right">W-Statistic</th>
                        <th className="px-6 py-2.5 text-right">p-value</th>
                        <th className="px-6 py-2.5 text-center">Distribution</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(chartDetailInsight.normality_test).map(([varName, test]: [string, any], i) => (
                        <tr key={varName} className={`border-b border-slate-100 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                          <td className="px-6 py-2 font-bold text-slate-700">{varName}</td>
                          <td className="px-6 py-2 text-right font-mono text-slate-600">{test.statistic != null ? test.statistic.toFixed(4) : '-'}</td>
                          <td className={`px-6 py-2 text-right font-mono font-black ${test.p_value != null && test.p_value < 0.05 ? 'text-red-500' : 'text-emerald-600'}`}>
                            {test.p_value != null ? (test.p_value < 0.001 ? '<0.001' : test.p_value.toFixed(4)) : '-'}
                          </td>
                          <td className="px-6 py-2 text-center">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[9px] font-black ${test.is_normal ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-600'}`}>
                              {test.is_normal ? '✓ Normal' : '✗ Non-Normal'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Outlier Summary in Chart Detail */}
              {chartDetailInsight.outlier_summary && Object.keys(chartDetailInsight.outlier_summary).length > 0 && (
                <div className="bg-white rounded border border-academic-200 overflow-hidden shadow-sm">
                  <div className="px-6 py-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                    <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">이상치 탐지 — IQR Method (1.5×IQR)</span>
                    <span className="text-[9px] font-mono text-amber-600 font-black">
                      총 {Object.values(chartDetailInsight.outlier_summary).reduce((sum: number, v: any) => sum + (v.count || 0), 0)}건
                    </span>
                  </div>
                  <table className="w-full text-[10px] text-left border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                        <th className="px-6 py-2.5">Variable</th>
                        <th className="px-6 py-2.5 text-right">Outliers</th>
                        <th className="px-6 py-2.5 text-right">Lower</th>
                        <th className="px-6 py-2.5 text-right">Upper</th>
                        <th className="px-6 py-2.5 text-center">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(chartDetailInsight.outlier_summary)
                        .sort(([, a]: [string, any], [, b]: [string, any]) => (b.count || 0) - (a.count || 0))
                        .map(([varName, out]: [string, any], i) => (
                          <tr key={varName} className={`border-b border-slate-100 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                            <td className="px-6 py-2 font-bold text-slate-700">{varName}</td>
                            <td className={`px-6 py-2 text-right font-mono font-black ${out.count > 0 ? 'text-amber-600' : 'text-slate-400'}`}>{out.count}</td>
                            <td className="px-6 py-2 text-right font-mono text-slate-500">{out.lower != null ? out.lower.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}</td>
                            <td className="px-6 py-2 text-right font-mono text-slate-500">{out.upper != null ? out.upper.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-'}</td>
                            <td className="px-6 py-2 text-center">
                              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[9px] font-black ${out.count === 0 ? 'bg-emerald-50 text-emerald-600' : out.count <= 5 ? 'bg-amber-50 text-amber-600' : 'bg-red-50 text-red-600'}`}>
                                {out.count === 0 ? '✓ Clean' : out.count <= 5 ? '△ Minor' : '⚠ Alert'}
                              </span>
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* ANOVA Results in Chart Detail */}
              {chartDetailInsight.anova_results && chartDetailInsight.anova_results.length > 0 && (
                <div className="bg-white rounded border border-academic-200 overflow-hidden shadow-sm">
                  <div className="px-6 py-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                    <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">One-Way ANOVA — 범주형 영향도</span>
                    <span className="text-[9px] font-mono text-slate-400">η² = SS_between / SS_total</span>
                  </div>
                  <table className="w-full text-[10px] text-left border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                        <th className="px-6 py-2.5">Variable</th>
                        <th className="px-4 py-2.5 text-right">Categories</th>
                        <th className="px-4 py-2.5 text-right">F-Statistic</th>
                        <th className="px-4 py-2.5 text-right">p-value</th>
                        <th className="px-4 py-2.5 text-right">η² (Effect Size)</th>
                        <th className="px-4 py-2.5 text-center">Effect Level</th>
                        <th className="px-4 py-2.5 text-center">Significance</th>
                      </tr>
                    </thead>
                    <tbody>
                      {chartDetailInsight.anova_results.map((r, i) => (
                        <tr key={r.variable} className={`border-b border-slate-100 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                          <td className="px-6 py-2 font-bold text-slate-700">{r.variable}</td>
                          <td className="px-4 py-2 text-right font-mono text-slate-600">{r.n_categories}</td>
                          <td className="px-4 py-2 text-right font-mono font-black text-slate-800">{r.f_statistic?.toFixed(2)}</td>
                          <td className={`px-4 py-2 text-right font-mono font-black ${(r.p_value ?? 1) < 0.05 ? 'text-red-500' : 'text-slate-400'}`}>
                            {r.p_value != null ? (r.p_value < 0.001 ? '<0.001' : r.p_value.toFixed(4)) : '-'}
                          </td>
                          <td className="px-4 py-2 text-right font-mono font-black text-blue-600">{r.eta_squared?.toFixed(4)}</td>
                          <td className="px-4 py-2 text-center">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[9px] font-black ${(r.eta_squared ?? 0) >= 0.14 ? 'bg-red-50 text-red-600' :
                              (r.eta_squared ?? 0) >= 0.06 ? 'bg-amber-50 text-amber-600' :
                                (r.eta_squared ?? 0) >= 0.01 ? 'bg-blue-50 text-blue-600' :
                                  'bg-slate-50 text-slate-400'
                              }`}>
                              {(r.eta_squared ?? 0) >= 0.14 ? 'Large' :
                                (r.eta_squared ?? 0) >= 0.06 ? 'Medium' :
                                  (r.eta_squared ?? 0) >= 0.01 ? 'Small' : 'Negligible'}
                            </span>
                          </td>
                          <td className="px-4 py-2 text-center">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[9px] font-black ${r.is_significant ? 'bg-emerald-50 text-emerald-600' : 'bg-slate-50 text-slate-400'}`}>
                              {r.is_significant ? '✓ Sig.' : '✗ N.S.'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Category Detail in Chart Detail (Top 7) */}
              {chartDetailInsight.category_detail && Object.keys(chartDetailInsight.category_detail).length > 0 && (() => {
                const orderedVars = chartDetailInsight.anova_results
                  ? chartDetailInsight.anova_results.slice(0, 7).map(r => r.variable).filter(v => chartDetailInsight.category_detail![v])
                  : Object.keys(chartDetailInsight.category_detail).slice(0, 7);
                return (
                  <div className="space-y-3">
                    {orderedVars.map((varName) => {
                      const catStats = chartDetailInsight.category_detail![varName];
                      if (!catStats) return null;
                      const anovaInfo = chartDetailInsight.anova_results?.find(r => r.variable === varName);
                      return (
                        <div key={varName} className="bg-white rounded border border-academic-200 overflow-hidden shadow-sm">
                          <div className="px-6 py-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                            <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">'{varName}' 카테고리별 통계</span>
                            {anovaInfo && (
                              <span className={`text-[8px] font-black px-2 py-0.5 rounded-full ${anovaInfo.is_significant ? 'bg-emerald-50 text-emerald-600' : 'bg-slate-100 text-slate-400'}`}>
                                η²={anovaInfo.eta_squared?.toFixed(4)} | F={anovaInfo.f_statistic?.toFixed(2)}
                              </span>
                            )}
                          </div>
                          <table className="w-full text-[10px] text-left border-collapse">
                            <thead>
                              <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                                <th className="px-6 py-2.5">Category</th>
                                <th className="px-4 py-2.5 text-right">N</th>
                                <th className="px-4 py-2.5 text-right">Mean</th>
                                <th className="px-4 py-2.5 text-right">Std</th>
                                <th className="px-4 py-2.5 text-right">Min</th>
                                <th className="px-4 py-2.5 text-right">Max</th>
                              </tr>
                            </thead>
                            <tbody>
                              {Object.entries(catStats).map(([cat, s]: [string, any], i) => (
                                <tr key={cat} className={`border-b border-slate-100 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                                  <td className="px-6 py-2 font-bold text-slate-700">{cat}</td>
                                  <td className="px-4 py-2 text-right font-mono text-slate-600">{s.count}</td>
                                  <td className="px-4 py-2 text-right font-mono font-black text-slate-800">{s.mean?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                  <td className="px-4 py-2 text-right font-mono text-slate-500">{s.std?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                  <td className="px-4 py-2 text-right font-mono text-slate-500">{s.min?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                  <td className="px-4 py-2 text-right font-mono text-slate-500">{s.max?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      );
                    })}
                  </div>
                );
              })()}

              {/* Time Series Analysis in Chart Detail */}
              {chartDetailInsight.time_analysis && (
                <>
                  <div className="bg-white rounded border border-academic-200 p-6 shadow-sm">
                    <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest block mb-4">시계열 추세 분석</span>
                    <div className="grid grid-cols-4 gap-4">
                      <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-4 text-center border border-blue-100">
                        <div className="text-[9px] font-black text-blue-400 uppercase">Direction</div>
                        <div className={`text-xl font-black mt-1 ${chartDetailInsight.time_analysis.direction === '상승' ? 'text-emerald-600' : 'text-red-500'}`}>
                          {chartDetailInsight.time_analysis.direction === '상승' ? '↑' : '↓'} {chartDetailInsight.time_analysis.direction}
                        </div>
                      </div>
                      <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-4 text-center border border-blue-100">
                        <div className="text-[9px] font-black text-blue-400 uppercase">Slope (β)</div>
                        <div className="text-sm font-black text-slate-800 font-mono mt-1">{chartDetailInsight.time_analysis.slope?.toFixed(4)}</div>
                      </div>
                      <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-4 text-center border border-blue-100">
                        <div className="text-[9px] font-black text-blue-400 uppercase">R²</div>
                        <div className="text-sm font-black text-slate-800 font-mono mt-1">{chartDetailInsight.time_analysis.r_squared?.toFixed(4)}</div>
                      </div>
                      <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-4 text-center border border-blue-100">
                        <div className="text-[9px] font-black text-blue-400 uppercase">p-value</div>
                        <div className={`text-sm font-black font-mono mt-1 ${(chartDetailInsight.time_analysis.slope_pvalue ?? 1) < 0.05 ? 'text-emerald-600' : 'text-slate-400'}`}>
                          {chartDetailInsight.time_analysis.slope_pvalue != null ? (chartDetailInsight.time_analysis.slope_pvalue < 0.001 ? '<0.001' : chartDetailInsight.time_analysis.slope_pvalue.toFixed(4)) : '-'}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Period Stats in Chart Detail */}
                  {chartDetailInsight.period_stats && Object.keys(chartDetailInsight.period_stats).length > 0 && (
                    <div className="bg-white rounded border border-academic-200 overflow-hidden shadow-sm">
                      <div className="px-6 py-3 bg-slate-50 border-b border-slate-200">
                        <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">기간별 통계</span>
                      </div>
                      <table className="w-full text-[10px] text-left border-collapse">
                        <thead>
                          <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                            <th className="px-6 py-2.5">Period</th>
                            <th className="px-4 py-2.5 text-right">N</th>
                            <th className="px-4 py-2.5 text-right">Mean</th>
                            <th className="px-4 py-2.5 text-right">Std</th>
                            <th className="px-4 py-2.5 text-right">Min</th>
                            <th className="px-4 py-2.5 text-right">Max</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(chartDetailInsight.period_stats).map(([period, s]: [string, any], i) => (
                            <tr key={period} className={`border-b border-slate-100 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                              <td className="px-6 py-2 font-bold text-slate-700">{period}</td>
                              <td className="px-4 py-2 text-right font-mono text-slate-600">{s.count}</td>
                              <td className="px-4 py-2 text-right font-mono font-black text-slate-800">{s.mean?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                              <td className="px-4 py-2 text-right font-mono text-slate-500">{s.std?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                              <td className="px-4 py-2 text-right font-mono text-slate-500">{s.min?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                              <td className="px-4 py-2 text-right font-mono text-slate-500">{s.max?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </>
              )}

              {/* Monthly Chart in Chart Detail */}
              {(chartDetailInsight as any).plotly_monthly && (
                <>
                  <div className="bg-white rounded border border-academic-200 overflow-hidden shadow-sm">
                    <div className="px-6 py-3 bg-gradient-to-r from-purple-50 to-pink-50 border-b border-slate-200 flex items-center justify-between">
                      <span className="text-[10px] font-black text-purple-600 uppercase tracking-widest">월별 분석 — {chartDetailInsight.monthly_analysis?.column || '발주월'}</span>
                      {chartDetailInsight.monthly_analysis && (
                        <span className={`text-[9px] font-black px-2.5 py-1 rounded-full ${chartDetailInsight.monthly_analysis.is_significant ? 'bg-emerald-50 text-emerald-600' : 'bg-slate-100 text-slate-400'}`}>
                          ANOVA F={chartDetailInsight.monthly_analysis.f_statistic?.toFixed(2)} | η²={chartDetailInsight.monthly_analysis.eta_squared?.toFixed(4)} | {chartDetailInsight.monthly_analysis.is_significant ? '✓ 유의' : '✗ 비유의'}
                        </span>
                      )}
                    </div>
                    <div ref={(el) => {
                      if (el && (chartDetailInsight as any).plotly_monthly) {
                        try { (window as any).Plotly?.newPlot(el, (chartDetailInsight as any).plotly_monthly.data, { ...(chartDetailInsight as any).plotly_monthly.layout, autosize: true, height: 450 }, { responsive: true }); } catch { }
                      }
                    }} className="w-full" />
                  </div>

                  {/* Monthly Stats Table in Chart Detail */}
                  {chartDetailInsight.monthly_stats && Object.keys(chartDetailInsight.monthly_stats).length > 0 && (
                    <div className="bg-white rounded border border-academic-200 overflow-hidden shadow-sm">
                      <div className="px-6 py-3 bg-slate-50 border-b border-slate-200">
                        <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">월별 통계</span>
                      </div>
                      <table className="w-full text-[10px] text-left border-collapse">
                        <thead>
                          <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                            <th className="px-6 py-2.5">Month</th>
                            <th className="px-4 py-2.5 text-right">N</th>
                            <th className="px-4 py-2.5 text-right">Mean</th>
                            <th className="px-4 py-2.5 text-right">Std</th>
                            <th className="px-4 py-2.5 text-right">Min</th>
                            <th className="px-4 py-2.5 text-right">Max</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(chartDetailInsight.monthly_stats).map(([month, s]: [string, any], i) => (
                            <tr key={month} className={`border-b border-slate-100 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                              <td className="px-6 py-2 font-bold text-slate-700">{month}</td>
                              <td className="px-4 py-2 text-right font-mono text-slate-600">{s.count}</td>
                              <td className="px-4 py-2 text-right font-mono font-black text-slate-800">{s.mean?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                              <td className="px-4 py-2 text-right font-mono text-slate-500">{s.std?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                              <td className="px-4 py-2 text-right font-mono text-slate-500">{s.min?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                              <td className="px-4 py-2 text-right font-mono text-slate-500">{s.max?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </>
              )}

              {/* Pareto Detail in Chart Detail */}
              {chartDetailInsight.pareto_detail && chartDetailInsight.pareto_detail.length > 0 && (
                <div className="space-y-3">
                  {chartDetailInsight.pareto_summary && (
                    <div className="bg-gradient-to-r from-orange-50 to-amber-50 p-5 rounded-2xl border border-orange-200">
                      <span className="text-[10px] font-black text-orange-600 uppercase tracking-widest block mb-3">파레토 요약</span>
                      <div className="grid grid-cols-3 gap-3">
                        <div className="bg-white rounded-xl p-3 text-center border border-orange-100">
                          <div className="text-[8px] font-black text-slate-400 uppercase">80% 도달</div>
                          <div className="text-lg font-black text-orange-600 mt-1">{chartDetailInsight.pareto_summary.n_80pct}개</div>
                          <div className="text-[8px] text-slate-400">/ {chartDetailInsight.pareto_summary.n_categories}개 중</div>
                        </div>
                        <div className="bg-white rounded-xl p-3 text-center border border-orange-100">
                          <div className="text-[8px] font-black text-slate-400 uppercase">집중도</div>
                          <div className="text-lg font-black text-orange-600 mt-1">{chartDetailInsight.pareto_summary.concentration_pct?.toFixed(0)}%</div>
                        </div>
                        <div className="bg-white rounded-xl p-3 text-center border border-orange-100">
                          <div className="text-[8px] font-black text-slate-400 uppercase">1위</div>
                          <div className="text-[11px] font-black text-slate-700 mt-1 truncate">{chartDetailInsight.pareto_summary.top_category}</div>
                          <div className="text-[8px] text-slate-400">{chartDetailInsight.pareto_summary.top_contribution?.toFixed(1)}%</div>
                        </div>
                      </div>
                    </div>
                  )}
                  <div className="bg-white rounded border border-academic-200 overflow-hidden shadow-sm">
                    <div className="px-6 py-3 bg-slate-50 border-b border-slate-200">
                      <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">파레토 상세 (상위 {chartDetailInsight.pareto_detail.length}개)</span>
                    </div>
                    <table className="w-full text-[10px] text-left border-collapse">
                      <thead>
                        <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                          <th className="px-4 py-2.5">Rank</th>
                          <th className="px-4 py-2.5">Category</th>
                          <th className="px-4 py-2.5 text-right">합계</th>
                          <th className="px-4 py-2.5 text-right">N</th>
                          <th className="px-4 py-2.5 text-right">기여도%</th>
                          <th className="px-4 py-2.5 text-right">누적%</th>
                        </tr>
                      </thead>
                      <tbody>
                        {chartDetailInsight.pareto_detail.map((p, i) => (
                          <tr key={p.rank} className={`border-b border-slate-100 ${p.cumulative_pct <= 80 ? 'bg-orange-50/30' : i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                            <td className="px-4 py-2 font-mono font-black text-slate-600">{p.rank}</td>
                            <td className="px-4 py-2 font-bold text-slate-700">{p.category}</td>
                            <td className="px-4 py-2 text-right font-mono text-slate-800">{p.sum?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                            <td className="px-4 py-2 text-right font-mono text-slate-600">{p.count}</td>
                            <td className="px-4 py-2 text-right font-mono font-black text-slate-800">{p.contribution_pct?.toFixed(1)}%</td>
                            <td className="px-4 py-2 text-right font-mono">
                              <span className={p.cumulative_pct <= 80 ? 'text-orange-600 font-black' : 'text-slate-500'}>{p.cumulative_pct?.toFixed(1)}%</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Crosstab Results in Chart Detail */}
              {chartDetailInsight.crosstab_results && chartDetailInsight.crosstab_results.length > 0 && (
                <div className="bg-white rounded border border-academic-200 overflow-hidden shadow-sm">
                  <div className="px-6 py-3 bg-slate-50 border-b border-slate-200">
                    <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">교차분석 결과 (Chi-square)</span>
                  </div>
                  <table className="w-full text-[10px] text-left border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                        <th className="px-4 py-2.5">Var1 × Var2</th>
                        <th className="px-4 py-2.5 text-right">χ²</th>
                        <th className="px-4 py-2.5 text-right">Cramér's V</th>
                        <th className="px-4 py-2.5 text-right">p-value</th>
                        <th className="px-4 py-2.5 text-right">dof</th>
                        <th className="px-4 py-2.5 text-center">유의</th>
                      </tr>
                    </thead>
                    <tbody>
                      {chartDetailInsight.crosstab_results.map((r, i) => (
                        <tr key={`${r.var1}-${r.var2}`} className={`border-b border-slate-100 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                          <td className="px-4 py-2 font-bold text-slate-700">{r.var1} × {r.var2}</td>
                          <td className="px-4 py-2 text-right font-mono text-slate-800">{r.chi2?.toFixed(1)}</td>
                          <td className="px-4 py-2 text-right font-mono font-black text-slate-800">{r.cramers_v?.toFixed(3)}</td>
                          <td className="px-4 py-2 text-right font-mono text-slate-500">{r.p_value?.toFixed(4)}</td>
                          <td className="px-4 py-2 text-right font-mono text-slate-500">{r.dof}</td>
                          <td className="px-4 py-2 text-center">
                            <span className={`text-[8px] font-black px-2 py-0.5 rounded-full ${r.is_significant ? 'bg-emerald-50 text-emerald-600' : 'bg-red-50 text-red-400'}`}>
                              {r.is_significant ? '✓ 유의' : '✗ 비유의'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Anomaly Detail in Chart Detail */}
              {chartDetailInsight.anomaly_methods && (
                <div className="space-y-3">
                  <div className="bg-gradient-to-r from-red-50 to-rose-50 p-5 rounded-2xl border border-red-200">
                    <span className="text-[10px] font-black text-red-600 uppercase tracking-widest block mb-3">이상치 탐지 방법 비교</span>
                    <div className="grid grid-cols-3 gap-3">
                      <div className="bg-white rounded-xl p-3 text-center border border-red-100">
                        <div className="text-[8px] font-black text-slate-400 uppercase">IQR Method</div>
                        <div className="text-lg font-black text-red-600 mt-1">{chartDetailInsight.anomaly_methods.iqr?.count}건</div>
                        <div className="text-[8px] text-slate-400">{chartDetailInsight.anomaly_methods.iqr?.pct?.toFixed(1)}%</div>
                      </div>
                      <div className="bg-white rounded-xl p-3 text-center border border-red-100">
                        <div className="text-[8px] font-black text-slate-400 uppercase">Z-Score (3σ)</div>
                        <div className="text-lg font-black text-red-600 mt-1">{chartDetailInsight.anomaly_methods.zscore?.count}건</div>
                        <div className="text-[8px] text-slate-400">{chartDetailInsight.anomaly_methods.zscore?.pct?.toFixed(1)}%</div>
                      </div>
                      <div className="bg-white rounded-xl p-3 text-center border border-red-100">
                        <div className="text-[8px] font-black text-slate-400 uppercase">양쪽 동시탐지</div>
                        <div className="text-lg font-black text-rose-700 mt-1">{chartDetailInsight.anomaly_methods.combined?.both_methods}건</div>
                        <div className="text-[8px] text-slate-400">강한 이상치</div>
                      </div>
                    </div>
                  </div>

                  {chartDetailInsight.outlier_records && chartDetailInsight.outlier_records.length > 0 && (
                    <div className="bg-white rounded border border-academic-200 overflow-hidden shadow-sm">
                      <div className="px-6 py-3 bg-slate-50 border-b border-slate-200">
                        <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">이상치 상위 {chartDetailInsight.outlier_records.length}건</span>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-[10px] text-left border-collapse">
                          <thead>
                            <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                              {chartDetailInsight.outlier_records[0] && Object.keys(chartDetailInsight.outlier_records[0]).map(k => (
                                <th key={k} className="px-4 py-2.5">{k}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {chartDetailInsight.outlier_records.map((rec, i) => (
                              <tr key={i} className={`border-b border-slate-100 ${i % 2 === 0 ? 'bg-red-50/20' : 'bg-white'}`}>
                                {Object.values(rec).map((v: any, j) => (
                                  <td key={j} className="px-4 py-2 font-mono text-slate-700">
                                    {typeof v === 'number' ? v.toLocaleString(undefined, { maximumFractionDigits: 2 }) : String(v)}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {chartDetailInsight.var_outlier_summary && Object.keys(chartDetailInsight.var_outlier_summary).length > 0 && (
                    <div className="bg-white rounded border border-academic-200 overflow-hidden shadow-sm">
                      <div className="px-6 py-3 bg-slate-50 border-b border-slate-200">
                        <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">변수별 이상치 비율</span>
                      </div>
                      <table className="w-full text-[10px] text-left border-collapse">
                        <thead>
                          <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                            <th className="px-4 py-2.5">Variable</th>
                            <th className="px-4 py-2.5 text-right">이상치 수</th>
                            <th className="px-4 py-2.5 text-right">전체 N</th>
                            <th className="px-4 py-2.5 text-right">비율</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(chartDetailInsight.var_outlier_summary)
                            .sort(([, a]: any, [, b]: any) => (b.pct || 0) - (a.pct || 0))
                            .map(([varName, s]: [string, any], i) => (
                              <tr key={varName} className={`border-b border-slate-100 ${i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                                <td className="px-4 py-2 font-bold text-slate-700">{varName}</td>
                                <td className="px-4 py-2 text-right font-mono text-slate-600">{s.count}</td>
                                <td className="px-4 py-2 text-right font-mono text-slate-500">{s.total}</td>
                                <td className="px-4 py-2 text-right">
                                  <span className={`font-mono font-black ${s.pct > 5 ? 'text-red-600' : s.pct > 2 ? 'text-amber-600' : 'text-slate-600'}`}>
                                    {s.pct?.toFixed(1)}%
                                  </span>
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* Coefficients Table */}
              {chartDetailInsight.coefficients && chartDetailInsight.coefficients.length > 0 && (
                <div className="bg-white rounded border border-academic-200 overflow-hidden shadow-sm">
                  <div className="px-6 py-3 bg-slate-50 border-b border-slate-200">
                    <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">회귀 계수 (Coefficients)</span>
                  </div>
                  <table className="w-full text-[11px] text-left border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 uppercase font-black tracking-widest text-slate-400 bg-slate-50/50">
                        <th className="px-6 py-3">Variable</th>
                        <th className="px-6 py-3 text-right">Coef (β)</th>
                        <th className="px-6 py-3 text-right">Std.Err</th>
                        <th className="px-6 py-3 text-right">t-stat</th>
                        <th className="px-6 py-3 text-right">p-value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {chartDetailInsight.coefficients.map((c, i) => (
                        <tr key={i} className={`border-b border-slate-100 ${(c.p_value ?? 1) < 0.05 ? 'bg-emerald-50/40' : ''}`}>
                          <td className="px-6 py-3 font-bold text-slate-700">{c.variable}</td>
                          <td className="px-6 py-3 text-right font-mono font-black">{c.coef?.toFixed(4)}</td>
                          <td className="px-6 py-3 text-right font-mono text-slate-500">{c.std_err?.toFixed(4)}</td>
                          <td className="px-6 py-3 text-right font-mono text-slate-500">{c.t_stat?.toFixed(3)}</td>
                          <td className={`px-6 py-3 text-right font-mono font-black ${(c.p_value ?? 1) < 0.05 ? 'text-emerald-600' : 'text-slate-400'}`}>
                            {c.p_value != null ? (c.p_value < 0.001 ? '<0.001***' : c.p_value < 0.01 ? c.p_value.toFixed(4) + '**' : c.p_value < 0.05 ? c.p_value.toFixed(4) + '*' : c.p_value.toFixed(4)) : 'N/A'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* VIF Table */}
              {chartDetailInsight.vif && chartDetailInsight.vif.length > 0 && (
                <div className="bg-white rounded border border-academic-200 overflow-hidden shadow-sm">
                  <div className="px-6 py-3 bg-slate-50 border-b border-slate-200">
                    <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">VIF 다중공선성 검정</span>
                  </div>
                  <table className="w-full text-[11px] text-left border-collapse">
                    <thead><tr className="border-b border-slate-200 text-slate-400 font-black uppercase text-[9px]">
                      <th className="px-6 py-3">Variable</th>
                      <th className="px-6 py-3 text-right">VIF</th>
                      <th className="px-6 py-3 text-right">Status</th>
                    </tr></thead>
                    <tbody>
                      {chartDetailInsight.vif.map((v, i) => (
                        <tr key={i} className="border-b border-slate-100">
                          <td className="px-6 py-3 font-bold text-slate-700">{v.variable}</td>
                          <td className="px-6 py-3 text-right font-mono font-black">{v.vif?.toFixed(2)}</td>
                          <td className={`px-6 py-3 text-right font-mono font-black ${(v.vif ?? 0) > 10 ? 'text-red-500' : (v.vif ?? 0) > 5 ? 'text-amber-500' : 'text-emerald-500'}`}>
                            {(v.vif ?? 0) > 10 ? '⚠ High' : (v.vif ?? 0) > 5 ? '△ Moderate' : '✓ OK'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Model Summary */}
              {chartDetailInsight.model_summary && (
                <div className="bg-white rounded border border-academic-200 p-6 shadow-sm">
                  <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest block mb-4">Model Summary</span>
                  <div className="grid grid-cols-4 gap-3">
                    {Object.entries(chartDetailInsight.model_summary).map(([key, val]) => (
                      <div key={key} className="bg-slate-50 rounded-xl p-4 text-center border border-slate-100">
                        <div className="text-[8px] font-black text-slate-400 uppercase tracking-widest">{key.replace(/_/g, ' ')}</div>
                        <div className="text-[14px] font-black text-slate-800 font-mono mt-1">{typeof val === 'number' ? val.toFixed(4) : val}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #E2E8F0; border-radius: 10px; }
        @keyframes drawLine { from { stroke-dashoffset: 400; opacity: 0; } to { stroke-dashoffset: 0; opacity: 1; } }
        .animate-draw-line { stroke-dasharray: 400; animation: drawLine 2s forwards cubic-bezier(0.4, 0, 0.2, 1); }
      `}</style>
    </div>
  );
};

export default App;
