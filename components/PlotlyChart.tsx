import React, { useEffect, useRef } from 'react';

declare global {
    interface Window {
        Plotly: any;
    }
}

interface PlotlyChartProps {
    data: any;
    className?: string;
    style?: React.CSSProperties;
}

/**
 * Plotly 차트를 렌더링하는 React 컴포넌트.
 * Python 백엔드에서 생성된 plotly figure JSON을 받아서 표시한다.
 * plotly.js CDN이 index.html에서 로드되어야 한다.
 */
const PlotlyChart: React.FC<PlotlyChartProps> = ({ data, className, style }) => {
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!containerRef.current || !data || !window.Plotly) return;

        const figureData = data.data || [];
        const layout = {
            ...(data.layout || {}),
            autosize: true,
            responsive: true,
        };
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
            displaylogo: false,
            locale: 'ko',
        };

        try {
            window.Plotly.newPlot(containerRef.current, figureData, layout, config);
        } catch (err) {
            console.error('Plotly render error:', err);
        }

        // cleanup
        return () => {
            if (containerRef.current) {
                try {
                    window.Plotly.purge(containerRef.current);
                } catch { }
            }
        };
    }, [data]);

    // 리사이즈 대응
    useEffect(() => {
        const handleResize = () => {
            if (containerRef.current && window.Plotly) {
                try {
                    window.Plotly.Plots.resize(containerRef.current);
                } catch { }
            }
        };

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    if (!data) {
        return (
            <div className={`flex items-center justify-center h-48 bg-slate-50 rounded-xl border border-dashed border-slate-200 ${className || ''}`}>
                <span className="text-slate-400 text-xs font-mono">No chart data</span>
            </div>
        );
    }

    return (
        <div
            ref={containerRef}
            className={className || ''}
            style={{ width: '100%', minHeight: '300px', ...style }}
        />
    );
};

export default PlotlyChart;
