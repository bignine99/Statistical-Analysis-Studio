declare module 'lucide-react' {
    import React from 'react';
    interface IconProps extends React.SVGAttributes<SVGElement> {
        size?: number | string;
        color?: string;
        strokeWidth?: number | string;
        absoluteStrokeWidth?: boolean;
        className?: string;
    }
    type Icon = React.FC<IconProps>;
    export const Cpu: Icon;
    export const Database: Icon;
    export const Activity: Icon;
    export const BrainCircuit: Icon;
    export const Binary: Icon;
    export const CheckCircle2: Icon;
    export const Clock: Icon;
    export const Sparkles: Icon;
    export const BarChart4: Icon;
    export const TrendingUp: Icon;
    export const Zap: Icon;
    export const Key: Icon;
    export const Search: Icon;
    export const SortAsc: Icon;
    export const ArrowDownWideNarrow: Icon;
    export const PieChart: Icon;
    export const LineChart: Icon;
    export const ShieldCheck: Icon;
    export const Target: Icon;
    export const AlertCircle: Icon;
    export const RotateCcw: Icon;
    export const Tag: Icon;
    export const RefreshCw: Icon;
    export const Layers: Icon;
    export const FileText: Icon;
    export const MousePointer2: Icon;
    export const ArrowRight: Icon;
    export const ChevronRight: Icon;
    export const ChevronDown: Icon;
    export const ChevronUp: Icon;
    export const X: Icon;
    export const GanttChart: Icon;
    export const FlaskConical: Icon;
    export const HardHat: Icon;
    export const BarChartHorizontal: Icon;
    export const FileDown: Icon;
    export const Wand2: Icon;
    export const Filter: Icon;
}

declare module 'papaparse' {
    const Papa: any;
    export default Papa;
}
