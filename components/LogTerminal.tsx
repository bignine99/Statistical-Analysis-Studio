
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useEffect, useRef } from 'react';

interface LogTerminalProps {
  logs: string[];
  type: 'flash' | 'thinking';
  streamText?: string;
}

export const LogTerminal: React.FC<LogTerminalProps> = ({ logs, type, streamText }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, streamText]);

  return (
    <div className="flex flex-col h-full bg-white rounded-2xl overflow-hidden font-mono text-[10px]">
      <div className="bg-slate-50 px-5 py-3 border-b border-slate-100 flex justify-between items-center">
        <span className="font-black text-slate-500 tracking-widest uppercase">
          AGENT_REASONING_ENGINE
        </span>
        <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.4)] animate-pulse"></div>
      </div>

      <div ref={scrollRef} className="flex-1 p-5 overflow-y-auto space-y-2 custom-scrollbar bg-white">
        {logs.map((log, i) => (
          <div key={i} className="flex items-start gap-3">
            <span className="text-slate-400 shrink-0 font-bold">[{new Date().toLocaleTimeString('ko-KR', { hour12: false })}]</span>
            <span className="text-slate-700 font-bold leading-relaxed">{'>'} {log}</span>
          </div>
        ))}
        {logs.length === 0 && <div className="text-slate-300 font-black italic uppercase tracking-wider">Engine initialized...</div>}
      </div>
    </div>
  );
};
