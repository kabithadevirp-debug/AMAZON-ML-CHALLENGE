import React from 'react';
import { Sliders, ShieldCheck, Zap } from 'lucide-react';

interface ThresholdSliderProps {
  threshold: number;
  onThresholdChange: (newThreshold: number) => void;
  isUpdating: boolean;
}

export const ThresholdSlider: React.FC<ThresholdSliderProps> = ({
  threshold,
  onThresholdChange,
  isUpdating,
}) => {
  return (
    <div className="bg-[#111420] border border-slate-800 rounded-2xl p-5 mb-8 shadow-xl">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
              <Sliders className="w-4 h-4" />
            </div>
            <h3 className="text-sm font-bold text-white tracking-tight">
              Macro F0.5 Precision Threshold Calibrator
            </h3>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
              {(threshold * 100).toFixed(1)}% Cutoff
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Dynamically re-scores candidate links in real-time without retraining. Higher threshold penalizes false merges (protects F0.5 score).
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs text-slate-300">
          <span className="px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-slate-400 flex items-center gap-1">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            Precision Priority: 2x
          </span>
          {isUpdating && (
            <span className="flex items-center gap-1 text-indigo-400 animate-pulse">
              <Zap className="w-3.5 h-3.5" /> Recomputing...
            </span>
          )}
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
          <span>0.50 (Permissive)</span>
          <span className="text-indigo-400 font-bold">0.75 (Recommended F0.5 Optimum)</span>
          <span>0.95 (Ultra-Conservative)</span>
        </div>
        <input
          type="range"
          min="0.50"
          max="0.95"
          step="0.01"
          value={threshold}
          onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
          className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
        />
      </div>
    </div>
  );
};
