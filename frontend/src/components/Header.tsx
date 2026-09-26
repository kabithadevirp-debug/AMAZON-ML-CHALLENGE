import React from 'react';
import { Database, ShieldCheck } from 'lucide-react';

interface HeaderProps {
  currentPreset: 'train' | 'test' | 'custom_upload' | null;
  onLoadPreset: (preset: 'train' | 'test') => void;
  isLoading: boolean;
}

export const Header: React.FC<HeaderProps> = ({ currentPreset, onLoadPreset, isLoading }) => {
  return (
    <header className="border-b border-slate-800/80 bg-[#0B0D14]/90 sticky top-0 z-40 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-sky-400 p-[1px] shadow-lg shadow-indigo-500/20">
            <div className="w-full h-full bg-[#0D0F18] rounded-[11px] flex items-center justify-center">
              <Database className="w-5 h-5 text-indigo-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-base font-bold tracking-tight text-white flex items-center gap-1.5">
                EntityMatch <span className="text-xs px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono font-medium">AI</span>
              </h1>
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 animate-pulse" />
                System Ready
              </span>
            </div>
            <p className="text-xs text-slate-400">High-Precision Business Entity Resolution Platform</p>
          </div>
        </div>

        {/* Preset quick actions */}
        <div className="flex items-center space-x-3">
          <div className="hidden sm:flex items-center bg-[#131722] p-1 rounded-lg border border-slate-800">
            <button
              onClick={() => onLoadPreset('train')}
              disabled={isLoading}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                currentPreset === 'train'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              Load Train Split (2.2M)
            </button>
            <button
              onClick={() => onLoadPreset('test')}
              disabled={isLoading}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                currentPreset === 'test'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              Load Test Split (1.7M)
            </button>
          </div>

          <div className="flex items-center pl-2 border-l border-slate-800 text-xs text-slate-400 font-mono">
            <span className="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300 flex items-center gap-1">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              Macro F0.5 Optimized
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
