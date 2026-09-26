import React from 'react';
import { Layers, Split, Cpu, CheckCircle2, FileOutput } from 'lucide-react';

interface PipelineStepperProps {
  currentStage: 'idle' | 'normalizing' | 'blocking' | 'features' | 'matching' | 'done';
}

export const PipelineStepper: React.FC<PipelineStepperProps> = ({ currentStage }) => {
  const stages = [
    { id: 'foundation', name: '1. Data Foundation', desc: 'TSV Schema & Ingestion', icon: Layers, activeOn: ['idle', 'normalizing', 'blocking', 'features', 'matching', 'done'] },
    { id: 'blocking', name: '2. Blocking Engine', desc: 'Multi-pass Candidate Gen', icon: Split, activeOn: ['normalizing', 'blocking', 'features', 'matching', 'done'] },
    { id: 'features', name: '3. Feature Engineering', desc: 'Name, Address & Token Sim', icon: Cpu, activeOn: ['features', 'matching', 'done'] },
    { id: 'matching', name: '4. Precision Matcher', desc: 'Macro F0.5 Classifier', icon: CheckCircle2, activeOn: ['matching', 'done'] },
    { id: 'export', name: '5. Results & Export', desc: 'Verified TSV Submission', icon: FileOutput, activeOn: ['done'] },
  ];

  return (
    <div className="w-full bg-[#10131C] border border-slate-800/80 rounded-2xl p-5 mb-8 shadow-xl shadow-black/40">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-sm font-semibold text-white tracking-wide uppercase flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-indigo-500 animate-ping" />
            Entity Resolution Pipeline Execution Stages
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">End-to-end transparent state progression without opaque spinners</p>
        </div>
        <div className="text-xs font-mono px-2.5 py-1 rounded-md bg-indigo-950/60 border border-indigo-800/50 text-indigo-300">
          Stage: <span className="font-semibold capitalize">{currentStage}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {stages.map((stage) => {
          const Icon = stage.icon;
          const isCurrent = (
            (stage.id === 'foundation' && currentStage === 'idle') ||
            (stage.id === 'blocking' && (currentStage === 'normalizing' || currentStage === 'blocking')) ||
            (stage.id === 'features' && currentStage === 'features') ||
            (stage.id === 'matching' && currentStage === 'matching') ||
            (stage.id === 'export' && currentStage === 'done')
          );
          const isPassed = stage.activeOn.includes(currentStage) && !isCurrent;

          return (
            <div
              key={stage.id}
              className={`relative p-3.5 rounded-xl border transition-all ${
                isCurrent
                  ? 'bg-gradient-to-b from-indigo-950/40 to-slate-900 border-indigo-500/60 shadow-lg shadow-indigo-500/10'
                  : isPassed
                  ? 'bg-[#141724] border-emerald-500/30'
                  : 'bg-[#0E1017]/60 border-slate-800/50 opacity-60'
              }`}
            >
              <div className="flex items-start gap-3">
                <div
                  className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
                    isCurrent
                      ? 'bg-indigo-600 text-white shadow-md shadow-indigo-500/30'
                      : isPassed
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      : 'bg-slate-800/60 text-slate-400'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                </div>
                <div className="min-w-0">
                  <p className="text-xs font-semibold text-slate-200 truncate">{stage.name}</p>
                  <p className="text-[11px] text-slate-400 truncate mt-0.5">{stage.desc}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
