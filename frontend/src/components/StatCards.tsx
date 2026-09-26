import React from 'react';
import { PipelineStats } from '../types';
import { Layers, FileSpreadsheet, Database, Users, CheckCircle2, UserX, Target } from 'lucide-react';

interface StatCardsProps {
  stats: PipelineStats;
}

export const StatCards: React.FC<StatCardsProps> = ({ stats }) => {
  const cards = [
    {
      title: 'Source 1 (Reference)',
      value: stats.s1_count > 0 ? stats.s1_count.toLocaleString() : '—',
      subtext: 'Clean deduplicated baseline',
      icon: Layers,
      color: 'from-blue-500/10 to-indigo-500/10 border-blue-500/20 text-blue-400',
    },
    {
      title: 'Source 2 (Provider A)',
      value: stats.s2_count > 0 ? stats.s2_count.toLocaleString() : '—',
      subtext: 'Noisy provider records',
      icon: FileSpreadsheet,
      color: 'from-purple-500/10 to-indigo-500/10 border-purple-500/20 text-purple-400',
    },
    {
      title: 'Source 3 (Provider B)',
      value: stats.s3_count > 0 ? stats.s3_count.toLocaleString() : '—',
      subtext: 'Noisy provider records',
      icon: Database,
      color: 'from-sky-500/10 to-cyan-500/10 border-sky-500/20 text-sky-400',
    },
    {
      title: 'Candidate Pairs',
      value: stats.candidate_pairs > 0 ? stats.candidate_pairs.toLocaleString() : '—',
      subtext: 'Post-blocking candidate space',
      icon: Users,
      color: 'from-amber-500/10 to-yellow-500/10 border-amber-500/20 text-amber-400',
    },
    {
      title: 'Confirmed Matches',
      value: stats.confirmed_matches > 0 ? stats.confirmed_matches.toLocaleString() : '—',
      subtext: 'Predicted links to S2/S3',
      icon: CheckCircle2,
      color: 'from-emerald-500/10 to-teal-500/10 border-emerald-500/20 text-emerald-400',
    },
    {
      title: 'No-Match Entities',
      value: stats.singletons > 0 ? stats.singletons.toLocaleString() : '—',
      subtext: 'Distinct verified singletons',
      icon: UserX,
      color: 'from-rose-500/10 to-pink-500/10 border-rose-500/20 text-rose-400',
    },
    {
      title: 'Avg. Model Confidence',
      value: stats.avg_confidence > 0 ? `${(stats.avg_confidence * 100).toFixed(1)}%` : '—',
      subtext: 'Precision boundary tuned',
      icon: Target,
      color: 'from-indigo-500/10 to-violet-500/10 border-indigo-500/20 text-indigo-400',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-7 gap-3 mb-8">
      {cards.map((card, idx) => {
        const Icon = card.icon;
        return (
          <div
            key={idx}
            className="p-4 rounded-xl bg-[#11131C] border border-slate-800/80 hover:border-slate-700 transition-all flex flex-col justify-between"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">{card.title}</span>
              <div className={`p-1.5 rounded-md border ${card.color}`}>
                <Icon className="w-3.5 h-3.5" />
              </div>
            </div>
            <div>
              <p className="text-xl font-bold tracking-tight text-white font-mono">{card.value}</p>
              <p className="text-[11px] text-slate-400 mt-0.5 truncate">{card.subtext}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
};
