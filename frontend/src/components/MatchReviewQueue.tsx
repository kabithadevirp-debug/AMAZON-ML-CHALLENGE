import React, { useState } from 'react';
import { Check, X, Eye, AlertTriangle, CheckCircle2, UserX, Search } from 'lucide-react';

interface CandidateItem {
  target_id: string;
  confidence: number;
  norm_name: string;
  norm_addr: string;
  is_match: boolean;
}

interface MatchRecord {
  s1_id: string;
  business_name: string;
  business_address: string;
  country: string;
  matched_ids: string[];
  status: 'matched' | 'singleton';
  candidates: CandidateItem[];
}

interface MatchReviewQueueProps {
  records: MatchRecord[];
  onDecision: (s1_id: string, target_id: string, action: 'accept' | 'reject') => void;
  onInspectEntity: (record: MatchRecord) => void;
}

export const MatchReviewQueue: React.FC<MatchReviewQueueProps> = ({
  records,
  onDecision,
  onInspectEntity,
}) => {
  const [filterMode, setFilterMode] = useState<'all' | 'matched' | 'singleton' | 'review'>('all');
  const [searchTerm, setSearchTerm] = useState('');

  const filteredRecords = records.filter((r) => {
    if (filterMode === 'matched' && r.status !== 'matched') return false;
    if (filterMode === 'singleton' && r.status !== 'singleton') return false;
    if (filterMode === 'review') {
      const hasBorderline = r.candidates.some((c) => c.confidence >= 0.40 && c.confidence < 0.75);
      if (!hasBorderline) return false;
    }

    const term = searchTerm.toLowerCase();
    if (!term) return true;
    return (
      r.s1_id.toLowerCase().includes(term) ||
      r.business_name.toLowerCase().includes(term) ||
      r.business_address.toLowerCase().includes(term) ||
      r.matched_ids.some((id) => id.toLowerCase().includes(term))
    );
  });

  return (
    <div className="bg-[#10131E] border border-slate-800 rounded-2xl overflow-hidden shadow-2xl mb-8">
      <div className="p-5 border-b border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4 bg-[#131724]">
        <div>
          <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <span>Entity Match Prediction & Review Queue</span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Surfaces predicted business links and borderline candidates for precision audits.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center bg-[#0B0D14] p-1 rounded-lg border border-slate-800 text-xs">
            <button
              onClick={() => setFilterMode('all')}
              className={`px-3 py-1 rounded-md font-medium transition-all ${
                filterMode === 'all'
                  ? 'bg-slate-700 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              All ({records.length})
            </button>
            <button
              onClick={() => setFilterMode('matched')}
              className={`px-3 py-1 rounded-md font-medium transition-all ${
                filterMode === 'matched'
                  ? 'bg-emerald-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Matched ({records.filter((r) => r.status === 'matched').length})
            </button>
            <button
              onClick={() => setFilterMode('singleton')}
              className={`px-3 py-1 rounded-md font-medium transition-all ${
                filterMode === 'singleton'
                  ? 'bg-rose-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Singletons ({records.filter((r) => r.status === 'singleton').length})
            </button>
            <button
              onClick={() => setFilterMode('review')}
              className={`px-3 py-1 rounded-md font-medium transition-all flex items-center gap-1 ${
                filterMode === 'review'
                  ? 'bg-amber-600 text-white shadow-sm'
                  : 'text-amber-400 hover:text-amber-300'
              }`}
            >
              <AlertTriangle className="w-3 h-3" />
              Review Queue
            </button>
          </div>

          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Filter by ID, name, address..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-8 pr-3 py-1.5 rounded-lg bg-[#0B0D14] border border-slate-800 text-xs text-slate-200 placeholder-slate-400 focus:outline-none focus:border-indigo-500 font-mono w-56"
            />
          </div>
        </div>
      </div>

      <div className="overflow-x-auto max-h-[550px] overflow-y-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead className="bg-[#151928] sticky top-0 z-10 text-slate-300 font-semibold uppercase tracking-wider text-[11px] border-b border-slate-800">
            <tr>
              <th className="py-3 px-4 w-28">Source 1 ID</th>
              <th className="py-3 px-4">Business Reference</th>
              <th className="py-3 px-4">Address</th>
              <th className="py-3 px-4">Match Status</th>
              <th className="py-3 px-4">Predicted S2 / S3 Entities</th>
              <th className="py-3 px-4 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {filteredRecords.map((rec) => {
              const isSingleton = rec.status === 'singleton';
              const topCand = rec.candidates[0];

              return (
                <tr
                  key={rec.s1_id}
                  className="hover:bg-indigo-950/20 transition-colors odd:bg-[#0B0D14]/30 even:bg-[#0E111A]/40"
                >
                  <td className="py-3 px-4 font-bold text-indigo-400 whitespace-nowrap">
                    {rec.s1_id}
                  </td>
                  <td className="py-3 px-4 font-sans font-medium text-white max-w-xs truncate">
                    {rec.business_name}
                  </td>
                  <td className="py-3 px-4 font-sans text-slate-400 max-w-sm truncate">
                    {rec.business_address || <span className="italic text-slate-400">&lt;empty&gt;</span>}
                  </td>
                  <td className="py-3 px-4 whitespace-nowrap">
                    {isSingleton ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/10 text-rose-400 border border-rose-500/20">
                        <UserX className="w-3 h-3" />
                        Singleton (1.0)
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                        <CheckCircle2 className="w-3 h-3" />
                        {rec.matched_ids.length} Match
                        {rec.matched_ids.length > 1 ? 'es' : ''}
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex flex-wrap gap-1.5 items-center">
                      {rec.candidates.slice(0, 3).map((cand) => (
                        <div
                          key={cand.target_id}
                          className={`flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] border ${
                            cand.is_match
                              ? 'bg-indigo-500/15 text-indigo-300 border-indigo-500/30 font-bold'
                              : cand.confidence >= 0.40
                              ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
                              : 'bg-slate-800/40 text-slate-400 border-slate-700/40'
                          }`}
                        >
                          <span>{cand.target_id}</span>
                          <span className="text-[10px] font-mono opacity-80">
                            {(cand.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                      {rec.candidates.length > 3 && (
                        <span className="text-[10px] text-slate-400">
                          +{rec.candidates.length - 3} more
                        </span>
                      )}
                      {rec.candidates.length === 0 && (
                        <span className="text-slate-400 text-xs italic">No candidates generated</span>
                      )}
                    </div>
                  </td>
                  <td className="py-3 px-4 text-right whitespace-nowrap">
                    <div className="flex items-center justify-end gap-1.5">
                      {topCand && (
                        <>
                          <button
                            onClick={() => onDecision(rec.s1_id, topCand.target_id, 'accept')}
                            title="Accept Match"
                            className="p-1 rounded bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/20 transition-colors"
                          >
                            <Check className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => onDecision(rec.s1_id, topCand.target_id, 'reject')}
                            title="Reject Match"
                            className="p-1 rounded bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/20 transition-colors"
                          >
                            <X className="w-3.5 h-3.5" />
                          </button>
                        </>
                      )}
                      <button
                        onClick={() => onInspectEntity(rec)}
                        title="Side-by-side Inspection"
                        className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors"
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
            {filteredRecords.length === 0 && (
              <tr>
                <td colSpan={6} className="py-12 text-center text-slate-400 font-sans">
                  No records matching the selected filter criteria.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
