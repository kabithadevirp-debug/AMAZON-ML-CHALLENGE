import React from 'react';
import { X, Check, Globe, CheckCircle2 } from 'lucide-react';

interface EntityDetailModalProps {
  record: any;
  onClose: () => void;
  onDecision: (s1_id: string, target_id: string, action: 'accept' | 'reject') => void;
}

export const EntityDetailModal: React.FC<EntityDetailModalProps> = ({
  record,
  onClose,
  onDecision,
}) => {
  if (!record) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="bg-[#0E111A] border border-slate-800 w-full max-w-4xl rounded-2xl shadow-2xl shadow-black/90 flex flex-col max-h-[85vh] overflow-hidden">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-[#121522]">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                {record.s1_id}
              </span>
              <h3 className="text-base font-bold text-white">
                Side-by-Side Entity Match Comparison
              </h3>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Inspect normalized field diffs and similarity confidence across candidate providers
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6">
          {/* Reference Entity S1 Card */}
          <div className="p-4 rounded-xl bg-indigo-950/20 border border-indigo-500/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-mono uppercase tracking-wider text-indigo-400 font-bold">
                Source 1 (Reference Entity)
              </span>
              <span className="inline-flex items-center gap-1 text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                <Globe className="w-3 h-3 text-slate-400" />
                {record.country || 'US'}
              </span>
            </div>
            <p className="text-base font-bold text-white font-sans">{record.business_name}</p>
            <p className="text-xs text-slate-300 font-sans mt-1">
              {record.business_address || <span className="text-slate-400 italic">No address provided</span>}
            </p>
          </div>

          {/* Candidates Comparison List */}
          <div className="space-y-4">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Generated Candidate Records ({record.candidates?.length || 0})
            </h4>

            {record.candidates && record.candidates.length > 0 ? (
              record.candidates.map((cand: any, idx: number) => {
                const confPercent = (cand.confidence * 100).toFixed(1);
                const isMatch = cand.is_match;

                return (
                  <div
                    key={idx}
                    className={`p-4 rounded-xl border transition-all ${
                      isMatch
                        ? 'bg-emerald-950/15 border-emerald-500/40'
                        : cand.confidence >= 0.40
                        ? 'bg-amber-950/10 border-amber-500/30'
                        : 'bg-[#121522] border-slate-800'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2 font-mono text-xs">
                        <span className="font-bold text-indigo-400">{cand.target_id}</span>
                        <span className="text-slate-400">({cand.target_id.slice(0, 2)})</span>
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            cand.confidence >= 0.75
                              ? 'bg-emerald-500/20 text-emerald-300'
                              : cand.confidence >= 0.40
                              ? 'bg-amber-500/20 text-amber-300'
                              : 'bg-slate-800 text-slate-400'
                          }`}
                        >
                          Confidence: {confPercent}%
                        </span>
                      </div>

                      <div className="flex items-center gap-2">
                        {isMatch ? (
                          <span className="inline-flex items-center gap-1 text-xs text-emerald-400 font-semibold">
                            <CheckCircle2 className="w-3.5 h-3.5" />
                            Confirmed Link
                          </span>
                        ) : (
                          <span className="text-xs text-slate-400">Unlinked (Sub-threshold)</span>
                        )}
                        <button
                          onClick={() => {
                            onDecision(record.s1_id, cand.target_id, isMatch ? 'reject' : 'accept');
                          }}
                          className={`px-3 py-1 rounded text-xs font-semibold transition-colors flex items-center gap-1 ${
                            isMatch
                              ? 'bg-rose-600/20 text-rose-300 hover:bg-rose-600/30 border border-rose-500/30'
                              : 'bg-emerald-600 text-white hover:bg-emerald-500'
                          }`}
                        >
                          {isMatch ? (
                            <>
                              <X className="w-3 h-3" /> Unlink Record
                            </>
                          ) : (
                            <>
                              <Check className="w-3 h-3" /> Confirm Match
                            </>
                          )}
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3 pt-3 border-t border-slate-800/80 font-mono text-xs">
                      <div>
                        <span className="text-[10px] uppercase text-slate-400 block mb-0.5">Normalized Name</span>
                        <p className="text-slate-200 font-sans">{cand.norm_name || '<empty>'}</p>
                      </div>
                      <div>
                        <span className="text-[10px] uppercase text-slate-400 block mb-0.5">Normalized Address</span>
                        <p className="text-slate-400 font-sans">{cand.norm_addr || '<empty>'}</p>
                      </div>
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="p-8 text-center rounded-xl bg-slate-900/40 border border-slate-800 text-slate-400 text-xs">
                No candidate records produced by the blocking engine for this entity.
              </div>
            )}
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-[#121522] flex items-center justify-between text-xs text-slate-400">
          <span>Manual decisions update the submission file immediately</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-white font-medium transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
