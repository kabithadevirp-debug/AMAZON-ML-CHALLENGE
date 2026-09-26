import React, { useState } from 'react';
import { X, Search, FileSpreadsheet, ArrowUpDown, Globe } from 'lucide-react';
import { SourceMeta, SourceRow, GroundTruthRow } from '../types';

interface DataPreviewModalProps {
  sourceKey: string;
  sourceMeta: SourceMeta;
  onClose: () => void;
}

export const DataPreviewModal: React.FC<DataPreviewModalProps> = ({
  sourceKey,
  sourceMeta,
  onClose,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortField, setSortField] = useState<string>('entity_id');
  const [sortAsc, setSortAsc] = useState(true);

  const isGroundTruth = sourceKey === 'ground_truth';
  const rows = (sourceMeta.sample_rows || []) as Array<SourceRow | GroundTruthRow>;

  const filteredRows = rows.filter((r) => {
    const term = searchTerm.toLowerCase();
    return Object.values(r).some((val) => String(val).toLowerCase().includes(term));
  });

  const sortedRows = [...filteredRows].sort((a: any, b: any) => {
    const valA = a[sortField] || '';
    const valB = b[sortField] || '';
    if (valA < valB) return sortAsc ? -1 : 1;
    if (valA > valB) return sortAsc ? 1 : -1;
    return 0;
  });

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(true);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="bg-[#0F121C] border border-slate-800 w-full max-w-5xl rounded-2xl shadow-2xl shadow-black/80 flex flex-col max-h-[85vh] overflow-hidden">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-[#121522]">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
              <FileSpreadsheet className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-base font-bold text-white capitalize">
                  {sourceKey.replace('_', ' ')} Dataset Preview
                </h3>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  Showing first 20 records
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                Total rows: {sourceMeta.total_rows.toLocaleString()} | Path: {sourceMeta.filename}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Filter Toolbar */}
        <div className="px-6 py-3 border-b border-slate-800/80 bg-[#0E1018] flex items-center justify-between gap-4">
          <div className="relative flex-1 max-w-md">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search sample preview records..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-200 placeholder-slate-400 focus:outline-none focus:border-indigo-500 font-mono"
            />
          </div>
          <div className="text-xs text-slate-400 font-mono">
            Filtered: <span className="text-white font-bold">{sortedRows.length}</span> / {rows.length}
          </div>
        </div>

        {/* Table View */}
        <div className="flex-1 overflow-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-[#141724] sticky top-0 z-10 text-slate-300 font-semibold uppercase tracking-wider text-[11px] border-b border-slate-800">
              {isGroundTruth ? (
                <tr>
                  <th
                    onClick={() => handleSort('source1_entity_id')}
                    className="py-3 px-4 cursor-pointer hover:text-white"
                  >
                    <div className="flex items-center gap-1.5">
                      <span>Source 1 Entity ID</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                  <th
                    onClick={() => handleSort('matched_entity_ids')}
                    className="py-3 px-4 cursor-pointer hover:text-white"
                  >
                    <div className="flex items-center gap-1.5">
                      <span>Matched Entity IDs (Comma-separated)</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                  <th className="py-3 px-4">Match Status</th>
                </tr>
              ) : (
                <tr>
                  <th
                    onClick={() => handleSort('entity_id')}
                    className="py-3 px-4 cursor-pointer hover:text-white"
                  >
                    <div className="flex items-center gap-1.5">
                      <span>Entity ID</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                  <th
                    onClick={() => handleSort('business_name')}
                    className="py-3 px-4 cursor-pointer hover:text-white"
                  >
                    <div className="flex items-center gap-1.5">
                      <span>Business Name</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                  <th
                    onClick={() => handleSort('business_address')}
                    className="py-3 px-4 cursor-pointer hover:text-white"
                  >
                    <div className="flex items-center gap-1.5">
                      <span>Business Address</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                  <th
                    onClick={() => handleSort('country')}
                    className="py-3 px-4 cursor-pointer hover:text-white w-28"
                  >
                    <div className="flex items-center gap-1.5">
                      <span>Country</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                </tr>
              )}
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {sortedRows.map((row: any, idx) => (
                <tr
                  key={idx}
                  className="hover:bg-indigo-950/20 transition-colors odd:bg-[#0B0D14]/40 even:bg-[#0F121A]/40"
                >
                  {isGroundTruth ? (
                    <>
                      <td className="py-2.5 px-4 font-semibold text-indigo-400">
                        {row.source1_entity_id}
                      </td>
                      <td className="py-2.5 px-4 text-slate-200">
                        {row.matched_entity_ids ? (
                          <div className="flex flex-wrap gap-1">
                            {String(row.matched_entity_ids)
                              .split(',')
                              .map((id, i) => (
                                <span
                                  key={i}
                                  className="px-1.5 py-0.5 rounded text-[11px] bg-indigo-500/10 text-indigo-300 border border-indigo-500/20"
                                >
                                  {id.trim()}
                                </span>
                              ))}
                          </div>
                        ) : (
                          <span className="text-slate-400 italic">None (Singleton)</span>
                        )}
                      </td>
                      <td className="py-2.5 px-4">
                        {row.matched_entity_ids ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                            {String(row.matched_entity_ids).split(',').length} Matches
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20">
                            Singleton (1.0 Credit)
                          </span>
                        )}
                      </td>
                    </>
                  ) : (
                    <>
                      <td className="py-2.5 px-4 font-semibold text-indigo-400 whitespace-nowrap">
                        {row.entity_id}
                      </td>
                      <td className="py-2.5 px-4 font-sans font-medium text-white max-w-xs truncate">
                        {row.business_name}
                      </td>
                      <td className="py-2.5 px-4 font-sans text-slate-400 max-w-md truncate">
                        {row.business_address || <span className="italic text-slate-400">&lt;empty&gt;</span>}
                      </td>
                      <td className="py-2.5 px-4 whitespace-nowrap">
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-slate-800 text-slate-300 border border-slate-700">
                          <Globe className="w-3 h-3 text-slate-400" />
                          {row.country || 'N/A'}
                        </span>
                      </td>
                    </>
                  )}
                </tr>
              ))}
              {sortedRows.length === 0 && (
                <tr>
                  <td
                    colSpan={isGroundTruth ? 3 : 4}
                    className="py-12 text-center text-slate-400 font-sans"
                  >
                    No preview records matching your search query.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-[#121522] flex items-center justify-between text-xs text-slate-400">
          <span>Displaying verified sample schema fields</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-white font-medium transition-colors"
          >
            Close Preview
          </button>
        </div>
      </div>
    </div>
  );
};
