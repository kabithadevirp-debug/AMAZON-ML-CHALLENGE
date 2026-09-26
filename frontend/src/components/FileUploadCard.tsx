import React, { useRef, useState } from 'react';
import { UploadCloud, CheckCircle, AlertTriangle, XCircle, Eye, FileText, Globe } from 'lucide-react';
import { SourceMeta } from '../types';

interface FileUploadCardProps {
  sourceKey: 'source1' | 'source2' | 'source3' | 'ground_truth';
  label: string;
  expectedPrefix: string;
  sourceMeta: SourceMeta;
  onUpload: (sourceKey: string, file: File) => void;
  onPreview: (sourceKey: string) => void;
  isUploading: boolean;
}

export const FileUploadCard: React.FC<FileUploadCardProps> = ({
  sourceKey,
  label,
  expectedPrefix,
  sourceMeta,
  onUpload,
  onPreview,
  isUploading,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onUpload(sourceKey, e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onUpload(sourceKey, e.target.files[0]);
    }
  };

  const isReady = sourceMeta.status === 'ready' && sourceMeta.total_rows > 0;
  const isError = sourceMeta.status === 'error';

  return (
    <div
      className={`relative rounded-2xl border p-5 transition-all flex flex-col justify-between ${
        isDragOver
          ? 'bg-indigo-950/40 border-indigo-500 shadow-lg shadow-indigo-500/20'
          : isReady
          ? 'bg-[#121520] border-slate-800 hover:border-slate-700'
          : isError
          ? 'bg-rose-950/20 border-rose-800/60'
          : 'bg-[#10121A] border-slate-800/80 hover:border-slate-700'
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".tsv,.txt"
        className="hidden"
        onChange={handleFileChange}
      />

      <div>
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded text-[11px] font-mono font-medium bg-slate-800 text-slate-300 border border-slate-700">
                {expectedPrefix}
              </span>
              <h3 className="text-sm font-bold text-white tracking-tight">{label}</h3>
            </div>
            <p className="text-xs text-slate-400 mt-1 font-mono">
              {sourceMeta.filename || 'No file loaded yet'}
            </p>
          </div>

          {isReady ? (
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs">
              <CheckCircle className="w-3.5 h-3.5" />
              <span>Valid TSV</span>
            </div>
          ) : isError ? (
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs">
              <XCircle className="w-3.5 h-3.5" />
              <span>Schema Error</span>
            </div>
          ) : (
            <span className="text-xs text-slate-400 px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
              Pending
            </span>
          )}
        </div>

        {/* Status / Counts / Country Pills */}
        {isReady ? (
          <div className="my-4 p-3 rounded-xl bg-slate-900/80 border border-slate-800">
            <div className="flex items-baseline justify-between mb-2">
              <span className="text-xs text-slate-400">Total Valid Records</span>
              <span className="text-lg font-bold font-mono text-white">
                {sourceMeta.total_rows.toLocaleString()}
              </span>
            </div>

            {/* Country Distribution */}
            {Object.keys(sourceMeta.country_counts).length > 0 && (
              <div className="flex items-center gap-1.5 flex-wrap pt-2 border-t border-slate-800/80">
                <Globe className="w-3 h-3 text-slate-400 shrink-0" />
                {Object.entries(sourceMeta.country_counts).map(([country, count]) => (
                  <span
                    key={country}
                    className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700"
                  >
                    {country}: {count.toLocaleString()}
                  </span>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div
            onClick={() => fileInputRef.current?.click()}
            className="my-4 border border-dashed border-slate-700/80 hover:border-indigo-500/50 rounded-xl p-6 text-center cursor-pointer bg-slate-900/30 hover:bg-slate-900/60 transition-all group"
          >
            <UploadCloud className="w-8 h-8 text-slate-400 group-hover:text-indigo-400 mx-auto mb-2 transition-colors" />
            <p className="text-xs font-medium text-slate-300 group-hover:text-white">
              {isUploading ? 'Uploading & Validating...' : 'Drop .tsv file here, or click to browse'}
            </p>
            <p className="text-[11px] text-slate-400 mt-1 font-mono">
              Strict Tab-separated UTF-8 format
            </p>
          </div>
        )}

        {/* Errors / Warnings */}
        {sourceMeta.errors && sourceMeta.errors.length > 0 && (
          <div className="mb-3 p-2.5 rounded-lg bg-rose-950/40 border border-rose-800/50 text-[11px] text-rose-300">
            {sourceMeta.errors.map((err, i) => (
              <p key={i} className="flex items-center gap-1.5">
                <AlertTriangle className="w-3 h-3 text-rose-400 shrink-0" />
                {err}
              </p>
            ))}
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2 pt-3 border-t border-slate-800/80">
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          className="flex-1 py-1.5 px-3 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors flex items-center justify-center gap-1.5"
        >
          <FileText className="w-3.5 h-3.5" />
          {isReady ? 'Replace File' : 'Select TSV'}
        </button>

        {isReady && (
          <button
            onClick={() => onPreview(sourceKey)}
            className="py-1.5 px-3 rounded-lg text-xs font-medium bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/30 transition-colors flex items-center justify-center gap-1.5"
          >
            <Eye className="w-3.5 h-3.5" />
            Preview 20 Rows
          </button>
        )}
      </div>
    </div>
  );
};
