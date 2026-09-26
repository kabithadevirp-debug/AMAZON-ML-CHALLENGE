import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { PipelineStepper } from './components/PipelineStepper';
import { StatCards } from './components/StatCards';
import { FileUploadCard } from './components/FileUploadCard';
import { DataPreviewModal } from './components/DataPreviewModal';
import { DatasetSummaryResponse, SourceMeta } from './types';
import { Sparkles, RefreshCw, AlertCircle, ArrowRight, ShieldCheck } from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

export function App() {
  const [dataSummary, setDataSummary] = useState<DatasetSummaryResponse | null>(null);
  const [actionLoading, setActionLoading] = useState<boolean>(false);
  const [previewKey, setPreviewKey] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fetchSummary = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/datasets/summary`);
      if (res.ok) {
        const data = await res.json();
        setDataSummary(data);
      }
    } catch (err: any) {
      console.error('Failed to fetch summary:', err);
      setErrorMessage('Backend connection unavailable. Ensure FastAPI server is running on port 8000.');
    }
  };

  useEffect(() => {
    fetchSummary();
  }, []);

  const handleLoadPreset = async (preset: 'train' | 'test') => {
    setActionLoading(true);
    setErrorMessage(null);
    try {
      const res = await fetch(`${API_BASE}/api/datasets/load-preset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preset_type: preset }),
      });
      if (res.ok) {
        const data = await res.json();
        setDataSummary(data.summary);
      } else {
        const err = await res.json();
        setErrorMessage(err.detail || 'Failed to load dataset preset.');
      }
    } catch (err: any) {
      setErrorMessage(err.message || 'Error communicating with backend.');
    } finally {
      setActionLoading(false);
    }
  };

  const handleUploadFile = async (sourceKey: string, file: File) => {
    setActionLoading(true);
    setErrorMessage(null);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('source_type', sourceKey);

    try {
      const res = await fetch(`${API_BASE}/api/datasets/upload`, {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        setDataSummary(data.summary);
      } else {
        const err = await res.json();
        setErrorMessage(err.detail || 'File upload failed validation.');
      }
    } catch (err: any) {
      setErrorMessage(err.message || 'Network error during upload.');
    } finally {
      setActionLoading(false);
    }
  };

  const defaultStats = {
    s1_count: 0,
    s2_count: 0,
    s3_count: 0,
    candidate_pairs: 0,
    confirmed_matches: 0,
    singletons: 0,
    avg_confidence: 0,
    macro_f05: 0,
    blocking_recall: 0,
    pipeline_stage: 'idle' as const,
  };

  const currentStats = dataSummary?.stats || defaultStats;
  const currentSources = dataSummary?.sources || {
    source1: { file_path: null, total_rows: 0, status: 'missing', country_counts: {}, sample_rows: [] },
    source2: { file_path: null, total_rows: 0, status: 'missing', country_counts: {}, sample_rows: [] },
    source3: { file_path: null, total_rows: 0, status: 'missing', country_counts: {}, sample_rows: [] },
    ground_truth: { file_path: null, total_rows: 0, status: 'none', country_counts: {}, sample_rows: [] },
  };

  const previewSourceMeta: SourceMeta | undefined = previewKey
    ? (currentSources as any)[previewKey]
    : undefined;

  const isDataReady = currentSources.source1.status === 'ready' && currentSources.source2.status === 'ready';

  return (
    <div className="min-h-screen bg-[#090B10] text-slate-100 flex flex-col font-sans">
      <Header
        currentPreset={dataSummary?.current_preset || null}
        onLoadPreset={handleLoadPreset}
        isLoading={actionLoading}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Alert Banner */}
        {errorMessage && (
          <div className="mb-6 p-4 rounded-xl bg-rose-950/40 border border-rose-800/80 flex items-center justify-between text-rose-300 text-sm">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
              <span>{errorMessage}</span>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="text-xs underline hover:text-rose-200"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Pipeline Execution Stepper */}
        <PipelineStepper currentStage={currentStats.pipeline_stage} />

        {/* Milestone 1 Header Banner */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6 bg-gradient-to-r from-[#121624] via-[#10141F] to-[#0D101A] p-6 rounded-2xl border border-slate-800">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-1 rounded-md bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 text-xs font-mono font-semibold">
                Milestone 1
              </span>
              <h2 className="text-xl font-extrabold text-white tracking-tight">
                Data Foundation & Stream Ingestion
              </h2>
            </div>
            <p className="text-sm text-slate-400 mt-1 max-w-2xl">
              Upload Source 1 (Reference), Source 2, Source 3, and Ground Truth TSVs with strict column and delimiter validation, or load pre-bundled datasets with instant sub-second verification.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => fetchSummary()}
              disabled={actionLoading}
              className="p-2.5 rounded-xl bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors"
              title="Refresh Stats"
            >
              <RefreshCw className={`w-4 h-4 ${actionLoading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={() => handleLoadPreset('train')}
              disabled={actionLoading}
              className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white font-semibold text-xs shadow-lg shadow-indigo-500/25 transition-all flex items-center gap-2"
            >
              <Sparkles className="w-4 h-4" />
              Auto-Load Training Set (2.2M)
            </button>
          </div>
        </div>

        {/* Stat Cards */}
        <StatCards stats={currentStats} />

        {/* Source File Cards Grid */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <span>Data Source Ingestion & Schema Verification</span>
            </h3>
            <span className="text-xs text-slate-400 font-mono">
              Format: Tab-Separated Values (.tsv) • Encoding: UTF-8
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <FileUploadCard
              sourceKey="source1"
              label="Source 1 (Reference S1)"
              expectedPrefix="S1-"
              sourceMeta={currentSources.source1}
              onUpload={handleUploadFile}
              onPreview={(key) => setPreviewKey(key)}
              isUploading={actionLoading}
            />
            <FileUploadCard
              sourceKey="source2"
              label="Source 2 (Provider S2)"
              expectedPrefix="S2-"
              sourceMeta={currentSources.source2}
              onUpload={handleUploadFile}
              onPreview={(key) => setPreviewKey(key)}
              isUploading={actionLoading}
            />
            <FileUploadCard
              sourceKey="source3"
              label="Source 3 (Provider S3)"
              expectedPrefix="S3-"
              sourceMeta={currentSources.source3}
              onUpload={handleUploadFile}
              onPreview={(key) => setPreviewKey(key)}
              isUploading={actionLoading}
            />
            <FileUploadCard
              sourceKey="ground_truth"
              label="Ground Truth (Train Only)"
              expectedPrefix="S1 ➔ S2/S3"
              sourceMeta={currentSources.ground_truth}
              onUpload={handleUploadFile}
              onPreview={(key) => setPreviewKey(key)}
              isUploading={actionLoading}
            />
          </div>
        </div>

        {/* Next Step Action bar */}
        <div className="p-5 rounded-2xl bg-[#0F121C] border border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 shrink-0">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <p className="text-sm font-semibold text-white">
                {isDataReady ? 'Data Ingestion Complete & Validated' : 'Awaiting Dataset Ingestion'}
              </p>
              <p className="text-xs text-slate-400">
                {isDataReady
                  ? 'All schemas verified. Ready for Milestone 2: Normalization & Multi-Pass Candidate Blocking.'
                  : 'Load the bundled dataset or upload TSV files to proceed.'}
              </p>
            </div>
          </div>

          <button
            disabled={!isDataReady}
            className={`px-5 py-2.5 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all ${
              isDataReady
                ? 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-500/20 cursor-pointer'
                : 'bg-slate-800 text-slate-400 cursor-not-allowed border border-slate-700'
            }`}
          >
            <span>Proceed to Milestone 2: Blocking Engine</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </main>

      {/* Data Preview Modal */}
      {previewKey && previewSourceMeta && (
        <DataPreviewModal
          sourceKey={previewKey}
          sourceMeta={previewSourceMeta}
          onClose={() => setPreviewKey(null)}
        />
      )}
    </div>
  );
}
export default App;
