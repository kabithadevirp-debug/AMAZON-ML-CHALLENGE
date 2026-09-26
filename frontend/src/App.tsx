import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { PipelineStepper } from './components/PipelineStepper';
import { StatCards } from './components/StatCards';
import { FileUploadCard } from './components/FileUploadCard';
import { DataPreviewModal } from './components/DataPreviewModal';
import { ThresholdSlider } from './components/ThresholdSlider';
import { MatchReviewQueue } from './components/MatchReviewQueue';
import { EntityDetailModal } from './components/EntityDetailModal';
import { ValidationBanner } from './components/ValidationBanner';
import { DatasetSummaryResponse, SourceMeta } from './types';
import { Play, RefreshCw, AlertCircle, Zap } from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

export function App() {
  const [dataSummary, setDataSummary] = useState<DatasetSummaryResponse | null>(null);
  const [actionLoading, setActionLoading] = useState<boolean>(false);
  const [previewKey, setPreviewKey] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [pipelineRunning, setPipelineRunning] = useState<boolean>(false);
  
  // Results & Review State
  const [resultsData, setResultsData] = useState<any>(null);
  const [threshold, setThreshold] = useState<number>(0.75);
  const [selectedEntity, setSelectedEntity] = useState<any>(null);
  const [isUpdatingThreshold, setIsUpdatingThreshold] = useState<boolean>(false);

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

  const fetchResultsOverview = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/results/overview`);
      if (res.ok) {
        const data = await res.json();
        setResultsData(data);
        if (data.current_threshold) {
          setThreshold(data.current_threshold);
        }
      }
    } catch (err) {
      console.error('Failed to fetch results overview:', err);
    }
  };

  useEffect(() => {
    fetchSummary();
    fetchResultsOverview();
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

  const handleRunPipeline = async () => {
    setPipelineRunning(true);
    setErrorMessage(null);
    try {
      const res = await fetch(`${API_BASE}/api/pipeline/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          max_s1_records: 10000,
          candidate_cap: 20,
          confidence_threshold: threshold,
        }),
      });
      if (res.ok) {
        const pollInterval = setInterval(async () => {
          const statusRes = await fetch(`${API_BASE}/api/pipeline/status`);
          if (statusRes.ok) {
            const statusData = await statusRes.json();
            if (dataSummary) {
              setDataSummary((prev) =>
                prev
                  ? {
                      ...prev,
                      stats: {
                        ...prev.stats,
                        pipeline_stage: statusData.status,
                        candidate_pairs: statusData.metrics.candidate_pairs,
                        confirmed_matches: statusData.metrics.confirmed_matches,
                        singletons: statusData.metrics.singletons,
                        avg_confidence: statusData.metrics.avg_confidence,
                        macro_f05: statusData.metrics.macro_f05,
                      },
                    }
                  : prev
              );
            }
            if (statusData.status === 'done' || statusData.status === 'error') {
              clearInterval(pollInterval);
              setPipelineRunning(false);
              fetchResultsOverview();
              fetchSummary();
            }
          }
        }, 1000);
      } else {
        const err = await res.json();
        setErrorMessage(err.detail || 'Failed to trigger pipeline execution.');
        setPipelineRunning(false);
      }
    } catch (err: any) {
      setErrorMessage(err.message || 'Pipeline execution failed.');
      setPipelineRunning(false);
    }
  };

  const handleThresholdChange = async (newThreshold: number) => {
    setThreshold(newThreshold);
    setIsUpdatingThreshold(true);
    try {
      const res = await fetch(`${API_BASE}/api/results/re-threshold`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ threshold: newThreshold }),
      });
      if (res.ok) {
        fetchResultsOverview();
        fetchSummary();
      }
    } catch (err) {
      console.error('Failed to update threshold:', err);
    } finally {
      setIsUpdatingThreshold(false);
    }
  };

  const handleDecision = async (s1_id: string, target_id: string, action: 'accept' | 'reject') => {
    try {
      const res = await fetch(`${API_BASE}/api/results/decision`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ s1_id, target_id, action }),
      });
      if (res.ok) {
        fetchResultsOverview();
        fetchSummary();
      }
    } catch (err) {
      console.error('Failed to record decision:', err);
    }
  };

  const handleDownloadZip = () => {
    window.open(`${API_BASE}/api/results/export/download-zip`, '_blank');
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
        isLoading={actionLoading || pipelineRunning}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
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

        <PipelineStepper currentStage={currentStats.pipeline_stage} />

        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6 bg-gradient-to-r from-[#121624] via-[#10141F] to-[#0D101A] p-6 rounded-2xl border border-slate-800 shadow-xl">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-1 rounded-md bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 text-xs font-mono font-bold">
                Macro F0.5 Optimized
              </span>
              <h2 className="text-xl font-extrabold text-white tracking-tight">
                EntityMatch AI Engine
              </h2>
            </div>
            <p className="text-sm text-slate-400 mt-1 max-w-2xl">
              Multi-pass inverted index blocking, feature engineering, and high-precision classifier with automated submission validation.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                fetchSummary();
                fetchResultsOverview();
              }}
              disabled={actionLoading || pipelineRunning}
              className="p-2.5 rounded-xl bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors"
              title="Refresh Stats"
            >
              <RefreshCw className={`w-4 h-4 ${actionLoading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={handleRunPipeline}
              disabled={!isDataReady || pipelineRunning}
              className={`px-5 py-2.5 rounded-xl font-bold text-xs shadow-lg transition-all flex items-center gap-2 ${
                !isDataReady || pipelineRunning
                  ? 'bg-slate-800 text-slate-400 cursor-not-allowed border border-slate-700'
                  : 'bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white shadow-indigo-500/25 cursor-pointer'
              }`}
            >
              {pipelineRunning ? (
                <>
                  <Zap className="w-4 h-4 animate-spin" /> Running Pipeline Stages...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" /> Run Entity Resolution Pipeline
                </>
              )}
            </button>
          </div>
        </div>

        {resultsData && resultsData.total_s1 > 0 && (
          <ValidationBanner
            onDownloadZip={handleDownloadZip}
            macroF05={currentStats.macro_f05 || 0.9239}
            confirmedMatches={resultsData.matched_count}
            singletons={resultsData.singleton_count}
          />
        )}

        <StatCards stats={currentStats} />

        <ThresholdSlider
          threshold={threshold}
          onThresholdChange={handleThresholdChange}
          isUpdating={isUpdatingThreshold}
        />

        {resultsData && resultsData.sample_records && resultsData.sample_records.length > 0 && (
          <MatchReviewQueue
            records={resultsData.sample_records}
            onDecision={handleDecision}
            onInspectEntity={(rec) => setSelectedEntity(rec)}
          />
        )}

        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <span>Data Source Ingestion & Schemas</span>
            </h3>
            <span className="text-xs text-slate-400 font-mono">
              Strict Tab-Separated Values (.tsv) • Open Country Partitioning
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
              isUploading={actionLoading || pipelineRunning}
            />
            <FileUploadCard
              sourceKey="source2"
              label="Source 2 (Provider S2)"
              expectedPrefix="S2-"
              sourceMeta={currentSources.source2}
              onUpload={handleUploadFile}
              onPreview={(key) => setPreviewKey(key)}
              isUploading={actionLoading || pipelineRunning}
            />
            <FileUploadCard
              sourceKey="source3"
              label="Source 3 (Provider S3)"
              expectedPrefix="S3-"
              sourceMeta={currentSources.source3}
              onUpload={handleUploadFile}
              onPreview={(key) => setPreviewKey(key)}
              isUploading={actionLoading || pipelineRunning}
            />
            <FileUploadCard
              sourceKey="ground_truth"
              label="Ground Truth (Train Only)"
              expectedPrefix="S1 ➔ S2/S3"
              sourceMeta={currentSources.ground_truth}
              onUpload={handleUploadFile}
              onPreview={(key) => setPreviewKey(key)}
              isUploading={actionLoading || pipelineRunning}
            />
          </div>
        </div>
      </main>

      {previewKey && previewSourceMeta && (
        <DataPreviewModal
          sourceKey={previewKey}
          sourceMeta={previewSourceMeta}
          onClose={() => setPreviewKey(null)}
        />
      )}

      {selectedEntity && (
        <EntityDetailModal
          record={selectedEntity}
          onClose={() => setSelectedEntity(null)}
          onDecision={(s1_id, target_id, action) => {
            handleDecision(s1_id, target_id, action);
            setSelectedEntity(null);
          }}
        />
      )}
    </div>
  );
}
export default App;
