export interface SourceRow {
  entity_id: string;
  business_name: string;
  business_address: string;
  country: string;
}

export interface GroundTruthRow {
  source1_entity_id: string;
  matched_entity_ids: string;
}

export interface SourceMeta {
  file_path: string | null;
  filename?: string;
  total_rows: number;
  status: 'ready' | 'missing' | 'error' | 'none';
  country_counts: Record<string, number>;
  sample_rows: Array<SourceRow | GroundTruthRow>;
  warnings?: string[];
  errors?: string[];
}

export interface PipelineStats {
  s1_count: number;
  s2_count: number;
  s3_count: number;
  candidate_pairs: number;
  confirmed_matches: number;
  singletons: number;
  avg_confidence: number;
  macro_f05: number;
  blocking_recall: number;
  pipeline_stage: 'idle' | 'normalizing' | 'blocking' | 'features' | 'matching' | 'done';
}

export interface DatasetSummaryResponse {
  sources: {
    source1: SourceMeta;
    source2: SourceMeta;
    source3: SourceMeta;
    ground_truth: SourceMeta;
  };
  stats: PipelineStats;
  current_preset: 'train' | 'test' | 'custom_upload' | null;
}
