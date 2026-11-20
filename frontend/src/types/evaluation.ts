// TypeScript interfaces for Verity evaluation data

export interface EvaluationSubmitRequest {
  question: string;
  ai_response: string;
  reference_answer?: string;
  source_document?: string;
}

export interface RetrievedEvidenceItem {
  chunk_id: string;
  text: string;
  similarity_score: number;
  dataset_name: string;
  source_record_id: string;
  title?: string | null;
  category?: string | null;
}

export interface RelevanceJudgeOutput {
  agent: string;
  score: number;
  classification: 'fully_relevant' | 'mostly_relevant' | 'partially_relevant' | 'off_topic' | 'unrelated';
  reasoning: string;
  key_points_addressed: string[];
  key_points_missed: string[];
  confidence: 'high' | 'medium' | 'low';
  evaluator_model: string;
}

export interface OrchestratedEvaluationResult {
  submission_id: string;
  status: 'pending' | 'processing' | 'completed' | 'partial' | 'failed';
  relevance?: RelevanceJudgeOutput | null;
  retrieved_evidence: RetrievedEvidenceItem[];
  overall_score?: number | null;
  verdict?: string | null;
  created_at: string;
}

export interface EvaluationResponse {
  id: number;
  question: string;
  ai_response: string;
  reference_answer: string | null;
  source_document: string | null;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'partial';
  created_at: string;
  evaluation_result?: OrchestratedEvaluationResult | null;
}

export interface EvaluationListItem {
  id: number;
  question: string; // truncated preview (up to 100 chars from server)
  status: 'pending' | 'running' | 'completed' | 'failed' | 'partial';
  created_at: string;
}

export interface PaginatedEvaluationsResponse {
  items: EvaluationListItem[];
  total: number;
  page: number;
  page_size: number;
}
