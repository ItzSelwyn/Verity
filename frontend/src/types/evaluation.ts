// TypeScript interfaces for Verity evaluation data

export interface EvaluationSubmitRequest {
  question: string;
  ai_response: string;
  reference_answer?: string;
  source_document?: string;
}

export interface EvaluationResponse {
  id: number;
  question: string;
  ai_response: string;
  reference_answer: string | null;
  source_document: string | null;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'partial';
  created_at: string;
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
