import axios, { AxiosError } from 'axios';
import type {
  EvaluationSubmitRequest,
  EvaluationResponse,
  PaginatedEvaluationsResponse,
  OrchestratedEvaluationResult,
} from '../types/evaluation';

// Base URL comes from the Vite environment variable (set in .env.local)
const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Submit a new evaluation request to the backend.
 * POST /api/v1/evaluations/
 */
export async function submitEvaluation(
  data: EvaluationSubmitRequest
): Promise<EvaluationResponse> {
  const response = await api.post<EvaluationResponse>('/api/v1/evaluations/', data);
  return response.data;
}

/**
 * Fetch a single evaluation submission by ID.
 * GET /api/v1/evaluations/{id}
 */
export async function getEvaluation(id: number): Promise<EvaluationResponse> {
  const response = await api.get<EvaluationResponse>(`/api/v1/evaluations/${id}`);
  return response.data;
}

/**
 * List evaluation submissions with optional pagination.
 * GET /api/v1/evaluations/?page={page}&page_size={pageSize}
 */
export async function listEvaluations(
  page: number = 1,
  pageSize: number = 10
): Promise<PaginatedEvaluationsResponse> {
  const response = await api.get<PaginatedEvaluationsResponse>('/api/v1/evaluations/', {
    params: { page, page_size: pageSize },
  });
  return response.data;
}

/**
 * Run multi-agent evaluation on a stored submission.
 * POST /api/v1/judge/submissions/{id}/evaluate
 */
export async function runSubmissionEvaluation(
  id: number
): Promise<OrchestratedEvaluationResult> {
  const response = await api.post<OrchestratedEvaluationResult>(
    `/api/v1/judge/submissions/${id}/evaluate`
  );
  return response.data;
}

/**
 * Extract a user-friendly error message from an Axios error.
 * Checks response.data.detail first (FastAPI validation errors), then falls
 * back to the HTTP status text or a generic message.
 */
export function extractApiError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const axiosErr = error as AxiosError<{ detail?: string | { msg: string }[] }>;
    const detail = axiosErr.response?.data?.detail;

    if (typeof detail === 'string') {
      return detail;
    }

    // FastAPI validation errors come back as an array of objects
    if (Array.isArray(detail) && detail.length > 0) {
      return detail.map((d) => d.msg).join('; ');
    }

    if (axiosErr.response?.statusText) {
      return `Error ${axiosErr.response.status}: ${axiosErr.response.statusText}`;
    }

    if (axiosErr.message) {
      return axiosErr.message;
    }
  }

  return 'An unexpected error occurred. Please try again.';
}
