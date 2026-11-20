import React, { useEffect, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  RotateCw,
  Inbox,
  FileText,
  Clock,
  Calendar,
  AlertCircle,
  Hash,
  ChevronRight,
  Zap,
  Database,
  Check,
} from '../components/icons';
import {
  listEvaluations,
  getEvaluation,
  runSubmissionEvaluation,
  extractApiError,
} from '../api/evaluationApi';
import type { EvaluationListItem, EvaluationResponse } from '../types/evaluation';
import StatusBadge from '../components/StatusBadge';

/** Format an ISO date string as a human-readable relative time */
function formatRelativeTime(isoString: string): string {
  const date = new Date(isoString);
  const now = new Date();
  const diffSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (diffSeconds < 60) return 'just now';
  if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)}m ago`;
  if (diffSeconds < 86400) return `${Math.floor(diffSeconds / 3600)}h ago`;
  if (diffSeconds < 604800) return `${Math.floor(diffSeconds / 86400)}d ago`;

  return date.toLocaleDateString();
}

/** Truncate string to maxLen characters */
function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen) + '…';
}

/** Format an ISO date string for detailed view */
function formatFullDate(isoString: string): string {
  return new Date(isoString).toLocaleString('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  });
}

/** Detail section card */
const DetailSection: React.FC<{ label: string; value: string | null; placeholder?: string }> = ({
  label,
  value,
  placeholder = 'Not provided by user',
}) => (
  <div style={detailStyles.section}>
    <div style={detailStyles.sectionHeader}>
      <span style={detailStyles.sectionLabel}>{label}</span>
      {value && <span style={detailStyles.charCount}>{value.length} characters</span>}
    </div>
    <div style={detailStyles.sectionContent}>
      {value ? (
        <p style={detailStyles.sectionText}>{value}</p>
      ) : (
        <p style={detailStyles.sectionMuted}>{placeholder}</p>
      )}
    </div>
  </div>
);

const detailStyles: Record<string, React.CSSProperties> = {
  section: {
    marginBottom: '24px',
  },
  sectionHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '8px',
  },
  sectionLabel: {
    fontSize: '13px',
    fontWeight: '800',
    textTransform: 'uppercase',
    letterSpacing: '0.6px',
    color: '#09090b',
  },
  charCount: {
    fontSize: '12px',
    color: '#71717a',
    fontWeight: '500',
  },
  sectionContent: {
    backgroundColor: '#ffffff',
    border: '1px solid #e4e4e7',
    borderRadius: '8px',
    padding: '16px 18px',
    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.02)',
  },
  sectionText: {
    fontSize: '15px',
    color: '#09090b',
    lineHeight: '1.7',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },
  sectionMuted: {
    fontSize: '14px',
    color: '#71717a',
    fontStyle: 'italic',
  },
};

function getRelevanceBadgeStyle(classification?: string): React.CSSProperties {
  switch (classification) {
    case 'fully_relevant':
      return { backgroundColor: '#09090b', color: '#ffffff' };
    case 'mostly_relevant':
      return { backgroundColor: '#27272a', color: '#ffffff' };
    case 'partially_relevant':
      return { backgroundColor: '#f4f4f5', color: '#09090b', border: '1px solid #d4d4d8' };
    case 'off_topic':
    case 'unrelated':
      return { backgroundColor: '#ffffff', color: '#09090b', border: '1px solid #09090b' };
    default:
      return { backgroundColor: '#f4f4f5', color: '#71717a' };
  }
}

const resultStyles: Record<string, React.CSSProperties> = {
  resultsContainer: {
    marginBottom: '32px',
    border: '1px solid #e4e4e7',
    borderRadius: '12px',
    padding: '24px',
    backgroundColor: '#ffffff',
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.04)',
  },
  resultsHeader: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    marginBottom: '20px',
    borderBottom: '1px solid #f4f4f5',
    paddingBottom: '16px',
    flexWrap: 'wrap',
    gap: '12px',
  },
  resultsPill: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '11px',
    fontWeight: '700',
    letterSpacing: '0.6px',
    color: '#09090b',
    backgroundColor: '#f4f4f5',
    padding: '4px 10px',
    borderRadius: '100px',
    marginBottom: '8px',
  },
  resultsTitle: {
    fontSize: '20px',
    fontWeight: '800',
    color: '#09090b',
    margin: '0 0 4px 0',
  },
  resultsSubtitle: {
    fontSize: '13px',
    color: '#71717a',
    margin: 0,
  },
  rerunBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 14px',
    fontSize: '13px',
    fontWeight: '600',
    color: '#09090b',
    backgroundColor: '#ffffff',
    border: '1px solid #e4e4e7',
    borderRadius: '6px',
    cursor: 'pointer',
    transition: 'all 0.15s ease',
  },
  card: {
    border: '1px solid #e4e4e7',
    borderRadius: '10px',
    padding: '20px',
    backgroundColor: '#fafafa',
    marginBottom: '20px',
  },
  cardHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '14px',
    flexWrap: 'wrap',
    gap: '10px',
  },
  cardBadgeGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    flexWrap: 'wrap',
  },
  cardTypeTag: {
    fontSize: '11px',
    fontWeight: '800',
    letterSpacing: '0.8px',
    color: '#09090b',
  },
  classificationBadge: {
    fontSize: '11px',
    fontWeight: '700',
    padding: '3px 9px',
    borderRadius: '100px',
    letterSpacing: '0.5px',
  },
  scoreBox: {
    display: 'flex',
    alignItems: 'baseline',
    gap: '4px',
  },
  scoreNumber: {
    fontSize: '28px',
    fontWeight: '900',
    color: '#09090b',
    lineHeight: 1,
  },
  scoreMax: {
    fontSize: '14px',
    fontWeight: '600',
    color: '#71717a',
  },
  metaInfoRow: {
    display: 'flex',
    gap: '8px',
    marginBottom: '16px',
    flexWrap: 'wrap',
  },
  metaPill: {
    fontSize: '11px',
    color: '#52525b',
    backgroundColor: '#ffffff',
    border: '1px solid #e4e4e7',
    padding: '3px 8px',
    borderRadius: '4px',
    fontWeight: '500',
  },
  reasoningBox: {
    backgroundColor: '#ffffff',
    border: '1px solid #e4e4e7',
    borderRadius: '8px',
    padding: '14px 16px',
    marginBottom: '16px',
  },
  sectionSubtitle: {
    fontSize: '12px',
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    color: '#09090b',
    margin: '0 0 6px 0',
  },
  reasoningText: {
    fontSize: '14px',
    color: '#18181b',
    lineHeight: '1.6',
    margin: 0,
  },
  pointsGrid: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    marginTop: '12px',
  },
  pointGroupLabel: {
    fontSize: '12px',
    fontWeight: '700',
    color: '#09090b',
    display: 'block',
    marginBottom: '6px',
  },
  tagRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '6px',
  },
  tagAddressed: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '5px',
    fontSize: '12px',
    fontWeight: '600',
    backgroundColor: '#ffffff',
    color: '#09090b',
    border: '1px solid #d4d4d8',
    padding: '4px 10px',
    borderRadius: '6px',
  },
  tagMissed: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '5px',
    fontSize: '12px',
    fontWeight: '500',
    backgroundColor: '#ffffff',
    color: '#71717a',
    border: '1px solid #e4e4e7',
    padding: '4px 10px',
    borderRadius: '6px',
  },
  evidenceIntro: {
    fontSize: '13px',
    color: '#71717a',
    marginBottom: '14px',
    marginTop: 0,
  },
  evidenceCountBadge: {
    fontSize: '12px',
    fontWeight: '600',
    color: '#52525b',
    backgroundColor: '#ffffff',
    border: '1px solid #e4e4e7',
    padding: '3px 8px',
    borderRadius: '100px',
  },
  noEvidenceText: {
    fontSize: '13px',
    color: '#71717a',
    fontStyle: 'italic',
    padding: '12px',
    backgroundColor: '#ffffff',
    border: '1px dashed #d4d4d8',
    borderRadius: '6px',
    margin: 0,
  },
  evidenceList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  evidenceItem: {
    backgroundColor: '#ffffff',
    border: '1px solid #e4e4e7',
    borderRadius: '8px',
    padding: '14px',
  },
  evidenceItemHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '10px',
    flexWrap: 'wrap',
  },
  similarityBadge: {
    fontSize: '12px',
    fontWeight: '800',
    color: '#09090b',
    backgroundColor: '#f4f4f5',
    padding: '2px 8px',
    borderRadius: '4px',
  },
  datasetBadge: {
    fontSize: '11px',
    fontWeight: '700',
    letterSpacing: '0.5px',
    color: '#ffffff',
    backgroundColor: '#09090b',
    padding: '2px 7px',
    borderRadius: '4px',
  },
  evidenceTopicBadge: {
    fontSize: '11px',
    fontWeight: '600',
    color: '#52525b',
    backgroundColor: '#f4f4f5',
    padding: '2px 7px',
    borderRadius: '4px',
  },
  sourceIdBadge: {
    fontSize: '11px',
    fontWeight: '500',
    color: '#71717a',
    marginLeft: 'auto',
  },
  evidenceTextQuote: {
    fontSize: '13.5px',
    lineHeight: '1.6',
    color: '#18181b',
    borderLeft: '3px solid #09090b',
    paddingLeft: '12px',
    backgroundColor: '#fafafa',
    padding: '10px 12px',
    borderRadius: '0 6px 6px 0',
  },
  actionBanner: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#ffffff',
    border: '1px solid #e4e4e7',
    borderRadius: '10px',
    padding: '18px 20px',
    marginBottom: '24px',
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.03)',
    flexWrap: 'wrap',
    gap: '16px',
  },
  actionBannerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '14px',
    maxWidth: '560px',
  },
  actionBannerIcon: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '40px',
    height: '40px',
    backgroundColor: '#f4f4f5',
    borderRadius: '8px',
    flexShrink: 0,
  },
  actionBannerTitle: {
    fontSize: '15px',
    fontWeight: '700',
    color: '#09090b',
    margin: '0 0 3px 0',
  },
  actionBannerDesc: {
    fontSize: '13px',
    color: '#71717a',
    margin: 0,
    lineHeight: '1.5',
  },
  primaryRunBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '8px',
    backgroundColor: '#09090b',
    color: '#ffffff',
    border: 'none',
    padding: '11px 20px',
    borderRadius: '6px',
    fontSize: '14px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'opacity 0.15s ease',
  },
  errorBanner: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    backgroundColor: '#fafafa',
    border: '1px solid #09090b',
    padding: '12px 16px',
    borderRadius: '6px',
    marginBottom: '20px',
    fontSize: '13.5px',
    color: '#09090b',
    fontWeight: '500',
  },
};

const EvaluationHistoryPage: React.FC = () => {
  const { id } = useParams<{ id?: string }>();
  const navigate = useNavigate();

  // List state
  const [items, setItems] = useState<EvaluationListItem[]>([]);
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);

  // Detail state
  const [detail, setDetail] = useState<EvaluationResponse | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  // Evaluation execution state
  const [evaluating, setEvaluating] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);

  const selectedId = id ? parseInt(id, 10) : null;

  const handleRunEvaluation = async () => {
    if (!detail) return;
    setEvaluating(true);
    setEvalError(null);
    try {
      const result = await runSubmissionEvaluation(detail.id);
      setDetail((prev) => (prev ? { ...prev, status: 'completed', evaluation_result: result } : null));
      setItems((prev) =>
        prev.map((item) => (item.id === detail.id ? { ...item, status: 'completed' } : item))
      );
    } catch (err) {
      setEvalError(extractApiError(err));
    } finally {
      setEvaluating(false);
    }
  };

  // Load list
  const loadList = useCallback(async () => {
    setListLoading(true);
    setListError(null);
    try {
      const data = await listEvaluations(1, 25);
      setItems(data.items);
    } catch (err) {
      setListError(extractApiError(err));
    } finally {
      setListLoading(false);
    }
  }, []);

  useEffect(() => {
    loadList();
  }, [loadList]);

  // Load detail on id change
  useEffect(() => {
    if (selectedId === null) {
      setDetail(null);
      return;
    }

    let cancelled = false;
    setDetailLoading(true);
    setDetailError(null);
    setDetail(null);

    getEvaluation(selectedId)
      .then((data) => {
        if (!cancelled) setDetail(data);
      })
      .catch((err) => {
        if (!cancelled) setDetailError(extractApiError(err));
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  const handleSelectItem = (itemId: number) => {
    navigate(`/history/${itemId}`);
  };

  return (
    <div style={styles.page}>
      {/* ── Left Column: Submissions List ── */}
      <aside style={styles.listPanel}>
        <div style={styles.listHeader}>
          <div style={styles.listTitleGroup}>
            <h2 style={styles.listTitle}>Submissions</h2>
            {items.length > 0 && (
              <span style={styles.countBadge}>{items.length}</span>
            )}
          </div>
          <button
            style={styles.refreshBtn}
            onClick={loadList}
            disabled={listLoading}
            title="Refresh history"
            aria-label="Refresh history"
          >
            <RotateCw
              size={15}
              style={{
                animation: listLoading ? 'spin 1s linear infinite' : undefined,
              }}
            />
          </button>
        </div>

        {listLoading && (
          <div style={styles.centerMessage}>
            <RotateCw size={18} style={{ animation: 'spin 1s linear infinite' }} />
            <span>Loading submissions…</span>
          </div>
        )}

        {listError && (
          <div style={styles.errorBox}>
            <AlertCircle size={16} />
            <p>{listError}</p>
            <button style={styles.retryBtn} onClick={loadList}>
              Retry
            </button>
          </div>
        )}

        {!listLoading && !listError && items.length === 0 && (
          <div style={styles.emptyState}>
            <Inbox size={32} strokeWidth={1.5} color="#71717a" />
            <h4 style={styles.emptyTitle}>No Submissions Yet</h4>
            <p style={styles.emptyText}>
              Create a new evaluation to start building your evaluation history.
            </p>
          </div>
        )}

        {!listLoading && items.length > 0 && (
          <ul style={styles.list}>
            {items.map((item) => {
              const isSelected = item.id === selectedId;
              return (
                <li
                  key={item.id}
                  style={{
                    ...styles.listItem,
                    ...(isSelected ? styles.listItemActive : {}),
                  }}
                  onClick={() => handleSelectItem(item.id)}
                >
                  <div style={styles.itemHeader}>
                    <span style={styles.itemId}>#{item.id}</span>
                    <StatusBadge status={item.status} size="sm" />
                  </div>
                  <p style={styles.itemQuestion}>{truncate(item.question, 95)}</p>
                  <div style={styles.itemFooter}>
                    <span style={styles.itemDate}>
                      <Clock size={12} />
                      {formatRelativeTime(item.created_at)}
                    </span>
                    <ChevronRight size={14} color="#71717a" />
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </aside>

      {/* ── Right Column: Submission Detail View ── */}
      <section style={styles.detailPanel}>
        {selectedId === null && (
          <div style={styles.emptyPlaceholder}>
            <div style={styles.placeholderIconCircle}>
              <FileText size={32} strokeWidth={1.5} color="#09090b" />
            </div>
            <h3 style={styles.placeholderTitle}>Select an Evaluation</h3>
            <p style={styles.placeholderSubtitle}>
              Click on any submission on the left to inspect its question, response, and reference context.
            </p>
          </div>
        )}

        {selectedId !== null && detailLoading && (
          <div style={styles.emptyPlaceholder}>
            <RotateCw size={24} style={{ animation: 'spin 1s linear infinite' }} />
            <p style={{ marginTop: '12px', fontSize: '15px', color: '#52525b' }}>
              Fetching submission #{selectedId}…
            </p>
          </div>
        )}

        {selectedId !== null && !detailLoading && detailError && (
          <div style={styles.emptyPlaceholder}>
            <AlertCircle size={32} color="#09090b" />
            <h4 style={{ marginTop: '12px', fontSize: '16px', color: '#09090b' }}>
              Could not load submission
            </h4>
            <p style={{ fontSize: '14px', color: '#52525b' }}>{detailError}</p>
          </div>
        )}

        {detail && !detailLoading && (
          <div style={styles.detailContainer}>
            {/* Detail Top Header */}
            <div style={styles.detailHeader}>
              <div style={styles.detailTitleGroup}>
                <div style={styles.detailIdPill}>
                  <Hash size={14} />
                  <span>{detail.id}</span>
                </div>
                <h1 style={styles.detailTitle}>Evaluation Record</h1>
                <StatusBadge status={detail.status} size="md" />
              </div>

              <div style={styles.detailMetaRow}>
                <span style={styles.metaItem}>
                  <Calendar size={14} />
                  <span>Submitted {formatFullDate(detail.created_at)}</span>
                </span>
              </div>
            </div>

            {evalError && (
              <div style={resultStyles.errorBanner}>
                <AlertCircle size={16} color="#09090b" />
                <span>{evalError}</span>
              </div>
            )}

            {detail.evaluation_result ? (
              /* Multi-Agent Evaluation Results Dashboard */
              <div style={resultStyles.resultsContainer}>
                <div style={resultStyles.resultsHeader}>
                  <div>
                    <div style={resultStyles.resultsPill}>
                      <Zap size={13} color="#09090b" />
                      <span>MILESTONE 2.1 EVALUATION REPORT</span>
                    </div>
                    <h2 style={resultStyles.resultsTitle}>Evaluation Results</h2>
                    <p style={resultStyles.resultsSubtitle}>
                      Calculated by Relevance Judge Agent & Knowledge Base Retrieval
                    </p>
                  </div>
                  <button
                    onClick={handleRunEvaluation}
                    disabled={evaluating}
                    style={resultStyles.rerunBtn}
                  >
                    <RotateCw
                      size={14}
                      style={{ animation: evaluating ? 'spin 1s linear infinite' : undefined }}
                    />
                    <span>{evaluating ? 'Evaluating…' : 'Re-run Evaluation'}</span>
                  </button>
                </div>

                {/* Relevance Judge Card */}
                {detail.evaluation_result.relevance && (
                  <div style={resultStyles.card}>
                    <div style={resultStyles.cardHeader}>
                      <div style={resultStyles.cardBadgeGroup}>
                        <span style={resultStyles.cardTypeTag}>RELEVANCE JUDGE AGENT</span>
                        <span
                          style={{
                            ...resultStyles.classificationBadge,
                            ...getRelevanceBadgeStyle(detail.evaluation_result.relevance.classification),
                          }}
                        >
                          {detail.evaluation_result.relevance.classification.replace('_', ' ').toUpperCase()}
                        </span>
                      </div>

                      <div style={resultStyles.scoreBox}>
                        <span style={resultStyles.scoreNumber}>
                          {detail.evaluation_result.relevance.score.toFixed(1)}
                        </span>
                        <span style={resultStyles.scoreMax}>/ 10.0</span>
                      </div>
                    </div>

                    <div style={resultStyles.metaInfoRow}>
                      <span style={resultStyles.metaPill}>
                        Confidence: {detail.evaluation_result.relevance.confidence.toUpperCase()}
                      </span>
                      <span style={resultStyles.metaPill}>
                        Model: {detail.evaluation_result.relevance.evaluator_model}
                      </span>
                    </div>

                    <div style={resultStyles.reasoningBox}>
                      <h5 style={resultStyles.sectionSubtitle}>Assessment Reasoning</h5>
                      <p style={resultStyles.reasoningText}>
                        {detail.evaluation_result.relevance.reasoning}
                      </p>
                    </div>

                    {/* Addressed & Missed Key Points */}
                    {((detail.evaluation_result.relevance.key_points_addressed?.length ?? 0) > 0 ||
                      (detail.evaluation_result.relevance.key_points_missed?.length ?? 0) > 0) && (
                      <div style={resultStyles.pointsGrid}>
                        {(detail.evaluation_result.relevance.key_points_addressed?.length ?? 0) > 0 && (
                          <div>
                            <span style={resultStyles.pointGroupLabel}>Key Points Addressed:</span>
                            <div style={resultStyles.tagRow}>
                              {detail.evaluation_result.relevance.key_points_addressed.map((pt, idx) => (
                                <span key={idx} style={resultStyles.tagAddressed}>
                                  <Check size={12} strokeWidth={2.5} />
                                  <span>{pt}</span>
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {(detail.evaluation_result.relevance.key_points_missed?.length ?? 0) > 0 && (
                          <div>
                            <span style={resultStyles.pointGroupLabel}>Key Points Missed / Ignored:</span>
                            <div style={resultStyles.tagRow}>
                              {detail.evaluation_result.relevance.key_points_missed.map((pt, idx) => (
                                <span key={idx} style={resultStyles.tagMissed}>
                                  <span>• {pt}</span>
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* ChromaDB RAG Retrieved Evidence Card */}
                <div style={resultStyles.card}>
                  <div style={resultStyles.cardHeader}>
                    <div style={resultStyles.cardBadgeGroup}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Database size={16} color="#09090b" />
                        <span style={resultStyles.cardTypeTag}>RETRIEVED KNOWLEDGE BASE EVIDENCE (CHROMADB)</span>
                      </div>
                    </div>
                    <span style={resultStyles.evidenceCountBadge}>
                      {detail.evaluation_result.retrieved_evidence?.length || 0} chunks extracted
                    </span>
                  </div>

                  <p style={resultStyles.evidenceIntro}>
                    Top semantically matching passages retrieved from TruthfulQA & SQuAD knowledge base:
                  </p>

                  {(!detail.evaluation_result.retrieved_evidence ||
                    detail.evaluation_result.retrieved_evidence.length === 0) ? (
                    <p style={resultStyles.noEvidenceText}>
                      No relevant benchmark passages matched the query with similarity ≥ 25%.
                    </p>
                  ) : (
                    <div style={resultStyles.evidenceList}>
                      {detail.evaluation_result.retrieved_evidence.map((chunk, idx) => (
                        <div key={chunk.chunk_id || idx} style={resultStyles.evidenceItem}>
                          <div style={resultStyles.evidenceItemHeader}>
                            <span style={resultStyles.similarityBadge}>
                              {(chunk.similarity_score * 100).toFixed(1)}% Match
                            </span>
                            <span style={resultStyles.datasetBadge}>
                              {chunk.dataset_name.toUpperCase()}
                            </span>
                            {chunk.title && (
                              <span style={resultStyles.evidenceTopicBadge}>{chunk.title}</span>
                            )}
                            {chunk.category && !chunk.title && (
                              <span style={resultStyles.evidenceTopicBadge}>Topic: {chunk.category}</span>
                            )}
                            <span style={resultStyles.sourceIdBadge}>#{chunk.source_record_id}</span>
                          </div>
                          <div style={resultStyles.evidenceTextQuote}>
                            {chunk.text}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              /* Run Evaluation Action Banner */
              <div style={resultStyles.actionBanner}>
                <div style={resultStyles.actionBannerLeft}>
                  <div style={resultStyles.actionBannerIcon}>
                    <Zap size={20} color="#09090b" strokeWidth={2.2} />
                  </div>
                  <div>
                    <h4 style={resultStyles.actionBannerTitle}>Ready for Automated Evaluation</h4>
                    <p style={resultStyles.actionBannerDesc}>
                      Run the multi-agent pipeline: retrieve factual benchmark evidence from ChromaDB and evaluate prompt-response relevance.
                    </p>
                  </div>
                </div>

                <button
                  onClick={handleRunEvaluation}
                  disabled={evaluating}
                  style={resultStyles.primaryRunBtn}
                >
                  <Zap size={16} color="#ffffff" />
                  <span>{evaluating ? 'Running Evaluation…' : 'Run Evaluation'}</span>
                </button>
              </div>
            )}

            {/* Sections */}
            <DetailSection label="Question / Prompt" value={detail.question} />
            <DetailSection label="AI-Generated Response" value={detail.ai_response} />
            <DetailSection
              label="Reference Answer (Ground Truth)"
              value={detail.reference_answer}
              placeholder="No reference answer provided with this submission."
            />
            <DetailSection
              label="Source Document Context"
              value={detail.source_document}
              placeholder="No source document text supplied with this submission."
            />
          </div>
        )}
      </section>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  page: {
    display: 'flex',
    height: '100%',
    overflow: 'hidden',
    backgroundColor: '#ffffff',
  },

  // Left List Panel
  listPanel: {
    width: '400px',
    minWidth: '320px',
    borderRight: '1px solid #e4e4e7',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    backgroundColor: '#fafafa',
  },
  listHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '20px 18px',
    borderBottom: '1px solid #e4e4e7',
    backgroundColor: '#ffffff',
  },
  listTitleGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  listTitle: {
    fontSize: '16px',
    fontWeight: '800',
    color: '#09090b',
    letterSpacing: '-0.3px',
  },
  countBadge: {
    fontSize: '11px',
    fontWeight: '800',
    padding: '1px 6px',
    borderRadius: '10px',
    backgroundColor: '#f4f4f6',
    border: '1px solid #e4e4e7',
    color: '#09090b',
  },
  refreshBtn: {
    background: '#ffffff',
    border: '1px solid #e4e4e7',
    borderRadius: '6px',
    cursor: 'pointer',
    padding: '6px 8px',
    color: '#09090b',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'background-color 0.15s',
  },
  list: {
    listStyle: 'none',
    overflowY: 'auto',
    flex: 1,
    padding: '10px',
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  listItem: {
    padding: '14px 16px',
    borderRadius: '8px',
    cursor: 'pointer',
    backgroundColor: '#ffffff',
    border: '1px solid #e4e4e7',
    transition: 'all 0.15s ease',
  },
  listItemActive: {
    borderColor: '#09090b',
    backgroundColor: '#ffffff',
    boxShadow: '0 2px 6px rgba(0, 0, 0, 0.06)',
  },
  itemHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '8px',
  },
  itemId: {
    fontSize: '13px',
    fontWeight: '800',
    color: '#09090b',
    fontVariantNumeric: 'tabular-nums',
  },
  itemQuestion: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#09090b',
    lineHeight: '1.5',
    marginBottom: '10px',
  },
  itemFooter: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  itemDate: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '4px',
    fontSize: '12px',
    color: '#71717a',
    fontWeight: '500',
  },
  centerMessage: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    padding: '40px 16px',
    fontSize: '14px',
    color: '#71717a',
  },
  errorBox: {
    margin: '16px',
    padding: '16px',
    borderRadius: '8px',
    border: '1px solid #09090b',
    backgroundColor: '#ffffff',
    fontSize: '13px',
    color: '#09090b',
  },
  retryBtn: {
    marginTop: '10px',
    padding: '6px 14px',
    fontSize: '12px',
    fontWeight: '600',
    backgroundColor: '#09090b',
    color: '#ffffff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  emptyState: {
    padding: '60px 24px',
    textAlign: 'center',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  emptyTitle: {
    fontSize: '15px',
    fontWeight: '700',
    color: '#09090b',
    marginTop: '12px',
    marginBottom: '4px',
  },
  emptyText: {
    fontSize: '13px',
    color: '#71717a',
    lineHeight: '1.5',
  },

  // Right Detail Panel
  detailPanel: {
    flex: 1,
    overflowY: 'auto',
    backgroundColor: '#ffffff',
    minWidth: 0,
  },
  emptyPlaceholder: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    padding: '48px',
    textAlign: 'center',
  },
  placeholderIconCircle: {
    width: '64px',
    height: '64px',
    borderRadius: '50%',
    backgroundColor: '#f4f4f6',
    border: '1px solid #e4e4e7',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: '16px',
  },
  placeholderTitle: {
    fontSize: '18px',
    fontWeight: '700',
    color: '#09090b',
    marginBottom: '6px',
    letterSpacing: '-0.3px',
  },
  placeholderSubtitle: {
    fontSize: '14px',
    color: '#52525b',
    maxWidth: '420px',
    lineHeight: '1.6',
  },
  detailContainer: {
    padding: '36px 40px 64px',
    maxWidth: '900px',
  },
  detailHeader: {
    paddingBottom: '20px',
    borderBottom: '1px solid #e4e4e7',
    marginBottom: '24px',
  },
  detailTitleGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    marginBottom: '8px',
    flexWrap: 'wrap',
  },
  detailIdPill: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '3px',
    padding: '2px 8px',
    backgroundColor: '#09090b',
    color: '#ffffff',
    borderRadius: '4px',
    fontSize: '12px',
    fontWeight: '800',
  },
  detailTitle: {
    fontSize: '24px',
    fontWeight: '800',
    letterSpacing: '-0.5px',
    color: '#09090b',
  },
  detailMetaRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
  },
  metaItem: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '13px',
    color: '#71717a',
    fontWeight: '500',
  },
  pendingCallout: {
    display: 'flex',
    gap: '14px',
    padding: '16px 18px',
    backgroundColor: '#fafafa',
    border: '1px solid #e4e4e7',
    borderRadius: '8px',
    marginBottom: '28px',
  },
  calloutIcon: {
    color: '#09090b',
    paddingTop: '2px',
    flexShrink: 0,
  },
  calloutBody: {
    flex: 1,
  },
  calloutTitle: {
    fontSize: '14px',
    fontWeight: '700',
    color: '#09090b',
    marginBottom: '4px',
  },
  calloutText: {
    fontSize: '13px',
    color: '#52525b',
    lineHeight: '1.6',
  },
};

export default EvaluationHistoryPage;
