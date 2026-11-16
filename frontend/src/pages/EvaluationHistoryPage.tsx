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
  Info,
  ChevronRight,
} from '../components/icons';
import { listEvaluations, getEvaluation, extractApiError } from '../api/evaluationApi';
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

  const selectedId = id ? parseInt(id, 10) : null;

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

            {/* Pipeline Notice Callout */}
            <div style={styles.pendingCallout}>
              <div style={styles.calloutIcon}>
                <Info size={18} strokeWidth={2.2} />
              </div>
              <div style={styles.calloutBody}>
                <h5 style={styles.calloutTitle}>Pipeline State: Ingested & Stored</h5>
                <p style={styles.calloutText}>
                  This submission has been validated and recorded. Automated multi-agent scoring
                  (Relevance, Accuracy, Hallucination, and Completeness) will be executed once the evaluation orchestrator is connected in Milestone 2.
                </p>
              </div>
            </div>

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
