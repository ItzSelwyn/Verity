import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Send,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  ArrowRight,
  Sparkles,
} from '../components/icons';
import { submitEvaluation, extractApiError } from '../api/evaluationApi';
import type { EvaluationSubmitRequest } from '../types/evaluation';

interface FormState {
  question: string;
  ai_response: string;
  reference_answer: string;
  source_document: string;
}

interface FormErrors {
  question?: string;
  ai_response?: string;
}

function validate(form: FormState): FormErrors {
  const errors: FormErrors = {};

  if (!form.question.trim()) {
    errors.question = 'Question is required.';
  } else if (form.question.trim().length < 10) {
    errors.question = 'Question must be at least 10 characters.';
  }

  if (!form.ai_response.trim()) {
    errors.ai_response = 'AI Response is required.';
  } else if (form.ai_response.trim().length < 10) {
    errors.ai_response = 'AI Response must be at least 10 characters.';
  }

  return errors;
}

const EMPTY_FORM: FormState = {
  question: '',
  ai_response: '',
  reference_answer: '',
  source_document: '',
};

const NewEvaluationPage: React.FC = () => {
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [errors, setErrors] = useState<FormErrors>({});
  const [loading, setLoading] = useState(false);
  const [successId, setSuccessId] = useState<number | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const handleChange = (field: keyof FormState) =>
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setForm((prev) => ({ ...prev, [field]: e.target.value }));
      if (errors[field as keyof FormErrors]) {
        setErrors((prev) => ({ ...prev, [field]: undefined }));
      }
    };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setApiError(null);
    setSuccessId(null);

    const validationErrors = validate(form);
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }

    const payload: EvaluationSubmitRequest = {
      question: form.question.trim(),
      ai_response: form.ai_response.trim(),
    };
    if (form.reference_answer.trim()) {
      payload.reference_answer = form.reference_answer.trim();
    }
    if (form.source_document.trim()) {
      payload.source_document = form.source_document.trim();
    }

    setLoading(true);
    try {
      const result = await submitEvaluation(payload);
      setSuccessId(result.id);
      setForm(EMPTY_FORM);
      setErrors({});
    } catch (err) {
      setApiError(extractApiError(err));
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setForm(EMPTY_FORM);
    setErrors({});
    setApiError(null);
    setSuccessId(null);
  };

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        {/* Header */}
        <div style={styles.header}>
          <div style={styles.headerBadge}>
            <Sparkles size={14} />
            <span>SUBMISSION MODULE</span>
          </div>
          <h1 style={styles.title}>New Evaluation</h1>
          <p style={styles.subtitle}>
            Submit an AI-generated response alongside optional ground-truth evidence to verify accuracy, hallucination risks, and completeness.
          </p>
        </div>

        {/* Success Banner */}
        {successId !== null && (
          <div style={styles.successCard}>
            <div style={styles.bannerIconBox}>
              <CheckCircle2 size={22} color="#09090b" strokeWidth={2.2} />
            </div>
            <div style={styles.bannerContent}>
              <h4 style={styles.bannerTitle}>Submission #{successId} Queued</h4>
              <p style={styles.bannerText}>
                Your submission has been safely persisted to the database with status <strong>Pending</strong>.
              </p>
              <Link to={`/history/${successId}`} style={styles.bannerActionBtn}>
                <span>Open in History</span>
                <ArrowRight size={15} />
              </Link>
            </div>
          </div>
        )}

        {/* Error Banner */}
        {apiError && (
          <div style={styles.errorCard}>
            <div style={styles.bannerIconBox}>
              <AlertCircle size={22} color="#09090b" strokeWidth={2.2} />
            </div>
            <div style={styles.bannerContent}>
              <h4 style={styles.bannerTitle}>Validation / API Error</h4>
              <p style={styles.bannerText}>{apiError}</p>
            </div>
          </div>
        )}

        {/* Evaluation Form */}
        <form onSubmit={handleSubmit} noValidate style={styles.form}>
          {/* Question / Prompt */}
          <div style={styles.fieldCard}>
            <div style={styles.labelRow}>
              <label style={styles.label} htmlFor="question">
                Question or Prompt <span style={styles.requiredMark}>*</span>
              </label>
              <span style={styles.charCounter}>{form.question.length} / 5000</span>
            </div>
            <p style={styles.hint}>
              The primary prompt or query that was issued to the language model.
            </p>
            <textarea
              id="question"
              rows={4}
              style={{
                ...styles.textarea,
                ...(errors.question ? styles.textareaError : {}),
              }}
              placeholder="e.g. What causes the Northern Lights and where are they most visible?"
              value={form.question}
              onChange={handleChange('question')}
              disabled={loading}
            />
            {errors.question && (
              <p style={styles.errorMsg}>
                <AlertCircle size={13} /> {errors.question}
              </p>
            )}
          </div>

          {/* AI Response */}
          <div style={styles.fieldCard}>
            <div style={styles.labelRow}>
              <label style={styles.label} htmlFor="ai_response">
                AI-Generated Response <span style={styles.requiredMark}>*</span>
              </label>
              <span style={styles.charCounter}>{form.ai_response.length} / 20000</span>
            </div>
            <p style={styles.hint}>
              The complete response generated by the AI model that requires factual verification.
            </p>
            <textarea
              id="ai_response"
              rows={7}
              style={{
                ...styles.textarea,
                ...(errors.ai_response ? styles.textareaError : {}),
              }}
              placeholder="Paste the candidate AI response here for automated evaluation..."
              value={form.ai_response}
              onChange={handleChange('ai_response')}
              disabled={loading}
            />
            {errors.ai_response && (
              <p style={styles.errorMsg}>
                <AlertCircle size={13} /> {errors.ai_response}
              </p>
            )}
          </div>

          {/* Reference Answer (Optional) */}
          <div style={styles.fieldCard}>
            <div style={styles.labelRow}>
              <div style={styles.labelGroup}>
                <label style={styles.label} htmlFor="reference_answer">
                  Reference Answer
                </label>
                <span style={styles.optionalPill}>Optional</span>
              </div>
              <span style={styles.charCounter}>{form.reference_answer.length} / 10000</span>
            </div>
            <p style={styles.hint}>
              A verified ground-truth answer to measure exact factual agreement against.
            </p>
            <textarea
              id="reference_answer"
              rows={4}
              style={styles.textarea}
              placeholder="If you have an established ground-truth answer, provide it here..."
              value={form.reference_answer}
              onChange={handleChange('reference_answer')}
              disabled={loading}
            />
          </div>

          {/* Source Document Text (Optional) */}
          <div style={styles.fieldCard}>
            <div style={styles.labelRow}>
              <div style={styles.labelGroup}>
                <label style={styles.label} htmlFor="source_document">
                  Source Reference Document Text
                </label>
                <span style={styles.optionalPill}>Optional</span>
              </div>
              <span style={styles.charCounter}>{form.source_document.length} / 50000</span>
            </div>
            <p style={styles.hint}>
              Supporting context, passage text, or source document supplied to the model.
            </p>
            <textarea
              id="source_document"
              rows={5}
              style={styles.textarea}
              placeholder="Paste raw reference material or benchmark context passages..."
              value={form.source_document}
              onChange={handleChange('source_document')}
              disabled={loading}
            />
          </div>

          {/* Actions */}
          <div style={styles.actionsRow}>
            <button
              type="submit"
              disabled={loading}
              style={{
                ...styles.primaryBtn,
                ...(loading ? styles.btnDisabled : {}),
              }}
            >
              {loading ? (
                <>
                  <span style={styles.btnSpinner} />
                  <span>Submitting Evaluation…</span>
                </>
              ) : (
                <>
                  <Send size={16} strokeWidth={2.2} />
                  <span>Submit Evaluation</span>
                </>
              )}
            </button>

            <button
              type="button"
              onClick={handleClear}
              disabled={loading}
              style={styles.secondaryBtn}
            >
              <RotateCcw size={16} />
              <span>Clear Form</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  page: {
    padding: '36px 32px 64px',
    minHeight: '100%',
    backgroundColor: '#ffffff',
  },
  container: {
    maxWidth: '840px',
    margin: '0 auto',
  },
  header: {
    marginBottom: '32px',
  },
  headerBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    padding: '4px 10px',
    borderRadius: '4px',
    backgroundColor: '#f4f4f6',
    border: '1px solid #e4e4e7',
    fontSize: '11px',
    fontWeight: '800',
    letterSpacing: '0.6px',
    color: '#09090b',
    marginBottom: '12px',
  },
  title: {
    fontSize: '28px',
    fontWeight: '800',
    letterSpacing: '-0.6px',
    color: '#09090b',
    marginBottom: '8px',
  },
  subtitle: {
    fontSize: '15px',
    color: '#52525b',
    lineHeight: '1.6',
    maxWidth: '680px',
  },

  // Banners
  successCard: {
    display: 'flex',
    gap: '16px',
    padding: '18px 20px',
    backgroundColor: '#ffffff',
    border: '2px solid #09090b',
    borderRadius: '8px',
    marginBottom: '28px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
  },
  errorCard: {
    display: 'flex',
    gap: '16px',
    padding: '18px 20px',
    backgroundColor: '#ffffff',
    border: '2px solid #09090b',
    borderRadius: '8px',
    marginBottom: '28px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
  },
  bannerIconBox: {
    paddingTop: '2px',
    flexShrink: 0,
  },
  bannerContent: {
    flex: 1,
  },
  bannerTitle: {
    fontSize: '16px',
    fontWeight: '700',
    color: '#09090b',
    marginBottom: '4px',
    letterSpacing: '-0.2px',
  },
  bannerText: {
    fontSize: '14px',
    color: '#52525b',
    lineHeight: '1.5',
    marginBottom: '10px',
  },
  bannerActionBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '13px',
    fontWeight: '700',
    color: '#09090b',
    textDecoration: 'none',
    borderBottom: '1.5px solid #09090b',
    paddingBottom: '2px',
  },

  // Form
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '24px',
  },
  fieldCard: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    padding: '20px',
    backgroundColor: '#ffffff',
    border: '1px solid #e4e4e7',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.02)',
  },
  labelRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  labelGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  label: {
    fontSize: '15px',
    fontWeight: '700',
    color: '#09090b',
    letterSpacing: '-0.2px',
  },
  requiredMark: {
    color: '#09090b',
    fontWeight: '800',
  },
  optionalPill: {
    fontSize: '11px',
    fontWeight: '700',
    color: '#52525b',
    backgroundColor: '#f4f4f6',
    border: '1px solid #e4e4e7',
    borderRadius: '4px',
    padding: '2px 8px',
    textTransform: 'uppercase',
    letterSpacing: '0.4px',
  },
  charCounter: {
    fontSize: '12px',
    fontWeight: '500',
    color: '#71717a',
    fontVariantNumeric: 'tabular-nums',
  },
  hint: {
    fontSize: '13px',
    color: '#52525b',
    marginBottom: '4px',
  },
  textarea: {
    width: '100%',
    padding: '12px 14px',
    backgroundColor: '#fcfcfd',
    color: '#09090b',
    border: '1px solid #d4d4d8',
    borderRadius: '6px',
    fontSize: '15px',
    lineHeight: '1.6',
    fontFamily: 'inherit',
    resize: 'vertical',
    outline: 'none',
    transition: 'border-color 0.15s, background-color 0.15s',
  },
  textareaError: {
    borderColor: '#09090b',
    borderWidth: '2px',
  },
  errorMsg: {
    display: 'flex',
    alignItems: 'center',
    gap: '5px',
    fontSize: '13px',
    fontWeight: '600',
    color: '#09090b',
    marginTop: '4px',
  },

  // Actions
  actionsRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    marginTop: '8px',
  },
  primaryBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    padding: '12px 24px',
    backgroundColor: '#09090b',
    color: '#ffffff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '15px',
    fontWeight: '700',
    letterSpacing: '-0.2px',
    cursor: 'pointer',
    transition: 'background-color 0.15s, transform 0.05s',
  },
  btnDisabled: {
    opacity: 0.65,
    cursor: 'not-allowed',
  },
  secondaryBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    padding: '12px 20px',
    backgroundColor: '#ffffff',
    color: '#09090b',
    border: '1px solid #e4e4e7',
    borderRadius: '6px',
    fontSize: '15px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'background-color 0.15s, border-color 0.15s',
  },
  btnSpinner: {
    display: 'inline-block',
    width: '16px',
    height: '16px',
    border: '2px solid rgba(255, 255, 255, 0.3)',
    borderTopColor: '#ffffff',
    borderRadius: '50%',
    animation: 'spin 0.8s linear infinite',
  },
};

export default NewEvaluationPage;
