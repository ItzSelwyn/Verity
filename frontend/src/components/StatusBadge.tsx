import React from 'react';
import { Clock, Loader2, CheckCircle2, XCircle, AlertCircle } from './icons';
import type { EvaluationListItem } from '../types/evaluation';

type Status = EvaluationListItem['status'];

interface StatusBadgeProps {
  status: Status;
  size?: 'sm' | 'md';
}

const STATUS_CONFIG: Record<
  Status,
  { label: string; icon: React.ComponentType<{ size?: number; className?: string }> }
> = {
  pending: { label: 'Pending', icon: Clock },
  running: { label: 'Running', icon: Loader2 },
  completed: { label: 'Completed', icon: CheckCircle2 },
  failed: { label: 'Failed', icon: XCircle },
  partial: { label: 'Partial', icon: AlertCircle },
};

/**
 * Minimal monochromatic badge using crisp black icons and clean borders.
 */
const StatusBadge: React.FC<StatusBadgeProps> = ({ status, size = 'md' }) => {
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.pending;
  const Icon = config.icon;
  const isRunning = status === 'running';

  const isSmall = size === 'sm';

  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: isSmall ? '5px' : '6px',
        padding: isSmall ? '2px 8px' : '4px 10px',
        borderRadius: '20px',
        fontSize: isSmall ? '12px' : '13px',
        fontWeight: '600',
        letterSpacing: '0.2px',
        color: '#09090b',
        backgroundColor: '#f4f4f6',
        border: '1px solid #e4e4e7',
        whiteSpace: 'nowrap',
        lineHeight: 1.2,
      }}
    >
      <span
        style={{
          display: 'inline-flex',
          animation: isRunning ? 'spin 1s linear infinite' : undefined,
        }}
      >
        <Icon size={isSmall ? 13 : 14} />
      </span>
      <span>{config.label}</span>
    </span>
  );
};

export default StatusBadge;
