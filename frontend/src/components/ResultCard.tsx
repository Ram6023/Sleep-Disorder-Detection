import React from 'react';
import {
  Info,
  Clock,
  ArrowRight,
} from 'lucide-react';
import type { SleepPredictionOutput } from '../types/index.ts';

interface ResultCardProps {
  result: SleepPredictionOutput | null;
  isLoading: boolean;
  onViewExplainability?: () => void;
  onNewAssessment?: () => void;
}

export const ResultCard: React.FC<ResultCardProps> = ({
  result,
  isLoading,
  onViewExplainability,
  onNewAssessment,
}) => {
  if (isLoading) {
    return (
      <div className="result-card glass-panel loading-skeleton" aria-live="polite">
        <div className="skeleton-pill" />
        <div className="skeleton-title" />
        <div className="skeleton-bars">
          <div className="skeleton-bar" />
          <div className="skeleton-bar" />
          <div className="skeleton-bar" />
        </div>
        <p className="loading-caption">
          Querying trained Random Forest model &amp; evaluating decision paths&hellip;
        </p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="result-card glass-panel empty-state">
        <div className="empty-icon" aria-hidden="true">
          <Clock size={36} className="text-secondary" />
        </div>
        <h3 className="empty-title">Awaiting Sleep Assessment</h3>
        <p className="empty-desc">
          Fill out the patient health parameters in the assessment form or choose one of the pre-loaded clinical sample presets to view machine learning predictions and calibrated probabilities.
        </p>
        {onNewAssessment && (
          <button type="button" className="btn btn-secondary mt-3" onClick={onNewAssessment}>
            <span>Go to Sleep Assessment Form</span>
          </button>
        )}
      </div>
    );
  }

  // Determine styling based on predicted_class
  const isHealthy = result.predicted_class === 0;
  const isInsomnia = result.predicted_class === 1;
  const isApnea = result.predicted_class === 2;

  const statusClass = isHealthy
    ? 'status-none'
    : isInsomnia
    ? 'status-insomnia'
    : 'status-apnea';

  const highestProbKey = Object.keys(result.probabilities).reduce((a, b) =>
    result.probabilities[a] > result.probabilities[b] ? a : b
  );
  const highestProbVal = result.probabilities[highestProbKey] ?? 0;
  const highestProbPct = (highestProbVal * 100).toFixed(1);

  return (
    <article className="result-card glass-panel" aria-labelledby="result-heading">
      <div className="result-header">
        <div className="result-badge-row">
          <span className={`prediction-pill ${statusClass}`}>
            <span className="dot" aria-hidden="true" />
            Class {result.predicted_class}: {result.predicted_class_name}
          </span>
          <span className="confidence-chip" title="Model posterior probability for top class">
            {highestProbPct}% Probability
          </span>
        </div>
        <h3 id="result-heading" className="result-class-name">
          {result.predicted_class_name}
        </h3>
        <p className="result-sub">
          {isHealthy && 'Lifestyle and physiological indicators align with healthy non-disordered sleep patterns.'}
          {isInsomnia && 'Elevated risk pattern characterized by reduced sleep duration, elevated occupational stress, and disrupted sleep quality.'}
          {isApnea && 'Elevated risk pattern characterized by elevated blood pressure, BMI indicators, and fragmented restorative sleep.'}
        </p>
      </div>

      {/* Probabilities Distribution */}
      <section className="probabilities-section" aria-label="Calibrated Class Probabilities">
        <h4 className="section-subtitle">Calibrated Probability Distribution</h4>
        <div className="probability-list">
          {Object.entries(result.probabilities).map(([className, prob]) => {
            const pct = (prob * 100).toFixed(1);
            const isTarget = className === result.predicted_class_name;
            const barClass =
              className === 'No recorded sleep disorder'
                ? 'bar-healthy'
                : className === 'Insomnia'
                ? 'bar-insomnia'
                : 'bar-apnea';

            return (
              <div key={className} className={`prob-row ${isTarget ? 'is-predicted' : ''}`}>
                <div className="prob-label-row">
                  <span className="prob-name">
                    {className}
                    {isTarget && <span className="predicted-tag">Predicted</span>}
                  </span>
                  <span className="prob-value font-mono">{pct}%</span>
                </div>
                <div className="prob-track" role="progressbar" aria-valuenow={parseFloat(pct)} aria-valuemin={0} aria-valuemax={100}>
                  <div
                    className={`prob-fill ${barClass}`}
                    style={{ width: `${Math.max(parseFloat(pct), 2)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Clinical Interpretation Notice */}
      <div className="clinical-interpretation-box">
        <Info size={18} className="box-icon-svg" />
        <div className="box-content">
          <strong>Interpretation Guidance</strong>
          <p>
            This output represents model posterior probabilities across 25 transformed features. While probabilities indicate algorithmic confidence, they reflect correlation within the study sample rather than clinical diagnosis or physiological certainty.
          </p>
        </div>
      </div>

      {/* SHAP Callout */}
      {result.explanation ? (
        <div className="shap-callout">
          <div className="shap-callout-text">
            <strong>Local Explainability Available</strong>
            <p>
              SHAP feature attributions have been computed for this assessment. Inspect the top contributing factors that influenced this prediction.
            </p>
          </div>
          {onViewExplainability && (
            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={onViewExplainability}
            >
              <span>Inspect SHAP Explanations</span>
              <ArrowRight size={14} />
            </button>
          )}
        </div>
      ) : (
        <div className="shap-callout-disabled">
          <Info size={15} className="mr-1 inline-icon" />
          <span>Local SHAP attributions were omitted for this query. Enable &lsquo;Include Local SHAP Explainability&rsquo; in the form to view feature-level contributions.</span>
        </div>
      )}

      {/* Technical Metadata Footer */}
      <footer className="result-metadata-footer">
        <div className="meta-item">
          <span className="meta-label">Prediction ID:</span>
          <span className="meta-value font-mono" title={result.prediction_id}>
            {result.prediction_id.slice(0, 13)}&hellip;
          </span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Model Pipeline:</span>
          <span className="meta-value">Random Forest v{result.model_version}</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Contract:</span>
          <span className="meta-value">v{result.contract_version}</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Inference Time:</span>
          <span className="meta-value font-mono">
            {new Date(result.timestamp_utc).toLocaleTimeString()}
          </span>
        </div>
      </footer>
    </article>
  );
};
