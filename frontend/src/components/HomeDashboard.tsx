import React from 'react';
import { ArrowRight } from 'lucide-react';
import type { SleepPredictionOutput, HealthStatus } from '../types/index.ts';
import { DisclaimerBanner } from './DisclaimerBanner.tsx';

interface HomeDashboardProps {
  onStartAssessment: () => void;
  onViewModel: () => void;
  onViewResults: () => void;
  latestResult: SleepPredictionOutput | null;
  health: HealthStatus | null;
}

export const HomeDashboard: React.FC<HomeDashboardProps> = ({
  onStartAssessment,
  onViewModel,
  onViewResults,
  latestResult,
  health,
}) => {
  return (
    <div className="overview-page">
      <DisclaimerBanner compact />

      {/* Page Heading */}
      <section className="page-header-editorial">
        <h2 className="page-title">Overview</h2>
        <p className="page-lead">
          An academic machine learning prototype for exploring sleep-disorder classification using lifestyle and health data.
        </p>

        <div className="page-actions-editorial">
          <button type="button" className="btn btn-primary" onClick={onStartAssessment}>
            <span>Start Assessment</span>
            <ArrowRight size={15} />
          </button>
          <button type="button" className="btn btn-secondary" onClick={onViewModel}>
            <span>View Full Model Details</span>
          </button>
          {latestResult && (
            <button type="button" className="btn btn-secondary" onClick={onViewResults}>
              <span>View Latest Result</span>
            </button>
          )}
        </div>

        {health && (
          <div className="system-meta-text">
            <span>System status: <strong>{health.status === 'healthy' ? 'Operational' : 'Unavailable'}</strong></span>
            <span className="meta-sep">&bull;</span>
            <span>Model: <strong>Random Forest v{health.model_version}</strong></span>
            <span className="meta-sep">&bull;</span>
            <span>Contract: <strong>v{health.contract_version}</strong></span>
          </div>
        )}
      </section>

      <hr className="divider" />

      {/* Model Summary: Compact Aligned Table / Information Row */}
      <section className="content-section" aria-labelledby="model-summary-title">
        <h3 id="model-summary-title" className="section-title">Model Summary</h3>
        <p className="section-desc">
          Key architecture and dataset parameters configured for the classification pipeline.
        </p>

        <div className="table-wrap">
          <table className="editorial-table">
            <thead>
              <tr>
                <th scope="col">Parameter</th>
                <th scope="col">Specification</th>
                <th scope="col">Details</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Dataset Size</strong></td>
                <td>374 patient records</td>
                <td className="text-secondary">Sleep Health and Lifestyle Dataset; 12 raw features</td>
              </tr>
              <tr>
                <td><strong>Target Classes</strong></td>
                <td>3 categories</td>
                <td className="text-secondary">No recorded sleep disorder (58.6%), Insomnia (20.6%), Sleep Apnea (20.9%)</td>
              </tr>
              <tr>
                <td><strong>Selected Model</strong></td>
                <td>Random Forest Classifier</td>
                <td className="text-secondary">100 estimators, max depth 6, balanced class weighting, fixed seed (42)</td>
              </tr>
              <tr>
                <td><strong>Holdout Evaluation Accuracy</strong></td>
                <td className="text-highlight">96.0%</td>
                <td className="text-secondary">Evaluated on unseen 20% holdout test set (N=75, leakage-free)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <hr className="divider" />

      {/* Workflow: Simple Numbered Sequence */}
      <section className="content-section" aria-labelledby="workflow-title">
        <h3 id="workflow-title" className="section-title">Inference Workflow</h3>
        <p className="section-desc">
          Sequential execution flow from raw patient inputs to local game-theoretic attributions.
        </p>

        <ol className="editorial-process-list">
          <li className="process-item">
            <span className="process-step-num">1</span>
            <div className="process-details">
              <strong>Health and Lifestyle Inputs</strong>
              <p>
                Collection of 12 self-reported and physiological parameters, including sleep duration, quality of sleep, physical activity level, BMI category, resting heart rate, and blood pressure.
              </p>
            </div>
          </li>
          <li className="process-item">
            <span className="process-step-num">2</span>
            <div className="process-details">
              <strong>Data Preprocessing &amp; Feature Scaling</strong>
              <p>
                Strict physiological boundary verification, standard numeric scaling (fitted only on training data), and categorical one-hot encoding expand inputs to 25 transformed features.
              </p>
            </div>
          </li>
          <li className="process-item">
            <span className="process-step-num">3</span>
            <div className="process-details">
              <strong>Random Forest Inference</strong>
              <p>
                Ensemble of 100 decision trees aggregates predictions to output class probabilities across the three supported conditions with zero data leakage.
              </p>
            </div>
          </li>
          <li className="process-item">
            <span className="process-step-num">4</span>
            <div className="process-details">
              <strong>Local SHAP Attribution (Optional)</strong>
              <p>
                TreeExplainer computes additive Shapley values &#981;<sub>i</sub>(x) showing which specific physiological biomarkers increased or decreased risk relative to population baseline.
              </p>
            </div>
          </li>
        </ol>
      </section>

      <hr className="divider" />

      {/* Evaluation Summary: Clean Table */}
      <section className="content-section" aria-labelledby="eval-summary-title">
        <h3 id="eval-summary-title" className="section-title">Holdout Evaluation Summary</h3>
        <p className="section-desc">
          Verified classification metrics measured on the independent 20% holdout test split ($N=75$), completely isolated during model training and feature selection.
        </p>

        <div className="table-wrap">
          <table className="editorial-table">
            <thead>
              <tr>
                <th scope="col">Metric</th>
                <th scope="col">Holdout Test Value</th>
                <th scope="col">5-Fold CV Benchmark</th>
                <th scope="col">Interpretation</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Overall Accuracy</strong></td>
                <td className="font-mono text-highlight">96.0%</td>
                <td className="font-mono text-secondary">89.30% &plusmn; 2.50%</td>
                <td className="text-secondary">72 of 75 holdout patients correctly identified</td>
              </tr>
              <tr>
                <td><strong>Balanced Accuracy</strong></td>
                <td className="font-mono">93.47%</td>
                <td className="font-mono text-secondary">86.45% &plusmn; 2.10%</td>
                <td className="text-secondary">Accounts for class distribution imbalance</td>
              </tr>
              <tr>
                <td><strong>Macro F1-Score</strong></td>
                <td className="font-mono">93.52%</td>
                <td className="font-mono text-secondary">86.93% &plusmn; 2.00%</td>
                <td className="text-secondary">Unweighted harmonic mean across all three classes</td>
              </tr>
              <tr>
                <td><strong>Multiclass ROC-AUC (OvR)</strong></td>
                <td className="font-mono">0.9957</td>
                <td className="font-mono text-secondary">&mdash;</td>
                <td className="text-secondary">Macro-averaged One-vs-Rest discrimination ability</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
