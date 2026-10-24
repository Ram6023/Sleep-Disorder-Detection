import React from 'react';
import { Cpu, Layers, GitBranch } from 'lucide-react';
import type { ModelContractInfo } from '../types/index.ts';
import { ModelComparison } from './ModelComparison.tsx';

interface AboutModelProps {
  modelInfo: ModelContractInfo | null;
  loading: boolean;
}

export const AboutModel: React.FC<AboutModelProps> = ({ modelInfo, loading }) => {
  return (
    <div className="about-model-container">
      {/* Overview & Architecture Header */}
      <section className="glass-panel content-section">
        <div className="section-head">
          <div className="badge-row">
            <span className="badge badge-brand">Model Architecture</span>
            <span className="badge badge-subtle">Scikit-Learn Pipeline</span>
            {modelInfo && <span className="badge badge-subtle">Contract v{modelInfo.contract_version}</span>}
          </div>
          <h2 className="section-title mt-2">
            Intelligent Random Forest Classification Pipeline
          </h2>
          <p className="section-sub">
            A productionized, leakage-free multi-class machine learning system trained to detect sleep disorders from physiological biomarkers and lifestyle metrics.
          </p>
        </div>

        <div className="info-grid grid-3">
          <div className="info-box">
            <div className="info-box-header">
              <Cpu size={16} className="info-icon" />
              <span className="info-label">Base Classifier</span>
            </div>
            <strong className="info-value">Random Forest Ensemble</strong>
            <p className="info-desc">100 estimators, max depth 6, balanced split criterion, fixed seed (42).</p>
          </div>
          <div className="info-box">
            <div className="info-box-header">
              <Layers size={16} className="info-icon" />
              <span className="info-label">Feature Processing</span>
            </div>
            <strong className="info-value">ColumnTransformer</strong>
            <p className="info-desc">
              StandardScaler for numerical values; OneHotEncoder (<code>handle_unknown=&apos;ignore&apos;</code>) for categorical features.
            </p>
          </div>
          <div className="info-box">
            <div className="info-box-header">
              <GitBranch size={16} className="info-icon" />
              <span className="info-label">Dimensionality</span>
            </div>
            <strong className="info-value">12 Raw &rarr; 25 Transformed</strong>
            <p className="info-desc">Zero leakage barrier: all transformations fitted exclusively on training splits.</p>
          </div>
        </div>

        {/* Supported Classes */}
        <div className="mt-4">
          <h3 className="section-subtitle">Supported Sleep Disorder Classes</h3>
          <div className="classes-grid grid-3">
            <div className="class-card class-card-none">
              <div className="class-header">
                <span className="class-code font-mono">Index 0</span>
                <span className="class-status-badge">Baseline</span>
              </div>
              <h4 className="class-title">No recorded sleep disorder</h4>
              <p className="class-desc">
                Represents regular, restorative sleep architecture without chronic pathological sleep disruption.
              </p>
            </div>

            <div className="class-card class-card-insomnia">
              <div className="class-header">
                <span className="class-code font-mono">Index 1</span>
                <span className="class-status-badge insomnia">Disorder</span>
              </div>
              <h4 className="class-title">Insomnia</h4>
              <p className="class-desc">
                Characterized by difficulty initiating or maintaining sleep, early awakenings, and high occupational stress.
              </p>
            </div>

            <div className="class-card class-card-apnea">
              <div className="class-header">
                <span className="class-code font-mono">Index 2</span>
                <span className="class-status-badge apnea">Disorder</span>
              </div>
              <h4 className="class-title">Sleep Apnea</h4>
              <p className="class-desc">
                Characterized by repeated airway collapse and oxygen desaturations, strongly correlated with elevated blood pressure and BMI.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Model Benchmark Comparison & Evaluation */}
      <ModelComparison />

      {/* Validation Bounds Reference Table */}
      <section className="glass-panel content-section">
        <div className="section-head">
          <h3 className="section-title">Authoritative Physiological Validation Bounds</h3>
          <p className="section-sub">
            Guardrails enforced both client-side and verified authoritatively by Pydantic v2 schemas on the FastAPI backend.
          </p>
        </div>

        {loading ? (
          <p className="loading-caption">Fetching live schema bounds from /api/model-info&hellip;</p>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Feature Name</th>
                  <th scope="col">Type</th>
                  <th scope="col">Enforced Minimum</th>
                  <th scope="col">Enforced Maximum</th>
                  <th scope="col">Clinical Relevance</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="font-mono">age</td>
                  <td>Integer</td>
                  <td>1 year</td>
                  <td>120 years</td>
                  <td>Age-related sleep architecture changes</td>
                </tr>
                <tr>
                  <td className="font-mono">sleep_duration</td>
                  <td>Float</td>
                  <td>0.1 hours</td>
                  <td>24.0 hours</td>
                  <td>Average daily sleep duration</td>
                </tr>
                <tr>
                  <td className="font-mono">quality_of_sleep</td>
                  <td>Integer</td>
                  <td>1 (Poor)</td>
                  <td>10 (Excellent)</td>
                  <td>Subjective restorative quality</td>
                </tr>
                <tr>
                  <td className="font-mono">physical_activity_level</td>
                  <td>Integer</td>
                  <td>0 min</td>
                  <td>1440 min</td>
                  <td>Daily active movement volume</td>
                </tr>
                <tr>
                  <td className="font-mono">stress_level</td>
                  <td>Integer</td>
                  <td>1 (Low)</td>
                  <td>10 (High)</td>
                  <td>Perceived psychological stress</td>
                </tr>
                <tr>
                  <td className="font-mono">heart_rate</td>
                  <td>Integer</td>
                  <td>30 BPM</td>
                  <td>220 BPM</td>
                  <td>Resting cardiovascular rate</td>
                </tr>
                <tr>
                  <td className="font-mono">daily_steps</td>
                  <td>Integer</td>
                  <td>0 steps</td>
                  <td>100,000 steps</td>
                  <td>Daily ambulatory volume</td>
                </tr>
                <tr>
                  <td className="font-mono">bp_systolic</td>
                  <td>Float</td>
                  <td>50.0 mmHg</td>
                  <td>260.0 mmHg</td>
                  <td>Systolic arterial blood pressure</td>
                </tr>
                <tr>
                  <td className="font-mono">bp_diastolic</td>
                  <td>Float</td>
                  <td>30.0 mmHg</td>
                  <td>160.0 mmHg</td>
                  <td>Diastolic arterial blood pressure</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
};
