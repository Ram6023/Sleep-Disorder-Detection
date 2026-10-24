import React from 'react';
import { Database, ShieldAlert, Search, Lock } from 'lucide-react';
import { DisclaimerBanner } from './DisclaimerBanner.tsx';

export const ResearchEthics: React.FC = () => {
  return (
    <div className="research-ethics-container">
      <DisclaimerBanner />

      <section className="glass-panel content-section mt-4">
        <div className="section-head">
          <h2 className="section-title">Research Integrity &amp; Ethical Guardrails</h2>
          <p className="section-sub">
            Academic documentation regarding dataset provenance, model assumptions, and responsible healthcare AI design principles.
          </p>
        </div>

        <div className="ethics-grid grid-2 mt-4">
          <div className="ethics-card">
            <div className="ethics-card-header">
              <Database size={18} className="ethics-icon" />
              <h3 className="ethics-card-title">Dataset Provenance &amp; Limits</h3>
            </div>
            <p className="ethics-card-text">
              This system is trained on the Kaggle <em>Sleep Health and Lifestyle Dataset</em> (374 patient instances, 13 features). While calibrated to simulate authentic demographic and clinical patterns, the dataset is synthetic and lacks long-term polysomnography (PSG) sleep-study recordings.
            </p>
          </div>

          <div className="ethics-card">
            <div className="ethics-card-header">
              <ShieldAlert size={18} className="ethics-icon text-warning" />
              <h3 className="ethics-card-title">Non-Diagnostic Guardrails</h3>
            </div>
            <p className="ethics-card-text">
              The model provides statistical probability scores for research inquiry. It is not approved by the FDA, EMA, or CE medical authorities, and must never be used to guide pharmacotherapy, CPAP titration, or surgical intervention.
            </p>
          </div>

          <div className="ethics-card">
            <div className="ethics-card-header">
              <Search size={18} className="ethics-icon" />
              <h3 className="ethics-card-title">Explainability Transparency</h3>
            </div>
            <p className="ethics-card-text">
              Local SHAP TreeExplainer outputs reveal the mathematical attributions applied by the model ensemble. High attribution indicates algorithmic correlation within the synthetic training space, not biological causation.
            </p>
          </div>

          <div className="ethics-card">
            <div className="ethics-card-header">
              <Lock size={18} className="ethics-icon" />
              <h3 className="ethics-card-title">Patient Privacy &amp; Data Minimization</h3>
            </div>
            <p className="ethics-card-text">
              Inference requests are stateless. Patient parameters submitted via the assessment interface are processed transiently in server memory and are neither stored in databases nor utilized to retrain model pipelines.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
