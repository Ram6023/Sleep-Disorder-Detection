import React from 'react';
import { MODEL_EVALUATION_DATA } from '../data/modelEvaluationData.ts';

export const ModelComparison: React.FC = () => {
  const { testMetrics, benchmarkComparison, cvMetrics } = MODEL_EVALUATION_DATA;

  return (
    <div className="model-comparison-container">
      {/* Top Metric Highlights */}
      <div className="metric-cards-grid grid-4">
        <div className="metric-stat-card glass-panel">
          <span className="stat-label">Holdout Test Accuracy</span>
          <span className="stat-value highlight">{(testMetrics.accuracy * 100).toFixed(1)}%</span>
          <span className="stat-sub">75 holdout test patients (leakage-free)</span>
        </div>
        <div className="metric-stat-card glass-panel">
          <span className="stat-label">Balanced Accuracy</span>
          <span className="stat-value">{(testMetrics.balancedAccuracy * 100).toFixed(2)}%</span>
          <span className="stat-sub">Imbalance-adjusted mean recall</span>
        </div>
        <div className="metric-stat-card glass-panel">
          <span className="stat-label">Macro F1-Score</span>
          <span className="stat-value">{(testMetrics.macroF1 * 100).toFixed(2)}%</span>
          <span className="stat-sub">Equal weighting across all 3 classes</span>
        </div>
        <div className="metric-stat-card glass-panel">
          <span className="stat-label">Macro ROC-AUC (OvR)</span>
          <span className="stat-value font-mono">{(testMetrics.macroRocAuc).toFixed(4)}</span>
          <span className="stat-sub">One-vs-Rest multiclass discrimination</span>
        </div>
      </div>

      {/* Per-Class Performance Table */}
      <section className="glass-panel content-section" aria-labelledby="per-class-heading">
        <div className="section-head">
          <h3 id="per-class-heading" className="section-title">
            Per-Class Holdout Test Performance
          </h3>
          <p className="section-sub">
            Strict evaluation on unseen test partition ($N=75$) across the three target sleep disorder classes.
          </p>
        </div>

        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Disorder Class</th>
                <th scope="col">Precision</th>
                <th scope="col">Recall (Sensitivity)</th>
                <th scope="col">F1-Score</th>
                <th scope="col">ROC-AUC</th>
                <th scope="col">Support</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(testMetrics.perClass).map(([clsName, metrics]) => (
                <tr key={clsName}>
                  <td className="font-semibold">{clsName}</td>
                  <td className="font-mono">{(metrics.precision * 100).toFixed(1)}%</td>
                  <td className="font-mono">{(metrics.recall * 100).toFixed(1)}%</td>
                  <td className="font-mono">{(metrics.f1 * 100).toFixed(1)}%</td>
                  <td className="font-mono">{metrics.rocAuc.toFixed(4)}</td>
                  <td className="font-mono">{metrics.support} instances</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Model Benchmark Comparison Table */}
      <section className="glass-panel content-section" aria-labelledby="benchmark-heading">
        <div className="section-head">
          <div className="badge-row">
            <span className="badge badge-brand">5-Fold Stratified Cross-Validation</span>
            <span className="badge badge-subtle">9 Algorithms Benchmarked</span>
          </div>
          <h3 id="benchmark-heading" className="section-title mt-2">
            Cross-Validation Algorithm Benchmark
          </h3>
          <p className="section-sub">
            Comparison of top machine learning models trained on identical stratified CV folds with strictly isolated feature scaling and categorical encoding.
          </p>
        </div>

        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Algorithm</th>
                <th scope="col">Feature Regime</th>
                <th scope="col">CV Macro F1 (Mean &plusmn; Std)</th>
                <th scope="col">CV Balanced Acc</th>
                <th scope="col">CV Accuracy</th>
                <th scope="col">Status</th>
              </tr>
            </thead>
            <tbody>
              {benchmarkComparison.map((bm) => {
                const isSelected = bm.status === 'selected';
                return (
                  <tr key={bm.model} className={isSelected ? 'selected-row' : ''}>
                    <td className="font-semibold">
                      <span>{bm.model}</span>
                      {isSelected && <span className="winner-tag">Selected</span>}
                    </td>
                    <td className="text-secondary">{bm.configuration}</td>
                    <td className="font-mono font-bold">
                      {(bm.cvMacroF1 * 100).toFixed(2)}%
                      {isSelected && <span className="text-muted font-normal"> &plusmn; {(cvMetrics.macroF1Std * 100).toFixed(1)}%</span>}
                    </td>
                    <td className="font-mono">{(bm.cvBalancedAcc * 100).toFixed(2)}%</td>
                    <td className="font-mono">{(bm.cvAccuracy * 100).toFixed(2)}%</td>
                    <td>
                      <span className={`status-badge ${isSelected ? 'badge-winner' : 'badge-evaluated'}`}>
                        {isSelected ? 'Selected Model' : 'Evaluated'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
