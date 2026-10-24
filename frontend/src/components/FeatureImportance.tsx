import {
  BarChart2,
  Info,
} from 'lucide-react';
import type { PredictionExplanation, SleepPredictionOutput } from '../types/index.ts';

interface FeatureImportanceProps {
  result: SleepPredictionOutput | null;
  onNavigateToForm?: () => void;
}

export const FeatureImportance: React.FC<FeatureImportanceProps> = ({
  result,
  onNavigateToForm,
}) => {
  const explanation: PredictionExplanation | null | undefined = result?.explanation;

  if (!result) {
    return (
      <div className="glass-panel empty-state" aria-label="No prediction active">
        <div className="empty-icon" aria-hidden="true">
          <BarChart2 size={36} className="text-secondary" />
        </div>
        <h3 className="empty-title">No Active Prediction for Explainability</h3>
        <p className="empty-desc">
          Run a sleep health assessment with the &ldquo;Include Local SHAP Explainability&rdquo; option enabled to view instance-level feature attributions computed via SHAP TreeExplainer.
        </p>
        {onNavigateToForm && (
          <button type="button" className="btn btn-primary mt-3" onClick={onNavigateToForm}>
            <span>Go to Sleep Assessment</span>
          </button>
        )}
      </div>
    );
  }

  if (!explanation) {
    return (
      <div className="glass-panel empty-state">
        <div className="empty-icon" aria-hidden="true">
          <Info size={36} className="text-secondary" />
        </div>
        <h3 className="empty-title">SHAP Explanation Omitted for this Ingestion</h3>
        <p className="empty-desc">
          This prediction was generated with <code>explain=false</code>. To compute local Shapley feature attributions, re-run the assessment with the SHAP checkbox checked.
        </p>
        {onNavigateToForm && (
          <button type="button" className="btn btn-secondary mt-3" onClick={onNavigateToForm}>
            <span>Re-run with SHAP Enabled</span>
          </button>
        )}
      </div>
    );
  }

  // Max absolute SHAP value for scaling bars
  const maxAbsShap = Math.max(
    ...explanation.top_factors.map((f) => Math.abs(f.shap_value)),
    0.01
  );

  return (
    <article className="explainability-card glass-panel" aria-labelledby="xai-heading">
      <div className="card-header-modular">
        <div className="header-titles">
          <div className="badge-row">
            <span className="badge badge-brand">Local Interpretability</span>
            <span className="badge badge-subtle">{explanation.method}</span>
          </div>
          <h2 id="xai-heading" className="card-title mt-2">
            Local SHAP Feature Attribution
          </h2>
          <p className="card-desc">
            Deconstructing how individual physiological and lifestyle values shifted the model&rsquo;s prediction toward <strong>&ldquo;{result.predicted_class_name}&rdquo;</strong>.
          </p>
        </div>

        <div className="base-value-card">
          <strong className="base-label">Model Baseline</strong>
          <span className="base-val font-mono">{explanation.base_value.toFixed(4)}</span>
          <span className="base-sub">
            The model&apos;s starting output before accounting for the input features.
          </span>
        </div>
      </div>

      {/* Attribution Bars Section */}
      <section className="factors-section" aria-label="Top Influencing Features">
        <h3 className="section-subtitle">Top Contributing Factors</h3>
        <div className="factors-list">
          {explanation.top_factors.map((factor, idx) => {
            const isPositive = factor.shap_value >= 0;
            const widthPct = (Math.abs(factor.shap_value) / maxAbsShap) * 100;
            const sign = isPositive ? '+' : '';

            return (
              <div key={`${factor.feature}-${idx}`} className="factor-row">
                <div className="factor-meta">
                  <div className="factor-title-row">
                    <span className="factor-name font-mono">{factor.feature}</span>
                    <span className="factor-patient-val">
                      Value: <code className="value-code">{String(factor.value)}</code>
                    </span>
                  </div>
                  <div className="factor-impact-row">
                    <span
                      className={`direction-pill ${
                        isPositive ? 'direction-support' : 'direction-oppose'
                      }`}
                    >
                      {isPositive ? '▲ Supports' : '▼ Opposes'}
                    </span>
                    <span className="shap-val font-mono">
                      {sign}{factor.shap_value.toFixed(4)}
                    </span>
                  </div>
                </div>

                <div className="waterfall-track" aria-hidden="true">
                  <div className="center-divider" />
                  <div className="bar-container">
                    {isPositive ? (
                      <div
                        className="bar-positive"
                        style={{ width: `${Math.min(widthPct / 2, 50)}%` }}
                      />
                    ) : (
                      <div
                        className="bar-negative"
                        style={{
                          width: `${Math.min(widthPct / 2, 50)}%`,
                          marginRight: '50%',
                        }}
                      />
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Methodological Context & Guardrails */}
      <footer className="xai-guardrails-footer">
        <div className="xai-explanation-card">
          <p className="xai-summary-text">
            SHAP estimates how much each input feature contributes to a model&apos;s prediction relative to a baseline. Positive contributions move the model output in one direction, while negative contributions move it in the opposite direction. These attributions explain model behavior and do not establish causation.
          </p>

          <details className="formula-disclosure">
            <summary className="formula-summary">
              <span>View mathematical formula</span>
            </summary>
            <div className="formula-dropdown-content">
              <div
                className="formula-math-display"
                role="math"
                aria-label="Shapley value formula: phi_i(f, x) equals sum over subsets S of F excluding i of (|S|!(|F| - |S| - 1)! / |F|!) times [f_x(S union {i}) - f_x(S)]"
              >
                <div className="math-equation">
                  <span className="math-term">
                    <span className="math-var">ϕ</span>
                    <sub className="math-sub">i</sub>
                    <span className="math-parens">(</span>
                    <span className="math-var">f</span>
                    <span className="math-punct">,</span>
                    <span className="math-var">x</span>
                    <span className="math-parens">)</span>
                  </span>

                  <span className="math-operator">=</span>

                  <span className="math-sigma-group">
                    <span className="math-sigma">&#8721;</span>
                    <span className="math-sigma-sub">
                      <span className="math-var">S</span>
                      <span className="math-sym">&#8838;</span>
                      <span className="math-var">F</span>
                      <span className="math-sym">&#8726;</span>
                      <span className="math-set">&#123;<span className="math-var">i</span>&#125;</span>
                    </span>
                  </span>

                  <span className="math-fraction">
                    <span className="math-numerator">
                      |<span className="math-var">S</span>|!
                      (|<span className="math-var">F</span>| &minus; |<span className="math-var">S</span>| &minus; 1)!
                    </span>
                    <span className="math-fraction-bar" aria-hidden="true" />
                    <span className="math-denominator">
                      |<span className="math-var">F</span>|!
                    </span>
                  </span>

                  <span className="math-term">
                    <span className="math-bracket">[</span>
                    <span className="math-var">f</span>
                    <sub className="math-sub">x</sub>
                    <span className="math-parens">(</span>
                    <span className="math-var">S</span>
                    <span className="math-sym">&#8746;</span>
                    <span className="math-set">&#123;<span className="math-var">i</span>&#125;</span>
                    <span className="math-parens">)</span>
                    <span className="math-operator">&minus;</span>
                    <span className="math-var">f</span>
                    <sub className="math-sub">x</sub>
                    <span className="math-parens">(</span>
                    <span className="math-var">S</span>
                    <span className="math-parens">)</span>
                    <span className="math-bracket">]</span>
                  </span>
                </div>
              </div>

              <div className="symbol-definitions">
                <h4 className="symbol-definitions-title">Symbol Definitions</h4>
                <div className="table-wrap">
                  <table className="editorial-table symbol-table">
                    <thead>
                      <tr>
                        <th scope="col">Symbol</th>
                        <th scope="col">Definition</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="font-mono"><strong>ϕ<sub>i</sub>(f, x)</strong></td>
                        <td>Shapley value (marginal attribution) of feature <em>i</em> for prediction model <em>f</em> on patient instance <em>x</em></td>
                      </tr>
                      <tr>
                        <td className="font-mono"><strong>F</strong></td>
                        <td>Total set of all input features (|F| = 25 transformed features)</td>
                      </tr>
                      <tr>
                        <td className="font-mono"><strong>S</strong></td>
                        <td>Subset of features drawn from all features excluding feature <em>i</em> (S &#8838; F \ &#123;i&#125;)</td>
                      </tr>
                      <tr>
                        <td className="font-mono"><strong>|S|, |F|</strong></td>
                        <td>Cardinality (number of features) in subset <em>S</em> and total feature set <em>F</em></td>
                      </tr>
                      <tr>
                        <td className="font-mono"><strong>f<sub>x</sub>(S)</strong></td>
                        <td>Model prediction evaluated using only the features present in subset <em>S</em></td>
                      </tr>
                      <tr>
                        <td className="font-mono"><strong>f<sub>x</sub>(S &#8746; &#123;i&#125;) &minus; f<sub>x</sub>(S)</strong></td>
                        <td>Marginal contribution gained by adding feature <em>i</em> to coalition <em>S</em></td>
                      </tr>
                      <tr>
                        <td className="font-mono"><strong>|S|!(|F| &minus; |S| &minus; 1)! / |F|!</strong></td>
                        <td>Combinatorial weighting factor averaging over all possible feature permutations</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </details>
        </div>

        <div className="guardrail-item">
          <strong className="guardrail-title">Absence of Clinical Causality</strong>
          <p className="guardrail-desc">
            SHAP highlights the mathematical features most strongly utilized by the Random Forest classifier. High attribution does not imply biological cause and effect; lifestyle parameters may correlate with unmeasured physiological or demographic confounding factors.
          </p>
        </div>
      </footer>
    </article>
  );
};
