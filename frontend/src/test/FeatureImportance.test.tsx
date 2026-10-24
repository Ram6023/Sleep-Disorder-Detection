import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { FeatureImportance } from '../components/FeatureImportance.tsx';
import type { SleepPredictionOutput } from '../types/index.ts';

describe('FeatureImportance Component Tests', () => {
  const mockResultWithXai: SleepPredictionOutput = {
    prediction_id: 'xai-uuid-1',
    model_version: '1.0.0',
    contract_version: '1.0.0',
    predicted_class: 2,
    predicted_class_name: 'Sleep Apnea',
    probabilities: {
      'No recorded sleep disorder': 0.05,
      Insomnia: 0.15,
      'Sleep Apnea': 0.8,
    },
    explanation: {
      method: 'SHAP TreeExplainer',
      base_value: 0.425,
      top_factors: [
        {
          feature: 'BP_Systolic',
          value: 142.0,
          shap_value: 0.152,
          direction: 'supports_prediction',
        },
        {
          feature: 'Sleep Duration',
          value: 5.6,
          shap_value: -0.045,
          direction: 'opposes_prediction',
        },
      ],
    },
    disclaimer: 'Disclaimer text',
    timestamp_utc: '2026-10-24T12:00:00Z',
  };

  it('renders empty message when no prediction is loaded', () => {
    render(<FeatureImportance result={null} />);
    expect(screen.getByText(/No Active Prediction for Explainability/i)).toBeInTheDocument();
  });

  it('renders omitted notice when prediction has no explanation', () => {
    const resultNoXai = { ...mockResultWithXai, explanation: null };
    render(<FeatureImportance result={resultNoXai} />);
    expect(screen.getByText(/SHAP Explanation Omitted for this Ingestion/i)).toBeInTheDocument();
  });

  it('renders SHAP method, simplified Model Baseline, and feature attribution factors', () => {
    render(<FeatureImportance result={mockResultWithXai} />);

    expect(screen.getByText(/Local SHAP Feature Attribution/i)).toBeInTheDocument();
    expect(screen.getByText('SHAP TreeExplainer')).toBeInTheDocument();
    
    // Model Baseline section
    expect(screen.getByText('Model Baseline')).toBeInTheDocument();
    expect(screen.getByText('0.4250')).toBeInTheDocument();
    expect(
      screen.getByText("The model's starting output before accounting for the input features.")
    ).toBeInTheDocument();
    expect(screen.queryByText(/Baseline Expected Value/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Prior class probability/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/E\[f\(x\)\]/i)).not.toBeInTheDocument();

    expect(screen.getByText('BP_Systolic')).toBeInTheDocument();
    expect(screen.getAllByText(/Value:/i)).toHaveLength(2);
    expect(screen.getByText('+0.1520')).toBeInTheDocument();
    expect(screen.getByText(/▲ Supports/i)).toBeInTheDocument();

    expect(screen.getByText('Sleep Duration')).toBeInTheDocument();
    expect(screen.getByText('-0.0450')).toBeInTheDocument();
    expect(screen.getByText(/▼ Opposes/i)).toBeInTheDocument();
  });

  it('renders clinical non-causality warning statement', () => {
    render(<FeatureImportance result={mockResultWithXai} />);
    expect(screen.getByText(/Absence of Clinical Causality/i)).toBeInTheDocument();
  });

  it('renders the preferred non-technical visible explanation', () => {
    render(<FeatureImportance result={mockResultWithXai} />);
    expect(
      screen.getByText(
        /SHAP estimates how much each input feature contributes to a model's prediction relative to a baseline\. Positive contributions move the model output in one direction, while negative contributions move it in the opposite direction\. These attributions explain model behavior and do not establish causation\./i
      )
    ).toBeInTheDocument();
  });

  it('renders the expandable "View mathematical formula" section with mathematical notation and symbol definitions', () => {
    render(<FeatureImportance result={mockResultWithXai} />);

    // Expandable summary
    expect(screen.getByText('View mathematical formula')).toBeInTheDocument();

    // Mathematical formula container with accessible role and aria-label
    const mathDisplay = screen.getByRole('math');
    expect(mathDisplay).toBeInTheDocument();
    expect(mathDisplay).toHaveAttribute(
      'aria-label',
      expect.stringContaining('Shapley value formula')
    );

    // Symbol definitions table
    expect(screen.getByText('Symbol Definitions')).toBeInTheDocument();
    expect(screen.getByText(/Shapley value \(marginal attribution\) of feature/i)).toBeInTheDocument();
    expect(screen.getByText(/Total set of all input features/i)).toBeInTheDocument();
    expect(screen.getByText(/Combinatorial weighting factor/i)).toBeInTheDocument();
  });

  it('verifies that no raw LaTeX commands are visible in the rendered interface', () => {
    const { container } = render(<FeatureImportance result={mockResultWithXai} />);
    const textContent = container.textContent || '';

    // Verify common LaTeX tokens are not present as raw unparsed strings
    expect(textContent).not.toContain('\\phi');
    expect(textContent).not.toContain('\\sum');
    expect(textContent).not.toContain('\\frac');
    expect(textContent).not.toContain('\\subseteq');
    expect(textContent).not.toContain('\\setminus');
    expect(textContent).not.toContain('\\cup');
    expect(textContent).not.toContain('$\\phi');
  });
});

