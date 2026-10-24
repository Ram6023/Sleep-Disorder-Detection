import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ResultCard } from '../components/ResultCard.tsx';
import type { SleepPredictionOutput } from '../types/index.ts';

describe('ResultCard Component Tests', () => {
  const mockHealthyResult: SleepPredictionOutput = {
    prediction_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
    model_version: '1.0.0',
    contract_version: '1.0.0',
    predicted_class: 0,
    predicted_class_name: 'No recorded sleep disorder',
    probabilities: {
      'No recorded sleep disorder': 0.8905,
      Insomnia: 0.0744,
      'Sleep Apnea': 0.0351,
    },
    explanation: {
      method: 'SHAP TreeExplainer',
      base_value: 0.5794,
      top_factors: [
        {
          feature: 'BP_Systolic',
          value: 120,
          shap_value: 0.0934,
          direction: 'supports_prediction',
        },
      ],
    },
    disclaimer: 'Educational research disclaimer text.',
    timestamp_utc: '2026-10-24T12:00:00Z',
  };

  it('renders empty state when result is null and not loading', () => {
    const handleNew = vi.fn();
    render(
      <ResultCard
        result={null}
        isLoading={false}
        onNewAssessment={handleNew}
      />
    );

    expect(screen.getByText(/Awaiting Sleep Assessment/i)).toBeInTheDocument();
    const btn = screen.getByRole('button', { name: /Go to Sleep Assessment Form/i });
    fireEvent.click(btn);
    expect(handleNew).toHaveBeenCalledTimes(1);
  });

  it('renders skeleton loading state when isLoading is true', () => {
    render(
      <ResultCard
        result={null}
        isLoading={true}
      />
    );

    expect(screen.getByText(/Querying trained Random Forest model/i)).toBeInTheDocument();
  });

  it('renders prediction results with class name, status badge, and probabilities', () => {
    render(
      <ResultCard
        result={mockHealthyResult}
        isLoading={false}
      />
    );

    expect(screen.getByRole('heading', { name: 'No recorded sleep disorder' })).toBeInTheDocument();
    expect(screen.getByText(/Class 0: No recorded sleep disorder/i)).toBeInTheDocument();
    expect(screen.getByText(/89\.[01]% Probability/)).toBeInTheDocument();

    expect(screen.getByText(/^89\.[01]%$/)).toBeInTheDocument();
    expect(screen.getByText('7.4%')).toBeInTheDocument();
    expect(screen.getByText('3.5%')).toBeInTheDocument();
  });

  it('shows SHAP callout button when explanation is present and calls onViewExplainability', () => {
    const handleExplain = vi.fn();
    render(
      <ResultCard
        result={mockHealthyResult}
        isLoading={false}
        onViewExplainability={handleExplain}
      />
    );

    expect(screen.getByText(/Local Explainability Available/i)).toBeInTheDocument();
    const btn = screen.getByRole('button', { name: /Inspect SHAP Explanations/i });
    fireEvent.click(btn);
    expect(handleExplain).toHaveBeenCalledTimes(1);
  });

  it('displays technical metadata including prediction ID and model version', () => {
    render(
      <ResultCard
        result={mockHealthyResult}
        isLoading={false}
      />
    );

    expect(screen.getByText(/Random Forest v1.0.0/i)).toBeInTheDocument();
    expect(screen.getAllByText(/v1\.0\.0/i).length).toBeGreaterThan(0);
  });
});
