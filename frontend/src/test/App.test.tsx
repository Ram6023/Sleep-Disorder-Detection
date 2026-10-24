import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { App } from '../App.tsx';

describe('App Full Integration Tests', () => {
  beforeEach(() => {
    vi.restoreAllMocks();

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      const urlStr = String(url);
      if (urlStr.includes('/health')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            status: 'healthy',
            model_loaded: true,
            model_version: '1.0.0',
            contract_version: '1.0.0',
            timestamp_utc: '2026-10-24T12:00:00Z',
          }),
        } as Response;
      }
      if (urlStr.includes('/api/model-info')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            model_name: 'Random Forest (Base Features Only)',
            model_version: '1.0.0',
            contract_version: '1.0.0',
            target_classes: ['None', 'Insomnia', 'Sleep Apnea'],
            display_class_names: ['No recorded sleep disorder', 'Insomnia', 'Sleep Apnea'],
            input_feature_count: 12,
            transformed_feature_count: 25,
            transformed_features: ['Gender_Female', 'Gender_Male'],
            validation_bounds: {},
            disclaimer: 'Academic disclaimer',
          }),
        } as Response;
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({}),
      } as Response;
    });
  });

  it('renders branding and navigates across all tabs smoothly', async () => {
    render(<App />);

    expect(screen.getByText('Sleep Disorder Detection')).toBeInTheDocument();
    expect(screen.getAllByText(/ML-Based Intelligent System/i).length).toBeGreaterThan(0);

    // Verify health status pill
    await waitFor(() => {
      expect(screen.getByText(/Model Online \(v1.0.0\)/i)).toBeInTheDocument();
    });

    // Test Navigation to Sleep Assessment Tab
    const assessTab = screen.getByRole('button', { name: /^Sleep Assessment$/i });
    fireEvent.click(assessTab);
    expect(screen.getByRole('heading', { name: /Sleep Health Assessment/i })).toBeInTheDocument();

    // Test Navigation to About Model Tab
    const modelTab = screen.getByRole('button', { name: /^About the Model$/i });
    fireEvent.click(modelTab);
    expect(
      screen.getByRole('heading', { name: /Intelligent Random Forest Classification Pipeline/i })
    ).toBeInTheDocument();
    expect(screen.getByText(/Holdout Test Accuracy/i)).toBeInTheDocument();

    // Test Navigation to Ethics & Safety Tab
    const ethicsTab = screen.getByRole('button', { name: /^Ethics & Safety$/i });
    fireEvent.click(ethicsTab);
    expect(
      screen.getByRole('heading', { name: /Research Integrity & Ethical Guardrails/i })
    ).toBeInTheDocument();

    // Return to Overview
    const overviewTab = screen.getByRole('button', { name: /^Overview$/i });
    fireEvent.click(overviewTab);
    expect(screen.getByRole('heading', { name: /^Overview$/i })).toBeInTheDocument();
  });
});
