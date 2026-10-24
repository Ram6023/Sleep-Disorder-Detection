import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  predictSleepDisorder,
  fetchModelInfo,
  checkHealth,
  formatApiErrorDetail,
  ApiClientError,
} from '../services/api.ts';
import type { SleepPredictionInput } from '../types/index.ts';

describe('API Service Unit Tests', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  const mockValidInput: SleepPredictionInput = {
    gender: 'Male',
    age: 42,
    occupation: 'Software Engineer',
    sleep_duration: 7.2,
    quality_of_sleep: 8,
    physical_activity_level: 60,
    stress_level: 4,
    bmi_category: 'Normal',
    blood_pressure: '120/80',
    bp_systolic: 120,
    bp_diastolic: 80,
    heart_rate: 70,
    daily_steps: 7500,
    person_id: 101,
  };

  it('formats complex FastAPI error detail structures correctly', () => {
    expect(formatApiErrorDetail('Simple error string')).toBe('Simple error string');

    const detailList = [
      { loc: ['body', 'Age'], msg: 'Input should be between 1 and 120' },
      { loc: ['body', 'Sleep Duration'], msg: 'Field required' },
    ];
    const formatted = formatApiErrorDetail(detailList);
    expect(formatted).toContain('Age: Input should be between 1 and 120');
    expect(formatted).toContain('Sleep Duration: Field required');
  });

  it('sends correct POST request with explain=false by default', async () => {
    const mockResponseData = {
      prediction_id: 'test-uuid-1234',
      model_version: '1.0.0',
      contract_version: '1.0.0',
      predicted_class: 0,
      predicted_class_name: 'No recorded sleep disorder',
      probabilities: {
        'No recorded sleep disorder': 0.89,
        Insomnia: 0.08,
        'Sleep Apnea': 0.03,
      },
      disclaimer: 'Academic disclaimer',
      timestamp_utc: '2026-10-24T12:00:00Z',
    };

    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockResponseData,
    } as Response);

    const result = await predictSleepDisorder(mockValidInput, false, 'http://127.0.0.1:8000');

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [callUrl, callOptions] = fetchSpy.mock.calls[0];
    expect(callUrl).toBe('http://127.0.0.1:8000/api/predict?explain=false');
    expect(callOptions?.method).toBe('POST');
    expect(JSON.parse(callOptions?.body as string)).toEqual(mockValidInput);

    expect(result.prediction_id).toBe('test-uuid-1234');
    expect(result.predicted_class).toBe(0);
    expect(result.probabilities['No recorded sleep disorder']).toBe(0.89);
  });

  it('sends explain=true query parameter when requested', async () => {
    const mockResponseData = {
      prediction_id: 'test-uuid-5678',
      model_version: '1.0.0',
      contract_version: '1.0.0',
      predicted_class: 1,
      predicted_class_name: 'Insomnia',
      probabilities: {
        'No recorded sleep disorder': 0.15,
        Insomnia: 0.75,
        'Sleep Apnea': 0.1,
      },
      explanation: {
        method: 'SHAP TreeExplainer',
        base_value: 0.33,
        top_factors: [
          {
            feature: 'Sleep Duration',
            value: 5.5,
            shap_value: 0.25,
            direction: 'supports_prediction',
          },
        ],
      },
      disclaimer: 'Academic disclaimer',
      timestamp_utc: '2026-10-24T12:00:00Z',
    };

    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockResponseData,
    } as Response);

    const result = await predictSleepDisorder(mockValidInput, true, 'http://127.0.0.1:8000');

    expect(fetchSpy.mock.calls[0][0]).toBe('http://127.0.0.1:8000/api/predict?explain=true');
    expect(result.explanation).toBeDefined();
    expect(result.explanation?.top_factors[0].feature).toBe('Sleep Duration');
  });

  it('handles HTTP 422 validation errors without fabricating predictions', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: false,
      status: 422,
      statusText: 'Unprocessable Entity',
      json: async () => ({
        detail: 'Systolic pressure (80 mmHg) must be strictly greater than diastolic pressure (120 mmHg).',
      }),
    } as Response);

    await expect(
      predictSleepDisorder(mockValidInput, false, 'http://127.0.0.1:8000')
    ).rejects.toThrow(ApiClientError);
  });

  it('handles HTTP 503 service unavailable when model artifacts are not loaded', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: false,
      status: 503,
      statusText: 'Service Unavailable',
      json: async () => ({
        detail: 'Model service is currently unavailable. Serialized model artifacts could not be loaded.',
      }),
    } as Response);

    try {
      await predictSleepDisorder(mockValidInput, false, 'http://127.0.0.1:8000');
      expect.fail('Should have thrown an error');
    } catch (err) {
      expect(err).toBeInstanceOf(ApiClientError);
      const apiErr = err as ApiClientError;
      expect(apiErr.statusCode).toBe(503);
      expect(apiErr.message).toContain('Model service is currently unavailable');
    }
  });

  it('handles network connection failures gracefully', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValueOnce(new TypeError('Failed to fetch'));

    try {
      await predictSleepDisorder(mockValidInput, false, 'http://127.0.0.1:8000');
      expect.fail('Should have thrown an error');
    } catch (err) {
      expect(err).toBeInstanceOf(ApiClientError);
      const apiErr = err as ApiClientError;
      expect(apiErr.isNetworkError).toBe(true);
      expect(apiErr.message).toContain('Unable to connect to prediction service');
    }
  });

  it('fetches model info successfully from GET /api/model-info', async () => {
    const mockModelInfo = {
      model_name: 'Random Forest (Base Features Only)',
      model_version: '1.0.0',
      contract_version: '1.0.0',
      target_classes: ['None', 'Insomnia', 'Sleep Apnea'],
      display_class_names: ['No recorded sleep disorder', 'Insomnia', 'Sleep Apnea'],
      input_feature_count: 12,
      transformed_feature_count: 25,
      transformed_features: ['Gender_Female', 'Gender_Male'],
      validation_bounds: {
        age: { min: 1, max: 120 },
      },
      disclaimer: 'Academic disclaimer',
    };

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockModelInfo,
    } as Response);

    const data = await fetchModelInfo('http://127.0.0.1:8000');
    expect(data.model_name).toBe('Random Forest (Base Features Only)');
    expect(data.target_classes).toHaveLength(3);
  });

  it('fetches health status from GET /health', async () => {
    const mockHealth = {
      status: 'healthy',
      model_loaded: true,
      model_version: '1.0.0',
      contract_version: '1.0.0',
      timestamp_utc: '2026-10-24T12:00:00Z',
    };

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockHealth,
    } as Response);

    const health = await checkHealth('http://127.0.0.1:8000');
    expect(health.status).toBe('healthy');
    expect(health.model_loaded).toBe(true);
  });
});
