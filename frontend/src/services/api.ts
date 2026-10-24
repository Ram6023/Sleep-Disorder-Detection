import type {
  SleepPredictionInput,
  SleepPredictionOutput,
  ModelContractInfo,
  HealthStatus,
  ApiErrorResponse,
  ApiErrorDetail,
} from '../types/index.ts';

export const API_BASE_URL =
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE_URL) ||
  'http://127.0.0.1:8000';

export class ApiClientError extends Error {
  statusCode?: number;
  details?: string | ApiErrorDetail[];
  isNetworkError: boolean;

  constructor(
    message: string,
    options?: {
      statusCode?: number;
      details?: string | ApiErrorDetail[];
      isNetworkError?: boolean;
    }
  ) {
    super(message);
    this.name = 'ApiClientError';
    this.statusCode = options?.statusCode;
    this.details = options?.details;
    this.isNetworkError = options?.isNetworkError ?? false;
  }
}

/**
 * Parses FastAPI response body error details into human-readable text
 */
export function formatApiErrorDetail(details?: string | ApiErrorDetail[]): string {
  if (!details) return '';
  if (typeof details === 'string') return details;
  if (Array.isArray(details)) {
    return details
      .map((item) => {
        const field = item.loc ? item.loc.filter((l) => l !== 'body').join(' -> ') : '';
        return field ? `${field}: ${item.msg}` : item.msg || 'Invalid field value';
      })
      .join('; ');
  }
  return String(details);
}

/**
 * Predict sleep disorder using validated lifestyle/health parameters
 */
export async function predictSleepDisorder(
  input: SleepPredictionInput,
  explain: boolean = false,
  baseUrl: string = API_BASE_URL
): Promise<SleepPredictionOutput> {
  const url = `${baseUrl.replace(/\/+$/, '')}/api/predict?explain=${explain ? 'true' : 'false'}`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(input),
    });
  } catch (err) {
    const errorMsg =
      err instanceof Error ? err.message : 'Unknown network connection failure';
    throw new ApiClientError(
      `Unable to connect to prediction service at ${baseUrl}. Ensure backend is running. (${errorMsg})`,
      { isNetworkError: true }
    );
  }

  if (!response.ok) {
    let errorData: ApiErrorResponse | null = null;
    try {
      errorData = (await response.json()) as ApiErrorResponse;
    } catch {
      // Non-JSON error body
    }

    const detailText = formatApiErrorDetail(errorData?.detail);

    if (response.status === 422) {
      throw new ApiClientError(
        detailText || 'Input validation failed. Please check the entered parameters.',
        {
          statusCode: 422,
          details: errorData?.detail,
        }
      );
    }

    if (response.status === 503) {
      throw new ApiClientError(
        detailText ||
          'Model service is currently unavailable. Serialized model artifacts could not be loaded.',
        {
          statusCode: 503,
          details: errorData?.detail,
        }
      );
    }

    throw new ApiClientError(
      detailText || `Prediction request failed with HTTP ${response.status} (${response.statusText}).`,
      {
        statusCode: response.status,
        details: errorData?.detail,
      }
    );
  }

  const data = (await response.json()) as SleepPredictionOutput;
  return data;
}

/**
 * Fetch model metadata, versioning, feature schemas, and validation ranges
 */
export async function fetchModelInfo(
  baseUrl: string = API_BASE_URL
): Promise<ModelContractInfo> {
  const url = `${baseUrl.replace(/\/+$/, '')}/api/model-info`;

  let response: Response;
  try {
    response = await fetch(url, {
      headers: { Accept: 'application/json' },
    });
  } catch (err) {
    const errorMsg =
      err instanceof Error ? err.message : 'Network error';
    throw new ApiClientError(
      `Failed to retrieve model info from ${baseUrl}. (${errorMsg})`,
      { isNetworkError: true }
    );
  }

  if (!response.ok) {
    let errorData: ApiErrorResponse | null = null;
    try {
      errorData = (await response.json()) as ApiErrorResponse;
    } catch {
      // Ignored
    }
    throw new ApiClientError(
      formatApiErrorDetail(errorData?.detail) ||
        `Model info request failed with HTTP ${response.status}.`,
      { statusCode: response.status }
    );
  }

  return (await response.json()) as ModelContractInfo;
}

/**
 * Health check endpoint for system readiness
 */
export async function checkHealth(
  baseUrl: string = API_BASE_URL
): Promise<HealthStatus> {
  const url = `${baseUrl.replace(/\/+$/, '')}/health`;

  let response: Response;
  try {
    response = await fetch(url, {
      headers: { Accept: 'application/json' },
    });
  } catch (err) {
    const errorMsg =
      err instanceof Error ? err.message : 'Network error';
    throw new ApiClientError(
      `Failed to reach health endpoint at ${baseUrl}. (${errorMsg})`,
      { isNetworkError: true }
    );
  }

  if (!response.ok) {
    throw new ApiClientError(
      `Health check failed with HTTP ${response.status}.`,
      { statusCode: response.status }
    );
  }

  return (await response.json()) as HealthStatus;
}
