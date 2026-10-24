/**
 * API Contract Types for Sleep Disorder Detection System
 * Complies with docs/api_contract.md (v1.0.0)
 */

export type TargetClassName =
  | 'No recorded sleep disorder'
  | 'Insomnia'
  | 'Sleep Apnea';

export interface SleepPredictionInput {
  gender: string;
  age: number;
  occupation: string;
  sleep_duration: number;
  quality_of_sleep: number;
  physical_activity_level: number;
  stress_level: number;
  bmi_category: string;
  blood_pressure?: string;
  bp_systolic?: number;
  bp_diastolic?: number;
  heart_rate: number;
  daily_steps: number;
  person_id?: number;
}

export interface FeatureAttribution {
  feature: string;
  value: number | string;
  shap_value: number;
  direction: 'supports_prediction' | 'opposes_prediction' | string;
}

export interface PredictionExplanation {
  method: string;
  base_value: number;
  top_factors: FeatureAttribution[];
}

export interface SleepPredictionOutput {
  prediction_id: string;
  model_version: string;
  contract_version: string;
  predicted_class: number;
  predicted_class_name: string;
  probabilities: Record<string, number>;
  explanation?: PredictionExplanation | null;
  disclaimer: string;
  timestamp_utc: string;
}

export interface ValidationBound {
  min?: number;
  max?: number;
}

export interface ModelContractInfo {
  model_name: string;
  model_version: string;
  contract_version: string;
  target_classes: string[];
  display_class_names: string[];
  input_feature_count: number;
  transformed_feature_count: number;
  transformed_features: string[];
  validation_bounds: Record<string, ValidationBound>;
  disclaimer: string;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  model_version: string;
  contract_version: string;
  timestamp_utc: string;
}

export interface ApiErrorDetail {
  loc?: (string | number)[];
  msg?: string;
  type?: string;
}

export interface ApiErrorResponse {
  detail?: string | ApiErrorDetail[];
}

export interface FormValidationErrors {
  gender?: string;
  age?: string;
  occupation?: string;
  sleep_duration?: string;
  quality_of_sleep?: string;
  physical_activity_level?: string;
  stress_level?: string;
  bmi_category?: string;
  blood_pressure?: string;
  bp_systolic?: string;
  bp_diastolic?: string;
  heart_rate?: string;
  daily_steps?: string;
  general?: string;
}

export interface SampleProfile {
  name: string;
  description: string;
  expectedClass: string;
  payload: SleepPredictionInput;
}
