export interface ModelEvaluationSummary {
  selectedModel: string;
  architecture: string;
  hyperparameters: Record<string, unknown>;
  cvMetrics: {
    macroF1: number;
    macroF1Std: number;
    balancedAccuracy: number;
    accuracy: number;
  };
  testMetrics: {
    accuracy: number;
    balancedAccuracy: number;
    macroF1: number;
    weightedF1: number;
    macroRocAuc: number;
    weightedRocAuc: number;
    perClass: Record<
      string,
      {
        precision: number;
        recall: number;
        f1: number;
        support: number;
        rocAuc: number;
      }
    >;
  };
  benchmarkComparison: Array<{
    model: string;
    configuration: string;
    cvMacroF1: number;
    cvBalancedAcc: number;
    cvAccuracy: number;
    status: 'selected' | 'evaluated';
  }>;
}

export const MODEL_EVALUATION_DATA: ModelEvaluationSummary = {
  selectedModel: 'Random Forest (Base Features Only)',
  architecture: 'Scikit-Learn Pipeline with OneHotEncoder, RobustScaler & 100 Trees',
  hyperparameters: {
    n_estimators: 100,
    max_depth: 6,
    min_samples_split: 4,
    min_samples_leaf: 1,
    random_state: 42,
    class_weight: 'None (Balanced by base performance)',
  },
  cvMetrics: {
    macroF1: 0.8693,
    macroF1Std: 0.02,
    balancedAccuracy: 0.8673,
    accuracy: 0.8962,
  },
  testMetrics: {
    accuracy: 0.96,
    balancedAccuracy: 0.9347,
    macroF1: 0.9352,
    weightedF1: 0.9599,
    macroRocAuc: 0.9957,
    weightedRocAuc: 0.9973,
    perClass: {
      'No recorded sleep disorder': {
        precision: 1.0,
        recall: 1.0,
        f1: 1.0,
        support: 44,
        rocAuc: 1.0,
      },
      Insomnia: {
        precision: 0.9286,
        recall: 0.8667,
        f1: 0.8966,
        support: 15,
        rocAuc: 0.9944,
      },
      'Sleep Apnea': {
        precision: 0.8824,
        recall: 0.9375,
        f1: 0.9091,
        support: 16,
        rocAuc: 0.9926,
      },
    },
  },
  benchmarkComparison: [
    {
      model: 'Random Forest',
      configuration: 'Base Features Only (leakage-free)',
      cvMacroF1: 0.8693,
      cvBalancedAcc: 0.8673,
      cvAccuracy: 0.8962,
      status: 'selected',
    },
    {
      model: 'Logistic Regression',
      configuration: 'Base + Engineered + SMOTE',
      cvMacroF1: 0.8658,
      cvBalancedAcc: 0.8667,
      cvAccuracy: 0.8894,
      status: 'evaluated',
    },
    {
      model: 'Bagging Classifier',
      configuration: 'Base + Engineered',
      cvMacroF1: 0.8651,
      cvBalancedAcc: 0.8639,
      cvAccuracy: 0.8895,
      status: 'evaluated',
    },
    {
      model: 'Gradient Boosting',
      configuration: 'Base Features Only',
      cvMacroF1: 0.8523,
      cvBalancedAcc: 0.8512,
      cvAccuracy: 0.8814,
      status: 'evaluated',
    },
    {
      model: 'Support Vector Machine (SVM)',
      configuration: 'RBF Kernel (Standardized)',
      cvMacroF1: 0.8492,
      cvBalancedAcc: 0.8465,
      cvAccuracy: 0.8782,
      status: 'evaluated',
    },
    {
      model: 'K-Nearest Neighbors (KNN)',
      configuration: 'k=5, Distance Weighting',
      cvMacroF1: 0.8415,
      cvBalancedAcc: 0.8398,
      cvAccuracy: 0.8712,
      status: 'evaluated',
    },
  ],
};
