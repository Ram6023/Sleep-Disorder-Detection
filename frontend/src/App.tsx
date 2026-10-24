import { useState, useEffect, useCallback } from 'react';
import type {
  SleepPredictionInput,
  SleepPredictionOutput,
  ModelContractInfo,
  HealthStatus,
} from './types/index.ts';
import {
  predictSleepDisorder,
  fetchModelInfo,
  checkHealth,
  ApiClientError,
} from './services/api.ts';
import { Header, type NavTab } from './components/Header.tsx';
import { HomeDashboard } from './components/HomeDashboard.tsx';
import { PredictionForm } from './components/PredictionForm.tsx';
import { ResultCard } from './components/ResultCard.tsx';
import { FeatureImportance } from './components/FeatureImportance.tsx';
import { AboutModel } from './components/AboutModel.tsx';
import { ResearchEthics } from './components/ResearchEthics.tsx';
import './App.css';

export function App() {
  const [activeTab, setActiveTab] = useState<NavTab>('overview');

  // Backend Health & Metadata State
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);
  const [modelInfo, setModelInfo] = useState<ModelContractInfo | null>(null);
  const [modelInfoLoading, setModelInfoLoading] = useState<boolean>(true);

  // Prediction State
  const [latestResult, setLatestResult] = useState<SleepPredictionOutput | null>(null);
  const [isLoadingPrediction, setIsLoadingPrediction] = useState<boolean>(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  // Poll initial health and model metadata
  const loadSystemStatus = useCallback(async () => {
    setHealthLoading(true);
    try {
      const h = await checkHealth();
      setHealth(h);
    } catch {
      setHealth(null);
    } finally {
      setHealthLoading(false);
    }

    setModelInfoLoading(true);
    try {
      const m = await fetchModelInfo();
      setModelInfo(m);
    } catch {
      // Model info failed (backend offline)
    } finally {
      setModelInfoLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSystemStatus();
  }, [loadSystemStatus]);

  // Handle Form Submission
  const handlePredictionSubmit = async (
    input: SleepPredictionInput,
    explain: boolean
  ) => {
    setIsLoadingPrediction(true);
    setPredictionError(null);

    try {
      const res = await predictSleepDisorder(input, explain);
      setLatestResult(res);
      // Auto-navigate to Results tab upon successful prediction
      setActiveTab('results');
    } catch (err) {
      setLatestResult(null);
      setActiveTab('assess');
      if (err instanceof ApiClientError) {
        setPredictionError(err.message);
      } else if (err instanceof Error) {
        setPredictionError(err.message);
      } else {
        setPredictionError('An unexpected error occurred during prediction.');
      }
    } finally {
      setIsLoadingPrediction(false);
    }
  };

  return (
    <div className="app-container">
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        health={health}
        healthLoading={healthLoading}
        hasResult={latestResult !== null}
        hasExplanation={Boolean(latestResult?.explanation)}
      />

      <main className="main-content" id="main-content">
        {activeTab === 'overview' && (
          <HomeDashboard
            onStartAssessment={() => setActiveTab('assess')}
            onViewModel={() => setActiveTab('model')}
            onViewResults={() => setActiveTab('results')}
            latestResult={latestResult}
            health={health}
          />
        )}

        {activeTab === 'assess' && (
          <div className="tab-assess-layout">
            <PredictionForm
              onSubmit={handlePredictionSubmit}
              isLoading={isLoadingPrediction}
              serverError={predictionError}
            />
          </div>
        )}

        {activeTab === 'results' && (
          <div className="tab-results-layout">
            <ResultCard
              result={latestResult}
              isLoading={isLoadingPrediction}
              onViewExplainability={() => setActiveTab('explain')}
              onNewAssessment={() => setActiveTab('assess')}
            />
          </div>
        )}

        {activeTab === 'explain' && (
          <div className="tab-explain-layout">
            <FeatureImportance
              result={latestResult}
              onNavigateToForm={() => setActiveTab('assess')}
            />
          </div>
        )}

        {activeTab === 'model' && (
          <div className="tab-model-layout">
            <AboutModel modelInfo={modelInfo} loading={modelInfoLoading} />
          </div>
        )}

        {activeTab === 'ethics' && (
          <div className="tab-ethics-layout">
            <ResearchEthics />
          </div>
        )}
      </main>

      <footer className="site-footer">
        <div className="footer-inner">
          <div>
            <strong>ML-Based Intelligent System for Automated Sleep Disorder Detection Using Lifestyle and Health Data</strong>
            <p>Academic Machine Learning Research Prototype &bull; Educational and Evaluation Use Only</p>
          </div>
          <div>
            <span>Non-Diagnostic Clinical Disclaimer &bull; Research &amp; Educational Use</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
