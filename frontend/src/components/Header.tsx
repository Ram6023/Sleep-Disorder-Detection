import React from 'react';
import type { HealthStatus } from '../types/index.ts';

export type NavTab = 'overview' | 'assess' | 'results' | 'explain' | 'model' | 'ethics';

interface HeaderProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  health: HealthStatus | null;
  healthLoading: boolean;
  hasResult: boolean;
  hasExplanation: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onTabChange,
  health,
  healthLoading,
  hasResult,
  hasExplanation,
}) => {
  return (
    <header className="site-header">
      <div className="header-inner">
        <div className="brand" onClick={() => onTabChange('overview')} role="button" tabIndex={0}>
          <div className="brand-text">
            <h1 className="brand-title">Sleep Disorder Detection</h1>
            <span className="brand-subtitle">ML-Based Intelligent System Using Lifestyle and Health Data</span>
          </div>
        </div>

        <nav className="header-nav" aria-label="Main Navigation">
          <button
            type="button"
            className={`nav-link ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => onTabChange('overview')}
            aria-current={activeTab === 'overview' ? 'page' : undefined}
          >
            Overview
          </button>
          <button
            type="button"
            className={`nav-link ${activeTab === 'assess' ? 'active' : ''}`}
            onClick={() => onTabChange('assess')}
            aria-current={activeTab === 'assess' ? 'page' : undefined}
          >
            Sleep Assessment
          </button>
          <button
            type="button"
            className={`nav-link ${activeTab === 'results' ? 'active' : ''}`}
            onClick={() => onTabChange('results')}
            aria-current={activeTab === 'results' ? 'page' : undefined}
          >
            Prediction Results {hasResult && <span className="tab-indicator-dot" />}
          </button>
          <button
            type="button"
            className={`nav-link ${activeTab === 'explain' ? 'active' : ''}`}
            onClick={() => onTabChange('explain')}
            aria-current={activeTab === 'explain' ? 'page' : undefined}
          >
            SHAP Explainability {hasExplanation && <span className="tab-indicator-dot active-dot" />}
          </button>
          <button
            type="button"
            className={`nav-link ${activeTab === 'model' ? 'active' : ''}`}
            onClick={() => onTabChange('model')}
            aria-current={activeTab === 'model' ? 'page' : undefined}
          >
            About the Model
          </button>
          <button
            type="button"
            className={`nav-link ${activeTab === 'ethics' ? 'active' : ''}`}
            onClick={() => onTabChange('ethics')}
            aria-current={activeTab === 'ethics' ? 'page' : undefined}
          >
            Ethics &amp; Safety
          </button>
        </nav>

        <div className="header-status">
          {healthLoading ? (
            <div className="status-pill status-loading" title="Checking backend health...">
              <span className="status-dot loading" />
              <span>Checking API...</span>
            </div>
          ) : health?.model_loaded ? (
            <div className="status-pill status-healthy" title={`Connected to FastAPI backend (Model v${health.model_version})`}>
              <span className="status-dot online" />
              <span>Model Online (v{health.model_version})</span>
            </div>
          ) : health ? (
            <div className="status-pill status-degraded" title="Backend connected but model is not loaded">
              <span className="status-dot degraded" />
              <span>Model Unavailable</span>
            </div>
          ) : (
            <div className="status-pill status-offline" title="Cannot reach backend server">
              <span className="status-dot offline" />
              <span>Backend Offline</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
