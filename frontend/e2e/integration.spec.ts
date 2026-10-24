import { test, expect } from '@playwright/test';

test.describe('Task 1.11 Full-Stack Integration & End-to-End Test Suite', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to frontend root
    await page.goto('/');
  });

  test('Scenario 1: Application loads and retrieves live backend health status', async ({ page }) => {
    // Verify project title and subtitle
    await expect(page.locator('.brand-title')).toContainText('Sleep Disorder Detection');
    await expect(page.locator('.brand-subtitle')).toContainText('ML-Based Intelligent System');

    // Verify live backend health pill (Model Online)
    const healthPill = page.locator('.header-status .status-pill');
    await expect(healthPill).toBeVisible();
    await expect(healthPill).toContainText('Model Online (v1.0.0)');
  });

  test('Scenario 2: Model metadata loads and displays correctly on About Model tab', async ({ page }) => {
    // Click on "About the Model" navigation button
    await page.getByRole('button', { name: /^About the Model$/i }).click();

    // Verify model architecture overview
    await expect(
      page.getByRole('heading', { name: /Intelligent Random Forest Classification Pipeline/i })
    ).toBeVisible();
    await expect(page.getByText('12 Raw → 25 Transformed')).toBeVisible();

    // Verify 3 supported classes
    await expect(page.getByRole('heading', { name: 'No recorded sleep disorder' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Insomnia' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Sleep Apnea' })).toBeVisible();

    // Verify holdout evaluation metrics
    await expect(page.getByText('96.0%')).toBeVisible(); // Accuracy
    await expect(page.getByText('0.9957')).toBeVisible(); // ROC-AUC

    // Verify physiological validation bounds table is rendered
    await expect(page.getByRole('heading', { name: /Authoritative Physiological Validation Bounds/i })).toBeVisible();
    await expect(page.getByRole('cell', { name: 'bp_systolic' })).toBeVisible();
  });

  test('Scenario 3: Valid assessment submission renders actual live API prediction', async ({ page }) => {
    // Go to assessment form
    await page.getByRole('button', { name: /^Sleep Assessment$/i }).click();
    await expect(page.getByRole('heading', { name: /Sleep Health Assessment/i })).toBeVisible();

    // Click Healthy preset
    await page.getByRole('button', { name: /Healthy/i }).click();

    // Ensure Explainability checkbox is checked
    const explainCheckbox = page.locator('#include-explain-checkbox');
    await expect(explainCheckbox).toBeChecked();

    // Submit assessment
    await page.locator('#submit-prediction-btn').click();

    // Should automatically navigate to results tab
    await expect(page.locator('.result-class-name')).toBeVisible();
    await expect(page.locator('.result-class-name')).toContainText('No recorded sleep disorder');
    await expect(page.locator('.confidence-chip')).toContainText('Probability');

    // Verify calibrated probability bars
    await expect(page.locator('.prob-row.is-predicted')).toContainText('No recorded sleep disorder');
    await expect(page.locator('.prob-row').filter({ hasText: 'Insomnia' })).toBeVisible();
    await expect(page.locator('.prob-row').filter({ hasText: 'Sleep Apnea' })).toBeVisible();

    // Verify technical metadata
    await expect(page.locator('.result-metadata-footer')).toContainText('Random Forest v1.0.0');
    await expect(page.locator('.result-metadata-footer')).toContainText('Contract:');
  });

  test('Scenario 4: Both compound blood pressure and split systolic/diastolic modes work against real backend', async ({ page }) => {
    // Go to assessment
    await page.getByRole('button', { name: /^Sleep Assessment$/i }).click();

    // 1. Test Compound Mode: Load Apnea preset (140/95)
    await page.getByRole('button', { name: /Sleep Apnea/i }).click();
    await expect(page.locator('#bp-compound-input')).toHaveValue('140/95');
    await page.locator('#submit-prediction-btn').click();

    // Verify prediction rendered
    await expect(page.locator('.result-class-name')).toHaveText('Sleep Apnea');

    // 2. Return to form and switch to Split Mode
    await page.getByRole('button', { name: /^Sleep Assessment$/i }).click();
    await page.getByRole('button', { name: /Split Systolic \/ Diastolic/i }).click();

    await expect(page.locator('#bp-systolic-input')).toBeVisible();
    await expect(page.locator('#bp-diastolic-input')).toBeVisible();

    // Set split values: 130 / 85 (Insomnia pattern)
    await page.locator('#bp-systolic-input').fill('130');
    await page.locator('#bp-diastolic-input').fill('85');
    await page.locator('#age-input').fill('45');
    await page.locator('#sleep-duration-input').fill('6.0');

    await page.locator('#submit-prediction-btn').click();

    // Verify prediction updated
    await expect(page.locator('.result-class-name')).toBeVisible();
    await expect(page.locator('.prediction-pill')).toBeVisible();
  });

  test('Scenario 5: Optional SHAP explanations are requested and rendered with direction and base value', async ({ page }) => {
    await page.getByRole('button', { name: /^Sleep Assessment$/i }).click();

    // Load preset with SHAP checked
    await page.getByRole('button', { name: /Healthy/i }).click();
    await expect(page.locator('#include-explain-checkbox')).toBeChecked();
    await page.locator('#submit-prediction-btn').click();

    // On Results card, click "Inspect SHAP Explanations"
    await expect(page.locator('.shap-callout')).toBeVisible();
    await page.getByRole('button', { name: /Inspect SHAP Explanations/i }).click();

    // Verify SHAP tab rendered
    await expect(page.getByRole('heading', { name: /Local SHAP Feature Attribution/i })).toBeVisible();
    await expect(page.locator('.base-val')).toBeVisible(); // Baseline E[f(x)]
    await expect(page.locator('.badge-subtle')).toContainText('SHAP TreeExplainer');

    // Verify waterfall attribution rows exist
    const factorRows = page.locator('.factor-row');
    await expect(factorRows.first()).toBeVisible();
    await expect(page.locator('.direction-pill').first()).toBeVisible();

    // Verify non-causality warning
    await expect(page.getByText(/Absence of Clinical Causality/i)).toBeVisible();
  });

  test('Scenario 6: Missing or invalid inputs show understandable validation errors', async ({ page }) => {
    await page.getByRole('button', { name: /^Sleep Assessment$/i }).click();

    // Enter out-of-bounds age
    await page.locator('#age-input').fill('180');

    // Enter inverted blood pressure (systolic < diastolic)
    await page.locator('#bp-compound-input').fill('80/120');

    await page.locator('#submit-prediction-btn').click();

    // Check error alerts
    await expect(page.locator('#age-error')).toContainText('Age must be an integer between 1 and 120');
    await expect(page.locator('#bp-error')).toContainText('must be strictly greater than diastolic');

    // Verify submission did not navigate away
    await expect(page.getByRole('heading', { name: /Sleep Health Assessment/i })).toBeVisible();
  });

  test('Scenario 7: Backend HTTP 422 errors are displayed gracefully without crashing the UI', async ({ page }) => {
    // Intercept POST /api/predict to simulate server-side 422
    await page.route('**/api/predict*', async (route) => {
      await route.fulfill({
        status: 422,
        contentType: 'application/json',
        body: JSON.stringify({
          detail: 'Unprocessable Entity: Systolic pressure (70 mmHg) out of valid clinical bounds.',
        }),
      });
    });

    await page.getByRole('button', { name: /^Sleep Assessment$/i }).click();
    await page.locator('#submit-prediction-btn').click();

    // Verify error alert card
    const errorAlert = page.locator('.alert-error');
    await expect(errorAlert).toBeVisible();
    await expect(errorAlert).toContainText('Systolic pressure (70 mmHg) out of valid clinical bounds.');
    // Form remains intact
    await expect(page.locator('#submit-prediction-btn')).toBeEnabled();
  });

  test('Scenario 8: Backend HTTP 503 and network failures show useful error states', async ({ page }) => {
    // Intercept POST /api/predict to simulate 503 Service Unavailable
    await page.route('**/api/predict*', async (route) => {
      await route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({
          detail: 'Model service is currently unavailable. Serialized model artifacts could not be loaded.',
        }),
      });
    });

    await page.getByRole('button', { name: /^Sleep Assessment$/i }).click();
    await page.locator('#submit-prediction-btn').click();

    const errorAlert = page.locator('.alert-error');
    await expect(errorAlert).toBeVisible();
    await expect(errorAlert).toContainText('Model service is currently unavailable');

    // Verify that navigating to Results tab displays empty state rather than a stale result
    await page.getByRole('button', { name: /Prediction Results/i }).click();
    await expect(page.getByText(/Awaiting Sleep Assessment/i)).toBeVisible();
  });

  test('Scenario 9: Repeated submissions update results without stale or mismatched explanations', async ({ page }) => {
    await page.getByRole('button', { name: /^Sleep Assessment$/i }).click();

    // 1. Submit with SHAP enabled
    await page.getByRole('button', { name: /Healthy/i }).click();
    await page.locator('#include-explain-checkbox').check();
    await page.locator('#submit-prediction-btn').click();

    await expect(page.locator('.result-class-name')).toHaveText('No recorded sleep disorder');
    await expect(page.locator('.shap-callout')).toBeVisible();

    // 2. Submit second time with SHAP disabled
    await page.getByRole('button', { name: /^Sleep Assessment$/i }).click();
    await page.getByRole('button', { name: /Insomnia/i }).click();
    await page.locator('#include-explain-checkbox').uncheck();
    await page.locator('#submit-prediction-btn').click();

    // Result should update to Insomnia
    await expect(page.locator('.result-class-name')).toHaveText('Insomnia');
    // SHAP callout should show disabled notice, not stale explanation
    await expect(page.locator('.shap-callout-disabled')).toBeVisible();

    // Navigating to SHAP tab directly should show omitted notice
    await page.getByRole('button', { name: /SHAP Explainability/i }).click();
    await expect(page.getByText(/SHAP Explanation Omitted for this Ingestion/i)).toBeVisible();
  });

  test('Scenario 10: Non-diagnostic disclaimer remains visible across tabs', async ({ page }) => {
    // Overview tab
    await expect(page.locator('.disclaimer-banner')).toBeVisible();
    await expect(page.locator('.disclaimer-banner')).toContainText('Non-Diagnostic Tool');

    // Ethics & Safety tab
    await page.getByRole('button', { name: /^Ethics & Safety$/i }).click();
    await expect(page.locator('.disclaimer-banner')).toBeVisible();
    await expect(page.getByText(/Non-Diagnostic Guardrails/i)).toBeVisible();

    // Footer
    await expect(page.locator('.site-footer')).toContainText('Non-Diagnostic Clinical Disclaimer');
  });

  test('Scenario 11: Responsive layouts adapt cleanly across desktop, tablet, and mobile viewports', async ({ page }) => {
    // Desktop Viewport (1280 x 800)
    await page.setViewportSize({ width: 1280, height: 800 });
    await expect(page.locator('.brand-title')).toBeVisible();
    await expect(page.locator('.header-nav')).toBeVisible();

    // Tablet Viewport (768 x 1024)
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(page.locator('.brand-title')).toBeVisible();
    await expect(page.locator('.header-nav')).toBeVisible();

    // Mobile Viewport (375 x 667)
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.locator('.brand-title')).toBeVisible();
    // Nav links should still be scrollable/clickable on mobile
    await page.getByRole('button', { name: /^Sleep Assessment$/i }).click();
    await expect(page.getByRole('heading', { name: /Sleep Health Assessment/i })).toBeVisible();
  });

  test('Scenario 12: Keyboard navigation and accessible form labels are usable', async ({ page }) => {
    await page.getByRole('button', { name: /^Sleep Assessment$/i }).click();

    // Focus on age input
    const ageInput = page.locator('#age-input');
    await ageInput.focus();
    await expect(ageInput).toBeFocused();

    // Check aria-describedby connects to hints
    const describedBy = await ageInput.getAttribute('aria-describedby');
    expect(describedBy).toContain('age-hint');

    // Press Tab to navigate to next control
    await page.keyboard.press('Tab');
    const occupationSelect = page.locator('#occupation-select');
    await expect(occupationSelect).toBeFocused();
  });
});
