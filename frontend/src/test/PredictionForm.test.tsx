import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { PredictionForm } from '../components/PredictionForm.tsx';

describe('PredictionForm Component Tests', () => {
  it('renders all required lifestyle and physiological fields', () => {
    render(
      <PredictionForm
        onSubmit={vi.fn()}
        isLoading={false}
        serverError={null}
      />
    );

    expect(screen.getAllByText(/Biological Sex/i).length).toBeGreaterThan(0);
    expect(screen.getByLabelText(/Age \(years\)/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Occupation/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Sleep Duration \(hours\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Quality of Sleep/i)).toBeInTheDocument();
    expect(screen.getByText(/Stress Level/i)).toBeInTheDocument();
    expect(screen.getByText(/BMI Category/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Blood Pressure \(mmHg\)/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Heart Rate \(BPM\)/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Daily Steps/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Analyze Sleep Disorder Risk/i })).toBeInTheDocument();
  });

  it('validates physiological bounds for age (1-120)', async () => {
    const handleSubmit = vi.fn();
    render(
      <PredictionForm
        onSubmit={handleSubmit}
        isLoading={false}
        serverError={null}
      />
    );

    const ageInput = screen.getByLabelText(/Age \(years\)/i);
    fireEvent.change(ageInput, { target: { value: '150' } });

    const submitBtn = screen.getByRole('button', { name: /Analyze Sleep Disorder Risk/i });
    fireEvent.click(submitBtn);

    expect(await screen.findByText(/Age must be an integer between 1 and 120/i)).toBeInTheDocument();
    expect(handleSubmit).not.toHaveBeenCalled();
  });

  it('validates compound blood pressure format and systolic > diastolic constraint', async () => {
    const handleSubmit = vi.fn();
    render(
      <PredictionForm
        onSubmit={handleSubmit}
        isLoading={false}
        serverError={null}
      />
    );

    // Inverted BP
    const bpInput = screen.getByLabelText(/Blood Pressure \(mmHg\)/i);
    fireEvent.change(bpInput, { target: { value: '80/120' } });

    const submitBtn = screen.getByRole('button', { name: /Analyze Sleep Disorder Risk/i });
    fireEvent.click(submitBtn);

    expect(
      await screen.findByText(/Systolic pressure \(80\) must be strictly greater than diastolic \(120\)/i)
    ).toBeInTheDocument();
    expect(handleSubmit).not.toHaveBeenCalled();
  });

  it('allows toggling between compound BP and split BP modes', async () => {
    render(
      <PredictionForm
        onSubmit={vi.fn()}
        isLoading={false}
        serverError={null}
      />
    );

    const splitBtn = screen.getByRole('button', { name: /Split Systolic \/ Diastolic/i });
    fireEvent.click(splitBtn);

    expect(screen.getByLabelText(/Systolic \(50 – 260\)/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Diastolic \(30 – 160\)/i)).toBeInTheDocument();
  });

  it('loads test presets when clicking sample preset buttons', async () => {
    render(
      <PredictionForm
        onSubmit={vi.fn()}
        isLoading={false}
        serverError={null}
      />
    );

    const apneaPresetBtn = screen.getByRole('button', { name: /Sleep Apnea/i });
    fireEvent.click(apneaPresetBtn);

    const ageInput = screen.getByLabelText(/Age \(years\)/i) as HTMLInputElement;
    expect(ageInput.value).toBe('52');

    const bpInput = screen.getByLabelText(/Blood Pressure \(mmHg\)/i) as HTMLInputElement;
    expect(bpInput.value).toBe('140/95');
  });

  it('calls onSubmit with valid payload and explanation flag', async () => {
    const handleSubmit = vi.fn().mockResolvedValue(undefined);
    render(
      <PredictionForm
        onSubmit={handleSubmit}
        isLoading={false}
        serverError={null}
      />
    );

    const submitBtn = screen.getByRole('button', { name: /Analyze Sleep Disorder Risk/i });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(handleSubmit).toHaveBeenCalledTimes(1);
    });

    const [payload, explain] = handleSubmit.mock.calls[0];
    expect(payload.age).toBe(42);
    expect(payload.gender).toBe('Male');
    expect(payload.blood_pressure).toBe('120/80');
    expect(explain).toBe(true);
  });

  it('displays loading state and disables submit button when isLoading is true', () => {
    render(
      <PredictionForm
        onSubmit={vi.fn()}
        isLoading={true}
        serverError={null}
      />
    );

    const btn = screen.getByRole('button', { name: /Calculating Prediction/i });
    expect(btn).toBeDisabled();
  });

  it('renders server error alert when serverError prop is provided', () => {
    render(
      <PredictionForm
        onSubmit={vi.fn()}
        isLoading={false}
        serverError="Model service is currently unavailable. (HTTP 503)"
      />
    );

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/Model service is currently unavailable/i)).toBeInTheDocument();
  });
});
