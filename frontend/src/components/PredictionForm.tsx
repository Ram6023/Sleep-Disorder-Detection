import React, { useState } from 'react';
import {
  RotateCcw,
  AlertCircle,
  Activity,
  Check,
} from 'lucide-react';
import type {
  SleepPredictionInput,
  FormValidationErrors,
} from '../types/index.ts';
import { SAMPLE_PROFILES, COMMON_OCCUPATIONS } from '../data/sampleProfiles.ts';

interface PredictionFormProps {
  onSubmit: (input: SleepPredictionInput, explain: boolean) => Promise<void>;
  isLoading: boolean;
  serverError: string | null;
}

export const PredictionForm: React.FC<PredictionFormProps> = ({
  onSubmit,
  isLoading,
  serverError,
}) => {
  // Blood pressure input mode: 'compound' ("120/80") or 'split' (systolic & diastolic numbers)
  const [bpMode, setBpMode] = useState<'compound' | 'split'>('compound');
  const [includeExplain, setIncludeExplain] = useState<boolean>(true);

  // Form Fields State
  const [gender, setGender] = useState<string>('Male');
  const [age, setAge] = useState<string>('42');
  const [occupation, setOccupation] = useState<string>('Software Engineer');
  const [customOccupation, setCustomOccupation] = useState<string>('');
  const [isCustomOccupation, setIsCustomOccupation] = useState<boolean>(false);
  const [sleepDuration, setSleepDuration] = useState<string>('7.2');
  const [qualityOfSleep, setQualityOfSleep] = useState<number>(8);
  const [physicalActivityLevel, setPhysicalActivityLevel] = useState<string>('60');
  const [stressLevel, setStressLevel] = useState<number>(4);
  const [bmiCategory, setBmiCategory] = useState<string>('Normal');
  const [bloodPressure, setBloodPressure] = useState<string>('120/80');
  const [bpSystolic, setBpSystolic] = useState<string>('120');
  const [bpDiastolic, setBpDiastolic] = useState<string>('80');
  const [heartRate, setHeartRate] = useState<string>('70');
  const [dailySteps, setDailySteps] = useState<string>('7500');
  const [personId, setPersonId] = useState<string>('101');

  // Client-side validation errors
  const [errors, setErrors] = useState<FormValidationErrors>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});

  const markTouched = (field: string) => {
    setTouched((prev) => ({ ...prev, [field]: true }));
  };

  const loadPreset = (profileIndex: number) => {
    const profile = SAMPLE_PROFILES[profileIndex];
    if (!profile) return;
    const p = profile.payload;

    setGender(p.gender);
    setAge(String(p.age));

    if (COMMON_OCCUPATIONS.includes(p.occupation)) {
      setOccupation(p.occupation);
      setIsCustomOccupation(false);
      setCustomOccupation('');
    } else {
      setOccupation('Custom');
      setIsCustomOccupation(true);
      setCustomOccupation(p.occupation);
    }

    setSleepDuration(String(p.sleep_duration));
    setQualityOfSleep(p.quality_of_sleep);
    setPhysicalActivityLevel(String(p.physical_activity_level));
    setStressLevel(p.stress_level);
    setBmiCategory(p.bmi_category);

    if (p.blood_pressure) {
      setBloodPressure(p.blood_pressure);
      const parts = p.blood_pressure.split('/');
      if (parts.length === 2) {
        setBpSystolic(parts[0].trim());
        setBpDiastolic(parts[1].trim());
      }
    } else if (p.bp_systolic && p.bp_diastolic) {
      setBpSystolic(String(p.bp_systolic));
      setBpDiastolic(String(p.bp_diastolic));
      setBloodPressure(`${p.bp_systolic}/${p.bp_diastolic}`);
    }

    setHeartRate(String(p.heart_rate));
    setDailySteps(String(p.daily_steps));
    if (p.person_id !== undefined) {
      setPersonId(String(p.person_id));
    }

    setErrors({});
    setTouched({});
  };

  const resetForm = () => {
    setGender('Male');
    setAge('42');
    setOccupation('Software Engineer');
    setIsCustomOccupation(false);
    setCustomOccupation('');
    setSleepDuration('7.2');
    setQualityOfSleep(8);
    setPhysicalActivityLevel('60');
    setStressLevel(4);
    setBmiCategory('Normal');
    setBloodPressure('120/80');
    setBpSystolic('120');
    setBpDiastolic('80');
    setHeartRate('70');
    setDailySteps('7500');
    setPersonId('101');
    setErrors({});
    setTouched({});
  };

  const validate = (): { isValid: boolean; payload?: SleepPredictionInput } => {
    const newErrors: FormValidationErrors = {};

    // Gender
    if (!gender || !['Male', 'Female'].includes(gender)) {
      newErrors.gender = 'Please select a biological sex (Male or Female).';
    }

    // Age
    const ageNum = parseInt(age, 10);
    if (isNaN(ageNum) || ageNum < 1 || ageNum > 120) {
      newErrors.age = 'Age must be an integer between 1 and 120 years.';
    }

    // Occupation
    const occValue = isCustomOccupation ? customOccupation.trim() : occupation.trim();
    if (!occValue || occValue.length < 1 || occValue.length > 100) {
      newErrors.occupation = 'Occupation is required (1 - 100 characters).';
    }

    // Sleep Duration
    const sleepNum = parseFloat(sleepDuration);
    if (isNaN(sleepNum) || sleepNum <= 0.0 || sleepNum > 24.0) {
      newErrors.sleep_duration = 'Sleep duration must be between 0.1 and 24.0 hours/day.';
    }

    // Quality of Sleep
    if (qualityOfSleep < 1 || qualityOfSleep > 10) {
      newErrors.quality_of_sleep = 'Quality rating must be between 1 and 10.';
    }

    // Physical Activity Level
    const actNum = parseInt(physicalActivityLevel, 10);
    if (isNaN(actNum) || actNum < 0 || actNum > 1440) {
      newErrors.physical_activity_level = 'Activity level must be between 0 and 1440 minutes/day.';
    }

    // Stress Level
    if (stressLevel < 1 || stressLevel > 10) {
      newErrors.stress_level = 'Stress rating must be between 1 and 10.';
    }

    // BMI Category
    if (!['Normal', 'Overweight', 'Obese'].includes(bmiCategory)) {
      newErrors.bmi_category = 'Please choose a valid BMI category.';
    }

    // Blood Pressure
    let finalSystolic: number | undefined;
    let finalDiastolic: number | undefined;
    let finalBpStr: string | undefined;

    if (bpMode === 'compound') {
      const trimmedBp = bloodPressure.trim();
      const bpMatch = trimmedBp.match(/^(\d{2,3})\/(\d{2,3})$/);
      if (!bpMatch) {
        newErrors.blood_pressure = "Format must be 'Systolic/Diastolic' (e.g. '120/80').";
      } else {
        const sys = parseFloat(bpMatch[1]);
        const dia = parseFloat(bpMatch[2]);
        if (sys < 50 || sys > 260) {
          newErrors.blood_pressure = `Systolic pressure (${sys} mmHg) out of valid range (50-260).`;
        } else if (dia < 30 || dia > 160) {
          newErrors.blood_pressure = `Diastolic pressure (${dia} mmHg) out of valid range (30-160).`;
        } else if (sys <= dia) {
          newErrors.blood_pressure = `Systolic pressure (${sys}) must be strictly greater than diastolic (${dia}).`;
        } else {
          finalBpStr = trimmedBp;
          finalSystolic = sys;
          finalDiastolic = dia;
        }
      }
    } else {
      const sys = parseFloat(bpSystolic);
      const dia = parseFloat(bpDiastolic);
      let bpFailed = false;

      if (isNaN(sys) || sys < 50 || sys > 260) {
        newErrors.bp_systolic = 'Systolic pressure must be between 50 and 260 mmHg.';
        bpFailed = true;
      }
      if (isNaN(dia) || dia < 30 || dia > 160) {
        newErrors.bp_diastolic = 'Diastolic pressure must be between 30 and 160 mmHg.';
        bpFailed = true;
      }
      if (!bpFailed && sys <= dia) {
        newErrors.bp_systolic = `Systolic pressure (${sys}) must be strictly greater than diastolic (${dia}).`;
        bpFailed = true;
      }

      if (!bpFailed) {
        finalSystolic = sys;
        finalDiastolic = dia;
        finalBpStr = `${sys}/${dia}`;
      }
    }

    // Heart Rate
    const hrNum = parseInt(heartRate, 10);
    if (isNaN(hrNum) || hrNum < 30 || hrNum > 220) {
      newErrors.heart_rate = 'Heart rate must be between 30 and 220 BPM.';
    }

    // Daily Steps
    const stepsNum = parseInt(dailySteps, 10);
    if (isNaN(stepsNum) || stepsNum < 0 || stepsNum > 100000) {
      newErrors.daily_steps = 'Daily steps must be between 0 and 100,000.';
    }

    setErrors(newErrors);

    if (Object.keys(newErrors).length > 0) {
      return { isValid: false };
    }

    const payload: SleepPredictionInput = {
      gender,
      age: ageNum,
      occupation: occValue,
      sleep_duration: sleepNum,
      quality_of_sleep: qualityOfSleep,
      physical_activity_level: actNum,
      stress_level: stressLevel,
      bmi_category: bmiCategory,
      blood_pressure: bpMode === 'compound' ? finalBpStr : undefined,
      bp_systolic: bpMode === 'split' ? finalSystolic : undefined,
      bp_diastolic: bpMode === 'split' ? finalDiastolic : undefined,
      heart_rate: hrNum,
      daily_steps: stepsNum,
      person_id: personId ? parseInt(personId, 10) || undefined : undefined,
    };

    return { isValid: true, payload };
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setTouched({
      gender: true,
      age: true,
      occupation: true,
      sleep_duration: true,
      quality_of_sleep: true,
      physical_activity_level: true,
      stress_level: true,
      bmi_category: true,
      blood_pressure: true,
      bp_systolic: true,
      bp_diastolic: true,
      heart_rate: true,
      daily_steps: true,
    });

    const { isValid, payload } = validate();
    if (!isValid || !payload) return;

    await onSubmit(payload, includeExplain);
  };

  return (
    <section className="assessment-card glass-panel" aria-labelledby="form-heading">
      <div className="card-header-modular">
        <div className="header-titles">
          <h2 id="form-heading" className="card-title">Sleep Health Assessment</h2>
          <p className="card-desc">
            Provide patient lifestyle and physiological parameters to evaluate sleep disorder risk patterns.
          </p>
        </div>

        {/* Quick Sample Presets */}
        <div className="preset-bar" aria-label="Sample test presets">
          <span className="preset-label">Test Presets:</span>
          <div className="preset-buttons">
            <button
              type="button"
              className="btn-preset"
              onClick={() => loadPreset(0)}
              title="Load Healthy Baseline profile"
            >
              <span>Healthy Baseline</span>
            </button>
            <button
              type="button"
              className="btn-preset"
              onClick={() => loadPreset(1)}
              title="Load Insomnia Risk profile"
            >
              <span>Insomnia Case</span>
            </button>
            <button
              type="button"
              className="btn-preset"
              onClick={() => loadPreset(2)}
              title="Load Sleep Apnea Risk profile"
            >
              <span>Sleep Apnea Case</span>
            </button>
            <button
              type="button"
              className="btn-preset btn-reset"
              onClick={resetForm}
              title="Reset all form values to defaults"
            >
              <RotateCcw size={13} />
              <span>Reset Defaults</span>
            </button>
          </div>
        </div>
      </div>

      {serverError && (
        <div className="alert alert-error" role="alert">
          <AlertCircle size={18} className="alert-icon" />
          <div className="alert-body">
            <strong>Inference Request Failed</strong>
            <p>{serverError}</p>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} noValidate className="assessment-form">
        {/* Section 1: Personal & Lifestyle Information */}
        <fieldset className="form-section">
          <legend className="section-legend">Personal &amp; Lifestyle Information</legend>
          <div className="form-grid-2">
            {/* Biological Sex */}
            <div className="field-group">
              <label htmlFor="gender" className="field-label">
                Biological Sex <span className="req">*</span>
              </label>
              <span id="gender-hint" className="field-hint">Biological sex for metabolic baseline</span>
              <div className="segment-control" role="radiogroup" aria-labelledby="gender-hint">
                <button
                  type="button"
                  className={`segment-btn ${gender === 'Male' ? 'active' : ''}`}
                  onClick={() => {
                    setGender('Male');
                    markTouched('gender');
                  }}
                >
                  {gender === 'Male' && <Check size={14} />}
                  <span>Male</span>
                </button>
                <button
                  type="button"
                  className={`segment-btn ${gender === 'Female' ? 'active' : ''}`}
                  onClick={() => {
                    setGender('Female');
                    markTouched('gender');
                  }}
                >
                  {gender === 'Female' && <Check size={14} />}
                  <span>Female</span>
                </button>
              </div>
              {touched.gender && errors.gender && (
                <span className="field-error">{errors.gender}</span>
              )}
            </div>

            {/* Age */}
            <div className="field-group">
              <label htmlFor="age-input" className="field-label">
                Age (years) <span className="req">*</span>
              </label>
              <span id="age-hint" className="field-hint">Patient age between 1 and 120</span>
              <input
                id="age-input"
                type="number"
                min="1"
                max="120"
                className={`form-input ${touched.age && errors.age ? 'input-error' : ''}`}
                value={age}
                onChange={(e) => {
                  setAge(e.target.value);
                  markTouched('age');
                }}
                onBlur={() => markTouched('age')}
                aria-describedby="age-hint"
                required
              />
              {touched.age && errors.age && (
                <span id="age-error" className="field-error">{errors.age}</span>
              )}
            </div>

            {/* Occupation */}
            <div className="field-group">
              <label htmlFor="occupation-select" className="field-label">
                Occupation <span className="req">*</span>
              </label>
              <span id="occ-hint" className="field-hint">Professional category or custom title</span>
              <select
                id="occupation-select"
                className="form-select"
                value={isCustomOccupation ? 'Custom' : occupation}
                onChange={(e) => {
                  markTouched('occupation');
                  if (e.target.value === 'Custom') {
                    setIsCustomOccupation(true);
                  } else {
                    setIsCustomOccupation(false);
                    setOccupation(e.target.value);
                  }
                }}
                aria-describedby="occ-hint"
              >
                {COMMON_OCCUPATIONS.map((occ) => (
                  <option key={occ} value={occ}>
                    {occ}
                  </option>
                ))}
                <option value="Custom">Custom / Other Profession...</option>
              </select>

              {isCustomOccupation && (
                <input
                  id="custom-occupation-input"
                  type="text"
                  placeholder="Enter specific occupation..."
                  className={`form-input mt-2 ${touched.occupation && errors.occupation ? 'input-error' : ''}`}
                  value={customOccupation}
                  onChange={(e) => setCustomOccupation(e.target.value)}
                  onBlur={() => markTouched('occupation')}
                  maxLength={100}
                />
              )}
              {touched.occupation && errors.occupation && (
                <span className="field-error">{errors.occupation}</span>
              )}
            </div>

            {/* Physical Activity Level */}
            <div className="field-group">
              <label htmlFor="activity-input" className="field-label">
                Physical Activity Level <span className="unit">(mins/day)</span> <span className="req">*</span>
              </label>
              <span id="act-hint" className="field-hint">Moderate to vigorous exercise duration</span>
              <input
                id="activity-input"
                type="number"
                min="0"
                max="1440"
                className={`form-input ${touched.physical_activity_level && errors.physical_activity_level ? 'input-error' : ''}`}
                value={physicalActivityLevel}
                onChange={(e) => {
                  setPhysicalActivityLevel(e.target.value);
                  markTouched('physical_activity_level');
                }}
                onBlur={() => markTouched('physical_activity_level')}
                aria-describedby="act-hint"
                required
              />
              {touched.physical_activity_level && errors.physical_activity_level && (
                <span className="field-error">{errors.physical_activity_level}</span>
              )}
            </div>

            {/* Daily Steps */}
            <div className="field-group">
              <label htmlFor="daily-steps-input" className="field-label">
                Daily Steps <span className="req">*</span>
              </label>
              <span id="steps-hint" className="field-hint">Average step volume (0 - 100,000)</span>
              <input
                id="daily-steps-input"
                type="number"
                min="0"
                max="100000"
                step="100"
                className={`form-input ${touched.daily_steps && errors.daily_steps ? 'input-error' : ''}`}
                value={dailySteps}
                onChange={(e) => {
                  setDailySteps(e.target.value);
                  markTouched('daily_steps');
                }}
                onBlur={() => markTouched('daily_steps')}
                aria-describedby="steps-hint"
                required
              />
              {touched.daily_steps && errors.daily_steps && (
                <span className="field-error">{errors.daily_steps}</span>
              )}
            </div>
          </div>
        </fieldset>

        {/* Section 2: Sleep Information */}
        <fieldset className="form-section">
          <legend className="section-legend">Sleep Information</legend>
          <div className="form-grid-2">
            {/* Sleep Duration */}
            <div className="field-group">
              <label htmlFor="sleep-duration-input" className="field-label">
                Sleep Duration (hours) <span className="req">*</span>
              </label>
              <span id="sleep-hint" className="field-hint">Average daily sleep (0.1 - 24.0 hours)</span>
              <input
                id="sleep-duration-input"
                type="number"
                step="0.1"
                min="0.1"
                max="24.0"
                className={`form-input ${touched.sleep_duration && errors.sleep_duration ? 'input-error' : ''}`}
                value={sleepDuration}
                onChange={(e) => {
                  setSleepDuration(e.target.value);
                  markTouched('sleep_duration');
                }}
                onBlur={() => markTouched('sleep_duration')}
                aria-describedby="sleep-hint"
                required
              />
              {touched.sleep_duration && errors.sleep_duration && (
                <span id="sleep-error" className="field-error">{errors.sleep_duration}</span>
              )}
            </div>

            {/* Quality of Sleep (1-10) */}
            <div className="field-group">
              <div className="label-with-value">
                <label className="field-label">
                  Quality of Sleep <span className="req">*</span>
                </label>
                <span className="current-scale-val font-mono">{qualityOfSleep} / 10</span>
              </div>
              <span id="quality-hint" className="field-hint">Subjective restorative depth (1 = Poor, 10 = Restorative)</span>
              <div className="rating-selector" role="radiogroup" aria-labelledby="quality-hint">
                {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((score) => (
                  <button
                    key={`qual-${score}`}
                    type="button"
                    className={`rating-btn ${qualityOfSleep === score ? 'active' : ''}`}
                    onClick={() => {
                      setQualityOfSleep(score);
                      markTouched('quality_of_sleep');
                    }}
                  >
                    {score}
                  </button>
                ))}
              </div>
            </div>

            {/* Stress Level (1-10) */}
            <div className="field-group">
              <div className="label-with-value">
                <label className="field-label">
                  Stress Level <span className="req">*</span>
                </label>
                <span className="current-scale-val font-mono">{stressLevel} / 10</span>
              </div>
              <span id="stress-hint" className="field-hint">Perceived psychological strain (1 = Low, 10 = Severe)</span>
              <div className="rating-selector" role="radiogroup" aria-labelledby="stress-hint">
                {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((score) => (
                  <button
                    key={`stress-${score}`}
                    type="button"
                    className={`rating-btn ${stressLevel === score ? 'active' : ''}`}
                    onClick={() => {
                      setStressLevel(score);
                      markTouched('stress_level');
                    }}
                  >
                    {score}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </fieldset>

        {/* Section 3: Health Indicators & Cardiovascular */}
        <fieldset className="form-section">
          <legend className="section-legend">Health Indicators &amp; Cardiovascular</legend>
          <div className="form-grid-2">
            {/* BMI Category */}
            <div className="field-group">
              <label className="field-label">
                BMI Category <span className="req">*</span>
              </label>
              <span id="bmi-hint" className="field-hint">Body Mass Index stratification</span>
              <div className="segment-control" role="radiogroup" aria-labelledby="bmi-hint">
                {(['Normal', 'Overweight', 'Obese'] as const).map((cat) => (
                  <button
                    key={cat}
                    type="button"
                    className={`segment-btn ${bmiCategory === cat ? 'active' : ''}`}
                    onClick={() => {
                      setBmiCategory(cat);
                      markTouched('bmi_category');
                    }}
                  >
                    {bmiCategory === cat && <Check size={14} />}
                    <span>{cat}</span>
                  </button>
                ))}
              </div>
              {touched.bmi_category && errors.bmi_category && (
                <span className="field-error">{errors.bmi_category}</span>
              )}
            </div>

            {/* Heart Rate */}
            <div className="field-group">
              <label htmlFor="heart-rate-input" className="field-label">
                Heart Rate (BPM) <span className="req">*</span>
              </label>
              <span id="hr-hint" className="field-hint">Resting heart rate in beats per minute (30 - 220)</span>
              <input
                id="heart-rate-input"
                type="number"
                min="30"
                max="220"
                className={`form-input ${touched.heart_rate && errors.heart_rate ? 'input-error' : ''}`}
                value={heartRate}
                onChange={(e) => {
                  setHeartRate(e.target.value);
                  markTouched('heart_rate');
                }}
                onBlur={() => markTouched('heart_rate')}
                aria-describedby="hr-hint"
                required
              />
              {touched.heart_rate && errors.heart_rate && (
                <span className="field-error">{errors.heart_rate}</span>
              )}
            </div>

            {/* Blood Pressure Input (Full Width Span) */}
            <div className="field-group full-width">
              <div className="bp-header-row">
                <div>
                  <label htmlFor="bp-compound-input" className="field-label">
                    Blood Pressure (mmHg) <span className="req">*</span>
                  </label>
                  <span id="bp-hint" className="field-hint">Resting arterial pressure (Systolic &gt; Diastolic)</span>
                </div>
                <div className="mode-toggle-group" role="group" aria-label="Blood pressure input mode">
                  <button
                    type="button"
                    className={`mode-btn ${bpMode === 'compound' ? 'active' : ''}`}
                    onClick={() => setBpMode('compound')}
                    aria-pressed={bpMode === 'compound'}
                  >
                    Compound (120/80)
                  </button>
                  <button
                    type="button"
                    className={`mode-btn ${bpMode === 'split' ? 'active' : ''}`}
                    onClick={() => setBpMode('split')}
                    aria-pressed={bpMode === 'split'}
                  >
                    Split Systolic / Diastolic
                  </button>
                </div>
              </div>

              {bpMode === 'compound' ? (
                <div>
                  <input
                    id="bp-compound-input"
                    type="text"
                    placeholder="e.g. 120/80"
                    pattern="^\d{2,3}/\d{2,3}$"
                    className={`form-input font-mono ${touched.blood_pressure && errors.blood_pressure ? 'input-error' : ''}`}
                    value={bloodPressure}
                    onChange={(e) => {
                      setBloodPressure(e.target.value);
                      markTouched('blood_pressure');
                    }}
                    onBlur={() => markTouched('blood_pressure')}
                    aria-describedby="bp-hint"
                    required
                  />
                  {touched.blood_pressure && errors.blood_pressure && (
                    <span id="bp-error" className="field-error">{errors.blood_pressure}</span>
                  )}
                </div>
              ) : (
                <div className="split-bp-grid">
                  <div className="split-field">
                    <label htmlFor="bp-systolic-input" className="field-label-sub">
                      Systolic (50 – 260)
                    </label>
                    <input
                      id="bp-systolic-input"
                      type="number"
                      min="50"
                      max="260"
                      className={`form-input font-mono ${touched.bp_systolic && errors.bp_systolic ? 'input-error' : ''}`}
                      value={bpSystolic}
                      onChange={(e) => {
                        setBpSystolic(e.target.value);
                        markTouched('bp_systolic');
                      }}
                      onBlur={() => markTouched('bp_systolic')}
                      required
                    />
                    {touched.bp_systolic && errors.bp_systolic && (
                      <span className="field-error">{errors.bp_systolic}</span>
                    )}
                  </div>

                  <div className="split-field">
                    <label htmlFor="bp-diastolic-input" className="field-label-sub">
                      Diastolic (30 – 160)
                    </label>
                    <input
                      id="bp-diastolic-input"
                      type="number"
                      min="30"
                      max="160"
                      className={`form-input font-mono ${touched.bp_diastolic && errors.bp_diastolic ? 'input-error' : ''}`}
                      value={bpDiastolic}
                      onChange={(e) => {
                        setBpDiastolic(e.target.value);
                        markTouched('bp_diastolic');
                      }}
                      onBlur={() => markTouched('bp_diastolic')}
                      required
                    />
                    {touched.bp_diastolic && errors.bp_diastolic && (
                      <span className="field-error">{errors.bp_diastolic}</span>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </fieldset>

        {/* Submit Bar */}
        <div className="form-submit-bar">
          <label className="checkbox-row" htmlFor="include-explain-checkbox">
            <input
              id="include-explain-checkbox"
              type="checkbox"
              checked={includeExplain}
              onChange={(e) => setIncludeExplain(e.target.checked)}
            />
            <div className="checkbox-text">
              <span className="checkbox-title">Include Local SHAP Explainability</span>
              <span className="checkbox-desc">Compute exact game-theoretic Shapley feature attributions for this instance</span>
            </div>
          </label>

          <button
            id="submit-prediction-btn"
            type="submit"
            className="btn btn-primary btn-submit"
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <span className="spinner" aria-hidden="true" />
                <span>Calculating Prediction...</span>
              </>
            ) : (
              <>
                <Activity size={16} />
                <span>Analyze Sleep Disorder Risk</span>
              </>
            )}
          </button>
        </div>
      </form>
    </section>
  );
};
