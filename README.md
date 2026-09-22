# Aircraft Predictive Maintenance Self project 

A machine learning pipeline that predicts equipment failure from sensor telemetry before the failure happens and explains which sensors actually drove each prediction, not just the prediction itself.

## Why explainability matters here

Most failure-prediction demos stop at a yes/no output. For anything safety-critical — aircraft systems included — a prediction on its own isn't enough; you need to know *why* a model is raising a flag before acting on it. This project treats that as the actual point, not an afterthought: alongside the prediction, it extracts and ranks which sensor readings the model relied on most.

## How it works

1. **`generate_data.py`** — simulates 1,000 machines' sensor readings (air/process temperature, rotational speed, torque, tool wear). Failure isn't a fixed rule — it's drawn from a genuine risk-based probability, so outcomes carry real uncertainty rather than being trivially reconstructable from the inputs.
2. **`main.py`** — trains a Random Forest classifier on 80% of the data, evaluates it on the unseen 20%, and reports accuracy, precision, and recall rather than accuracy alone. It then extracts and plots feature importances.

## Results

- **78% accuracy** on held-out data, against a ~75% baseline for always predicting "no failure" — a real, if modest, lift.
- **Precision and recall reported separately** for the failure class, since accuracy alone is misleading when failures are the minority outcome.
- **Feature importance** confirms the model leans most on Torque and Tool Wear — the two sensors that actually drive risk in the underlying simulation.

## Built with

Python · pandas · scikit-learn (`RandomForestClassifier`) · matplotlib · NumPy

## Running it

```bash
pip install pandas scikit-learn matplotlib numpy
python generate_data.py
python main.py
```

This produces `telemetry_data.csv` (the simulated sensor data) and `feature_importance.png` (the explainability chart).

## Limitations

The sensor data is simulated, not real aircraft telemetry — a real-world version would need actual operational data. Recall on the failure class is currently modest (~30%), reflecting a deliberately cautious model rather than a trigger-happy one; a production system would tune the classification threshold explicitly based on the real cost of a missed failure versus a false alarm.
