import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'Air temperature [K]': np.random.normal(300, 10, n),
    'Process temperature [K]': np.random.normal(310, 5, n),
    'Rotational speed [rpm]': np.random.normal(1500, 100, n),
    'Torque [Nm]': np.random.normal(40, 10, n),
    'Tool wear [min]': np.random.uniform(0, 250, n)
})

# Build a RISK SCORE from three sensors, each contributing on its own scale,
# instead of a hard yes/no cutoff on just two of them.
risk_score = (
    (df['Torque [Nm]'] - 48) / 10
    + (df['Tool wear [min]'] - 175) / 125
    + (df['Rotational speed [rpm]'] - 1620) / 500
)

# Turn that score into a genuine PROBABILITY of failure (0 to 1) using a sigmoid curve, then draw an actual random outcome from that probability -
# so two machines with identical readings can still get different outcomes, same as real machines do.
failure_probability = 1 / (1 + np.exp(-risk_score))
df['Machine failure'] = np.random.binomial(1, failure_probability)

df.to_csv('telemetry_data.csv', index=False)

print("Failure rate in dataset:", df['Machine failure'].mean().round(3))
print(df.head())
