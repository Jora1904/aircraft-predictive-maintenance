import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('telemetry_data.csv')

X = df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
y = df['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate on the held-out 20% the model never saw during training.
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on held-out test data: {accuracy:.2%}")
print(f"(For comparison, always predicting 'no failure' would score about {(1 - y_test.mean()):.0%} here — accuracy alone can be misleading when failures are the minority class.)")
print("\nPrecision/recall per class:")
print(classification_report(y_test, y_pred, target_names=["No failure", "Failure"]))

importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importances:")
print(importances)

importances.plot(kind='bar', color='#2ca02c')
plt.title('Random Forest: Sensor Feature Importance')
plt.ylabel('Relative Weight')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png')
