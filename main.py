import numpy as np
from sklearn.ensemble import IsolationForest

X = np.array([[10], [11], [12], [10], [11], [100], [9], [12]])

model = IsolationForest(contamination=0.1, random_state=42)
labels = model.fit_predict(X)

for value, label in zip(X.ravel(), labels):
    print(value, "ANOMALY" if label == -1 else "NORMAL")
