import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

cancer_data = load_breast_cancer()

# we learn that malignant is cancerous, and benign is not concerous
# 0 is maglignat, 1 is benign

df = pd.DataFrame(cancer_data['data'], columns = cancer_data['feature_names'])
df['target'] = cancer_data['target']

# numpyarrays
X = df[cancer_data.feature_names].values
y = df['target'].values

# creating model
model = LogisticRegression(solver='liblinear')
model.fit(X, y)

# print(model.predict_proba(X))

acc = model.score(X, y)
print("We get a score of " + str(acc) + "%")

# plt.scatter()
# plt.show()

