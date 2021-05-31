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
XLR = df[cancer_data.feature_names].values
YLR = df['target'].values

# creating model
model = LogisticRegression(solver='liblinear')
model.fit(XLR, YLR)

acc = model.score(XLR, YLR)
print("We get a score of " + str(acc) + "%")

prediction = model.predict_proba(XLR)[:,1]

plt.figure(figsize=(15,8))
plt.hist(prediction[YLR==0], bins=20, label='Negatives')
plt.hist(prediction[YLR==1], bins=20, label='Positives', alpha=0.7, color='r')
plt.xlabel('Probability of Cancer', fontsize=25)
plt.ylabel('Trials', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)

plt.show()

