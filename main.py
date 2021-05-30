import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()

# we learn that malignant is cancerous, and benign is not concerous

df = pd.DataFrame(cancer_data['data'], columns = cancer_data['feature_names'])
print(df.head())