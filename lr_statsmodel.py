import statsmodels.api as sm 
import numpy as np 
import pandas as pd 
from sklearn import datasets

data = datasets.load_boston()

df = pd.DataFrame(data.data, columns = data.feature_names)


target = pd.DataFrame(data.target, columns=["MEDV"])


X = df["RM"]
y = target["MEDV"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()