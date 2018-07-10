from sklearn import linear_model
import pandas as pd 
import numpy as np
from sklearn import datasets
data = datasets.load_boston() ##load boston dataset from datset libarary'


#define the data/predictors as the pre-set feature names
df = pd.DataFrame(data.data, columns = data.feature_names)

#Put the target (housing value --MEDV) in another data frame
target = pd.DataFrame(data.target, columns=["MEDV"])

x=df
y= target["MEDV"]

lm = linear_model.LinearRegression()
model = lm.fit(x,y)

predictions = lm.predict(x)
print (predictions)

lm.score(x,y)
lm.coef_
lm.intercept_

