import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')  # Ignore warnings

df = pd.read_csv('machine.data') # Load the dataset (credits for dataset in README)
df.columns = ['VendorName', 'ModelName', 'MachCycle', 'MinMem', 'MaxMem', 'Cach', 'MinChan', 'MaxChan', 'PubPerf', 'ExpPerf']
df = df.drop(['VendorName', 'ModelName', 'ExpPerf'], axis = 1) # Drop useless Features

train, valid, test = np.split(df.sample(frac=1),[int(0.6*len(df)),int(0.8*len(df))]) # Split data for train and validation.
cols = ['MachCycle', 'MinMem', 'MaxMem', 'Cach', 'MinChan', 'MaxChan']


def getxy(df, ylabel, xlabel=None):
    _df = copy.deepcopy(df)
    if xlabel is None:
        x = _df[cols].values
    else:
        x = np.reshape(_df[xlabel].values, (-1,1))
    y = np.reshape(_df[ylabel], (-1,1))
    return x, y


######### Linear Regression for well... duh... regression


##### Single Variate Linear Regression

## To figure out the Feature which produces the highest R**2 value in single variate

r2 = {}
preds = {}
for i in cols:
    sing_linreg = LinearRegression()
    x, y = getxy(train, 'PubPerf', i)
    sing_linreg.fit(x,y)
    perf_pred_sing = sing_linreg.predict(np.reshape(valid[i],(-1,1)))
    x_valid, y_valid = getxy(valid, 'PubPerf', i)
    r2[i] = sing_linreg.score(x_valid, y_valid)
    preds[i] = perf_pred_sing
MaxRsq = max(r2, key= lambda x: r2[x])
print(MaxRsq) # Max R**2 value is given with MaxMem

plt.scatter(valid['MaxMem'].values, valid['PubPerf'].values)
plt.scatter(valid['MaxMem'].values, preds[MaxRsq])


##### Multi Variate Linear Regression

all_linreg = LinearRegression()
x, y = getxy(train, 'PubPerf')
all_linreg.fit(x, y)
perf_pred_all = all_linreg.predict(valid[cols])
plt.scatter(valid['MaxMem'].values, perf_pred_all)

plt.xlabel('MaxMem')
plt.ylabel('PubPerf')
plt.legend(['Actual Data', 'Single Variate Linear Regression', 'Multi Variate Regression'])
plt.show() # Final graph showing the regression outputs


