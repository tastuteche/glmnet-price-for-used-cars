import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

superclean_data = pd.read_csv('superclean_data.csv')

df_numeric_data = superclean_data[['price', 'yearOfRegistration',
                                   'powerPS', 'kilometer']].dropna()


df_numeric_data['yearOfRegistration'] = 2017 - \
    df_numeric_data['yearOfRegistration']


from sklearn import preprocessing

min_max_scaler_x = preprocessing.MinMaxScaler()
x = min_max_scaler_x.fit_transform(
    df_numeric_data[['yearOfRegistration', 'powerPS', 'kilometer']])
min_max_scaler_y = preprocessing.MinMaxScaler()
y = min_max_scaler_y.fit_transform(
    df_numeric_data[['price']])

# ln -s /r/a/p/usr/lib64/python3.5/site-packages/glmnet_python /r/a/p/usr/lib64/python3.5/site-packages/glmnet
# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(
#     '/r/a/p/usr/lib64/python3.5/site-packages/glmnet_python/__init__.py')))
import glmnet_python
from glmnet import glmnet
from glmnetPlot import glmnetPlot

import scipy
wts = scipy.ones((df_numeric_data.shape[0], 1), dtype=scipy.float64)
fit = glmnet(x=x.copy(), y=y.copy(), family='gaussian',
             weights=wts,
             alpha=0.2, nlambda=20
             )
glmnetPlot(fit, xvar='lambda', label=True)

plt.savefig('glmnet_price1.png', dpi=200)
plt.cla()
plt.clf()
plt.close()


from patsy import dmatrices
df_categorical = superclean_data[['price', 'yearOfRegistration',
                                  'powerPS', 'kilometer', 'vehicleType', 'gearbox', 'fuelType', 'notRepairedDamage', 'brand', 'model']].dropna()

df_categorical['yearOfRegistration'] = 2017 - \
    df_categorical['yearOfRegistration']

y, X = dmatrices('price ~' + 'yearOfRegistration+powerPS+kilometer+C(notRepairedDamage)+C(fuelType)+C(gearbox)+C(vehicleType)+C(brand)+C(model)', df_categorical,
                 return_type='dataframe')


min_max_scaler_x1 = preprocessing.MinMaxScaler()
x1 = min_max_scaler_x1.fit_transform(X)
min_max_scaler_y1 = preprocessing.MinMaxScaler()
y1 = min_max_scaler_y1.fit_transform(y)

fit1 = glmnet(x=x1.copy(), y=y1.copy(), family='gaussian',
              weights=wts,
              alpha=1, nlambda=100
              )
from glmnetCoef import glmnetCoef
c = glmnetCoef(fit1)
c = c[1:, -1]  # remove intercept and get the coefficients at the end of the path
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 15))
h = glmnetPlot(fit1, xvar='lambda', label=False)
# /r/a/p/usr/lib64/python3.5/site-packages/glmnet_python/glmnetPlot.py
ax1 = h['ax1']
xloc = plt.xlim()
xloc = xloc[0]

index = h['index']
xpos = min(index)

labels = X.columns.tolist()
for i in range(len(c)):
    ax1.text(1 / 2 * xpos + 1 / 2 * xloc, c[i], labels[i])

plt.savefig('glmnet_price2.png', dpi=200)
plt.cla()
plt.clf()
plt.close()


from glmnetPrint import glmnetPrint
glmnetPrint(fit1)

important_cols = []
for i, val in enumerate(glmnetCoef(fit1, s=scipy.float64([0.005]), exact=False)[1:, -1]):
    if val != 0:
        important_cols.append((labels[i], val, abs(val)))

df_important_cols = pd.DataFrame(important_cols, columns=[
                                 'col', 'coef', 'coef_abs'])

from tabulate import tabulate
with open('df_important_cols.txt', 'w') as f:
    f.write(tabulate(df_important_cols.sort_values(
        'coef_abs', ascending=False), headers="keys", tablefmt="pipe"))
