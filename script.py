'''
    DOCSTRING
'''
__author__ = 'Stefan A. Obada'
__date__ = 'Wed Mar  4 11:16:48 2020'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('fev.csv')
X = df[['smoke', 'age']]
y = df['fev'].values.reshape(-1,1)

# 2 (a)
X_fev = df.smoke.values.reshape(-1,1)

## Fitting the model
model = LinearRegression() # We use sklearn to avoid long matrix Calculation
model.fit(X_fev, y)

beta0 = model.intercept_[0]
beta1 = model.coef_[0][0]

print(f'beta0 = {beta0}')
print(f'beta1 = {beta1}')
##

fev_smokers = y[X_fev==1]
fev_nonsmokers = y[X_fev==2]

## Plot histograms
plt.subplot(3,2,1).set_title('Smokers \n Mean: {:.3} \n Count:{}'.format(fev_smokers.mean(), len(fev_smokers)))
plt.hist(fev_smokers, bins=8, density=True, alpha=0.7)
plt.vlines(fev_smokers.mean(), 0, 0.7, color='red')
plt.xlim(y.min(), y.max())

plt.subplot(3,2,2).set_title('Non-Smokers \n Mean: {:.3} \n Count:{}'.format(fev_nonsmokers.mean(), len(fev_nonsmokers)))
plt.hist(fev_nonsmokers, bins=8, density=True, alpha=0.7)
plt.vlines(fev_nonsmokers.mean(), 0, 0.5, color='red')
plt.xlim(y.min(), y.max())
##

## Mean difference
exp_diff = (beta0 + 1*beta1) - (beta0 + 2*beta1)
print('Fev(smokers) - Fev(nonsmokers): {}'.format(exp_diff))
##




# 2 (b)

## Fitting the model
model_b = LinearRegression()
model_b.fit(X, y)

gamma0, gamma1, gamma2 = model_b.intercept_[0], model_b.coef_[0][0], model_b.coef_[0][1]
print(f'gamma0 = {gamma0}')
print(f'gamma1 = {gamma1}')
print(f'gamma2 = {gamma2}')
##

## Mean difference
exp_diff = gamma1*1-gamma1*2
print('Fev(smokers) - Fev(nonsmokers): {}'.format(exp_diff))
##

## Plot histograms just for under 12 years old
under_twelve_years_smokers = y[(X.age<12) & (X.smoke==1)]
under_twelve_years_nonsmokers = y[(X.age<12) & (X.smoke==2)]

plt.subplot(3,2,3).set_title('<12 years smokers \n Mean: {:.3} \n Count:{}'.format(under_twelve_years_smokers.mean(), len(under_twelve_years_smokers)))
plt.hist(under_twelve_years_smokers, bins=4, density=True, alpha=0.7)
plt.vlines(under_twelve_years_smokers.mean(), 0, 0.63, color='red')
plt.xlim(y.min(), y.max())

plt.subplot(3,2,4).set_title('<12 years non-Smokers \n Mean: {:.3} \n Count:{}'.format(under_twelve_years_nonsmokers.mean(), len(under_twelve_years_nonsmokers)))
plt.hist(under_twelve_years_nonsmokers, bins=6, density=True, alpha=0.7)
plt.vlines(under_twelve_years_nonsmokers.mean(), 0, 0.57, color='red')
plt.xlim(y.min(), y.max())
##

## Plot histograms just for over 12 years old
over_twelve_years_smokers = y[(X.age>=12) & (X.smoke==1)]
over_twelve_years_nonsmokers = y[(X.age>=12) & (X.smoke==2)]

plt.subplot(3,2,5).set_title('>=12 years smokers \n Mean: {:.3} \n Count:{}'.format(over_twelve_years_smokers.mean(), len(over_twelve_years_smokers)))
plt.hist(over_twelve_years_smokers, bins=8, density=True, alpha=0.7)
plt.vlines(over_twelve_years_smokers.mean(), 0, 0.54, color='red')
plt.xlim(y.min(), y.max())

plt.subplot(3,2,6).set_title('>=12 years non-Smokers \n Mean: {:.3} \n Count:{}'.format(over_twelve_years_nonsmokers.mean(), len(over_twelve_years_nonsmokers)))
plt.hist(over_twelve_years_nonsmokers, bins=8, density=True, alpha=0.7)
plt.vlines(over_twelve_years_nonsmokers.mean(), 0, 0.4, color='red')
plt.xlim(y.min(), y.max())
##

## Variability
def S(a: np.array, b: np.array):
    '''Return S_{ab} as in the coursework'''
    return (a-a.mean()).dot(b-b.mean())

print('var(beta1)/var(gamma1) = {}'.format((1-S(X.smoke,X.age)**2/(S(X.smoke,X.smoke)*S(X.age, X.age)))))
##

plt.tight_layout()