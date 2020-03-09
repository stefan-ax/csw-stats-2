'''
    DOCSTRING
'''
__author__ = 'Stefan A. Obada'
__date__ = 'Wed Mar  4 11:16:48 2020'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('fev.csv')
X = df[['smoke', 'age']]
y = df['fev'].values.reshape(-1,1)

# Linear regression fit
class LinearRegression():
    """Simple LSE computing for Linear Regression"""
    
    @staticmethod
    def append_ones(X):
        # Appends a column of ones to X
        X = np.concatenate([np.ones(shape=(X.shape[0], 1)), X], axis=1)
        return X
    
    def fit(self, X, y):
        X = self.append_ones(X)
        # Least Squares method
        LSE = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.intercept = LSE[0]
        self.coef = LSE[1:]

# 2 (a)
X_a = df.smoke.values.reshape(-1,1)

## Fitting the model
model = LinearRegression() # We use sklearn to avoid long matrix Calculation
model.fit(X_a, y)

beta0 = model.intercept
beta1 = model.coef[0]

print(f'beta0 = {beta0}')
print(f'beta1 = {beta1}')
##

fev_smokers = y[X_a==1]
fev_nonsmokers = y[X_a==2]

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

gamma0, gamma1, gamma2 = model_b.intercept, model_b.coef[0], model_b.coef[1]
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
    return (a-a.mean()).T.dot(b-b.mean())

print('var(beta1)/var(gamma1) = {}'.format((1-S(X.smoke,X.age)**2/(S(X.smoke,X.smoke)*S(X.age, X.age)))))
##

## 2c)
alfa=0.05
n = X_a.shape[0]
X_a_with_ones = LinearRegression.append_ones(X_a)
RSS = np.linalg.norm(y-X_a_with_ones.dot(np.array([beta0, beta1])))**2
import scipy.stats
t_dist = scipy.stats.t
t_alfa = t_dist.ppf(1-alfa/2, df=n-2)
sum_smoke_squared = sum(X_a**2)
lower = beta1-t_alfa*np.sqrt(n*RSS/(n*sum_smoke_squared-(n**2)*((X_a.mean())**2))/(n-2))
upper = beta1+t_alfa*np.sqrt(n*RSS/(n*sum_smoke_squared-(n**2)*((X_a.mean())**2))/(n-2))
print('A {}% CI for beta1 is: ( {:.3}, {:.3} )'.format((1-alfa)*100, lower[0], upper[0]))

## F_test 2c)
alfa=0.05
n = X_a.shape[0]
r = 2
s = 1
X_a_without_smoke = np.ones(n).reshape(-1,1)

P_on_X = X_a_without_smoke.dot(np.linalg.inv(X_a_without_smoke.T.dot(X_a_without_smoke))).dot(X_a_without_smoke.T)
Q = np.eye(P_on_X.shape[0])-P_on_X
v2 = y.T.dot(Q).dot(y)

RSS0 = np.linalg.norm(y-X_a_without_smoke*beta0)**2
F = (RSS0-RSS)/RSS * (n-r)/(r-s)
import scipy.stats
F_dist = scipy.stats.f
c=F_dist.ppf(1-alfa, dfn=r-s,dfd=n-r)

print(f'F = {F}')
print(f'c = {c}')
##


## 2d)
alfa=0.05
n = X.shape[0]
X_with_ones = LinearRegression.append_ones(X)
RSS = np.linalg.norm(y-X_with_ones.dot(np.array([gamma0, gamma1, gamma2])))**2
import scipy.stats
t_dist = scipy.stats.t
t_alfa = t_dist.ppf(1-alfa/2, df=n-3)
e_dash = S(X.age, X.age)+n*(X.age.mean()**2)
lower = gamma1-t_alfa*np.sqrt(e_dash*RSS/(n*(S(X.smoke,X.smoke)*S(X.age,X.age)-S(X.smoke, X.age)**2))/(n-3))
upper = gamma1+t_alfa*np.sqrt(sum_smoke_squared*RSS/(n*sum_smoke_squared-(n**2)*((X_a.mean())**2))/(n-3))
print('A {}% CI for gamma1 is: ( {:.3}, {:.3} )'.format((1-alfa)*100, lower[0], upper[0]))

## F_test 2d)
alfa=0.05
n = X.shape[0]
r = 3
s = 2
X_without_smoke = LinearRegression.append_ones(X.age.values.reshape(-1,1))
RSS0 = np.linalg.norm(y-X_without_smoke.dot(np.array([gamma0, gamma2])))**2
F = (RSS0-RSS)/RSS * (n-r)/(r-s)
import scipy.stats
F_dist = scipy.stats.f
c=F_dist.ppf(1-alfa, dfn=r-s,dfd=n-r)

print(f'F = {F}')
print(f'c = {c}')
##

##

plt.tight_layout()