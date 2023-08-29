# Imports 

import numpy as np
from numpy.random import binomial, multivariate_normal, normal, uniform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import sys

# import package for the data
#import xarray as xr

# import packages specific to causal methods
#from nb21 import cumulative_gain, elast
import statsmodels.formula.api as smf
from matplotlib import style

from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor

from econml.dml import DML, CausalForestDML

from scipy.stats import skewnorm
from scipy.stats import beta

# drawing DAGs

import graphviz as gr
from matplotlib import style
style.use("default")


# Set seed
np.random.seed(42)

# set some basic variables
var_names = [r'uvb', r't2m', r'sp', r'stl1', r'windspeed', r'blh', r'relhum', r'tp', r'lai', r'no', r'no2', r'o3']

coeffs_df = pd.DataFrame(index = var_names, 
                         columns = var_names,
                         data = np.ones((len(var_names), len(var_names))))

coeffs_df['uvb']['t2m'] = 1.5e-4
coeffs_df['t2m']['stl1'] = 0.5
coeffs_df['uvb']['stl1'] = 0.5
coeffs_df['t2m']['blh'] = 10
coeffs_df['t2m']['relhum'] = 0.8
coeffs_df['sp']['relhum'] = 1e-5
coeffs_df['t2m']['no'] = -1
coeffs_df['t2m']['no2'] = -0.5
coeffs_df['uvb']['o3'] = 3.5e-4
coeffs_df['t2m']['o3'] = 0.7
coeffs_df['no']['o3'] = -0.2
coeffs_df['blh']['o3'] = -1e-2
coeffs_df['windspeed']['o3'] = -0.7

# How do I do this quickly, so I can cycle through a load of different DAGs and see if it is confounded?

# Let's make a big function to do the whole process, apart from the graph plotting


def generate_synthetic_data(seed, variables, coefficients, non_linear_ozone = False):

    '''
    Function to generate sythetic data from a given DAG, 
    with a given set of coefficients
    '''

    # Set a random seed
    random_state = np.random.default_rng(seed=seed)

    # Generate a random dataset of the same size as the original dataset, and check shape
    sim_data = random_state.uniform(low=0, high=140000, size = (10000, len(variables)))
    assert sim_data.shape == (10000, 12)

    sim_data_df = pd.DataFrame(sim_data, columns=variables)
    
    for i in range(0, 10000):
        sim_data_df.loc[i, 't2m'] = coefficients['uvb']['t2m'] * sim_data_df.loc[i, 'uvb'] + 3 + np.random.normal(0,5)   # uvb -> t2m
        sim_data_df.loc[i, 'sp'] = skewnorm.rvs(-6, 10000, 10000)  # sp
        sim_data_df.loc[i, 'stl1'] = coefficients['uvb']['stl1'] * sim_data_df.loc[i, 'uvb'] + coefficients['t2m']['stl1'] * sim_data_df.loc[i, 't2m'] + np.random.normal(0,2) # uvb, t2m -> stl1
        sim_data_df.loc[i, 'windspeed'] = 11 * beta.rvs(1.6, 6) # windspeed
        sim_data_df.loc[i, 'blh'] = coefficients['t2m']['blh'] * sim_data_df.loc[i, 't2m'] + 300 + np.random.normal(0, 10) # t2m -> blh
        sim_data_df.loc[i, 'relhum'] = coefficients['t2m']['relhum'] * sim_data_df.loc[i, 't2m'] + coefficients['sp']['relhum'] * sim_data_df.loc[i, 'sp'] + np.random.normal(0, 1) # t2m, sp -> relhum
        sim_data_df.loc[i, 'tp'] = 0.01 * beta.rvs(0.5, 0.5) # tp
        sim_data_df.loc[i, 'lai'] = np.random.normal(3,1) # lai
        sim_data_df.loc[i, 'no'] = coefficients['t2m']['no'] * sim_data_df.loc[i, 't2m'] + 30 + np.random.normal(0, 1) # t2m -> no
        sim_data_df.loc[i, 'no2'] = coefficients['t2m']['no2'] * sim_data_df.loc[i, 't2m'] + 25 + np.random.normal(0, 1) # t2m -> no2
        if non_linear_ozone == False:
            sim_data_df.loc[i, 'o3'] = coefficients['uvb']['o3'] * sim_data_df.loc[i, 'uvb'] + coefficients['t2m']['o3'] * sim_data_df.loc[i, 't2m'] + coefficients['no']['o3'] * sim_data_df.loc[i, 'no'] + coefficients['blh']['o3'] * sim_data_df.loc[i, 'blh'] + coefficients['windspeed']['o3'] * sim_data_df.loc[i, 'windspeed'] + 10 + np.random.normal(0, 4)   # uvb, t2m, no, blh, windspeed -> o3
        else:
            sim_data_df.loc[i, 'o3'] = coefficients['uvb']['o3'] * sim_data_df.loc[i, 'uvb'] + coefficients['t2m']['o3'] * sim_data_df.loc[i, 't2m'] + coefficients['no']['o3'] * sim_data_df.loc[i, 'no'] + coefficients['blh']['o3'] * sim_data_df.loc[i, 'blh'] + coefficients['windspeed']['o3'] * sim_data_df.loc[i, 'windspeed'] + 10 + np.random.normal(0, 4)   # uvb, t2m, no, blh, windspeed -> o3
            if sim_data_df.loc[i, 't2m'] > 20.0:
                sim_data_df.loc[i, 'o3'] = coefficients['uvb']['o3'] * sim_data_df.loc[i, 'uvb'] + ((coefficients['t2m']['o3']+0.7) * sim_data_df.loc[i, 't2m']) + coefficients['no']['o3'] * sim_data_df.loc[i, 'no'] + coefficients['blh']['o3'] * sim_data_df.loc[i, 'blh'] + coefficients['windspeed']['o3'] * sim_data_df.loc[i, 'windspeed'] + 10 + np.random.normal(0, 4)   # uvb, t2m, no, blh, windspeed -> o3
            else:
                sim_data_df.loc[i, 'o3'] = sim_data_df.loc[i, 'o3']
    
    return sim_data_df


    # Now, for each variable, calculate the value of the variable based on the coefficients and the values of the other variables

    #for i in range(len(variables)):
    #    data_copy[variables[i]] = coefficients[i] * data_copy[variables[i]] + np.random.normal(0, 1)

# generate linear synthetic data
syn_data = generate_synthetic_data(1, var_names, coeffs_df, non_linear_ozone = False)

# generate non-linear synthetic data
syn_data_nonlinear = generate_synthetic_data(1, var_names, coeffs_df, non_linear_ozone = True)

# ATE estimation function

def debias_denoise_ml(model, train_data, outcome, treatment, covariates):
    '''
    Debias and denoise using non-parametric ML methods.
    '''
    debias_m = model(max_depth=3)
    denoise_m = model(max_depth=3)

    train_pred = train_data.assign(treatment_res =  train_data[treatment] - cross_val_predict(debias_m, train_data[covariates], train_data[treatment], cv=5),
                          outcome_res =  train_data[outcome] - cross_val_predict(denoise_m, train_data[covariates], train_data[outcome], cv=5))
    return train_pred

def extract_params_dml(data, outcome, treatment, confounders):
    '''
    Function that simply returns the coefficient derived 
    from DML of treatment on outcome with confounders.
    '''
    data_residuals = debias_denoise_ml(LGBMRegressor, data, outcome, treatment, confounders)
    final_ols = smf.ols(formula='outcome_res ~ treatment_res', data=data_residuals).fit()
    return final_ols.params[1], final_ols.bse[1]


# Define confounders here...
temp_confs = ['uvb', 'blh', 'windspeed', 'no']
uvb_confs = ["t2m", "windspeed"]
no_confs = ["t2m"]
windspeed_confs = []
blh_confs = ['t2m']

naive_confs = []

# do linear case first...
# naively, extract the parameters for each of the variables that we are interested in, with no confounders considered

l_naive_temp_param, l_naive_temp_param_se = extract_params_dml(syn_data, 'o3', 't2m', naive_confs)
l_naive_uvb_param, l_naive_uvb_param_se = extract_params_dml(syn_data, 'o3', 'uvb', naive_confs)
l_naive_no_param, l_naive_no_param_se = extract_params_dml(syn_data, 'o3', 'no', naive_confs)
l_naive_windspeed_param, l_naive_windspeed_param_se = extract_params_dml(syn_data, 'o3', 'windspeed', naive_confs)
l_naive_blh_param, l_naive_blh_param_se = extract_params_dml(syn_data, 'o3', 'blh', naive_confs)

# extract the parameters for each of the variables that we are interested in

l_temp_param, l_temp_param_se = extract_params_dml(syn_data, 'o3', 't2m', temp_confs)
l_uvb_param, l_uvb_param_se = extract_params_dml(syn_data, 'o3', 'uvb', uvb_confs)
l_no_param, l_no_param_se = extract_params_dml(syn_data, 'o3', 'no', no_confs)
l_windspeed_param, l_windspeed_param_se = extract_params_dml(syn_data, 'o3', 'windspeed', windspeed_confs)
l_blh_param, l_blh_param_se = extract_params_dml(syn_data, 'o3', 'blh', blh_confs)


# Nonlinear case
# naively, extract the parameters for each of the variables that we are interested in, with no confounders considered

nl_naive_temp_param, nl_naive_temp_param_se = extract_params_dml(syn_data_nonlinear, 'o3', 't2m', naive_confs)
nl_naive_uvb_param, nl_naive_uvb_param_se = extract_params_dml(syn_data_nonlinear, 'o3', 'uvb', naive_confs)
nl_naive_no_param, nl_naive_no_param_se = extract_params_dml(syn_data_nonlinear, 'o3', 'no', naive_confs)
nl_naive_windspeed_param, nl_naive_windspeed_param_se = extract_params_dml(syn_data_nonlinear, 'o3', 'windspeed', naive_confs)
nl_naive_blh_param, nl_naive_blh_param_se = extract_params_dml(syn_data_nonlinear, 'o3', 'blh', naive_confs)

# extract the parameters for each of the variables that we are interested in

nl_temp_param, nl_temp_param_se = extract_params_dml(syn_data_nonlinear, 'o3', 't2m', temp_confs)
nl_uvb_param, nl_uvb_param_se = extract_params_dml(syn_data_nonlinear, 'o3', 'uvb', uvb_confs)
nl_no_param, nl_no_param_se = extract_params_dml(syn_data_nonlinear, 'o3', 'no', no_confs)
nl_windspeed_param, nl_windspeed_param_se = extract_params_dml(syn_data_nonlinear, 'o3', 'windspeed', windspeed_confs)
nl_blh_param, nl_blh_param_se = extract_params_dml(syn_data_nonlinear, 'o3', 'blh', blh_confs)

np.polyfit(syn_data['blh'], syn_data['o3'], 1)


# Plot linear effect first

x = ["Temp", "Sol. rad. x 1e3", "Wind speed", "NO", "BLH x 1e2"]
#x = [0, 1, 2, 3, 4]
y_true = [coeffs_df['t2m']['o3'], coeffs_df['uvb']['o3'] * 1e3, coeffs_df['windspeed']['o3'], coeffs_df['no']['o3'], coeffs_df['blh']['o3'] * 1e2]
y_naive = [l_naive_temp_param, l_naive_uvb_param * 1e3, l_naive_windspeed_param, l_naive_no_param, l_naive_blh_param * 1e2]
#y_naive = [1.786, 4.1e-4 * 1e3, -0.727, -1.759, 0.1753 * 100]
y_hat = [l_temp_param, l_uvb_param * 1e3, l_windspeed_param, l_no_param, l_blh_param * 1e2]
y_err = [l_temp_param_se, l_uvb_param_se * 1e3, l_windspeed_param_se, l_no_param_se, l_blh_param_se * 1e2]

plt.figure(figsize=(14,8))

plt.scatter(x, y_true, label = 'True', color = 'k')
plt.scatter(x, y_hat, label = 'DML estimate', color = 'orange', alpha=0.4)
plt.scatter(x, y_naive, label = 'Naive estimate', color = 'red', alpha=0.4)

plt.errorbar(x, y_hat, yerr=y_err, fmt="o", color='orange', alpha=0.4)

plt.title('True causal effect, estimated, and estimated without confounders', fontsize=20)
plt.ylabel('Average treatment effect / ppb unit$^{-1}$', fontsize=12)
plt.xlabel('Variables')
plt.xticks(rotation=90)
#plt.ylim(-2, 2)

plt.legend(fontsize=16)
plt.show()

# Plot nonlinear results

x = ["Temp", "Sol. rad. x 1e3", "Wind speed", "NO", "BLH x 1e2"]
#x = [0, 1, 2, 3, 4]
y_true = [coeffs_df['t2m']['o3'], coeffs_df['uvb']['o3'] * 1e3, coeffs_df['windspeed']['o3'], coeffs_df['no']['o3'], coeffs_df['blh']['o3'] * 1e2]
y_nl_true = [coeffs_df['t2m']['o3']+0.7, np.nan, np.nan, np.nan, np.nan]
y_naive = [nl_naive_temp_param, nl_naive_uvb_param * 1e3, nl_naive_windspeed_param, nl_naive_no_param, nl_naive_blh_param * 1e2]
#y_naive = [1.786, 4.1e-4 * 1e3, -0.727, -1.759, 0.1753 * 100]
y_hat = [nl_temp_param, nl_uvb_param * 1e3, nl_windspeed_param, nl_no_param, nl_blh_param * 1e2]
y_err = [nl_temp_param_se, nl_uvb_param_se * 1e3, nl_windspeed_param_se, nl_no_param_se, nl_blh_param_se * 1e2]

plt.figure(dpi=150)

plt.scatter(x, y_true, label = 'True', color = 'k')
plt.scatter(x, y_nl_true, label = 'True (nonlinear)', color = 'k', marker='x')
plt.scatter(x, y_hat, label = 'DML Estimated', color = 'orange', alpha=0.4)
#plt.scatter(x, y_naive, label = 'Naive corr. estimate', color = 'red')

plt.errorbar(x, y_hat, yerr=y_err, fmt="o", color='orange', alpha=0.4)

plt.title('NONLINEAR- Plot of true causal effect, estimated, and estimated without confounders', fontsize=12)
plt.xticks(rotation=90)
plt.ylim(-2, 2)

plt.legend()
plt.show()