# Tests

from gen_syn import generate_synthetic_data
import numpy as np
import pandas as pd


# test inputs
var_names = [r'uvb', r't2m', r'sp', r'stl1',
             r'windspeed', r'blh', r'relhum', r'tp',
             r'lai', r'no', r'no2', r'o3']

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

# TODO generalise tests for all dataset functions

def test_generate_synthetic_data_is_dataframe():

    """ Check if output is a dataframe"""
    df = generate_synthetic_data(generate_synthetic_data(1, var_names, coeffs_df, non_linear_ozone = False))
    assert isinstance(df, pd.DataFrame), "df is not a dataframe"

def test_generate_synthetic_data_has_correct_columns():

    """ Check if output has correct columns"""
    df = generate_synthetic_data(generate_synthetic_data(1, var_names, coeffs_df, non_linear_ozone = False))
    assert df.columns.tolist() == [r'uvb', r't2m', r'sp', r'stl1', r'windspeed', r'blh', r'relhum', r'tp', r'lai', r'no', r'no2', r'o3'], "df does not have correct columns"

def test_generate_synthetic_data_has_correct_shape():
    
        """ Check if output has correct shape"""
        df = generate_synthetic_data(generate_synthetic_data(1, var_names, coeffs_df, non_linear_ozone = False))
        assert df.shape == (1, 12), "df does not have correct shape"

def check_generate_synthetic_data_is_linear_non_linear():

    """ Check if output is linear or non-linear"""
    df = generate_synthetic_data(generate_synthetic_data(1, var_names, coeffs_df, non_linear_ozone = False))
    assert df['o3'].equals(df['t2m']*coeffs_df['t2m']['o3'] + df['no']*coeffs_df['no']['o3'] + df['blh']*coeffs_df['blh']['o3'] + df['windspeed']*coeffs_df['windspeed']['o3']), "df is not linear"