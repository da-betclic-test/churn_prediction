#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:18:14 2019

@author: davidazoulay
"""

import pandas as pd
import datetime as dt

from sklearn.externals import joblib
from training_part.training import DataPreprocessing


###################
### Macro Variables

# Column names
col_names = ['customer_key', 'registration_date', 'year_of_birth', 'gender', 
             'acquisition_channel_id', 'transaction_date', 'bet_nb', 'bet_amount',
		    'profit_loss', 'deposit_nb', 'deposit_amount', 'betclic_customer_segmentation']


def make_predictions(model_fit, test_set):
	
	model_predictions   = model_fit.predict(test_set)
	model_probabilities = model_fit.predict_proba(test_set)[:,1]
	
	return model_predictions, model_probabilities		
		

def main(f_name):

	dateParse = lambda x : dt.datetime.strptime(x, '%d/%m/%y')		
	
	df_churn_test = pd.read_csv(f_name, sep = ';', names = col_names, header = 0,
						      usecols = col_names[:-1], index_col = 0, 
                                 parse_dates = ['registration_date','transaction_date'], 
					          date_parser = dateParse, na_values = 'NaT', 
						      dtype = {'year_of_birth': object, 'acquisition_channel_id': object})
	
	df_churn_test = df_churn_test.reset_index().sort_values(
			       by=['customer_key','transaction_date']).set_index(['customer_key'])
	
	pre_process = DataPreprocessing(df_churn_test)					
	X_test, y_test = pre_process.process()

	pre_trained_model = joblib.load('training_part/classifier.joblib')						
	y_pred, y_probs = make_predictions(pre_trained_model, X_test)
	
	return y_pred, y_probs
