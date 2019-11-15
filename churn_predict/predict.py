#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:18:14 2019

@author: davidazoulay
"""


from training_part import training


def predict(X_test,model_fit):
	
	model_predictions = model_fit.predict(X_test)
	
	return model_predictions
	

def main(f_dir):
	
	# Column names
	col_names = ['customer_key', 'registration_date', 'year_of_birth', 'gender', 
	             'acquisition_channel_id', 'transaction_date', 'bet_nb', 'bet_amount', 'profit_loss',
			     'deposit_nb', 'deposit_amount', 'betclic_customer_segmentation']
	
	df_churn_test = training.read_csv_file(
						f_dir,';',col_names,0,col_names[:-1],0,
						['registration_date','transaction_date'],
						{'year_of_birth': object, 'acquisition_channel_id': object})
	
	pre_process = training.data_preprocessing(df_churn_test)					
	X_test, y_test = pre_process.process()					
	
	f_dir_training = input("Enter the training dataset filepath : ")
	training_model = training.main(f_dir_training)
	
	pred = training_model.predict(X_test)
	
	return pred
