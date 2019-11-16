#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:18:14 2019

@author: davidazoulay
"""


from training_part.training import read_csv_file, data_preprocessing, training_model


class predict(training_model):
	
	def __init__(self,target,features,test_set):
		
		training_model.__init__(self,target,features)
		self.test_set = test_set
	
	def predictions(self):
		
		model = training_model(self.target, self.features)
		model_fit = model.fit()
		model_predictions = model_fit.predict(self.test_set)
		
		return model_predictions		
	

def main(f_dir):
	
	# Column names
	col_names = ['customer_key', 'registration_date', 'year_of_birth', 'gender', 
	             'acquisition_channel_id', 'transaction_date', 'bet_nb', 'bet_amount', 'profit_loss',
			     'deposit_nb', 'deposit_amount', 'betclic_customer_segmentation']
	
	df_churn_test = read_csv_file(
						f_dir,';',col_names,0,col_names[:-1],0,
						['registration_date','transaction_date'],
						{'year_of_birth': object, 'acquisition_channel_id': object})
	
	pre_process = data_preprocessing(df_churn_test)					
	X_test, y_test = pre_process.process()						
	
	pred = training_model.predict(X_test)
	
	return pred
