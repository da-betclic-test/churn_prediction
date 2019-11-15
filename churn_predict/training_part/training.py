#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:18:40 2019

@author: davidazoulay
"""


#####################
### Attention: tous les indicateurs doivent être calculé avant la derniere date disponible - 90 jours
### Customer key unique lors de l'apprentissage du modele


###############
### Modules ###
###############

import pandas as pd
import numpy as np
import datetime as dt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier


# Macro variables
last_available_date  = dt.datetime(2019,7,23)
last_observable_date = last_available_date - dt.timedelta(days=90)


################
### Data Reading

def read_csv_file(file_directory,sep,col_names,header,usecols,index_col,col_dates,dtypes):
	
	dateParse = lambda x : dt.datetime.strptime(x, '%d/%m/%y')
					
	df = pd.read_csv(file_directory, sep = sep, names = col_names, header = header, 
				    usecols = usecols, index_col = index_col, 
				    parse_dates = col_dates, date_parser = dateParse, na_values = 'NaT', 
				    dtype = dtypes)
	
	return df
	

##################################
### Data preprocessing methods ###
##################################


class data_preprocessing:
	
	def __init__(self,df):
		
		self.df = df		


	def replace_missing(self, x, y=None):
	
		"""
		Replace the missing values within the DataFrame
		
		Parameters
		----------
		x : pandas.Series
			Series with potential missing values.
		
		y: pandas.Series
			Series to use to fill missing values in x.
		
		Returns
		-------
		Series with no missing values
		"""
				
		if x.isnull().any():
			
			if y is not None:
				
				x = x.fillna(y)
			
			else:
						
				if x.dtype in ['O','datetime64[ns]']:
					x = x.fillna(x.mode()[0])
				
				elif x.dtype == float:
					x = x.fillna(x.mean())							
			
		return x
	
	def replace_values(self, x, old_val, new_val, reg_expr=False):
	
		"""
		Replace a specific value within a Series by an other one
		
		Parameters
		----------
		x : pandas.Series
		
		old_val : int, float, str
			Value to be replaced
		
		new_val : int, float, str
			Replacement value
		
		reg_expr: boolean
			Specify if whether old_val is a regular expression or not
			
		Returns
		-------
		Series with new values 
		"""
		
		if reg_expr:
			x = x.replace(old_val, new_val, regex = reg_expr)
		else: x = x.replace(old_val, new_val)
		
		return x
	
	
	def discretization(self, x, bins):
		
		"""
		Variable discretization
		
		Parameters
		----------
		
		x : pandas.Series
			The variable that will be discretized.
		
		bins : list
			Defines the bin edges.
		
		Returns
		-------
		Series representing the respective bin for each value of x.
		"""
		
		if x.dtype == 'O':
			
			try:
				x = x.astype(int)
			except ValueError:
				raise ValueError("Object type series can't be discretized.")
				
		disc_x = pd.cut(x,bins)
		
		return disc_x
	
	
	def categorical_encoding(self, df, is_ordinal=[]):
	
		"""
		Encode the categorical variables
		
		Parameters
		----------
		df : pandas.DataFrame
		
		is_ordinal : list of boolean
			Specify if whether or not the categorical variable is ordinal
		
		Returns
		-------
		DataFrame with encoded categorical variables	
		"""
		
		df_cat = df.select_dtypes(include = ['object','category'])
		
		nvar_cat = df_cat.shape[1]
		df_new_cat = pd.DataFrame(index=df.index)
		
		df = df.drop(df_cat.columns,axis=1)
		
		if len(is_ordinal) == 0:
			
			is_ordinal = nvar_cat * [False]
		
		for i in range(nvar_cat):
			
			x = df_cat[df_cat.columns[i]]
			
			if is_ordinal[i]:
	
				le = LabelEncoder()
				df_new_cat[df_cat.columns[i]] = le.fit_transform(x)
			
			else:
				df_cat_dummies = pd.get_dummies(x,drop_first=True,prefix='col',prefix_sep='_')
				df_new_cat = pd.concat([df_new_cat,df_cat_dummies],axis=1)
						
		df = pd.concat([df,df_new_cat],axis=1)
						
		return df
	
	def count_days(self, date1, date2):
	
		"""
		Count the number of days between two dates
		
		Parameters
		----------
		
		date1 : pandas._libs.tslib.Timestamp
			First date
		
		date 2 : pandas._libs.tslib.Timestamp
			Second date
		
		Returns
		-------
		Either an integer or a pandas Series representing the number of days between each item of date1 
		and date2.	
		"""
		
		if isinstance(date1,dt.datetime) and isinstance(date2,dt.datetime):
			n_days = date1 - date2
			
		elif isinstance(date1,pd.Series) and isinstance(date2,dt.datetime):
			n_days = date1.apply(lambda x: (x-date2).days)
			
		elif isinstance(date1,dt.datetime) and isinstance(date2,pd.Series):
			n_days = date2.apply(lambda x: (date1-x).days)
		
		elif isinstance(date1,pd.Series) and isinstance(date2,pd.Series):
			n_days = (date1 - date2).apply(lambda x: x.days)
		
		return n_days
	

	def aggreg_stats(self, x, x_by, stat):
		
		"""
		Compute the corresponding aggregated statistics 
		
		Parameters
		----------
		
		x : pandas.Series
			Series on which the statistic will be computed.
		
		x_by: pandas.Series, pandas.Index
			Used to group the x's element.
			
		stat : str
			The statistic computed on the aggregation.
			
		Returns
		-------
		Series containing the values of the aggregated statistics
		"""
		
		if stat == 'min':
			x_aggreg = x.groupby(by=x_by).min()
			
		elif stat == 'max':
			x_aggreg = x.groupby(by=x_by).max()
			
		elif stat == 'mean':
			x_aggreg = x.groupby(by=x_by).mean()
			
		elif stat == 'sum':
			x_aggreg = x.groupby(by=x_by).sum()
			
		elif stat == 'first':
			x_aggreg = x.groupby(by=x_by).first()
			
		elif stat == 'last':
			x_aggreg = x.groupby(by=x_by).last()
		
		return x_aggreg


	def binarize(self, x, thres):
		
		"""
		Binary labeling depending on a specific threshold
		
		Parameters
		----------
		
		x : pandas.Series
			Series on which one applied the labelling.
			
		thres : int, float
			The threshold allowing to label x.
		
		Returns
		-------
		Binary variable corresponding to x labeling.
		"""
		
		if isinstance(thres,list):
			thres_x = 1*(x.isin(thres))
			
		else: thres_x = 1*(x>thres)
		
		return thres_x
	
	
	def process(self):
		
		############
		### Labeling
		
		n_inactive_days   = self.count_days(last_available_date,self.df.transaction_date)
		min_inactive_days = self.aggreg_stats(n_inactive_days,self.df.index,'min')
					
		df_target = self.binarize(min_inactive_days, 90).to_frame('target')
		self.df   = self.df.merge(df_target, how='inner', left_index=True, right_index=True)		
			
		#################
		### Data Cleaning
		
		self.df = self.df.apply(self.replace_missing,axis=0)
		
		# Replace the undefined value within gender by M (the most frequent level)
		self.df['gender'] = self.replace_values(self.df.gender, r'^((?![MF]).)*$',
			                                    self.df.gender.mode()[0], True)
		
		self.df['gender'] = self.binarize(self.df.gender,['F'])
		
		self.df['profit_loss'] = self.replace_values(self.df.profit_loss, r'[0-9,]{1,}E-[0-9]{1,}$',
											     '0.0', True)
		self.df['profit_loss'] = self.df.profit_loss.astype(float)
		
		# Discretization of the date of birth variable
		self.df['year_of_birth'] = self.discretization(self.df.year_of_birth,[1926,1959,1984,2001])
		
		# Binarize acquisition channel id splitting frequent channel from the unfrequent
		self.df['acquisition_channel_id'] = self.binarize(self.df.acquisition_channel_id,
										 ['10','11','12','14','16','17','20'])
		
		# Handle the categorical variable encoding
		self.df = self.categorical_encoding(self.df, [True])
		
		
		#######################
		### Feature Engineering
		
		# Statistics are computed until the last observable date
		self.df = self.df[self.df.transaction_date<=last_observable_date]
		
		# Customer fidelity
		ndays_registration = self.count_days(last_observable_date,
						   self.aggreg_stats(self.df.registration_date,
						   self.df.index,'first'))
		
		df_customer_fidelity = self.discretization(ndays_registration,[-1,365,365*2,365*3]
										       ).rename('customer_fidelity').to_frame()
		
		self.df = self.df.merge(df_customer_fidelity, how='inner', 
							  left_index = True, right_index = True)
		
		self.df = self.categorical_encoding(self.df,is_ordinal = [True])
		
		self.df = self.df.drop('registration_date',axis=1)
		
		# Total Bet Number per customer
		self.df['total_bet_nb'] = self.aggreg_stats(self.df.bet_nb,self.df.index,'sum')
		# Total Deposit Number per customer
		self.df['total_deposit_nb'] = self.aggreg_stats(self.df.deposit_nb,self.df.index,'sum')
		# Total profit/loss per customer
		self.df['total_profit_loss'] = self.aggreg_stats(self.df.profit_loss,self.df.index,'sum')	
	    
		# Mean Bet Amount per customer (remove deposit rows)
		self.df['mean_bet_amount'] = self.aggreg_stats(self.df[self.df.bet_nb!=0].bet_amount,
											       self.df[self.df.bet_nb!=0].index,'mean')
		
		self.df['mean_bet_amount'] = self.df.mean_bet_amount.replace(np.nan,0)
		
		# Mean Deposit Amount per customer (remove bet rows)
		self.df['mean_deposit_amount'] = self.aggreg_stats(
									  self.df[self.df.deposit_nb!=0].deposit_amount,
									  self.df[self.df.deposit_nb!=0].index,'mean')
		
		self.df['mean_deposit_amount'] = self.df.mean_deposit_amount.replace(np.nan,0)
		
		self.df = self.df.drop(['bet_nb','bet_amount','profit_loss','deposit_nb','deposit_amount'],
							   axis=1)
		
		# Number of days since the last transaction
		self.df = self.df.reset_index().sort_values(by=['customer_key','transaction_date']
		                                              ).set_index(['customer_key'])
		
		df_ndays_last_transaction = self.aggreg_stats(self.count_days(last_observable_date,
								 self.df.transaction_date),self.df.index,'last'
								 ).to_frame(name='ndays_last_transaction')
		
		self.df = pd.concat([self.df,df_ndays_last_transaction],axis=1)
		
		# Transactions frequency			
		transaction_date_shift = self.df.transaction_date.shift()
		customer_key_shift = [self.df.index[-1]]+list(self.df.index[:-1])
				
		ndays_between_transactions = self.count_days(self.df.transaction_date,
									         transaction_date_shift)
		
		# Number of days between two transactions
		self.df['interval_transactions'] = [ndays_between_transactions.iloc[i] if 
	                                         self.df.index[i] == customer_key_shift[i] else
	                                         np.nan for i in range(self.df.shape[0])]
	
		df_freq_transactions = self.aggreg_stats(self.df.interval_transactions.dropna(),
							 self.df[pd.notnull(self.df.interval_transactions)].index,
		                       'mean').to_frame(name='frequency_transactions')
		
		self.df = pd.concat([self.df,df_freq_transactions],axis=1)
		
		# Missing values in frequency transactions = customer with only one transaction
		self.df['frequency_transactions'] = self.replace_missing(self.df.frequency_transactions,
														    self.df.ndays_last_transaction)
										  		
		self.df = self.df.drop(['transaction_date','interval_transactions'],axis=1)
		
		# Drop duplicates (key = customer key)
		self.df = self.df.loc[~self.df.index.duplicated(keep='first')]
		
		# Features / Target
		y = self.df[['target']]
		X = self.df.drop('target',axis=1)
		
		return X, y


####################
### Cross Validation

def cross_validation(X, y, test_size, rnd_state):

	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,
												   random_state=rnd_state)
	
	return X_train, X_test, y_train, y_test


############
### Training

class training_model:
	
	def __init__(self,target,features,tuned_params,cv):
		
		self.target       = target
		self.features     = features
		self.tuned_params = tuned_params
		self.cv           = cv
		
		self.features = np.array(self.features)
		self.target   = np.array(self.target).flatten()


	def gridSearch(self):
		
		estimator = MLPClassifier()
							
		gs_cv = GridSearchCV(cv=self.cv, estimator=estimator, param_grid=self.tuned_params)									
		gs_cv_fit = gs_cv.fit(self.features, self.target)
		
		best_params = gs_cv_fit.best_params_
		
		return best_params
	
	
	def fit(self):
		
		best_params = self.gridSearch()
		
		ml_model = MLPClassifier(hidden_layer_sizes=best_params['hidden_layer_sizes'],
							   activation=best_params['activation'])
		
		ml_model_fit = ml_model.fit(self.features, self.target)
		
		return ml_model_fit
		
				

def main(f_dir):
	
	
	#################
	### Data Recovery					
	
	# Column names
	col_names = ['customer_key', 'registration_date', 'year_of_birth', 'gender', 
	             'acquisition_channel_id', 'transaction_date', 'bet_nb', 'bet_amount', 'profit_loss',
			    'deposit_nb', 'deposit_amount', 'betclic_customer_segmentation']	
	
	# Reading the data from f_dir
	df_churn = read_csv_file(f_dir,';',col_names,0,col_names[:-1],0,
						  ['registration_date','transaction_date'],
						  {'year_of_birth': object, 'acquisition_channel_id': object})	
	
	######################
	### Data preprocessing
	
	data_process = data_preprocessing(df_churn)
	X, y = data_process.process()							
	
	####################
	### Cross Validation
	
	X_train, X_test, y_train, y_test = cross_validation(X, y, 0.3, 123)
	
	############
	### Training	
		
	# Multilayer Perceptron
	mlp_training = training_model(y_train, X_train, [
			{'hidden_layer_sizes':[(14,7),(10,5),(8,)],'activation':['tanh','logistic']}], 4)
	mlp_fit = mlp_training.fit()
		
	return mlp_fit
										

