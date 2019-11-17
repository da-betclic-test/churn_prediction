#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:18:40 2019

@author: davidazoulay
"""


###############
### Modules ###
###############

import pandas as pd
import numpy as np
import datetime as dt

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


# Macro variables

last_available_date  = dt.datetime(2019,7,23)
last_observable_date = last_available_date - dt.timedelta(days=90)

current_year = last_available_date.year
minor_thres_year = current_year - 18          

# Column names
col_names = ['customer_key', 'registration_date', 'year_of_birth', 'gender', 
             'acquisition_channel_id', 'transaction_date', 'bet_nb', 'bet_amount', 
             'profit_loss', 'deposit_nb', 'deposit_amount', 'betclic_customer_segmentation']
	

##################################
### Data preprocessing methods ###
##################################


class DataPreprocessing:
	
	""" Data Preprocessing
	
	This class handle both the cleaning and the feature engineering of the input dataframe.
	
	Parameters
	----------
	
	df : dataframe
		The dataframe on which the preprocessing will be applied to.
		
	"""
	
	def __init__(self,df):
		
		self.df = df		

	def replace_missing(self, x, y=None):
	
		"""
		Replace the missing values within the DataFrame.
		
		Parameters
		----------
		x : pandas.Series
			Series with potential missing values.
		
		y: pandas.Series
			Value used in order to fill the missing values in x.
		
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
		Replace a specific value within a Series by an other one.
		
		Parameters
		----------
		x : pandas.Series
		
		old_val : int, float, str
			Value to be replaced
		
		new_val : int, float, str
			Replacement value
		
		reg_expr: boolean
			Specify if whether old_val is a regular expression or not.
			 
		"""
		
		if reg_expr:
			x = x.replace(old_val, new_val, regex = reg_expr)
		else: x = x.replace(old_val, new_val)
		
		return x
	
	
	def discretization(self, x, bins):
		
		"""
		Discretization of continuous variables.
		
		Parameters
		----------
		
		x : pandas.Series
			Variable to discretized.
		
		bins : list
			Defines the bin edges.
		
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
		Encode the categorical variables.
		
		Parameters
		----------
		df : pandas.DataFrame
		
		is_ordinal : list of boolean
			Specify if whether or not the categorical variable is ordinal
		
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
		Count the number of days between two dates.
		
		Parameters
		----------
		
		date1 : pandas._libs.tslib.Timestamp
			First date
		
		date 2 : pandas._libs.tslib.Timestamp
			Second date
		
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
		Compute the corresponding aggregated statistics.
		
		Parameters
		----------
		
		x : pandas.Series
			Series on which the statistic will be computed.
		
		x_by: pandas.Series, pandas.Index
			Used to group the x's element.
			
		stat : str
			The statistic computed on the aggregation.

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
		Binary labeling depending on a specific threshold.
		
		Parameters
		----------
		
		x : pandas.Series
			Series on which one applied the labelling.
			
		thres : int, float
			The threshold allowing to label x.
		
		"""
		
		if isinstance(thres,list):
			thres_x = 1*(x.isin(thres))
			
		else: thres_x = 1*(x>thres)
		
		return thres_x
	
	
	def process(self):
		
		
		############
		### Labeling
		
		n_inactive_days = self.count_days(last_available_date, self.aggreg_stats(
						self.df.transaction_date, self.df.index, 'last'))
					
		df_target = self.binarize(n_inactive_days, 90).to_frame('target')
		self.df   = self.df.merge(df_target, how='inner', left_index=True, right_index=True)		
			
		
		#################
		### Data Cleaning
		
		# Replacement of all the missing values in the dataframe
		self.df = self.df.apply(self.replace_missing,axis=0)
		
		# Replace values other than M or F in gender by M (the most frequent level)
		self.df['gender'] = self.replace_values(self.df.gender, r'^((?![MF]).)*$',
                                                 self.df.gender.mode()[0], True)		
		self.df['gender'] = self.binarize(self.df.gender,['F'])
		
		# Repace str values containing E- in profit_loss column by a 0 value
		self.df['profit_loss'] = self.replace_values(self.df.profit_loss, r'[0-9,]{1,}E-[0-9]{1,}$',
											     '0.0', True)
		self.df['profit_loss'] = self.df.profit_loss.astype(float)
		
		
		#######################
		### Feature Engineering
		
		
		### Categorical Variable
		
		# Year of birth discretization
		self.df['year_of_birth'] = self.discretization(
								self.df.year_of_birth,[1900,minor_thres_year-42,
							                           minor_thres_year-17,minor_thres_year])
			
		# Binarize acquisition channel id splitting frequent channel from the unfrequent
		self.df['acquisition_channel_id'] = self.binarize(self.df.acquisition_channel_id,
										['10','11','12','14','16','17','20'])		
										
		# Customer fidelity: number of years since registration
		self.df['customer_fidelity'] = self.count_days(last_observable_date,
		                                               self.df.registration_date)				
		self.df['customer_fidelity'] = self.discretization(self.df.customer_fidelity,
													  [-100,365,365*2,365*3])
				
		# Handle the categorical variable
		self.df = self.categorical_encoding(self.df, is_ordinal = [True,True])
		
		
		### Datetime variable
		
		# Following variables are computed until the last observable date
		self.df = self.df[self.df.transaction_date<=last_observable_date]
		
		# Number of days since the last transaction				
		df_ndays_last_transaction = self.aggreg_stats(self.count_days(last_observable_date,
								  self.df.transaction_date),self.df.index,'last'
								  ).to_frame(name='ndays_last_transaction')
		
		self.df = self.df.merge(df_ndays_last_transaction, how='inner', 
						      left_index=True, right_index=True)
		
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
							 self.df.interval_transactions.dropna().index,
		                       'mean').to_frame(name='frequency_transactions')
		
		self.df = self.df.merge(df_freq_transactions,how='left',left_index=True,right_index=True)
		
		# Missing values in frequency transactions = customer with only one transaction
		self.df['frequency_transactions'] = self.replace_missing(self.df.frequency_transactions,
														    self.df.ndays_last_transaction)
														
		
		### Continuous variable		
		
		# Total Bet Number per customer
		self.df['total_bet_nb'] = self.aggreg_stats(self.df.bet_nb,self.df.index,'sum')
		# Total Deposit Number per customer
		self.df['total_deposit_nb'] = self.aggreg_stats(self.df.deposit_nb,self.df.index,'sum')	
	    
		# Mean Bet Amount per customer (remove deposit rows)
		self.df['mean_bet_amount'] = self.aggreg_stats(self.df[self.df.bet_nb!=0].bet_amount,
											       self.df[self.df.bet_nb!=0].index,'mean')		
		self.df['mean_bet_amount'] = self.df.mean_bet_amount.replace(np.nan,0)							
		
		
		### Dropping useless rows and columns
										  		
		self.df = self.df.drop(['registration_date','transaction_date','interval_transactions',
                                 'bet_nb','bet_amount','profit_loss','deposit_nb','deposit_amount',
							  'gender'],axis=1)						
		
		self.df = self.df.loc[~self.df.index.duplicated(keep='first')]
		
		### Features / Target
		y = self.df[['target']]
		X = self.df.drop('target',axis=1)
		
		return X, y


############
### Training

class training_model:
	
	""" Logistic Regression model training
	
	Parameters
	----------
	
	target : array-like
		The dependant variable.
	
	features : matrix, dataFrame
		The explanatory variables.
	"""
	
	def __init__(self,target,features):
		
		self.target       = target
		self.features     = features
		
		self.features = np.array(self.features)
		self.target   = np.array(self.target).flatten()

	def fit(self):

		model = LogisticRegression(solver='liblinear')
		model_fit = model.fit(self.features, self.target)
		
		return model_fit					
				

def main(f_name):
		

	################
	### Data Reading
	
	dateParse = lambda x : dt.datetime.strptime(x, '%d/%m/%y')	
	
	# Reading the training dataset
	df_churn = pd.read_csv(f_name, sep = ';', names = col_names, header = 0,
						 usecols = col_names[:-1], index_col = 0, 
                            parse_dates = ['registration_date','transaction_date'], 
					      date_parser = dateParse, na_values = 'NaT', 
						 dtype = {'year_of_birth': object, 'acquisition_channel_id': object})

	# Sorting by customer key and tranasction date
	df_churn = df_churn.reset_index().sort_values(
			  by=['customer_key','transaction_date']).set_index(['customer_key'])	
	
	######################
	### Data preprocessing
	
	data_process = DataPreprocessing(df_churn)
	X, y = data_process.process()

	X_train = X.sample(frac=0.7,random_state=123)
	y_train = y.sample(frac=0.7,random_state=123)								
	
	############
	### Training	
		
	# Multilayer Perceptron
	mlp_training   = training_model(y_train, X_train)
	mlp_classifier = mlp_training.fit()
	
	joblib.dump(mlp_classifier, 'classifier.joblib')
		
	return mlp_classifier
										
