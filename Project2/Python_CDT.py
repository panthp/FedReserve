#!/usr/bin/env python

import numpy as np
import sys
import csv
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import pystan
import cdt
from hmmlearn.hmm import GaussianHMM
import warnings

def main():
#Read in data
	data1 = pd.read_csv('Processed_Data_PC.csv', header=0, delimiter =',')
	#print(data1.shape)
	cols = list(data1.columns.values)
	#print(cols)
	#pdb.set_trace()
	#test = cdt.causality.graph.PC()#.create_graph_from_data(data1)
	#test.create_graph_from_data(data1)
	#pdb.set_trace()
	#test.create_graph_from_data(data1)

	#testing_GES = cdt.causality.graph.GES()
	#testing_GES.create_graph_from_data(data1)
	for col in data1.columns: 
		print(col)

	training_data = data1.drop('DATE',axis=1)
	training_data_nonrec = training_data.loc[training_data['USREC'] == 0]
	training_data_rec = training_data.loc[training_data['USREC'] == 1]
	training_data_nonrec = training_data_nonrec.drop('USREC', axis=1)
	training_data_rec = training_data_rec.drop('USREC', axis=1)

	#print(training_data_nonrec.head(10))
	hmm_nonrecession = GaussianHMM(n_components = 7, covariance_type = 'diag', n_iter = 1000)
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		hmm_nonrecession.fit(training_data_nonrec)

	hmm_recession = GaussianHMM(n_components = 7, covariance_type = 'diag', n_iter = 1000)
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		hmm_recession.fit(training_data_rec)

	hidden_states_nonrec = hmm_nonrecession.score(training_data_nonrec)
	hidden_states_rec = hmm_recession.score(training_data_rec)
	hidden_states_nonrec2 = hmm_nonrecession.score(training_data_rec)
	hidden_states_rec2 = hmm_recession.score(training_data_nonrec)

	hidden_states_rec_current12 = hmm_recession.score(training_data_nonrec.tail(12))
	hidden_states_nonrec_current12 = hmm_nonrecession.score(training_data_nonrec.tail(12))\

	hidden_states_rec_current1 = hmm_recession.score(training_data_nonrec.tail(1))
	hidden_states_nonrec_current1 = hmm_nonrecession.score(training_data_nonrec.tail(1))

	print(hidden_states_rec)
	print(hidden_states_nonrec)
	print(hidden_states_rec2)
	print(hidden_states_nonrec2)
	print("Current Recession likelihood:", hidden_states_rec_current12)
	print("Current Nonrecession likelihood:", hidden_states_nonrec_current12)

	print("Current Recession likelihood:", hidden_states_rec_current1)
	print("Current Nonrecession likelihood:", hidden_states_nonrec_current1)






if __name__ == '__main__' :
	main()
