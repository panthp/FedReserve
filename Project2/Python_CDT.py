#!/usr/bin/env python

import numpy as np
import sys
import csv
import pandas as pd
import pdb
import matplotlib.pyplot as plt
#import pystan
#import cdt
from hmmlearn.hmm import GaussianHMM
import warnings
#import pgmpy
from pgmpy.estimators import ExhaustiveSearch
from pgmpy.estimators import ConstraintBasedEstimator
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BicScore
from pgmpy.models import BayesianModel
from pgmpy.inference import BeliefPropagation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pomegranate import BayesianNetwork
import pickle

def PCA_HMM(training_data):
	#PCA 
	pca = PCA(n_components=4)
	pca.fit(training_data)
	# pca = PCA().fit(training_data)
	# plt.plot(np.cumsum(pca.explained_variance_ratio_))
	# plt.xlabel('number of components')
	# plt.ylabel('cumulative explained variance');
	# plt.show()
	training_data_pca = pca.transform(training_data)
	print("original shape:   ", training_data.shape)
	print("transformed shape:", training_data_pca.shape)
	pdb.set_trace()
	training_data_ext = training_data_pca


def main():
	#Read in data
	data1 = pd.read_csv('Processed_Data_PC.csv', header=0, delimiter =',')
	data1_ext = pd.read_csv('Processed_Data_PC_test6.csv', header=0, delimiter =',')
	data1_bayes = pd.read_csv('Processed_Data_PC_bayes.csv', header=None, delimiter =',')

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
	training_data_ext = data1_ext.drop('DATE',axis=1)
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


	#PCA 
	pca = PCA(n_components=4)
	pca.fit(training_data)
	# pca = PCA().fit(training_data)
	# plt.plot(np.cumsum(pca.explained_variance_ratio_))
	# plt.xlabel('number of components')
	# plt.ylabel('cumulative explained variance');
	# plt.show()
	training_data_pca = pca.transform(training_data)
	print("original shape:   ", training_data.shape)
	print("transformed shape:", training_data_pca.shape)
	#pdb.set_trace()
	#training_data_ext_pca = training_data_pca
	training_data_ext_pca = pd.DataFrame(data = training_data_pca, columns=training_data.columns.values[:4])

	####pgmpy Structure and Inference####
	try:
		#skeleton, 
		bayesian_model = pickle.load(open("BayesModel.pickle", "rb"))
		#print("SUCCESS")
		#print("Part 1) Model:    ", best_model.edges())
	except (OSError, IOError) as e:
		#print("Model Creating")
		#pickle.dump(foo, open("var.pickle", "wb"))
		
		# est = ConstraintBasedEstimator(training_data) ####TRAINING DATA CHANGE####
		# skeleton, sep_sets = est.estimate_skeleton()
		# print("Part 1) Skeleton: ", skeleton.edges())

		# hc = HillClimbSearch(training_data_ext)#, scoring_method=BicScore(training_data))
		# best_model = hc.estimate()#tabu=10, white_list=skeleton.to_directed().edges())
		# with open('HCModel.pickle', 'wb') as f:
  		#  			pickle.dump([best_model], f)#skeleton,best_model], f)

		# print("Part 1) Model:    ", best_model.edges())

		# with open('HCModel.pickle', 'wb') as f:
  		#  			pickle.dump([best_model], f)#skeleton,best_model], f)
		
		data_Bayesian_model = data1.drop(['DATE','AvgHEar_PC1', 'Unemp_Gap'], axis=1)
		#data_Bayesian_model.drop(0)
		data1_bayes = data1_bayes.round(3)
		msk = np.random.rand(len(data1_bayes)) < 0.5
		training_data_bayes = data1_bayes[msk]
		testing_data_bayes = data1_bayes[~msk]
		testing_data_bayes[8].replace(0, np.nan,inplace=True)
		print(training_data_bayes.head(10))
		print(testing_data_bayes.head(10))
		#print("HERE2")
		#model = BayesianNetwork.from_samples(training_data_bayes, algorithm='chow-liu', state_names = data_Bayesian_model.columns.values)
		model = BayesianNetwork.from_samples(data1_bayes, algorithm='chow-liu', state_names = data_Bayesian_model.columns.values)
		#print("HERE3")
		print(model.structure)
		#print("HERE4")
		#model.plot()
		print(data1_bayes.head(10))
		model.predict_proba([[1.45, 13.7, 1.118, None, -0.011, -0.113, 0.009, 0.111, None]])#, [0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, None]])
		print(model.predict_proba([[1.45, 13.7, 1.118, None, -0.011, -0.113, 0.009, 0.111, None]]))
		print(model.predict([[1.45, 13.7, 1.118, None, -0.011, -0.113, 0.009, 0.111, None]]))
		#model.predict_proba([[1.45, 13.7, 1.118, None, -0.011, -0.113, 0.009, 0.111, None],[1.45, 13.7, 1.118, None, -0.011, -0.113, 0.009, 0.111, None], [1.45, 13.7, 1.118, None, -0.011, -0.113, 0.009, 0.111, None]])#, [0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, None]])
		#model.predict([[1.45, 13.7, 1.118, None, -0.011, -0.113, 0.009, 0.111, None],[1.45, 13.7, 1.118, None, -0.011, -0.113, 0.009, 0.111, None], [1.45, 13.7, 1.118, None, -0.011, -0.113, 0.009, 0.111, None]])#, [0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, None]])
		query_array = np.asarray([[1.45, 13.7, 1.118, None, -0.011, -0.113, 0.009, 0.111, None],[2.00, 17.0, 3.822, -0.121, 0.000, 0.166, 0.026, 0.195, None], [2.33,  15.8,  6.694, -0.204,  0.021,  0.221,  0.050,  0.231,  None]])
		answer_array = np.asarray(model.predict(query_array))
		print("Answered Query:", answer_array)
		#model.predict_proba({'USREC': 1})
		pdb.set_trace()

	   	#Create Model
	   	# AvgHEar_PC1	FEDFUNDS	Dur_UnEmp	Gross_Dom_Income_PC1	Unemp_Gap	Unemp_Ins_Claims_PC1	PAYEMS_PC1	PERMIT_PC1	Consumer_Expend_PC1	SP_Close_PC1	USREC
	

		data_Bayesian_model = data1.drop(['DATE','AvgHEar_PC1', 'Unemp_Gap'], axis=1)
		print(data_Bayesian_model.columns.values)
		bayesian_model = BayesianModel([('PERMIT_PC1', 'SP_Close_PC1'), ('PERMIT_PC1', 'USREC'), ('SP_Close_PC1', 'USREC'), ('Unemp_Ins_Claims_PC1', 'USREC'), ('USREC', 'PAYEMS_PC1'), ('USREC', 'Consumer_Expend_PC1'), ('Consumer_Expend_PC1', 'PAYEMS_PC1'), ('PAYEMS_PC1', 'FEDFUNDS'), ('PAYEMS_PC1', 'Dur_UnEmp'), ('Consumer_Expend_PC1', 'FEDFUNDS'), ('Consumer_Expend_PC1', 'Dur_UnEmp'), ('Consumer_Expend_PC1', 'PAYEMS_PC1')])
		bayesian_model.fit(data_Bayesian_model)
		#bayesian_model.edges()
		with open('BayesModel.pickle', 'wb') as f:
		 			pickle.dump([bayesian_model], f)#skeleton,best_model], f)

		# pdb.set_trace()
		# belief_propagation = BeliefPropagation(bayesian_model)
		# belief_propagation.map_query(variables=['USREC'], evidence={'A': 0, 'R': 0, 'G': 0, 'L': 1})


   	##Inference on Model##
	# model = BayesianModel(edges_list)
	# for node in nodes:
	# 	model.node[node] = nodes[node]
	# for edge in edges:
	# 	model.edge[edge] = edges[edge]

if __name__ == '__main__' :
	main()
