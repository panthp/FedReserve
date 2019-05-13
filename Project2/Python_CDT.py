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

def main():
#Read in data
	data1 = pd.read_csv('Monthly_data.csv', header=0, delimiter =' ')
	print(data1.shape)
	cols = list(data1.columns.values)
	print(cols)
	#pdb.set_trace()
	#test = cdt.causality.graph.PC()#.create_graph_from_data(data1)
	#test.create_graph_from_data(data1)
	#pdb.set_trace()
	#test.create_graph_from_data(data1)

	#testing_GES = cdt.causality.graph.GES()
	#testing_GES.create_graph_from_data(data1)
	training_data = data1
	hmm = GaussianHMM(n_components = 7, covariance_type = 'diag', n_iter = 1000)
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		hmm.fit(training_data)





if __name__ == '__main__' :
	main()
