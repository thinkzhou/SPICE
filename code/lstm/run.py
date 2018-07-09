# -*- coding: utf-8 -*-
import os
import sys

import numpy
import word2vec
import argparse

from sklearn.utils import shuffle

from trainvec import load_obj 
from trainvec import save_obj
from trainvec import load_data
from trainvec import get_vector
from trainvec import get_gaussian
from trainvec import line_to_vector_stack
from impl import parse_args
import models
from tensorflow.contrib import learn

def symbol_vec(prefix, gauss_filter, vect_dict, vect_dim):
	nkey = len(gauss_filter)
	prefix = prefix.split()
	prefix = [key.strip() for key in prefix][1:]
	if len(prefix) <= nkey:
		prefix = ['-1'] * (nkey - len(prefix)) + prefix
	else:
		prefix = prefix[len(prefix) - nkey :]

	A = line_to_vector_stack(vect_dict, gauss_filter, prefix, vect_dim)
	A = numpy.array([A]).astype('float32')	
	return A

def run(model, ranking_file, vect_dict, vect_dim, prefix_file, k, std):
	gauss_filter = get_gaussian(k, std)
	print('Load prefix from '+ prefix_file)
	with open(prefix_file) as f:
		cnt = 0
		for line in f.readlines():
			cnt = cnt + 1
			if (cnt == 1):
				n_samples, nout = line.split()
				n_samples,nout = int(n_samples), int(nout)
				test_data = numpy.zeros((n_samples, vect_dim*k), dtype=numpy.float32)
			if (cnt!=1):
				prefix = line
				vec = symbol_vec(prefix, gauss_filter, vect_dict, vect_dim)
				test_data[cnt-2] = vec

	test_data = test_data.reshape((-1, k, vect_dim))
	pred = model.predict(test_data)
	I = pred.argsort()[:,::-1][:,0:5]
	I[I==nout] = -1 
	
	with open(ranking_file,'w+') as f:
		for i in range(0, I.shape[0]):
			for j in range(0, I.shape[1]):
				rank = I[i][j]
				f.write(str(rank)+ ' ')
			f.write('\n')
	
	print('Finished.')
def starter(root, read_dir, model_dir, word_vec_dir, save_vec_dir, vect_dim, k, std):

	istart = 1
	n_problem = 16
	model_dir = root + model_dir
	for i in range(istart, n_problem):
		# State the problem number
		problem_id = str(i)

		if not os.path.exists(model_dir):
			os.makedirs(model_dir)

		model_file  = model_dir + '/' + problem_id + '.model'
		ranking_file = model_dir + '/' + problem_id + '.ranking'
		train_file  = root + save_vec_dir + '/' + problem_id + '.tr.npy'
		refer_file  = root + read_dir + '/' + problem_id + '.spice.train'
		prefix_file = root + 'SPiCe_Offline/prefixes/' + problem_id + '.spice.prefix.public'

		args = parse_args(train_file, model_file, val_portion=0.3)
		print ('batch_size: '+str(args.batch_size))
		print ('lr: '+ str(args.lr))

		with open(refer_file, 'r') as f:
			# # of samples and # of symbols
			first_line = f.readline()
			n_symbol   = int(first_line.split()[1])
		vector_dict = get_vector(root, problem_id, word_vec_dir, vect_dim, n_symbol)
		n_classes = n_symbol + 1
		model = models.LSTM()
		# learn the model
		print ("Start Learning...")
		model.train(args, n_classes, model_file)
		print ("Learning Ended...")
		
		run(model, ranking_file, vector_dict, vect_dim, 
				prefix_file, k, std)

if __name__ == '__main__':
	root = '/str/users/angus/Codes/yang/'
	vect_dim = 30
	read_dir = 'data'
	text_dir = 'text'
	k, std   = 15, 2

	try:
		vect_dim = int(sys.argv[1])
	except Exception as e:
		print('Params: ', sys.argv)
		pass
	print('Using word vector of dimension {:d}'.format(vect_dim))


	word_vec_dir = 'word_vec_' + str(vect_dim)
	save_vec_dir = 'tr_b_' + str(vect_dim) + '_' + str(k) + '_' + str(std)
	model_dir    = save_vec_dir + '_lstm_test'

	starter(root, read_dir, model_dir, word_vec_dir, save_vec_dir, vect_dim, k, std)	
