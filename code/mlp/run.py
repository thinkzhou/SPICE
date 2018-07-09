# -*- coding: utf-8 -*-
import os
import sys

import numpy
import word2vec
import mxnet as mx
import impl
from impl import get_model

from sklearn import svm
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier

from trainvec import load_obj 
from trainvec import save_obj
from trainvec import load_data
from trainvec import get_vector
from trainvec import get_gaussian
from trainvec import line_to_vector_stack

def next_symbols_ranking(model,model_type, prefix, gauss_filter, vect_dict, vect_dim):
    '''
    '''
    nkey = len(gauss_filter)
    prefix = prefix.split()
    prefix = [key.strip() for key in prefix][1:]
    if len(prefix) <= nkey:
    	prefix = ['-1'] * (nkey - len(prefix)) + prefix
    else:
    	prefix = prefix[len(prefix) - nkey :]

    A = line_to_vector_stack(vect_dict, gauss_filter, prefix, vect_dim)
    A = numpy.array([A]).astype('float32')
    if (model_type == 'cnn'):
        A = A.reshape((-1,1,15,vect_dim))

    A = mx.io.NDArrayIter(data = A)

    T = model.predict(X = A)
    T = T.ravel()

    I = sorted(range(len(T)), key=lambda k: T[k], reverse=True)
    I = I[:5]
    I = [-1 if idx == len(T) - 1 else idx for idx in I] 

    return I

def run_offline(model, model_type, ranking_file, vect_dict, vect_dim, prefix_file, problem_id, k, std):
	'''
	'''
	delimiter = '%20'
	gauss_filter = get_gaussian(k, std)
	print('Load prefix from '+ prefix_file)
	f_ranking = open(ranking_file,'w+')
	with open(prefix_file) as f:
		cnt = 0
		for line in f.readlines():
			cnt = cnt + 1
			if (cnt!=1):
				prefix = line
				#print(prefix)
				ranking = next_symbols_ranking(model,
                        model_type,prefix, gauss_filter, vect_dict, vect_dim)
				for rank in ranking:
					f_ranking.write(str(rank)+' ')
				f_ranking.write('\n')
	f_ranking.close()
	print('Save ranking to ' +ranking_file)

def starter(root, read_dir, model_dir, word_vec_dir, save_vec_dir, vect_dim, k, std):
	'''
	'''
	istart = 4
	n_problem = 5
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

		with open(refer_file, 'r') as f:
			# # of samples and # of symbols
			first_line = f.readline()
			n_symbol   = int(first_line.split()[1])
		vector_dict = get_vector(root, problem_id, word_vec_dir, vect_dim, n_symbol)
		
		# learn the model
		print ("Start Learning...")
		model,model_type = get_model(train_file, model_file, ndim=vect_dim*k, nout=n_symbol+1, islearning=False)
		print ("Learning Ended...")
		run_offline(model,model_type,
            ranking_file, 
			vector_dict, 
			vect_dim, 
			prefix_file, 
			problem_id, 
			k, std)
if __name__ == '__main__':
	'''
	'''
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
	model_dir    = save_vec_dir + '_mlp_2'

	starter(root, read_dir, model_dir, word_vec_dir, save_vec_dir, vect_dim, k, std)	
