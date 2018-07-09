# -*- coding: utf-8 -*-
import os
import numpy 
import word2vec
import six.moves.cPickle as pickle

from scipy import signal


def save_obj(obj_file, data):
	'''
	'''
	with open(obj_file + '.pkl', 'wb') as f:
		pickle.dump(data, f, protocol=-1)


def load_obj(obj_file):
	'''
	'''
	with open(obj_file + '.pkl', 'rb') as f:
		return pickle.load(f)


def get_gaussian(k=5, std=2):
	'''
	'''
	filter = signal.get_window(('gaussian', std), 2 * k - 1)
	return filter[:k]


def save_data(insts, labels, save_file, k=5, std=2):
	'''
	Save to the binary data.
	'''
	insts = numpy.array(insts)
	labels = numpy.array([labels]).transpose()

	data = numpy.hstack((insts, labels))
	numpy.save(save_file, data)
	# data = numpy.load(save_file + '.npy')
	# numpy.savetxt(save_file, data, delimiter=',')
	print('(', insts.shape, labels.shape, ')')
	print('Training data is saved to: ', save_file + '.npy')


def load_data(root, index, read_dir, read_file=None):
	'''
	Load the binary data.
	'''
	if not read_file:
		read_file = root + read_dir + '/' + str(index) + '.tr.npy'
	data = numpy.load(read_file)
	ncol = data.shape[1]
	x = data[:, 0 : ncol - 1]
	y = data[:, ncol - 1]
	print('Loading training data: ', read_file, x.shape)
	return x, y


def train_vector(root, index, vect_dim, read_dir, save_dir, debug):
	'''
	'''
	read_dir = root + read_dir
	save_dir = root + save_dir
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	read_file = read_dir + '/' + str(index) + '.spice.train'
	save_file = save_dir + '/' + str(index) + '.bin'

	print('Training words to vectors: ', read_file, '->', save_file)

	word2vec.word2vec(read_file, save_file, size = vect_dim, min_count = 1, verbose = debug)


def get_vector(root, index, read_dir, vect_dim, n_symbol, debug = False):
	'''
	'''
	read_file = root + read_dir + '/' + str(index) + '.bin'
	model = word2vec.load(read_file)

	vector_dict = {}
	for vocab, vector in zip(model.vocab, model.vectors):
		vector_dict.update({vocab : vector})

	numpy.random.seed(111)
	for idx in range(-1, n_symbol):
		try:
			vector_dict[str(idx)]
		except:
			print('Not find key: ', str(idx), ', randomly initialize it.')
			vector_dict[str(idx)] = numpy.random.randn(vect_dim) * 0.01

	vector_dict.update({'-1' : numpy.array([0] * vect_dim)})
	print('Loading word vectors: ', read_file, len(model.vocab))

	return vector_dict


def line_to_vector_stack(vector_dict, gauss_filter, line, vect_dim):
	'''
	Concatenate vectors.
	'''
	vector = []
	for key, scalar in zip(line, gauss_filter):
		try:
			# vector = numpy.hstack((vector, vector_dict[key] * scalar))
			vector = numpy.hstack((vector, vector_dict[key]))
		except Exception as e:
			print(vector_dict[key])
	return vector 


def line_to_vector_pool(vector_dict, gauss_filter, line, vect_dim):
	'''
	Weighted pool vectors.
	'''
	vector = [0] * vect_dim
	gauss_filter = get_gaussian(len(line), 2.5)
	for key, scalar in zip(line, gauss_filter):
		try:
			vector += vector_dict[key] * scalar
		except Exception as e:
			print(vector_dict[key])
	return vector 


def generate_insts_lstm(vector_dict, gauss_filter, line, vect_dim, k=5):
	'''
	Generate data for LSTM model.
	'''
	insts  = []
	labels = []
	line = [-1] + line + [len(vector_dict) - 2]
	for idx in range(1, len(line)):
		label   = 0
		subline = line[0 : idx]
		vector  = [int(key) for key in subline]
		try:
			label = int(line[idx])
		except Exception as e:
			# '-1' and '</s>'
			label = len(vector_dict) - 2
		insts.append(vector)
		labels.append(label)
		# print(label, vector)
	return insts, labels


def generate_insts_all(vector_dict, gauss_filter, line, vect_dim, k=5):
	'''
	Consider all symbols before the current one.
	'''
	insts  = []
	labels = []
	line = ['-1'] * k + line + ['</s>']
	for idx in range(k, len(line)):
		label   = 0
		subline = line[0 : idx]
		vector  = line_to_vector_pool(vector_dict, gauss_filter, subline, vect_dim)
		try:
			label = int(line[idx])
		except Exception as e:
			# '-1' and '</s>'
			label = len(vector_dict) - 2
		insts.append(vector)
		labels.append(label)
	return insts, labels 


def generate_insts_prek(vector_dict, gauss_filter, line, vect_dim, k=5):
	'''
	Consider only previous k symbols.
	'''
	insts  = []
	labels = []
	line = ['-1'] * k + line + ['</s>']
	for idx in range(len(line) - k):
		label   = 0
		subline = line[idx : idx + k]
		vector  = line_to_vector_stack(vector_dict, gauss_filter, subline, vect_dim)
		try:
			label = int(line[idx + k])
		except Exception as e:
			# '</s>'
			label = len(vector_dict) - 2
		insts.append(vector)
		labels.append(label)
		# print(label, subline)
	# print(len(insts), len(insts[0]))
	return insts, labels


def transform_data(
		root, 
		index, 
		vect_dim, 
		read_dir, 
		word_vec_dir, 
		save_vec_dir, 
		save_obj_dir='', debug=False, k=5, std=2):
	'''
	'''
	read_file    = root + read_dir + '/' + str(index) + '.spice.train'
	gauss_filter = get_gaussian(k, std)

	with open(read_file, 'r') as rlines:

		# # of samples and # of symbols
		first_line = rlines.readline()
		n_symbol   = int(first_line.split()[1])
		vector_dict  = get_vector(root, index, word_vec_dir, vect_dim, n_symbol)

		count  = 0
		insts  = []
		labels = []
		for rline in rlines:
			line = rline.split(' ')
			line = [key.strip() for key in line[1 : len(line)]]
			sub_insts, sub_labels = generate_insts_prek(vector_dict, gauss_filter, line, vect_dim, k)

			insts  += sub_insts
			labels += sub_labels

			# count  += 1
			# if count > 0:
			# 	break

		# write data
		save_vec_dir = root + save_vec_dir
		if not os.path.exists(save_vec_dir):
			os.makedirs(save_vec_dir)
		save_vec_file = save_vec_dir + '/' + str(index) + '.tr'
		save_data(insts, labels, save_vec_file)

		if save_obj_dir != '':
			obj_data = (insts, labels)

			save_obj_dir = root + save_obj_dir
			if not os.path.exists(save_obj_dir):
				os.makedirs(save_obj_dir)
			save_obj_file = save_obj_dir + '/' + str(index) + '.tr'
			save_obj(save_obj_file, obj_data)	


if __name__ == '__main__':
	'''
	'''
	root = '/str/users/angus/Codes/yang/'

	debug    = True
	data_ind = 0
	vect_dim = 30
	read_dir = 'data'
	text_dir = 'text'
	k, std   = 15, 2

	try:
		vect_dim = int(sys.argv[1])
	except Exception as e:
		pass
	print('Using word vector of dimension {:d}'.format(vect_dim))

	word_vec_dir = 'word_vec_' + str(vect_dim)
	save_vec_dir = 'tr_b_' + str(vect_dim) + '_' + str(k) + '_' + str(std)
	save_obj_dir = 'tr_o_' + str(vect_dim) + '_' + str(k) + '_' + str(std)
	save_obj_dir = ''


	# train_vector(root, data_ind, vect_dim, text_dir, word_vec_dir, debug)
	# transform_data(root, data_ind, vect_dim, read_dir, 
	# 	word_vec_dir, save_vec_dir, save_obj_dir, debug, k, std)


	for i in range(1, 16):
		data_ind = i 
		train_vector(root, data_ind, vect_dim, text_dir, word_vec_dir, debug)
		transform_data(root, data_ind, vect_dim, read_dir, 
			word_vec_dir, save_vec_dir, save_obj_dir, debug, k, std)
