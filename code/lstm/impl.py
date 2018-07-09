import os, sys
from sklearn.utils import shuffle
from trainvec import load_data
import numpy
import argparse


def split_data(args):
	X, y = load_data('', '', '', args.train_file)
	X, y = shuffle(X, y)
	X = X.reshape((-1,15,30))
	n = int(X.shape[0] * (1 - args.val_portion))
	y = y.astype('int')
	train_data  = X.astype('float32')
	train_label = y
	val_data    = X[n:, :].astype('float32')
	val_label   = y[n:]
	return train_data, train_label, val_data, val_label

def dense_to_one_hot(labels_dense, num_classes):
	"""Convert class labels from scalars to one-hot vectors."""
	num_labels = labels_dense.shape[0]
	labels_one_hot = numpy.zeros((num_labels, num_classes))
	labels_one_hot[numpy.arange(num_labels), labels_dense] = 1
	return labels_one_hot


def parse_args(train_file, model_file, val_portion):
    parser = argparse.ArgumentParser(description='train an LSTM')
    parser.add_argument('--network', type=str, default='lstm',
                        choices = ['lstm'],
                        help = 'the network to use')
    parser.add_argument('--gpus', type=str,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=.1,
                        help='the initial learning rate')
    parser.add_argument('--model-prefix', type=str,
                        help='the prefix of the model to load/save')
    parser.add_argument('--save-model-prefix', type=str,
                        help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=40000,
                        help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int,
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--kv-store', type=str, default='local',
                        help='the kvstore type')
    parser.add_argument('--lr-factor', type=float, default=1,
                        help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=float, default=1,
                        help='the number of epoch to factor the lr, could be .5')

    # customization
    parser.add_argument('--train-file', type=str, default=train_file,
                        help='the dataset on which the model is trained')
    parser.add_argument('--model-file', type=str, default=model_file,
                        help='save or load the model using pickle')
    parser.add_argument('--val-portion', type=float, default=val_portion,
                        help='the size of the validation set')
    return parser.parse_args()


def get_train_dataset(args, num_classes, onehot=True):
	train_data, train_label, val_data, val_label = split_data(args)
	if onehot:
		# Convert label into one hot
		train_label = dense_to_one_hot(train_label, num_classes)
		val_label = dense_to_one_hot(val_label, num_classes)
	dataset = DataSet(train_data,train_label,
				val_data, val_label)
	return dataset

class DataSet(object):
	def __init__(self, train_data, train_label, val_data, val_label):
		self.train_data = train_data
		self.train_label = train_label
		self.val_data = val_data
		self.val_label = val_label
		self.num_train = self.train_data.shape[0]
		self.num_val = self.val_data.shape[0]
		self._epochs_completed = 0
		self._index_in_epoch = 0

	def get_val_data(self):
		return self.val_data, self.val_label

	def num_train(self):
		return self.num_train
	
	def num_val(self):
		return self.num_val	

	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size):
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self.num_train:
			# End this epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = numpy.arange(self.num_train)
			numpy.random.shuffle(perm)
			self.train_data = self.train_data[perm]
			self.train_label = self.train_label[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size

		end = self._index_in_epoch
		return self.train_data[start:end], self.train_label[start:end]	
