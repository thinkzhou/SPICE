import mxnet as mx
import argparse
import os, sys
import models

from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata

from trainvec import load_obj
from trainvec import load_data




def get_mlp_spice(num_hidden):
    """
    multi-layer perceptron
    """
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=750)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 1000)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=num_hidden)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp

def get_cnn_spice(sentence_size=15, num_embed=30, num_label=34,
		filter_list=[10,11,12,13,14,15], num_filter=200, batch_size=128, dropout=0.0):
	input_x = mx.symbol.Variable('data')
	#input_y = mx.symbol.Variable('label')
	
	pooled_outputs =[]
	for i, filter_size in enumerate(filter_list):
		convi = mx.symbol.Convolution(data=input_x, kernel=(filter_size, num_embed),
					num_filter=num_filter)
		relui = mx.symbol.Activation(data=convi, act_type='relu')
		pooli = mx.symbol.Pooling(data=relui, pool_type='max',
					kernel=(sentence_size - filter_size + 1, 1), stride=(1,1))
		pooled_outputs.append(pooli)
	
	# combine all pooled outputs
	total_filters = num_filter * len(filter_list)
	concat = mx.symbol.Concat(*pooled_outputs, dim=1)
	h_pool = mx.symbol.Reshape(data=concat, shape=(-1, total_filters))

	# dropout layer
	if dropout > 0.0:
		h_drop = mx.symbol.Dropout(data=h_pool, p=dropout)
	else:
		h_drop = h_pool
	
	#fully connected
	fc = mx.symbol.FullyConnected(data = h_drop, name='fc', num_hidden=num_label)
	
	#softmax output
	sm = mx.symbol.SoftmaxOutput(data=fc, name='softmax')

	return sm


def get_test_iterator_spice(train_file):
    '''
    '''
    X, y  = load_data('', '', '', train_file)
    X, y  = shuffle(X, y)

    print('function->get_test_iterator_spice: ', X.shape)

    test_data  = X[:2, :].astype('float32')
    test_label = y[:2]

    data = mx.io.NDArrayIter(data = test_data)
    return data, label


def get_iterator_spice(data_shape):
    def get_iterator_impl(args, kv):
        
        X, y  = load_data('', '', '', args.train_file)
        X, y  = shuffle(X, y)

        print('function->get_iterator_spice: ', X.shape)

        n = int(X.shape[0] * (1 - args.val_portion)) 

        train_data  = X.astype('float32')
        train_label = y
        val_data    = X[n:, :].astype('float32')
        val_label   = y[n:]
        if (args.network == 'cnn'):
            train_data = train_data.reshape((-1,1,data_shape[0],data_shape[1]))
            val_data = val_data.reshape((-1,1,data_shape[0],data_shape[1]))

        train = mx.io.NDArrayIter(
            data  = train_data,
            label = train_label,
            batch_size  = args.batch_size,
            shuffle     = True)

        val   = mx.io.NDArrayIter(
            data  = val_data,
            label = val_label,
            batch_size  = args.batch_size,
            shuffle     = True)     
        
        return (train, val)
    return get_iterator_impl

def parse_args(train_file, model_file, islearning, val_portion):
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--network', type=str, default='mlp',
                        choices = ['mlp', 'cnn'],
                        help = 'the cnn to use')
    parser.add_argument('--data-dir', type=str, default='mnist/',
                        help='the input data directory')
    parser.add_argument('--gpus', type=str,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=.1,
                        help='the initial learning rate')
    parser.add_argument('--model-prefix', type=str,
                        help='the prefix of the model to load/save')
    parser.add_argument('--save-model-prefix', type=str,
                        help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=5,
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
    parser.add_argument('--islearning', type=bool, default=islearning,
                        help='train a new model if islearning, or load the existing model')
    parser.add_argument('--train-file', type=str, default=train_file,
                        help='the dataset on which the model is trained')
    parser.add_argument('--model-file', type=str, default=model_file,
                        help='save or load the model using pickle')
    parser.add_argument('--val-portion', type=float, default=val_portion,
                        help='the size of the validation set')
    return parser.parse_args()


def get_model(train_file, model_file, ndim=784, nout=10, val_fraction=0.3, islearning=True):
	'''
	'''
	args = parse_args(train_file, model_file, islearning, val_fraction)
	if args.network == 'mlp':
		data_shape = (ndim, )
		net = get_mlp_spice(nout)
	elif args.network == 'cnn':
		data_shape = (15, 30)#sentence_size, num_embed 
		net = get_cnn_spice(num_label=nout, batch_size=args.batch_size)

    # train
	model = models.fit(args, net, get_iterator_spice(data_shape))
	model = load_obj(model_file)
	return model, args.network
