from sklearn import metrics, preprocessing
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
from impl import split_data
from trainvec import save_obj

class LSTM():
	def __init__ (self, n_input=30, n_steps=15, n_hidden=32):
		self.n_input = n_input
		self.n_steps = n_steps
		self.n_hidden = n_hidden

	def set_classifier(classifier):
		self.classifier = classifier

	def rnn_model(self, X, y):
		X = tf.reshape(X, [-1, self.n_steps, self.n_input])  # (batch_size, n_steps, n_input)
		# # permute n_steps and batch_size
		X = tf.transpose(X, [1, 0, 2])
		# # Reshape to prepare input to hidden activation
		X = tf.reshape(X, [-1, self.n_input])  # (n_steps*batch_size, n_input)
		# # Split data because rnn cell needs a list of inputs for the RNN inner loop
		X = tf.split(0, self.n_steps, X)  # n_steps * (batch_size, n_input)

		# Define a GRU cell with tensorflow
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
		# Get lstm cell output
		_, encoding = tf.nn.rnn(lstm_cell, X, dtype=tf.float32)
		return learn.models.logistic_regression(encoding[-1], y)

	def get_classifier(self, n_classes, batch_size=128, learning_rate=0.1, training_steps=10):
		self.classifier = learn.TensorFlowEstimator(model_fn=self.rnn_model, n_classes=n_classes,
                                       batch_size=batch_size,
                                       steps=training_steps,
									   optimizer='SGD',
                                       learning_rate=learning_rate)
	def train(self, args, n_class, model_file):
		learning_rate = args.lr
		batch_size = args.batch_size
		num_epochs = args.num_epochs
		print("lr: "+str(learning_rate) + 
			" batch size: " + str(batch_size) + " num_epochs: "+str(num_epochs))
		self.get_classifier(n_class, batch_size, learning_rate, num_epochs)
		X_train, y_train, X_val, y_val = split_data(args)	

		self.classifier.fit(X_train, y_train)
		pred = self.classifier.predict(X_val)
		score = metrics.accuracy_score(y_val, pred)
		print('Accuracy: {0:f}'.format(score))
	
	def predict(self, X_test):
		return self.classifier.predict_proba(X_test)

