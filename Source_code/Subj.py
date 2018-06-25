# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:29:48 2016

@author: root
"""
import tensorflow as tf 
import utils
import math
import drnn
import cPickle as cp
import os
import numpy as np

class TrainConfig(object):	
 	"""Train config."""	
  	batch_size = 20	
  	hidden_size = [150] * 3
	dilations = [1,2,4]
	num_steps = None
	embedding_size = None	
	cell_type = "LSTM"
	nb_filter = 200
	class_num = 2
	fully_hidden = 200
	learning_rate = 5e-4	

class RNN_CNN_model(object):

	def __init__(self, config):
		self.batch_size = config.batch_size
		self.hidden_size = config.hidden_size
		self.dilations = config.dilations
		self.num_steps = config.num_steps
		self.embedding_size = config.embedding_size
		self.cell_type = config.cell_type
		self.output_channel = config.nb_filter
		self.class_num = config.class_num
		self.fully_hidden = config.fully_hidden

	def build_model(self):
		inputs = tf.placeholder(tf.float32, [None, self.num_steps, self.embedding_size], name = 'inputs')    # input
		targets = tf.placeholder(tf.float32, [None,self.class_num], name = 'targets')
		vaild_length = tf.placeholder(tf.int32, [None], name = 'vaild_len')
		keep_prob = tf.placeholder(tf.float32)
		
		assert len(self.hidden_size) == len(self.dilations)
		x = tf.contrib.layers.dropout( inputs, keep_prob = keep_prob ) 
		# outputs has a shape "layer" of "n_step" list of [batch,hidden], output shape "layer", [batch, n_step, hidden]
		outputs,output = drnn.drnn_layer_final(x, self.hidden_size, self.dilations, self.num_steps, input_dims=self.embedding_size)
		
		
		assert len(self.dilations) == len(outputs)
		
		assert len(self.dilations) == len(output)
		
		# output has [batch, n_step, hidden_size *len(self.dilations),channels ]
		output = tf.stack(output, axis=2)
		output = tf.reshape(output, [-1, self.num_steps, len(self.dilations)* self.hidden_size[0], 1])
		#output_drop = tf.contrib.layers.dropout( output, keep_prob = keep_prob )
		
		# conv_output has [batch, n_step, 1, self.output_channel]
		conv_output = tf.contrib.layers.conv2d(inputs = output, num_outputs = self.output_channel, kernel_size = [1,output.shape[2]], padding='VALID')
		
		# max_pool has [batch, 1, 1, self.output_channel]
		max_pool = tf.contrib.layers.max_pool2d(inputs = conv_output, kernel_size = [conv_output.shape[1],conv_output.shape[2]], stride = 1, padding='VALID')

		# max_pool has [batch, self.output_channel]
		max_pool = tf.reshape(max_pool, [-1, self.output_channel])
		max_pool = tf.contrib.layers.dropout( max_pool, keep_prob = keep_prob )
		
		# vaild_hidden has list of 'layer' of [batch,hidden]
		#vaild_hidden = []
		#for i in range(len(outputs)):
		#	L = []
		#	for j in range(self.batch_size):
		#		L.append ( tf.gather(outputs[i], tf.gather(vaild_length, j))[j,:] )
		#	vaild_hidden.append(tf.reshape(L, [self.batch_size,self.hidden_size[0]]))
		
		vaild_hidden = []
		for i in range(len(outputs)):
			vaild_hidden.append ( outputs[i][-1] )		
		
		# vaild_hidden has list of 'layer' of [batch,hidden] then vaild_concat_hidden is [ batch, hidden*(layer) ]
		vaild_concat_hidden = tf.stack(vaild_hidden, axis=1)
		vaild_concat_hidden = tf.reshape(vaild_concat_hidden, [-1, self.hidden_size[0]*len(self.dilations)])
		
		# lstm_aulu_task
		#lstm_fully_connected = tf.contrib.layers.legacy_fully_connected(x = vaild_concat_hidden ,num_output_units = self.fully_hidden)
		#lstm_fully_connected = tf.contrib.layers.dropout( lstm_fully_connected, keep_prob = keep_prob )
		#lstm_logits = tf.contrib.layers.legacy_fully_connected(x = lstm_fully_connected ,num_output_units = self.class_num)
		#aulu_ce = tf.nn.softmax_cross_entropy_with_logits(labels = targets, logits = lstm_logits, name = 'aulu_ce')
		
		# final_represent has shape of [batch, hidden*(layer) + num_output_units ]
		final_represent = tf.concat( [vaild_concat_hidden, max_pool], axis=1 )
		
		#final_represent_drop = tf.contrib.layers.dropout( final_represent, keep_prob = keep_prob )
		
		
		#fully_connected has shape of [batch,num_output_units]
		#fully_connected = tf.contrib.layers.legacy_fully_connected(x = final_represent_drop ,num_output_units = self.fully_hidden)
		fully_connected = tf.contrib.layers.legacy_fully_connected(x = final_represent ,num_output_units = self.fully_hidden)
		
		#logits has shape of [batch,self.class_num]
		#fully_connected = tf.contrib.layers.dropout( fully_connected, keep_prob = keep_prob )
		logits = tf.contrib.layers.legacy_fully_connected(x = fully_connected ,num_output_units = self.class_num)
		ce = tf.nn.softmax_cross_entropy_with_logits(labels = targets, logits = logits, name = 'ce')
		
		regularization_loss = 0.0
		for i in xrange (len(tf.trainable_variables())):
			regularization_loss += tf.nn.l2_loss(tf.trainable_variables()[i])
		
		loss = tf.reduce_sum(ce, name = 'loss') + 1e-3 * regularization_loss #+ tf.reduce_sum(aulu_ce)
		
		
		answer_probab = tf.nn.softmax(logits, name='answer_probab')
		predictions = tf.argmax(answer_probab,1)
		correct_predictions = tf.equal(tf.argmax(answer_probab,1), tf.argmax(targets,1))
		accuracy = tf.cast(correct_predictions, tf.float32)
		
		input_tensors = {
			'inputs' : inputs,
			'targets' : targets,
			'vaild_length' : vaild_length,
			'keep_prob' : keep_prob
		}
		
		return input_tensors, loss, predictions, accuracy	
		
	
			
def main():
	os.environ['CUDA_VISIBLE_DEVICES']='1'
	gpu_config = tf.ConfigProto()
	gpu_config.gpu_options.allow_growth = True
	
	config = TrainConfig()
	
	print 'Loading data-----------------------'
	train_data = cp.load(open("Subj/Subj.p","rb"))
	print 'Loading data completed-------------'
	config.num_steps = train_data[0].shape[1]
	config.embedding_size = train_data[0].shape[2]
	num_samples = train_data[0].shape[0]
	li = range(num_samples)
	np.random.shuffle(li)
	
	folds_train = []
	folds_test = []
	num_samples_per_fold = num_samples / 10
	p = 0
	for f in range(10):
		folds_train.append( np.array (np.concatenate([li[:p],li[p+num_samples_per_fold:]]), dtype=np.int) )
		folds_test.append( np.array (li[p:p+num_samples_per_fold], dtype=np.int) )
		p += num_samples_per_fold
	
	
	senctence, label, vaild_length = train_data
	
	with tf.Session(config = gpu_config) as sess:
		model = RNN_CNN_model(config = config)
		input_tensors, loss, _, accuracy = model.build_model()
		train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(loss)	
		Epoch = 200
		CV_max_acc = []
		for CV in range(10):
			sess.run(tf.global_variables_initializer())
			print '-------------------------Cross Vaildation %d-------------------------'%CV
			get_max_acc = []
			for i in range(Epoch):
				total_loss = 0.0
				total_train_accuracy = []
				print '----------Cross Vaildation_%d:Epoch %i----------'%(CV,i)
				for batch_input, batch_label, batch_vaild_length, _ in utils.CV_next_batch(config.batch_size, config.class_num, senctence[folds_train[CV]], label[folds_train[CV]], vaild_length[folds_train[CV]] ):
					batch_loss, _, batch_accuracy = sess.run([loss, train_op, accuracy], feed_dict={input_tensors['inputs']: batch_input, 
																		input_tensors['targets']: batch_label, 
																		input_tensors['vaild_length']: batch_vaild_length,
																		input_tensors['keep_prob'] : 0.2 }
					)
					total_loss += batch_loss
					total_train_accuracy.append(batch_accuracy)
				print "Loss:",total_loss,"Train acc:",np.mean(np.array(total_train_accuracy).reshape(-1))
				

				total_test_accuracy = []
				total_num = 0
				for batch_input, batch_label, batch_vaild_length, batch_size in utils.CV_next_batch(config.batch_size,config.class_num, senctence[folds_test[CV]], label[folds_test[CV]], vaild_length[folds_test[CV]] ):
					total_num += batch_size
					batch_accuracy = sess.run([accuracy], feed_dict={input_tensors['inputs']: batch_input, 
																		input_tensors['targets']: batch_label, 
																		input_tensors['vaild_length']: batch_vaild_length,
																		input_tensors['keep_prob'] : 1.0 }
					)

					total_test_accuracy.append(batch_accuracy)
				print "Test acc:",np.mean(np.array(total_test_accuracy).reshape(-1)[:total_num])
				get_max_acc.append(np.mean(np.array(total_test_accuracy).reshape(-1)[:total_num]))
			print 'Max accuracy',np.max(np.array(get_max_acc).reshape(-1))
			CV_max_acc.append(np.max(np.array(get_max_acc).reshape(-1)))
		print 'CV Max accuracy', np.array(CV_max_acc).reshape(-1)
		print 'CV Mean Max accuracy',np.mean(np.array(CV_max_acc).reshape(-1))
	
if __name__ == "__main__":
	main() 