# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:29:48 2016

@author: root
"""
import tensorflow as tf 
import numpy as np
import sys, time


def convertToOneHot(vector, num_classes=None):
    vector = np.array(vector,dtype = int)
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0
    assert num_classes is not None
	
    assert num_classes > 0
    vector = vector % num_classes

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(np.float32)

def next_batch(batch_size,class_label,data):
	senctence, label, vaild_length = data
	label = convertToOneHot(label,num_classes = class_label)
	assert senctence.shape[0] == label.shape[0]
	
	
	row = senctence.shape[0]
	batch_len = int( row / batch_size)
	left_row = row - batch_len * batch_size
	
	#打乱数据
	indices = np.random.permutation(senctence.shape[0])
	rand_senctence = senctence[indices]
	rand_label = label[indices]
	rand_vaild_length = vaild_length[indices]
	

	for i in xrange(batch_len):
		batch_input = rand_senctence[ i*batch_size : (i+1)*batch_size, :, :]
		batch_label = rand_label[ i*batch_size : (i+1)*batch_size, : ]
		batch_label_vaild_length = rand_vaild_length[i*batch_size : (i+1)*batch_size]
		yield (batch_input, batch_label, batch_label_vaild_length, batch_size)
		
	if left_row != 0:
		need_more = batch_size - left_row
		need_more = np.random.choice( np.arange(row), size = need_more )
		
		yield ( np.concatenate((rand_senctence[ -left_row: , :, : ], rand_senctence[need_more]), axis=0 ), np.concatenate( (rand_label[ -left_row: , : ], rand_label[ need_more ]),axis=0), 
			   np.concatenate( (rand_vaild_length[ -left_row : ] , rand_vaild_length[ need_more ]), axis=0 ), left_row)
			   

def CV_next_batch(batch_size,class_label, senctence, label, vaild_length):
	label = convertToOneHot(label,num_classes = class_label)
	assert senctence.shape[0] == label.shape[0]
	
	row = senctence.shape[0]
	batch_len = int( row / batch_size)
	left_row = row - batch_len * batch_size
	
	#打乱数据
	indices = np.random.permutation(senctence.shape[0])
	rand_senctence = senctence[indices]
	rand_label = label[indices]
	rand_vaild_length = vaild_length[indices]
	

	for i in xrange(batch_len):
		batch_input = rand_senctence[ i*batch_size : (i+1)*batch_size, :, :]
		batch_label = rand_label[ i*batch_size : (i+1)*batch_size, : ]
		batch_label_vaild_length = rand_vaild_length[i*batch_size : (i+1)*batch_size]
		yield (batch_input, batch_label, batch_label_vaild_length, batch_size)
		
	if left_row != 0:
		need_more = batch_size - left_row
		need_more = np.random.choice( np.arange(row), size = need_more )
		
		yield ( np.concatenate((rand_senctence[ -left_row: , :, : ], rand_senctence[need_more]), axis=0 ), np.concatenate( (rand_label[ -left_row: , : ], rand_label[ need_more ]),axis=0), 
			   np.concatenate( (rand_vaild_length[ -left_row : ] , rand_vaild_length[ need_more ]), axis=0 ), left_row)			   

			   
# def next_batch(batch_size,class_label,data):
	# senctence, label, vaild_length = data
	# label = convertToOneHot(label,num_classes = class_label)
	# assert senctence.shape[0] == label.shape[0]
	
	
	# row = senctence.shape[0]
	# batch_len = int( row / batch_size)
	# left_row = row - batch_len * batch_size
	
	#打乱数据
	# indices = np.random.permutation(senctence.shape[0])
	# rand_senctence = senctence[indices]
	# rand_label = label[indices]
	# rand_vaild_length = vaild_length[indices]
	

	# for i in xrange(batch_len):
		# batch_input = rand_senctence[ i*batch_size : (i+1)*batch_size, :, :]
		# batch_label = rand_label[ i*batch_size : (i+1)*batch_size, : ]
		# batch_label_vaild_length = rand_vaild_length[i*batch_size : (i+1)*batch_size]
		# yield (batch_input, batch_label, batch_label_vaild_length, len(batch_label_vaild_length) )
		
	# if left_row != 0:
		# yield ( rand_senctence[ -left_row: , :, : ], rand_label[ -left_row: , : ], rand_vaild_length[ -left_row : ], len(batch_label_vaild_length) )
		
		