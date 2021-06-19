# Copyright (c) 2020 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
# https://arxiv.org/abs/1905.09688

import numpy as np

import PyTsetlinMachineCUDA.kernels as kernels

import pycuda.curandom as curandom
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from time import time

g = curandom.XORWOWRandomNumberGenerator() 

class CommonTsetlinMachine():
	def __init__(self, number_of_clauses, T, s, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, max_weight=1, grid=(16*13,1,1), block=(128,1,1)):
		self.number_of_clauses = number_of_clauses
		self.number_of_clause_chunks = (number_of_clauses-1)/32 + 1
		self.number_of_state_bits = number_of_state_bits
		self.T = int(T)
		self.s = s
		self.max_weight = min(max_weight,255)
		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.append_negated = append_negated
		self.grid = grid
		self.block = block

		self.X_train = np.array([])
		self.Y_train = np.array([])
		self.X_test = np.array([])
		self.ta_state = np.array([])
		self.clause_weights = np.array([])

		mod_encode = SourceModule(kernels.code_encode, no_extern_c=True)
		self.prepare_encode = mod_encode.get_function("prepare_encode")
		self.encode = mod_encode.get_function("encode")

	def __getstate__(self):
		state = self.__dict__.copy()
		state['ta_state'] = np.empty(self.number_of_classes*self.number_of_clauses*self.number_of_ta_chunks*self.number_of_state_bits).astype(np.uint32)
		cuda.memcpy_dtoh(state['ta_state'], self.ta_state_gpu)
		state['clause_weights'] = np.empty(self.number_of_classes*self.number_of_clauses).astype(np.uint8)
		cuda.memcpy_dtoh(state['clause_weights'], self.clause_weights_gpu)
		
		if 'X_train' in state:
			del state['X_train']
			del state['Y_train']
			del state['encoded_Y_train']
			del state['encoded_X_training_gpu']
			del state['update']

		if 'X_test' in state:
			del state['X_test']

		if 'encoded_X_test_gpu' in state:
			del state['encoded_X_test_gpu']

		if 'evaluate' in state:
			del state['evaluate']

		if 'prepare_encode' in state:
			del state['prepare_encode']
			del state['prepare']
			del state['encode']
			del state['ta_state_gpu']
			del state['clause_weights_gpu']
			del state['clause_output_gpu']
			del state['class_sum_gpu']
			del state['Y_gpu']			

		return state

	def __setstate__(self, state):
		self.__dict__.update(state)
		self.X_train = np.array([])
		self.Y_train = np.array([])
		self.X_test = np.array([])

		mod_encode = SourceModule(kernels.code_encode, no_extern_c=True)
		self.prepare_encode = mod_encode.get_function("prepare_encode")
		self.encode = mod_encode.get_function("encode")
		
		self.ta_state_gpu = cuda.mem_alloc(self.number_of_classes*self.number_of_clauses*self.number_of_ta_chunks*self.number_of_state_bits*4)
		self.clause_weights_gpu = cuda.mem_alloc(self.number_of_classes*self.number_of_clauses)
		cuda.memcpy_htod(self.ta_state_gpu, self.ta_state)
		cuda.memcpy_htod(self.clause_weights_gpu, self.clause_weights)

	def encode_X(self, X, encoded_X_gpu):
		number_of_examples = X.shape[0]

		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)
		X_gpu = cuda.mem_alloc(Xm.nbytes)
		cuda.memcpy_htod(X_gpu, Xm)
		if self.append_negated:			
			self.prepare_encode(X_gpu, encoded_X_gpu, np.int32(number_of_examples), np.int32(self.dim[0]), np.int32(self.dim[1]), np.int32(self.dim[2]), np.int32(self.patch_dim[0]), np.int32(self.patch_dim[1]), np.int32(1), np.int32(0), grid=self.grid, block=self.block)
			cuda.Context.synchronize()
			self.encode(X_gpu, encoded_X_gpu, np.int32(number_of_examples), np.int32(self.dim[0]), np.int32(self.dim[1]), np.int32(self.dim[2]), np.int32(self.patch_dim[0]), np.int32(self.patch_dim[1]), np.int32(1), np.int32(0), grid=self.grid, block=self.block)
			cuda.Context.synchronize()
		else:
			self.prepare_encode(X_gpu, encoded_X_gpu, np.int32(number_of_examples), np.int32(self.dim[0]), np.int32(self.dim[1]), np.int32(self.dim[2]), np.int32(self.patch_dim[0]), np.int32(self.patch_dim[1]), np.int32(0), np.int32(0), grid=self.grid, block=self.block)
			cuda.Context.synchronize()
			self.encode(X_gpu, encoded_X_gpu, np.int32(number_of_examples), np.int32(self.dim[0]), np.int32(self.dim[1]), np.int32(self.dim[2]), np.int32(self.patch_dim[0]), np.int32(self.patch_dim[1]), np.int32(0), np.int32(0), grid=self.grid, block=self.block)
			cuda.Context.synchronize()

	def allocate_gpu_memory(self, number_of_examples):
		self.ta_state_gpu = cuda.mem_alloc(self.number_of_classes*self.number_of_clauses*self.number_of_ta_chunks*self.number_of_state_bits*4)
		self.clause_weights_gpu = cuda.mem_alloc(self.number_of_classes*self.number_of_clauses)
		self.clause_output_gpu = cuda.mem_alloc(self.number_of_classes*self.number_of_clauses*number_of_examples)
		self.class_sum_gpu = cuda.mem_alloc(self.number_of_classes*number_of_examples*4)

	def ta_action(self, mc_tm_class, clause, ta):
		if np.array_equal(self.ta_state, np.array([])):
			self.ta_state = np.empty(self.number_of_classes*self.number_of_clauses*self.number_of_ta_chunks*self.number_of_state_bits).astype(np.uint32)
			cuda.memcpy_dtoh(self.ta_state, self.ta_state_gpu)
		ta_state = self.ta_state.reshape((self.number_of_classes, self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits))

		return (ta_state[mc_tm_class, clause, ta // 32, self.number_of_state_bits-1] & (1 << (ta % 32))) > 0

	# Transform input data for processing at next layer
	def transform(self, X):
		number_of_examples = X.shape[0]
		
		encoded_X_gpu = cuda.mem_alloc(int(number_of_examples * self.number_of_patches * self.number_of_ta_chunks*4))
		self.encode_X(X, encoded_X_gpu)

		parameters = """
#define CLASSES %d
#define CLAUSES %d
#define FEATURES %d
#define STATE_BITS %d
#define BOOST_TRUE_POSITIVE_FEEDBACK %d
#define S %f
#define THRESHOLD %d

#define NEGATIVE_CLAUSES %d

#define PATCHES %d

#define NUMBER_OF_EXAMPLES %d

#define BATCH_SIZE %d

		""" % (self.number_of_classes, self.number_of_clauses, self.number_of_features, self.number_of_state_bits, self.boost_true_positive_feedback, self.s, self.T, self.negative_clauses, self.number_of_patches, number_of_examples, 100)

		mod = SourceModule(parameters + kernels.code_header + kernels.code_transform, no_extern_c=True)
		transform = mod.get_function("transform")

		X_transformed_gpu = cuda.mem_alloc(number_of_examples*self.number_of_classes*self.number_of_clauses*4)
		transform(self.ta_state_gpu, encoded_X_gpu, X_transformed_gpu, grid=self.grid, block=self.block)
		cuda.Context.synchronize()
		X_transformed = np.ascontiguousarray(np.empty(number_of_examples*self.number_of_classes*self.number_of_clauses, dtype=np.uint32))
		cuda.memcpy_dtoh(X_transformed, X_transformed_gpu)
		
		return X_transformed.reshape((number_of_examples, self.number_of_classes*self.number_of_clauses))

	def _fit(self, X, encoded_Y, epochs=100, incremental=False, batch_size=100):
		number_of_examples = X.shape[0]

		if (not np.array_equal(self.X_train, X)) or (not np.array_equal(self.encoded_Y_train, encoded_Y)):
			self.X_train = X
			self.encoded_Y_train = encoded_Y
			
			if len(X.shape) == 3:
				self.dim = (X.shape[1], X.shape[2],  1)
			elif len(X.shape) == 4:
				self.dim = X.shape[1:]

			if self.append_negated:
				self.number_of_features = int(self.patch_dim[0]*self.patch_dim[1]*self.dim[2] + (self.dim[0] - self.patch_dim[0]) + (self.dim[1] - self.patch_dim[1]))*2
			else:
				self.number_of_features = int(self.patch_dim[0]*self.patch_dim[1]*self.dim[2] + (self.dim[0] - self.patch_dim[0]) + (self.dim[1] - self.patch_dim[1]))

			self.number_of_patches = int((self.dim[0] - self.patch_dim[0] + 1)*(self.dim[1] - self.patch_dim[1] + 1))
			self.number_of_ta_chunks = int((self.number_of_features-1)/32 + 1)
		
			parameters = """
	#define CLASSES %d
	#define CLAUSES %d
	#define FEATURES %d
	#define STATE_BITS %d
	#define BOOST_TRUE_POSITIVE_FEEDBACK %d
	#define S %f
	#define THRESHOLD %d
	#define MAX_WEIGHT %d

	#define NEGATIVE_CLAUSES %d

	#define PATCHES %d

	#define NUMBER_OF_EXAMPLES %d

	#define BATCH_SIZE %d

""" % (self.number_of_classes, self.number_of_clauses, self.number_of_features, self.number_of_state_bits, self.boost_true_positive_feedback, self.s, self.T, self.max_weight, self.negative_clauses, self.number_of_patches, number_of_examples, batch_size)

			mod_prepare = SourceModule(parameters + kernels.code_header + kernels.code_prepare, no_extern_c=True)
			self.prepare = mod_prepare.get_function("prepare")

			self.allocate_gpu_memory(number_of_examples)

			self.prepare(self.ta_state_gpu, self.clause_weights_gpu, self.clause_output_gpu, self.class_sum_gpu, grid=self.grid, block=self.block)
			cuda.Context.synchronize()

			mod_update = SourceModule(parameters + kernels.code_header + kernels.code_update, no_extern_c=True)
			self.update = mod_update.get_function("update")
			self.update.prepare("PPPPPPPi")

			self.encoded_X_training_gpu = cuda.mem_alloc(int(number_of_examples * self.number_of_patches * self.number_of_ta_chunks*4))
			self.encode_X(X, self.encoded_X_training_gpu)
		
			self.Y_gpu = cuda.mem_alloc(encoded_Y.nbytes)
			cuda.memcpy_htod(self.Y_gpu, encoded_Y)
		elif incremental == False:
			self.prepare(self.ta_state_gpu, self.clause_weights_gpu, self.clause_output_gpu, self.class_sum_gpu, grid=self.grid, block=self.block)
			cuda.Context.synchronize()

		for epoch in range(epochs):
			for e in range(0, number_of_examples, batch_size):
				self.update.prepared_call(self.grid, self.block, g.state, self.ta_state_gpu, self.clause_weights_gpu, self.class_sum_gpu, self.clause_output_gpu, self.encoded_X_training_gpu, self.Y_gpu, np.int32(e))
				cuda.Context.synchronize()

		self.ta_state = np.array([])
		self.clause_weights = np.array([])
		
		return

	def _score(self, X):
		number_of_examples = X.shape[0]
		
		if not np.array_equal(self.X_test, X):
			self.X_test = X

			self.encoded_X_test_gpu = cuda.mem_alloc(int(number_of_examples * self.number_of_patches * self.number_of_ta_chunks*4))
			self.encode_X(X, self.encoded_X_test_gpu)

			parameters = """
#define CLASSES %d
#define CLAUSES %d
#define FEATURES %d
#define STATE_BITS %d
#define BOOST_TRUE_POSITIVE_FEEDBACK %d
#define S %f
#define THRESHOLD %d

#define NEGATIVE_CLAUSES %d

#define PATCHES %d

#define NUMBER_OF_EXAMPLES %d

#define BATCH_SIZE %d

		""" % (self.number_of_classes, self.number_of_clauses, self.number_of_features, self.number_of_state_bits, self.boost_true_positive_feedback, self.s, self.T, self.negative_clauses, self.number_of_patches, number_of_examples, 100)

			mod = SourceModule(parameters + kernels.code_header + kernels.code_evaluate, no_extern_c=True)
			self.evaluate = mod.get_function("evaluate")

		class_sum = np.ascontiguousarray(np.zeros(self.number_of_classes*number_of_examples)).astype(np.int32)
		class_sum_gpu = cuda.mem_alloc(class_sum.nbytes)
		cuda.memcpy_htod(class_sum_gpu, class_sum)

		self.evaluate(self.ta_state_gpu, self.clause_weights_gpu, class_sum_gpu, self.encoded_X_test_gpu, grid=self.grid, block=self.block)
		cuda.Context.synchronize()
		cuda.memcpy_dtoh(class_sum, class_sum_gpu)
		
		class_sum = np.clip(class_sum.reshape((self.number_of_classes, number_of_examples)), -self.T, self.T)

		return class_sum
	
class MultiClassConvolutionalTsetlinMachine2D(CommonTsetlinMachine):
	"""
	This class ...
	"""
	
	def __init__(self, number_of_clauses, T, s, patch_dim, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, max_weight=1, grid=(16*13,1,1), block=(128,1,1)):
		super().__init__(number_of_clauses, T, s, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, max_weight=max_weight, grid=grid, block=block)
		self.patch_dim = patch_dim
		self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False, batch_size = 100):
		self.number_of_classes = int(np.max(Y) + 1)
	
		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.empty((Y.shape[0], self.number_of_classes), dtype = np.int32)
		for i in range(self.number_of_classes):
			encoded_Y[:,i] = np.where(Y == i, self.T, -self.T)

		self._fit(X, encoded_Y, epochs=epochs, incremental=incremental, batch_size = batch_size)

	def score(self, X):
		return self._score(X)

	def predict(self, X):
		return np.argmax(self.score(X), axis=0)

class MultiClassTsetlinMachine(CommonTsetlinMachine):
	def __init__(self, number_of_clauses, T, s, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, max_weight=1, grid=(16*13,1,1), block=(128,1,1)):
		super().__init__(number_of_clauses, T, s, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, max_weight=max_weight, grid=grid, block=block)
		self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False, batch_size = 100):
		X = X.reshape(X.shape[0], X.shape[1], 1)

		self.number_of_classes = int(np.max(Y) + 1)
		self.patch_dim = (X.shape[1], 1, 1)

		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.empty((Y.shape[0], self.number_of_classes), dtype = np.int32)
		for i in range(self.number_of_classes):
			encoded_Y[:,i] = np.where(Y == i, self.T, -self.T)

		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental, batch_size = batch_size)

		return

	def score(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		return self._score(X)

	def predict(self, X):
		return np.argmax(self.score(X), axis=0)

class TsetlinMachine(CommonTsetlinMachine):
	def __init__(self, number_of_clauses, T, s, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, max_weight=1, grid=(16*13,1,1), block=(128,1,1)):
		super().__init__(number_of_clauses, T, s, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, max_weight=max_weight, grid=grid, block=block)
		self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False, batch_size = 100):
		X = X.reshape(X.shape[0], X.shape[1], 1)

		self.number_of_classes = 1
		self.patch_dim = (X.shape[1], 1, 1)
		
		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)

		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental, batch_size = batch_size)

		return

	def score(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		return self._score(X)[0,:]

	def predict(self, X):
		return self.score(X) >= 0

class RegressionTsetlinMachine(CommonTsetlinMachine):
	def __init__(self, number_of_clauses, T, s, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, max_weight=1, grid=(16*13,1,1), block=(128,1,1)):
		super().__init__(number_of_clauses, T, s, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, max_weight=max_weight, grid=grid, block=block)
		self.negative_clauses = 0

	def fit(self, X, Y, epochs=100, incremental=False, batch_size = 100):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		
		self.number_of_classes = 1
		self.patch_dim = (X.shape[1], 1, 1)

		self.max_y = np.max(Y)
		self.min_y = np.min(Y)
	
		encoded_Y = ((Y - self.min_y)/(self.max_y - self.min_y)*self.T).astype(np.int32)
			
		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental, batch_size = batch_size)

		return

	def predict(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		
		return 1.0*(self._score(X)[0,:])*(self.max_y - self.min_y)/(self.T) + self.min_y
