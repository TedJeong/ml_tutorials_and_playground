import tensorflow as tf
class RNN(object):
	def forward_single_step(self, x, prev_h, Wx, Wh, b):
		next_h, cache = None, None
		forward = tf.matmul(x, Wx) + tf.matmul(prev_h, Wh) + b
		next_h = tf.nn.tanh(forward)
		cache = x, Wx, prev_h, Wh, forward
		return next_h, cache
	def backward_single_step(self, dnext_h, cache):
		dx, dprev_h, dWx, dWh, db = None, None, None, None, None
		x, Wx, prev_h, Wh, forward = cache
		dforward = (1-tf.tanh(forward)**2)*dnext_h
		
		dx = tf.matmul(forward, Wx.T)
		dWx = tf.matmul(x.T, dforward)
		dprev_h = tf.matmul(dforward, Wh.T)
		dWh = tf.matmul(prev_h.T, dforward)
		
