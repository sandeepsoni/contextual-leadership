import numpy as np

def draw_exponential_random_variable (param, rng):
	# using the inverse method
	#return (1/float (param)) * np.log (rng.uniform (0,1))

	# using the built-in numpy function
	return rng.exponential (scale=param)

def exp_kernel (later_time,earlier_time,bandwidth):
	""" Calculates the exponential decay kernel

	Parameters:
	===========
	later_time (float): One of the two timestamps; this is the later one
	earlier_time (float): One of the two timestamps; this is the earlier one
	bandwidth (float): The scaling parameter for the kernel
	"""
	return np.exp (-bandwidth * (later_time-earlier_time))
