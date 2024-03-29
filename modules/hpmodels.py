import numpy as np
import hputils
from scipy.optimize import minimize

"""
class DCHP:
	# Discrete coarse HP evaluated
	@staticmethod
	def log_likelihood_single_cascade (params, \
									   cascade, \
									   activations, \
									   bandwidth, \
									   dims=5, \
									   sign=-1.0, \
									   epsilon=1e-5, \
									   verbose=False):

		# unflatten parameters
		mu = params[0*dims:1*dims]
		b = params[1*dims:2*dims]
		c = params[2*dims:3*dims]
		s = params[3*dims:4*dims]

		# initialization
		eta = np.zeros_like (mu)
		intensities = np.zeros_like (mu)
		total_intensities = np.zeros_like (mu)

		last_timestamp = -1
		log_intensities = 0
		se_gate = np.eye (dims)

		for timestamp in sorted (cascade.keys()):
			if last_timestamp < 0:
				eta = 0
			else:
				last_sources = cascade[last_timestamp]
				aggregator = np.zeros_like (mu)
				for last_source in last_sources:
					aggregator += ((b[last_source] * c) + se_gate[last_source,:])
				
				eta = hputils.exp_kernel (timestamp, last_timestamp, bandwidth) * (eta + aggregator)

			intensities = mu + eta
			current_sources = cascade[timestamp]
			log_intensities += np.log (intensities[current_sources] + epsilon).sum()
			total_intensities += intensities
			last_timestamp = timestamp


		ll = (sign) * (log_intensities - total_intensities.sum())
		return ll

"""

class DVHP:
	""" Discrete vanilla hawkes process. """
	@staticmethod
	def log_likelihood_single_cascade (params,
									   cascade,
									   bandwidth=1.0,
									   dims=5,
									   sign=-1.0,
									   epsilon=1e-5,
									   verbose=False):
		# unflatten parameters
		mu = params[0*dims: 1*dims]
		alpha = params[dims:].reshaps(dims, dims)

		# initialization
		eta = np.zeros_like (mu)
		intensities = np.zeros_like (mu)
		total_intensities = np.zeros_like (mu)

		last_timestamp = -1
		log_intensities = 0

		for timestamp in sorted (cascade.keys()):
			if last_timestamp < 0:
				eta = 0
			else:
				last_sources = cascade[last_timestamp]
				aggregator = np.zeros_like (mu)
				for last_source in last_sources:
					aggregator += alpha[last_source,:]

				eta = hputils.exp_kernel (timestamp, last_timestamp, bandwidth) * (eta + aggregator)

		intensities = mu + eta
		current_sources = cascade[timestamp]
		log_intensities += np.log (intensities[current_sources] + epsilon).sum()
		total_intensities += intensities
		last_timestamp = timestamp
        
		ll = (sign) * (log_intensities - total_intensities.sum())
		return ll

	@staticmethod
	def grad_mu (params, cascade, bandwidth=1.0, dims=5, sign=-1.0, epsilon=1e-5, verbose=False):
		pass
    
	@staticmethod
	def grad_alpha (params, cascade, bandwidth=1.0, dims=5, sign=-1.0, epsilon=1e-5, verbose=False):
		pass

	@staticmethod
	def log_likelihood_many_cascades (params, 
									  cascades, 
									  bandwidth=1.0, 
									  dims=5, 
									  sign=-1.0, 
									  epsilon=1e-5, 
									  verbose=False):
		n_cascades = len (cascades)
		total_log_likelihood = 0.0
		for i in range (n_cascades):
			log_likelihood = DCHP.log_likelihood_single_cascade (params, \
																 cascades[i], \
																 bandwidth, \
																 dims, \
																 sign, \
																 epsilon, \
																 verbose)
			total_log_likelihood += log_likelihood

		if verbose: print (total_log_likelihood/n_cascades)
		return total_log_likelihood/n_cascades

	@staticmethod
	def estimate (cascades, 
				  log_likelihood, 
				  gradient=None, 
				  bandwidth=1.0, 
				  dims=5,
				  seed=42):
		np.random.seed (seed)
		bounds = [(0,None) for i in range (dims + (dims * dims))] # set the non-negativity bounds on the parameters
		params = np.random.uniform (0, 1, size=dims+(dims*dims)) # random initialization of the parameters
		sign = -1.0 # Multiply so that we can minimize negative log likelihood
		epsilon = 1e-5
		result = minimize (log_likelihood,
						   params,
						   args=(cascades, bandwidth, dims, sign, epsilon, False),
						   method='L-BFGS-B',
						   jac=gradient,
						   bounds=bounds,
						   options={'ftol': 1e-5, "maxls": 50, "maxcor":50, "maxiter":5000, "maxfun": 5000, "disp": True})

		return result

class DCHP:
	""" Discrete coarse hawkes process. """
	@staticmethod
	def log_likelihood_single_cascade (params, 
									   cascade, 
									   bandwidth=1.0, 
									   dims=5, 
									   sign=-1.0, 
									   epsilon=1e-5, 
									   verbose=False):
		# unflatten parameters
		mu = params[0*dims:1*dims]
		b = params[1*dims:2*dims]
		c = params[2*dims:3*dims]
		s = params[3*dims:4*dims]
        
		# initialization
		eta = np.zeros_like (mu)
		intensities = np.zeros_like (mu)
		total_intensities = np.zeros_like (mu)
        
		last_timestamp = -1
		log_intensities = 0 
		se_gate = np.eye (dims)
        
		for timestamp in sorted(cascade.keys()):
			if last_timestamp < 0:
				eta = 0
			else:
				last_sources = cascade[last_timestamp]
				aggregator = np.zeros_like (mu)
				for last_source in last_sources:
					aggregator += ((b[last_source] * c) + (s*se_gate[last_source,:]))
                    
				eta = hputils.exp_kernel (timestamp, last_timestamp, bandwidth) * (eta + aggregator)

			intensities = mu + eta
			current_sources = cascade[timestamp]
			log_intensities += np.log (intensities[current_sources] + epsilon).sum()
			total_intensities += intensities
			last_timestamp = timestamp
        
		ll = (sign) * (log_intensities - total_intensities.sum())
		return ll

	@staticmethod
	def grad_mu (params, cascade, bandwidth=1.0, dims=5, sign=-1.0, epsilon=1e-5, verbose=False):
		pass
    
	@staticmethod
	def grad_alpha (params, cascade, bandwidth=1.0, dims=5, sign=-1.0, epsilon=1e-5, verbose=False):
		pass

	@staticmethod
	def log_likelihood_many_cascades (params, 
									  cascades, 
									  single_cascade_likelihood,
									  bandwidth=1.0, 
									  dims=5, 
									  sign=-1.0, 
									  epsilon=1e-5,
									  l2_coeff=0, 
									  verbose=False):
		n_cascades = len (cascades)
		total_log_likelihood = 0.0
		for i in range (n_cascades):
			log_likelihood = single_cascade_likelihood (params, \
														cascades[i], \
														bandwidth, \
														dims, \
														sign, \
														epsilon, \
														verbose)
			total_log_likelihood += log_likelihood

		total_log_likelihood = total_log_likelihood / n_cascades
		total_log_likelihood += (l2_coeff * np.linalg.norm (params))
		if verbose: print (total_log_likelihood)
		return total_log_likelihood
		
	@staticmethod
	def estimate (cascades, 
				  multiple_cascades_log_likelihood,
				  single_cascade_log_likelihood, 
				  log_optimizer=None,
				  gradient=None, 
				  bandwidth=1.0, 
				  dims=5,
				  l2_coeff=0,
				  seed=42):
		np.random.seed (seed)
		bounds = [(0,None) for i in range (4*dims)] # set the non-negativity bounds on the parameters
		params = np.random.uniform (0, 1, size=4*dims) # random initialization of the parameters
		sign = -1.0 # Multiply so that we can minimize negative log likelihood
		epsilon = 1e-5
		result = minimize (multiple_cascades_log_likelihood,
						   params,
						   args=(cascades, single_cascade_log_likelihood, bandwidth, dims, sign, epsilon, l2_coeff, False),
						   method='L-BFGS-B',
						   jac=gradient,
						   bounds=bounds,
						   options={'ftol': 1e-5, "maxls": 50, "maxcor":50, "maxiter":15000, "maxfun": 15000, "disp": True})

		return result
