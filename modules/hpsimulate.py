import numpy as np
import sklearn
from hputils import draw_exponential_random_variable, exp_kernel

def simulate_multivariate_cascades (mu, alpha, end_time, bandwidth=1.0, n_realizations=50, num_events=None, check_stability=False, seed=None):
	""" Modified implementation from https://github.com/stmorse/hawkes

	Parameters:
	===========
	mu(numpy.ndarray):  the base intensity vector
	alpha (numpy.ndarray): the excitation matrix
	end_time (float) : All the generated timestamps should lie within the end_time
	omega (float): the bandwidth parameter (default: 1.0)
	n_realizations (int) : the number of different cascades to be generated (default: 50)
	num_events (int) : The number of events to be generated (default: None)
                     If None, events are generated upto time `end_time`.
                     If not None, attempt to generate upto `num_events` events
                     under the condition that the time does not exceeed `end_time`.
	check_stability (bool): Before simulating, check the stability of HP by spectral analysis (default: False)
		
	seed (int): The seed for the random number generation (default: None)
              This ensures repeatability in the generation process.
              For debugging one should specify a constant seed. 
              But otherwise the seed should be `None`.
	
	Returns:
	========
	cascades (list of lists): Nested list. The inner lists are individual cascades
                            Every element of the list is a pair (source, timestamp)
                            of the generated event.
	"""

	if check_stability:
		w,v = np.linalg.eig (alpha)
		max_eig = np.amax (np.abs(w))
		if max_eig >= 1:
			print (f"(WARNING) Unstable ... max eigen value is: {max_eig}")
      
	prng = sklearn.utils.check_random_state (seed)
	dims = mu.shape[0]

	if num_events is None:
		n_expected = np.iinfo (np.int32).max
	else:
		n_expected = num_events
    
	cascades = list ()
	for i in range (n_realizations): 
		# Initialization
		cascade = list ()  
		n_total = 0
		istar = np.sum(mu)
		s = draw_exponential_random_variable (1./istar, prng)
    
		if s <=end_time and n_total < n_expected:
			# attribute (weighted random sample, since sum(mu)==Istar)
			n0 = int(prng.choice(np.arange(dims), 1, p=(mu / istar)))
			cascade.append((n0, s))
			n_total += 1

		# value of \lambda(t_k) where k is most recent event
		# starts with just the base rate
		last_rates = mu.copy()

		dec_istar = False
		while n_total < n_expected:
			uj, tj = int (cascade[-1][0]), cascade[-1][1]
			if dec_istar:
				# if last event was rejected, decrease istar
				istar = np.sum(rates)
				dec_istar = False
			else:
				# otherwise, we just had an event, so recalc Istar (inclusive of last event)
				istar = np.sum(last_rates) + alpha[uj,:].sum()
        
			s += draw_exponential_random_variable (1./istar, prng)
			if s > end_time:
				break

			# calc rates at time s (use trick to take advantage of rates at last event)
			rates = mu + exp_kernel (s,tj,bandwidth) * (alpha[uj,:] + last_rates - mu)

			# attribution/rejection test
			# handle attribution and thinning in one step as weighted random sample
			diff = istar - np.sum(rates)
			n0 = int (prng.choice(np.arange(dims+1), 1, p=(np.append(rates, diff) / istar)))
      
			if n0 < dims:
				cascade.append((n0, s))
				# update lastrates
				last_rates = rates.copy()
				n_total += 1
			else:
				dec_istar = True
        
		cascades.append (cascade)
	return cascades
