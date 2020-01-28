import numpy as np

class Distn(object):
	"""
	User should define:
	- __init__(params) to hold self.params_ = params
	- X_scoring(self, Y) 
	- D_X_scoring(self, Y)
	- sample(self, n)
	- fit(Y)
	- predict(self) mean, mode, whatever (method to call for point prediction
	"""

	def __getitem__(self, key):
		return self.__class__(self.params_[:,key])

	def __len__(self):
		return self.params_.shape[1]

	@property
	def params(self):
		return params_

	def fisher_info(self, n_mc_samples=100):
		grads = np.stack([self.D_nll(Y) for Y in self.sample(n_mc_samples)])
		return np.mean(np.einsum('sik,sij->sijk', grads, grads), axis=0)
		
	# autofit method
