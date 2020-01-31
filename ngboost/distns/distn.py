import numpy as np
from warnings import warn

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

	def __init__(self, params):
		self._params = params

	def __getitem__(self, key):
		return self.__class__(self._params[:,key])

	def __len__(self):
		return self._params.shape[1]

class RegressionDistn(Distn):
	def predict(self): # predictions for regression are typically conditional means
		return self.mean()

class ClassificationDistn(Distn):
	def predict(self): # returns class assignments
		return np.argmax(self.class_probs(), 1)

class SurvivalDistn(Distn):
	def predict(self): # predictions for regression are typically conditional means
		return self.mean()
