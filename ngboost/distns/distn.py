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

	def __init__(self, params):
		self._params = params

	def __getitem__(self, key):
		return self.__class__(self._params[:,key])

	def __len__(self):
		return self._params.shape[1]
		
def manifold(Score, Distribution):
	"""
	Mixes a scoring rule and a distribution together to create the resultant "Reimannian Manifold"
	(thus the name of the function). The resulting object has all the parameters of the distribution 
	can be sliced and indexed like one, and carries the distributions `fit` and `sample` methods, but 
	it also carries the appropriate `total_score` and `grad` methods that are inherited through 
	distribution-specific inheritence of the relevant implementation of the scoring rule
	"""
	try:
		DistScore = {S.__base__:S for S in Distribution.scores}[Score]
	except KeyError as err:
		raise ValueError(f'''The scoring rule {Score.__name__} is not implemented for the {Distribution.__name__} distribution.''') from err

	class Manifold(DistScore, Distribution):
		pass
	return Manifold
