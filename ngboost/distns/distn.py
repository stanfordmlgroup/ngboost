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

	@classmethod
	def implementation(cls, Score, scores=None):
		"""
		Finds the distribution-appropriate implementation of Score 
		(using the provided scores if cls.scores is empty) 
		"""
		if scores is None:
			scores = cls.scores
		if Score in scores:
			warn(f'Using Dist={Score.__name__} is unnecessary. NGBoost automatically selects the correct implementation when LogScore or CRPScore is used')
			return Score
		else:
			try:
				return {S.__bases__[-1]:S for S in scores}[Score]
			except KeyError as err:
				raise ValueError(f'The scoring rule {Score.__name__} is not implemented for the {cls.__name__} distribution.') from err

	@classmethod 
	def uncensor_score_implementation(cls, Score):
		"""
		This method does the following:
		1. it finds the distribution-appropriate implementation of the requested score
		2. it implements a version of it that is suitable for uncenscored outcomes
		3. it creates a new version of this distribution that is aware of the dynamically-generated 
			uncensored implementation 
		Returns:
			DistWithUncensoredScore: a new version of this distribution that is aware of the dynamically-generated 
			uncensored score implementation 
		"""
		class DistWithUncensoredScore(cls):
			scores = [cls.implementation(Score, cls.censored_scores).uncensor()]
		return DistWithUncensoredScore

	@classmethod
	def censor(cls):
		"""
		Creates a new dist class from a given dist. The new class has its implemented scores
		set to the censored versions of the scores implemente for distand expects a {time, event} 
		dict as Y instead of a numpy array.
		"""
		class CensoredDist(cls):
			scores = cls.censored_scores

			def fit(Y):
				return cls.fit(Y["Time"])
		return CensoredDist

class RegressionDistn(Distn):
	def predict(self): # predictions for regression are typically conditional means
		return self.mean()

class ClassificationDistn(Distn):
	def predict(self): # returns class assignments
		return np.argmax(self.class_probs(), 1)
