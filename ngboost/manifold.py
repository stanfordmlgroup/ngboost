def manifold(Score, Distribution):
	"""
	Mixes a scoring rule and a distribution together to create the resultant "Reimannian Manifold"
	(thus the name of the function). The resulting object has all the parameters of the distribution 
	can be sliced and indexed like one, and carries the distributions `fit` and `sample` methods, but 
	it also carries the appropriate `total_score` and `grad` methods that are inherited through 
	distribution-specific inheritence of the relevant implementation of the scoring rule
	"""
	if Score in Distribution.scores:
		DistScore = Score
		warn(f'Using Dist={Score.__name__} is unnecessary. NGBoost automatically selects the correct implementation when LogScore or CRPScore is used')
	else:
		try:
			DistScore = {S.__base__:S for S in Distribution.scores}[Score]
		except KeyError as err:
			raise ValueError(f'The scoring rule {Score.__name__} is not implemented for the {Distribution.__name__} distribution.') from err

	class Manifold(DistScore, Distribution):
		pass
	return Manifold
