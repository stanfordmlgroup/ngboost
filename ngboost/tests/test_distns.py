import pytest

@pytest.fixture(scope="module")
def dists_scores():
	from ngboost.distns import Bernoulli, k_categorical, Normal, LogNormal, Exponential
	from ngboost.scores import LogScore, CRPS
	return {
		Bernoulli:[LogScore],
		k_categorical(4): [LogScore], 
		Normal: [LogScore, CRPScore], 
		LogNormal: [LogScore, CRPScore], 
		Exponential: [LogScore, CRPScore]
		}

@pytest.fixture(scope="module")
def learners():
	from sklearn.tree import DecisionTreeRegressor
	return [
		DecisionTreeRegressor(criterion='friedman_mse', max_depth=5),
		DecisionTreeRegressor(criterion='friedman_mse', max_depth=3)
	]

class TestRegDistns():
	@pytest.fixture(scope="class")
	def reg_dists(dists_scores):
		from ngboost.distns import RegressionDistn
		return [dist for dist in dists_scores.keys() if isinstance(dist, RegressionDistn)]

	@pytest.fixture(scope="class")
	def reg_data():
		X, Y = load_boston(True)
		return train_test_split(X, Y, test_size=0.2)

	def test_which_reg_dists(reg_dists):
		assert set([Normal, LogNormal, Exponential]) == set(reg_dists)

	def test_dists(dists_scores, learners, reg_dists, reg_data):
		X_reg_train, X_reg_test, Y_reg_train, Y_reg_test = reg_data
		for Dist in reg_dists:
			for Score in dists_scores[Dist]:
				for Learner in learners:
					ngb = NGBRegressor(Dist=Dist, Score=Score, Base=Learner, verbose=False)
					ngb.fit(X_reg_train, Y_reg_train)
					y_pred = ngb.predict(X_reg_test)
					y_dist = ngb.pred_dist(X_reg_test)

# test slicing and ._params