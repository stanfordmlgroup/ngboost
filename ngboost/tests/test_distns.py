import pytest

from ngboost.distns import Normal, LogNormal, Exponential, Bernoulli, k_categorical
from ngboost.scores import LogScore, CRPScore
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBRegressor, NGBClassifier, NGBSurvival

from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split

import numpy as np

@pytest.fixture(scope="module")
def learners():
	# add some learners that aren't trees
	return [
		DecisionTreeRegressor(criterion='friedman_mse', max_depth=5),
		DecisionTreeRegressor(criterion='friedman_mse', max_depth=3)
	]

class TestRegDistns():

	@pytest.fixture(scope="class")
	def reg_dists(self):
		# try importing these in the class but outside the fn
		return {
			Normal: [LogScore, CRPScore], 
			LogNormal: [LogScore, CRPScore], 
			Exponential: [LogScore, CRPScore]
			}

	@pytest.fixture(scope="class")
	def reg_data(self):
		X, Y = load_boston(True)
		return train_test_split(X, Y, test_size=0.2)

	def test_dists(self, learners, reg_dists, reg_data):
		X_reg_train, X_reg_test, Y_reg_train, Y_reg_test = reg_data
		for Dist, Scores in reg_dists.items():
			for Score in Scores:
				for Learner in learners:
					# test early stopping features
					ngb = NGBRegressor(Dist=Dist, Score=Score, Base=Learner, verbose=False)
					ngb.fit(X_reg_train, Y_reg_train)
					y_pred = ngb.predict(X_reg_test)
					y_dist = ngb.pred_dist(X_reg_test)
					# test properties of output

	# test what happens when a dist that's not regression is passed in

# test survival stuff

class TestClsDistns():

	@pytest.fixture(scope="class")
	def cls_data(self):
		X, Y = load_breast_cancer(True)
		return train_test_split(X, Y, test_size=0.2)

	def test_bernoulli(self, learners, cls_data):
		X_cls_train, X_cls_test, Y_cls_train, Y_cls_test = cls_data
		for Learner in learners:
			# test early stopping features
			# test other args, n_trees, LR, minibatching- args as fixture
			ngb = NGBClassifier(Dist=Bernoulli, Score=LogScore, Base=Learner, verbose=False)
			ngb.fit(X_cls_train, Y_cls_train)
			y_pred = ngb.predict(X_cls_test)
			y_prob = ngb.predict_proba(X_cls_test)
			y_dist = ngb.pred_dist(X_cls_test)
			# test properties of output

	def test_categorical(self, learners, cls_data):
		X_cls_train, X_cls_test, Y_cls_train, Y_cls_test = cls_data
		for K in [2,4,7]:
			Dist = k_categorical(K)
			Y_cls_train = np.random.randint(0,K,(len(Y_cls_train)))

			for Learner in learners:
				# test early stopping features
				ngb = NGBClassifier(Dist=Dist, Score=LogScore, Base=Learner, verbose=False)
				ngb.fit(X_cls_train, Y_cls_train)
				y_pred = ngb.predict(X_cls_test)
				y_prob = ngb.predict_proba(X_cls_test)
				y_dist = ngb.pred_dist(X_cls_test)
				# test properties of output


# test slicing and ._params