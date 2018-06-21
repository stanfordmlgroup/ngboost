import torch
from scoring_rules import MLE_surv
from base_models import Base_Linear
from torch.distributions.log_normal import LogNormal
from torch.distributions.constraint_registry import transform_to

import numpy as np

class SurvBoost(object):
    def __init__(self, Dist=LogNormal, Score=MLE_surv, Base=Base_Linear, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.Dist = Dist
        self.D = lambda args: Dist(*[transform_to(constraint)(arg) for (param, constraint), arg in zip(Dist.arg_constraints.items(), args)])
        self.Score = Score
        self.Base = Base
        self.base_models = []

    def init_params(self, size):
        Z = lambda l: np.zeros(size).astype(np.float32)
        return [Z(size) for _ in  self.Dist.arg_constraints]

    def pred_param(self, X):
        m, n = X.shape
        params = self.init_params(m)

        for models in self.base_models:
            base_params = [model.predict(X) for model in models]
            params = [p + self.learning_rate * b for p, b in zip(params, base_params)]
        
        return [torch.tensor(p, requires_grad=True) for p in params]

    def sample(self, X, Y, C):
        return X, Y, C

    def fit_base(self, X, resids):
        models = [self.Base().fit(X, rs) for rs in resids]
        self.base_models.append(models)

    def fit(self, X, Y, C):
        for itr in range(self.n_estimators):
            X_batch, Y_batch, C_batch = self.sample(X, Y, C)

            params = self.pred_param(X_batch)

            Forecast = self.D(params)

            score = self.Score(Forecast, torch.Tensor(Y_batch), torch.Tensor(C_batch)).mean()
            print('[iter %d] loss=%f' % (itr, float(score.data.numpy())))

            score.backward()

            resids = [p.grad for p in params]

            self.fit_base(X_batch, resids)

    def pred_dist(self, X):
        params = self.pred_param(X)
        dist = self.D(params)
        return dist

    def pred_mean(self, X):
        dist = self.pred_dist(X)
        return dist.mean.data.numpy()

def main():
    m, n = 7, 3
    X = np.random.rand(m, n).astype(np.float32)
    Y = np.random.rand(m).astype(np.float32)
    C = (np.random.rand(m) > 0.5).astype(np.float32)

    sb = SurvBoost()
    sb.fit(X, Y, C)

    X_test = np.random.rand(m, n).astype(np.float32)
    Y_pred = sb.pred_mean(X_test)

    print(Y_pred)

if __name__ == '__main__':
    main()
