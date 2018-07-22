import matplotlib.pyplot as plt
import scipy.stats
import torch.distributions
from torch.distributions.log_normal import LogNormal, Normal
from torch.distributions.uniform import Uniform
from ngboost.scores import MLE_surv, CRPS_surv
import numpy as np
from torch.nn.parameter import Parameter
from torch.optim.adam import Adam
from torch.optim import LBFGS

def generate_data(total, frac_cens=0.5, mu=0., logstd=0.):
    Times = LogNormal(mu, float(np.exp(logstd)))
    logodds = np.log(frac_cens + 1e-6) - np.log(1 - frac_cens + 1e-6)
    Censor = LogNormal(mu - logodds, float(np.exp(logstd)))
    T = Times.sample(torch.Size([total]))
    U = Censor.sample(torch.Size([total]))
    Y = torch.min(T, U)
    C = T > U
    return Y, C.float()

def fit(Y, C, Score, mu_init=0., logstd_init=0.):
    lossfn = Score(K=1024)
    mu = torch.nn.Parameter(torch.tensor(mu_init))
    logstd = torch.nn.Parameter(torch.tensor(logstd_init))

    opt = Adam([mu, logstd], lr=0.1)
    opt = LBFGS([mu, logstd], lr=0.5, max_iter=20)

    Dist = LogNormal(mu, logstd.exp())

    prev_loss = 0.
    for i in range(10**10):
        opt.zero_grad()
        loss = lossfn(LogNormal(mu, logstd.exp()), Y, C).mean()
        loss.backward(retain_graph=True)
        opt.step(lambda: loss)
        curr_loss = loss.data.numpy()
        if np.abs(prev_loss - curr_loss) < 1e-4:
            break
        prev_loss = curr_loss

    return mu.data.numpy(), logstd.exp().data.numpy()

def main():
    mu, logstd = 0., np.log(1.)
    print('True mu=%.03f std=%.03f' % (mu, np.exp(logstd)))
    for frac in [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]:
        Y, C = generate_data(10000, frac_cens=frac, mu=mu, logstd=logstd)
        #print(Y[C==0].mean(), Y[C==1].mean())
        print('==== Censoring fraction %.2f ====' % torch.mean(C))
        mle_mu, mle_std = fit(Y, C, MLE_surv, mu_init=0., logstd_init=0.)
        print('MLE  mu=%.03f std=%.03f' % (mle_mu, mle_std))
        crps_mu, crps_std = fit(Y, C, CRPS_surv, mu_init=0., logstd_init=0.)
        print('CRPS mu=%.03f std=%.03f' % (crps_mu, crps_std))


if __name__ == '__main__':
    main()
