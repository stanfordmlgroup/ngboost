### NGBoost: Natural Gradient Boosting for Probabilistic Prediction

Last update: Sep 2019.

---

NGBoost is a Python library to use boosting for probabilistic prediction (regression, and survival predictions) built on top of [Jax](https://github.com/google/jax/tree/master/jax) and [Scikit-Learn](https://scikit-learn.org/stable/). It is designed to be scalable and modular with respect to choice of proper scoring rule, distribution, and base learners.

#### Gradient Boosting

We predict a parametric conditional distribution of an outcome using a combination of base learners [1], 
<p align="center"><img alt="$$&#10;y|x \sim p_\theta(x), \quad\quad \theta = f^{(0)}(x) - \eta\sum_{i=1}^n \sigma^{(i)}f^{(i)}(x).&#10;$$" src="svgs/6e88bf6b037886ee24b30f67cc37cea7.svg" align="middle" width="338.44368855pt" height="44.89738935pt"/></p>


In the training process, we first fit a base learner <img alt="$f^{(0)}(x)$" src="svgs/fbfd8683327ebfedc037cbf8cbd3cb33.svg" align="middle" width="49.64631704999999pt" height="29.190975000000005pt"/> to predict <img alt="$\theta$" src="svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg" align="middle" width="8.17352744999999pt" height="22.831056599999986pt"/> for the marginal distribution. Then we iteratively fit base learners <img alt="$f^{(i)}(x) $" src="svgs/9807bb9cb30fcdad61cb761a9920169a.svg" align="middle" width="47.74465409999999pt" height="29.190975000000005pt"/> to gradients of the proper scoring rule with respect to <img alt="$\theta$" src="svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg" align="middle" width="8.17352744999999pt" height="22.831056599999986pt"/>, calculating corresponding scaling parameters <img alt="$\sigma^{(i)}$" src="svgs/48f613508f39fc3fedd90b1e9a1d3892.svg" align="middle" width="24.907814249999987pt" height="29.190975000000005pt"/> via line search and taking steps of size <img alt="$\eta$" src="svgs/1d0496971a2775f4887d1df25cea4f7e.svg" align="middle" width="8.751954749999989pt" height="14.15524440000002pt"/>.

#### Proper Scoring Rules

Proper scoring rules are objective functions for forecasting that, when minimized, naturally yield calibrated predictions [2]. We provide support for maximum likelihood (MLE) and the continuous ranked probability score (CRPS), and their analogs in the classification and survival contexts. For parameters <img alt="$\theta$" src="svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg" align="middle" width="8.17352744999999pt" height="22.831056599999986pt"/> and an observed outcome <img alt="$y$" src="svgs/deceeaf6940a8c7a5a02373728002b0f.svg" align="middle" width="8.649225749999989pt" height="14.15524440000002pt"/>, these proper scoring rule are defined as,
<p align="center"><img alt="$$&#10;\mathcal{L}_\mathrm{MLE}(\theta,y) = -\log p_\theta(y)\quad\quad\quad\mathcal{L}_\mathrm{CRPS}(\theta,y) = \int_{-\infty}^\infty(F_\theta(x) - \mathbb{1}\{z \leq y\})^2dz&#10;$$" src="svgs/eff554d3f021190aaef3bb374b6a1121.svg" align="middle" width="529.6928307pt" height="39.61228755pt"/></p>
When the model is well-specified, both scoring rules will recover the true model. See [3] for a comprehensive discussion comparing the robustness of these scoring rules to model mis-specification.

The choice of proper scoring rule implies a choice of divergence between distributions [4]. The divergence between the empirical distribution <img alt="$p_\mathrm{data}$" src="svgs/58bc879565c6919a94b750e51deddcae.svg" align="middle" width="33.772988699999985pt" height="14.15524440000002pt"/> of training data and modeled distribution <img alt="$p_\theta$" src="svgs/5f8e143b80227a85682626ace43e37cb.svg" align="middle" width="14.88586109999999pt" height="14.15524440000002pt"/> is minimized in the training process. It turns out the implied divergences are the familiar KL-divergence for MLE, and the Cramer divergence for CRPS ¹.
<p align="center"><img alt="$$&#10;\begin{align*}&#10;D_{\mathrm{KL}}(p_\mathrm{data},p_\theta) &amp;= \int_{-\infty}^\infty p_\mathrm{data}(y)\log\frac{p_\mathrm{data}(y)}{p_\theta(y)}dy\\&#10;D_{\mathrm{Cramer}}(p_\mathrm{data}, p_\theta) &amp; = \int_{-\infty}^\infty \left(F_\mathrm{data}(y) - F_\theta(y)\right)^2dy&#10;\end{align*}&#10;$$" src="svgs/62ebb58cd9a3013b15c00ef4ec8889d2.svg" align="middle" width="345.5918169pt" height="85.98429674999998pt"/></p>

#### Distribution Choices

We  model the conditional distribution of the outcome as a parametric probability distribution. As a concrete example, consider heteroskedastic regression with a Normal distribution:
<p align="center"><img alt="$$&#10;p_\theta(x) = N(\mu, \sigma^2)\quad\quad\quad\mu = f_\mu(x), \sigma^2=f_{\sigma^2}(x)&#10;$$" src="svgs/69261c73e7f013d008b251585210b72d.svg" align="middle" width="334.41731565pt" height="18.905967299999997pt"/></p>


Each parameter <img alt="$f_\mu(x)$" src="svgs/768e74d928782da8363b7a87218a17fe.svg" align="middle" width="39.04318109999999pt" height="24.65753399999998pt"/> and <img alt="$f_{\sigma^2}(x)$" src="svgs/236392c07e915194689ac4cbc3e143cc.svg" align="middle" width="45.49624364999999pt" height="24.65753399999998pt"/> is learned via rounds of gradient boosting.

The choice of parametric distribution implies assumptions about the noise-generating process. For a well-specified model, miscalibration arises when the assumed noise-generating process does not match the true data-generating process [3]. In particular, a true noise distribution that has heavier (or lighter) tails that the assumed noise distribution will result W-shaped (or M-shaped) probability integral transformed histograms.

#### Base Learners

Any choice of base learner compatible with the Scikit-Learn API (specifically, implementing the `fit` and `predict` functions) may be used. We recommend using heavily regularized decision trees or linear base learners, in the spirit of ensembling a set of weak learners. This is also motivated by the empirical success of tree-based gradient boosting methods for certain modalities (such as Kaggle dataset).

#### Natural Gradient

The natural gradient [5] is typically motivated as the direction of steepest descent in parameter space. By leveraging information geometry, it results in gradient descent steps that are invariant to choice of distribution parameterization. 

For the MLE scoring rule, a natural gradient descent step is defined,
<p align="center"><img alt="$$&#10;\theta \leftarrow \theta -\eta \mathcal{I}(\theta)^{-1}\nabla_\theta\mathcal{L}_\mathrm{MLE}(\theta,y),&#10;$$" src="svgs/b5a0741a88fa241d6e8481a8ddb8e7d8.svg" align="middle" width="223.09870439999997pt" height="18.312383099999998pt"/></p>


where the Fisher information matrix is defined,
<p align="center"><img alt="$$&#10;\begin{align*}&#10;\mathcal{I}(\theta)&amp; =\mathrm{Var}_{y\sim p_\theta}[\nabla_\theta -\log p_\theta(y) ]\\&#10;                   &amp; = \mathbb{E}_{y\sim p_\theta}[(\nabla_\theta -\log p_\theta(y))(\nabla_\theta -\log p_\theta(y))^\intercal].&#10;\end{align*}&#10;$$" src="svgs/38a48440c02fe3f0ac16e514454b5f03.svg" align="middle" width="347.842671pt" height="41.68947585pt"/></p>
For exponential family distributions, it turns out the Fisher information matrix is equivalent to the Hessian in the natural parameter space, and so a natural gradient step is equivalent to a Newton-Raphson step. For any other choice of parameterization however, the MLE score is non-convex and the Hessian is not positive semi-definite, so a direct Newton-Raphson step is not recommended. However, for exponential families the Fisher information matrix turns out to be equivalent to the Generalized Gauss-Newton matrix, no matter what the choice of parameterization. We can therefore interpret natural gradient descent as a Newton-Raphson step that uses a positive semi-definite approximation to the Hessian [6]. 

For the CRPS scoring rule, a natural gradient descent step is defined,
<p align="center"><img alt="$$&#10;\theta \leftarrow \theta - \eta\left( 2 \int_{-\infty}^\infty \nabla_\theta F_\theta(z) \nabla_\theta F_\theta(z)^\intercal dz \right) \nabla_\theta \mathcal{L}_\mathrm{CRPS}(\theta, y).&#10;$$" src="svgs/ace80a1a727b53925a4a6d507565439e.svg" align="middle" width="403.05408165pt" height="40.1830407pt"/></p>


For heteroskedastic prediction tasks in particular the use of natural gradient significantly improves the speed of the training process. In the example below we fit the marginal distribution of set of i.i.d. normal observations <img alt="$x^{(i)} \sim N(0,1)$" src="svgs/107e8f18478079d9610c4194345099c2.svg" align="middle" width="98.58914174999998pt" height="29.190975000000005pt"/>, parameterizing the distribution with <img alt="$\mu,\log\sigma^2$" src="svgs/cfa5cfc2906fe6d5a3c22962c67576d9.svg" align="middle" width="57.71880509999998pt" height="26.76175259999998pt"/>. 

![natural_grad](examples/visualizations/natural_grad.png)

#### Usage

Installation:

```
pip3 install ngboost
```

Below we show an example of fitting a linear model using 1-dimensional covariates.

```python
# Todo.

```

The above examples result in the following prediction intervals.

For further details the `examples/` folder.

#### Footnotes

¹ While outside the scope of univariate regression tasks, we note that the multivariate generalization of the Cramer divergence is the Energy Distance [7,8], defined as
<p align="center"><img alt="$$&#10;D_{\mathrm{Energy}}(p_\mathrm{data}, p_\theta) = \frac{1}{2} \left(2 \mathbb{E}||X-Y||_2 - \mathbb{E}||X-X'||_2 -\mathbb{E}||Y-Y'||_2\right),&#10;$$" src="svgs/9fa4c771cec2b03ff38879e8d64092c0.svg" align="middle" width="494.46086084999996pt" height="32.990165999999995pt"/></p>
where expectations are taken over independent draws of random variables
<p align="center"><img alt="$$&#10;X,X' \sim p_\mathrm{data} \quad\quad Y,Y' \sim p_\theta.&#10;$$" src="svgs/6af188d0a4ed748de74c3698c282cf7d.svg" align="middle" width="207.97349924999997pt" height="16.3763325pt"/></p>
The multivariate generalization of the KL divergence is a straightforward extension of the univariate case.

#### References

[1] J. H. Friedman, Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics*, **29** (2001) 1189–1232.

[2] T. Gneiting & A. E. Raftery, Strictly Proper Scoring Rules, Prediction, and Estimation. *Journal of the American Statistical Association*, **102** (2007) 359–378.

[3] M. Gebetsberger, J. W. Messner, G. J. Mayr, & A. Zeileis, Estimation Methods for Nonhomogeneous Regression Models: Minimum Continuous Ranked Probability Score versus Maximum Likelihood. *Monthly Weather Review*, **146** (2018) 4323–4338. https://doi.org/10.1175/MWR-D-17-0364.1.

[4] A. P. Dawid, The geometry of proper scoring rules. *Annals of the Institute of Statistical Mathematics*, **59** (2007) 77–93. https://doi.org/10.1007/s10463-006-0099-8.

[5] S. Amari, Natural Gradient Works Efficiently in Learning. *Neural Computation*, (1998) 29.

[6] J. Martens, *New insights and perspectives on the natural gradient method* (2014).

[7] G. J. Székely & M. L. Rizzo, Energy statistics: A class of statistics based on distances. *Journal of Statistical Planning and Inference*, **143** (2013) 1249–1272. https://doi.org/10.1016/j.jspi.2013.03.018.

[8] M. G. Bellemare, I. Danihelka, W. Dabney, S. Mohamed, B. Lakshminarayanan, S. Hoyer, & R. Munos, *The Cramer Distance as a Solution to Biased Wasserstein Gradients* (2017).

#### License

This library is available under the MIT License.

