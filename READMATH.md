### NGBoost: Probabilistic Boosting in Python

Last update: May 2019.

---

NGBoost is a Python library to use boosting for probabilistic forecasting of classification, regression, and survival predictions, built on top of [Jax](https://github.com/google/jax/tree/master/jax) and [Scikit-Learn](https://scikit-learn.org/stable/). It is designed to be scalable and modular with respect to choice of proper scoring rule, distribution, and base learners.

#### Gradient Boosting

We predict a parametric conditional distribution of an outcome using a combination of base learners [1], 
$$
y|x \sim p_\theta(x), \quad\quad \theta = f^{(0)}(x) - \eta\sum_{i=1}^n \sigma^{(i)}f^{(i)}(x).
$$


In the training process, we first fit a base learner $f^{(0)}(x)$ to predict $\theta$ for the marginal distribution. Then we iteratively fit base learners $f^{(i)}(x) $ to gradients of the proper scoring rule with respect to $\theta$, calculating corresponding scaling parameters $\sigma^{(i)}$ via line search and taking steps of size $\eta$.

#### Proper Scoring Rules

Proper scoring rules are objective functions for forecasting that, when minimized, naturally yield calibrated predictions [2]. We provide support for maximum likelihood (MLE) and the continuous ranked probability score (CRPS), and their analogs in the classification and survival contexts. For parameters $\theta$ and an observed outcome $y$, these proper scoring rule are defined as,
$$
\mathcal{L}_\mathrm{MLE}(\theta,y) = -\log p_\theta(y)\quad\quad\quad\mathcal{L}_\mathrm{CRPS}(\theta,y) = \int_{-\infty}^\infty(F_\theta(x) - \mathbb{1}\{z \leq y\})^2dz
$$
When the model is well-specified, both scoring rules will recover the true model. See [3] for a comprehensive discussion comparing the robustness of these scoring rules to model mis-specification.

The choice of proper scoring rule implies a choice of divergence between distributions [4]. The divergence between the empirical distribution $p_\mathrm{data}$ of training data and modeled distribution $p_\theta$ is minimized in the training process. It turns out the implied divergences are the familiar KL-divergence for MLE, and the Cramer divergence for CRPS ¹.
$$
\begin{align*}
D_{\mathrm{KL}}(p_\mathrm{data},p_\theta) &= \int_{-\infty}^\infty p_\mathrm{data}(y)\log\frac{p_\mathrm{data}(y)}{p_\theta(y)}dy\\
D_{\mathrm{Cramer}}(p_\mathrm{data}, p_\theta) & = \int_{-\infty}^\infty \left(F_\mathrm{data}(y) - F_\theta(y)\right)^2dy
\end{align*}
$$

#### Distribution Choices

We  model the conditional distribution of the outcome as a parametric probability distribution. As a concrete example, consider heteroskedastic regression with a Normal distribution:
$$
p_\theta(x) = N(\mu, \sigma^2)\quad\quad\quad\mu = f_\mu(x), \sigma^2=f_{\sigma^2}(x)
$$


Each parameter $f_\mu(x)$ and $f_{\sigma^2}(x)$ is learned via rounds of gradient boosting.

The choice of parametric distribution implies assumptions about the noise-generating process. For a well-specified model, miscalibration arises when the assumed noise-generating process does not match the true data-generating process [3]. In particular, a true noise distribution that has heavier (or lighter) tails that the assumed noise distribution will result W-shaped (or M-shaped) probability integral transformed histograms.

#### Base Learners

Any choice of base learner compatible with the Scikit-Learn API (specifically, implementing the `fit` and `predict` functions) may be used. We recommend using heavily regularized decision trees or linear base learners, in the spirit of ensembling a set of weak learners. This is also motivated by the empirical success of tree-based gradient boosting methods for certain modalities (such as Kaggle dataset).

#### Natural Gradient

The natural gradient [5] is typically motivated as the direction of steepest descent in parameter space with respect to the divergence metric. By leveraging information geometry, it results in gradient descent steps that are invariant to choice of distribution parameterization. 

For the MLE scoring rule, a natural gradient descent step is defined,
$$
\theta \leftarrow \theta -\eta \mathcal{I}(\theta)^{-1}\nabla_\theta\mathcal{L}_\mathrm{MLE}(\theta,y),
$$


where the Fisher information matrix is defined,
$$
\begin{align*}
\mathcal{I}(\theta)& =\mathrm{Var}_{y\sim p_\theta}[\nabla_\theta -\log p_\theta(y) ]\\
                   & = \mathbb{E}_{y\sim p_\theta}[(\nabla_\theta -\log p_\theta(y))(\nabla_\theta -\log p_\theta(y))^\intercal].
\end{align*}
$$
For exponential family distributions, it turns out the Fisher information matrix is equivalent to the Hessian in the natural parameter space, and so a natural gradient step is equivalent to a Newton-Raphson step. For any other choice of parameterization however, the MLE score is non-convex and the Hessian is not positive semi-definite, so a direct Newton-Raphson step is not recommended. However, for exponential families the Fisher information matrix turns out to be equivalent to the Generalized Gauss-Newton matrix, no matter what the choice of parameterization. We can therefore interpret natural gradient descent as a Newton-Raphson step that uses a positive semi-definite approximation to the Hessian [6]. 

For the CRPS scoring rule, a natural gradient descent step is defined,
$$
\theta \leftarrow \theta - \eta\left( 2 \int_{-\infty}^\infty \nabla_\theta F_\theta(z) \nabla_\theta F_\theta(z)^\intercal dz \right) \nabla_\theta \mathcal{L}_\mathrm{CRPS}(\theta, y).
$$


For heteroskedastic prediction tasks in particular the use of natural gradient significantly improves the speed of the training process. In the example below we fit the marginal distribution of set of i.i.d. normal observations $x^{(i)} \sim N(0,1)$, parameterizing the distribution with $\mu,\log\sigma^2$. 

![natural_grad](examples/visualizations/vis_mle.png)

![natural_grad](examples/visualizations/vis_crps.png)

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
$$
D_{\mathrm{Energy}}(p_\mathrm{data}, p_\theta) = \frac{1}{2} \left(2 \mathbb{E}||X-Y||_2 - \mathbb{E}||X-X'||_2 -\mathbb{E}||Y-Y'||_2\right),
$$
where expectations are taken over independent draws of random variables
$$
X,X' \sim p_\mathrm{data} \quad\quad Y,Y' \sim p_\theta.
$$
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

#### Appendix

$$
\mathcal{I}(\theta) = 2\int_{-\infty}^\infty \nabla_\theta F_\theta(z) \nabla_\theta F(\theta)(z)^\intercal dz = \begin{bmatrix}
\frac{1}{\sqrt{\pi}\sigma} & 0\\ 0 & \frac{1}{2\sqrt\pi \sigma}
\end{bmatrix}
$$

