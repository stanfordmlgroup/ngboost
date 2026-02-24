# SymPy-Based Auto-Generated Distributions

## Why SymPy?

Hand-writing `d_score()` and `metric()` methods for NGBoost distributions is
tedious and error-prone. Each distribution requires:

- Correct symbolic derivatives (often involving digamma/polygamma functions)
- Chain-rule adjustments for log-transformed parameters
- Fisher Information computation (expected Hessian)

Two factories are provided:

- `make_distribution` — creates a **complete distribution class** ready for
  `NGBRegressor(Dist=...)` or `NGBClassifier(Dist=...)`. This is the
  recommended entry point for both regression and classification.
- `make_sympy_log_score` — creates just the **LogScore subclass** (for advanced
  use when you need full control over the distribution wrapper).

## Built-in Distributions

All SymPy-powered distributions are importable directly:

```python
from ngboost.distns import Beta, BetaBernoulli, BetaBinomial, BetaBinomialEstN, LogitNormal
```

| Distribution      | Type           | Factory used          | Use case                                  |
|------------------|----------------|-----------------------|-------------------------------------------|
| Beta             | Regression     | `make_distribution`   | Bounded (0,1) outcomes                    |
| LogitNormal      | Regression     | `make_distribution`   | Bounded (0,1) with logistic-normal model  |
| BetaBernoulli    | Classification | `make_distribution`   | Binary with calibrated uncertainty         |
| BetaBinomial     | Regression     | `make_distribution`   | Overdispersed count data (known n)        |
| BetaBinomialEstN | Regression     | `make_distribution`   | Overdispersed count data (unknown n)      |

## Quickstart: Using Built-in Distributions

All SymPy-powered distributions work out of the box — just import and go:

### Regression

```python
from ngboost import NGBRegressor
from ngboost.distns import Beta  # or LogitNormal, BetaBinomial, BetaBinomialEstN

ngb = NGBRegressor(Dist=Beta)
ngb.fit(X_train, Y_train)
dists = ngb.pred_dist(X_test)      # full predictive distributions
dists.ppf(0.1), dists.ppf(0.9)     # quantiles
```

### Classification

```python
from ngboost import NGBClassifier
from ngboost.distns import BetaBernoulli

ngb = NGBClassifier(Dist=BetaBernoulli)
ngb.fit(X_train, Y_train)
ngb.predict_proba(X_test)           # class probabilities with uncertainty
```

## Creating Custom Distributions with `make_distribution`

To define your own distribution, provide SymPy symbols and either a
`sympy.stats` distribution or a manual score expression:

### From sympy.stats (auto-derives everything)

```python
import sympy as sp
import sympy.stats as symstats
import scipy.stats
from ngboost.distns.sympy_utils import make_distribution

alpha, beta, y = sp.symbols("alpha beta y", positive=True)

MyBeta = make_distribution(
    params=[(alpha, True), (beta, True)],
    y=y,
    sympy_dist=symstats.Beta("Y", alpha, beta),
    scipy_dist_cls=scipy.stats.beta,
    scipy_kwarg_map={"a": alpha, "b": beta},
    name="MyBeta",
)
```

### Classification with class_prob_exprs

Pass `class_prob_exprs` — a list of SymPy expressions giving
`[P(Y=0), P(Y=1), ...]`. The factory produces a `ClassificationDistn`
with `class_probs()` and categorical `sample()` auto-generated:

```python
alpha, beta, y = sp.symbols("alpha beta y")
p = alpha / (alpha + beta)

MyBetaBernoulli = make_distribution(
    params=[(alpha, True), (beta, True)],
    y=y,
    sympy_dist=symstats.Bernoulli("Y", p),
    class_prob_exprs=[1 - p, p],
    name="MyBetaBernoulli",
)
```

The factory auto-derives score, gradients, and Fisher Information
from the SymPy distribution, and auto-generates fit/sample/mean (regression)
or class_probs/sample (classification).

## When You Need a Manual Score Expression

For distributions without a `sympy.stats` equivalent (e.g., LogitNormal), or
where the auto-derived density is too complex, you can provide the score
expression explicitly:

```python
mu, sigma, y = sp.symbols("mu sigma y", positive=True)
logit_y = sp.log(y / (1 - y))
score = (
    sp.Rational(1, 2) * sp.log(2 * sp.pi) + sp.log(sigma)
    + (logit_y - mu)**2 / (2 * sigma**2)
    + sp.log(y) + sp.log(1 - y)
)

LogitNormalLogScore = make_sympy_log_score(
    params=[(mu, False), (sigma, True)],
    y=y,
    score_expr=score,
    name="LogitNormalLogScore",
)
```

## Parameter Handling

### Log-transformed parameters

Most NGBoost parameters are log-transformed to ensure positivity (e.g.,
`alpha = exp(log_alpha)`). When `log_transformed=True`, the factory
automatically applies the chain rule:

```
d/d(log theta) = theta * d/d(theta)
```

### Identity parameters

Some parameters use an identity link (e.g., the mean `mu` of a Normal).
Set `log_transformed=False` and derivatives are computed directly.

### Extra (non-optimized) parameters

Parameters like `n` in BetaBinomial appear in the score but are not
optimized by NGBoost. Pass them via `extra_params` with a default value:

```python
make_distribution(
    params=[(alpha, True), (beta, True)],
    y=y,
    score_expr=score_expr,
    extra_params=[(n, 1)],  # not differentiated, default value 1
    fit_fn=my_fit,
    name="BetaBinomial",
)
```

The generated `__init__` accepts extra params as keyword arguments
(`Dist(params, n=20)`) and the score class reads `self.n` automatically.
Extra params are preserved through slicing (`dists[0:5]`).

To use a specific `n`, subclass with the value baked in:

```python
from ngboost.distns import BetaBinomial

class BetaBinomial20(BetaBinomial):
    def __init__(self, params):
        super().__init__(params, n=20)
    def fit(Y):
        return BetaBinomial.fit(Y, n=20)

ngb = NGBRegressor(Dist=BetaBinomial20)
```

For the lower-level `make_sympy_log_score`, pass bare symbols instead:

```python
make_sympy_log_score(
    params=[(alpha, True), (beta, True)],
    y=y,
    score_expr=score_expr,
    extra_params=[n],  # bare symbol, no default
)
```

## Fisher Information Strategy

The factory computes the Fisher Information (metric) analytically when
possible, using a three-tier strategy:

1. **y-free Hessian**: If the second derivatives of the score don't depend
   on `y`, then `FI = Hessian` directly (no expectation needed). This is
   the case for exponential family distributions like Beta and Gamma.

2. **SymPy E[]**: If the Hessian depends on `y` and `sympy_dist` is
   provided, the factory substitutes `y` with the random variable and
   computes `E[Hessian]` symbolically. This works for Normal, Bernoulli,
   and other distributions where SymPy can evaluate the expectation.

3. **Monte Carlo fallback**: If neither analytical approach succeeds, the
   class inherits `LogScore.metric()` which estimates the FI via Monte
   Carlo sampling. This is used for LogitNormal and BetaBinomial.

## Discrete Distribution Support

For discrete distributions like Bernoulli, `sympy.stats.density()` returns
a `Piecewise` expression. The factory automatically converts Bernoulli-like
distributions (support {0, 1}) to the smooth form `p^y * (1-p)^(1-y)` for
differentiation:

```python
alpha, beta, y = sp.symbols("alpha beta y")
p = alpha / (alpha + beta)

# Just pass the Bernoulli — Piecewise is handled automatically
BetaBernoulliLogScore = make_sympy_log_score(
    params=[(alpha, True), (beta, True)],
    y=y,
    sympy_dist=symstats.Bernoulli("Y", p),
)
```

## Worked Examples

### Continuous — auto-derived from sympy.stats (Beta)

```python
alpha, beta, y = sp.symbols("alpha beta y", positive=True)

BetaLogScore = make_sympy_log_score(
    params=[(alpha, True), (beta, True)],
    y=y,
    sympy_dist=symstats.Beta("Y", alpha, beta),
)
```

FI path: y-free Hessian (tier 1).

### Continuous with manual score (LogitNormal)

```python
mu, sigma, y = sp.symbols("mu sigma y", positive=True)
logit_y = sp.log(y / (1 - y))
score = (
    sp.Rational(1, 2) * sp.log(2 * sp.pi) + sp.log(sigma)
    + (logit_y - mu)**2 / (2 * sigma**2)
    + sp.log(y) + sp.log(1 - y)
)

LogitNormalLogScore = make_sympy_log_score(
    params=[(mu, False), (sigma, True)],
    y=y,
    score_expr=score,
)
```

FI path: Monte Carlo fallback (tier 3) — no `sympy_dist` available.

### Discrete classification (BetaBernoulli)

```python
alpha, beta, y = sp.symbols("alpha beta y")
p = alpha / (alpha + beta)

BetaBernoulli = make_distribution(
    params=[(alpha, True), (beta, True)],
    y=y,
    sympy_dist=symstats.Bernoulli("Y", p),
    class_prob_exprs=[1 - p, p],
    name="BetaBernoulli",
)
```

FI path: SymPy E[] (tier 2) — Hessian depends on y, but E[] is tractable.

### Extra non-optimized params (BetaBinomial)

```python
alpha, beta = sp.symbols("alpha beta", positive=True)
y, n = sp.symbols("y n", positive=True, integer=True)

score = -(
    sp.loggamma(n + 1) - sp.loggamma(y + 1) - sp.loggamma(n - y + 1)
    + sp.loggamma(y + alpha) + sp.loggamma(n - y + beta)
    - sp.loggamma(n + alpha + beta)
    + sp.loggamma(alpha + beta)
    - sp.loggamma(alpha) - sp.loggamma(beta)
)

BetaBinomial = make_distribution(
    params=[(alpha, True), (beta, True)],
    y=y,
    score_expr=score,
    extra_params=[(n, 1)],    # fixed, not optimized
    fit_fn=my_fit,
    sample_fn=my_sample,
    mean_fn=my_mean,
    name="BetaBinomial",
)
```

FI path: Monte Carlo fallback (tier 3) — E[] over BetaBinomial produces
unevaluated sums.

### Estimated n (BetaBinomialEstN)

When `n` is unknown, make it a third optimized parameter:

```python
BetaBinomialEstN = make_distribution(
    params=[(alpha, True), (beta, True), (n, True)],  # n is optimized
    y=y,
    score_expr=score,
    fit_fn=estn_fit,
    sample_fn=estn_sample,
    mean_fn=estn_mean,
    name="BetaBinomialEstN",
)
```

FI path: Monte Carlo fallback (tier 3). Note: estimating `n` from count
data is harder than fixing it — prefer `BetaBinomial` when `n` is known.

## Testing

### Gradient correctness

`tests/test_score.py` uses `scipy.optimize.approx_fprime` to compare
`d_score()` against finite differences of `score()`. All SymPy
distributions (Beta, LogitNormal, BetaBernoulli, BetaBinomial,
BetaBinomialEstN) are included.

### Metric correctness

`tests/test_score.py` compares analytical `metric()` against a Monte Carlo
estimate. Distributions with analytical FI (Beta, BetaBernoulli) are
included in the metric test.

### Existing distribution parity

`tests/test_sympy_existing_distns.py` verifies that SymPy-generated score
classes for Normal, Gamma, and Poisson match their hand-written
implementations numerically (score, d_score, and metric) — using only
`sympy_dist` (no manual score expressions).

## Example Notebooks

### Built-in distributions (import and go)

| Notebook                          | Distribution | Use case |
|----------------------------------|-------------|----------|
| `notebooks/example_beta.ipynb`           | `Beta` | Bounded (0,1) regression |
| `notebooks/example_logitnormal.ipynb`    | `LogitNormal` | Bounded (0,1) with logistic-normal model |
| `notebooks/example_betabernoulli.ipynb`  | `BetaBernoulli` | Binary classification with uncertainty |
| `notebooks/example_betabinomial.ipynb`   | `BetaBinomial` | Overdispersed counts with fixed `n` (subclass pattern) |
| `notebooks/example_betabinomial_estn.ipynb` | `BetaBinomialEstN` | Overdispersed counts with estimated `n` |
| `notebooks/example_laplace.ipynb`        | `Laplace` | Robust regression with heavy tails |
| `notebooks/example_t.ipynb`              | `T`, `TFixedDf`, `Cauchy` | Student's t with learnable or fixed degrees of freedom |
| `notebooks/example_gamma.ipynb`          | `Gamma` | Positive continuous outcomes (income, duration) |
| `notebooks/example_poisson.ipynb`        | `Poisson` | Count data (events per unit time) |
| `notebooks/example_weibull.ipynb`        | `Weibull` | Survival and reliability analysis |
| `notebooks/example_halfnormal.ipynb`     | `HalfNormal` | Positive outcomes with single scale parameter |
| `notebooks/example_multivariate_normal.ipynb` | `MultivariateNormal(k)` | Multi-output regression with correlated predictions |

### Factory demos and deep dives

| Notebook                          | Pattern demonstrated |
|----------------------------------|-----------------------------------------------|
| `notebooks/example_normal.ipynb`         | `make_distribution` from scratch, verified against built-in Normal |
| `notebooks/example_mixture_lognormal.ipynb` | Advanced: 8-param mixture of 3 log-normals with logsumexp path |
| `notebooks/sympy_normal_demo.ipynb`      | Deep dive: symbolic expressions, verification against hand-written code |

## Reference: SymPy-Powered Distributions

| Distribution      | Params (link)                         | Score source   | FI method      |
|------------------|---------------------------------------|----------------|----------------|
| Beta             | alpha (log), beta (log)               | auto from dist | y-free Hessian |
| BetaBernoulli    | alpha (log), beta (log)               | auto from dist | SymPy E[]      |
| BetaBinomial     | alpha (log), beta (log) + n (extra)   | manual         | Monte Carlo    |
| BetaBinomialEstN | alpha (log), beta (log), n (log)      | manual         | Monte Carlo    |
| LogitNormal      | mu (identity), sigma (log)            | manual         | Monte Carlo    |
