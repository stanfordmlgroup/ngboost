"""SymPy-based auto-generation of NGBoost distributions.

Provides two factories:

- ``make_sympy_log_score`` — creates a ``LogScore`` subclass from a SymPy
  distribution or symbolic expression (score, d_score, metric).
- ``make_distribution`` — creates a complete NGBoost distribution class
  (``RegressionDistn`` or ``ClassificationDistn`` subclass) ready for
  ``NGBRegressor(Dist=...)`` or ``NGBClassifier(Dist=...)``.
"""

import numpy as np
import sympy as sym
import sympy.stats as symstats
from scipy.optimize import minimize
from scipy.special import logsumexp as _scipy_logsumexp
from sympy.utilities.lambdify import lambdify

from ngboost.distns.distn import ClassificationDistn, RegressionDistn
from ngboost.scores import LogScore

# Clip internal (log-space) parameters before exp() to avoid overflow.
# np.exp(709.8) ≈ 1.8e308 (float64 max), np.exp(710) = inf.
# Using 700 keeps values safely within float64 range while being generous
# enough that no reasonable model would be affected.
_EXP_CLIP = 700

# Minimum value for log-transformed natural parameters (e.g. sigma, alpha).
# Lambdified expressions often contain 1/param^2; if param is too small,
# param^2 underflows to 0 → divide-by-zero, or 1/param^2 overflows.
# exp(-150)^2 = exp(-300) ≈ 5e-131 — safely positive in float64 and
# 1/exp(-300) = exp(300) ≈ 2e130 — safely below float64 max.
_LOG_PARAM_MIN = np.exp(-150)


def _density_from_piecewise(pw, y):
    """Convert a Piecewise density to a smooth expression for differentiation.

    SymPy represents discrete PMFs as ``Piecewise`` expressions (e.g.
    ``Piecewise((p, Eq(y, 1)), (1-p, Eq(y, 0)), (0, True))``).  These
    cannot be differentiated w.r.t. continuous parameters.

    For Bernoulli-like distributions with support {0, 1} this function
    returns the equivalent smooth form ``p1**y * p0**(1-y)``.
    """
    if not isinstance(pw, sym.Piecewise):
        return pw

    prob_map = {}
    for value, cond in pw.args:
        if isinstance(cond, sym.Eq):
            lhs, rhs = cond.args
            if lhs == y and rhs.is_number:
                prob_map[int(rhs)] = value
            elif rhs == y and lhs.is_number:
                prob_map[int(lhs)] = value

    # Bernoulli-like: support {0, 1} → p1^y * p0^(1-y)
    if set(prob_map.keys()) == {0, 1}:
        return prob_map[1] ** y * prob_map[0] ** (1 - y)

    return None


def _build_lambdified(expr, ordered_symbols, extra_symbols=None):
    """Lambdify a SymPy expression into a NumPy/SciPy-callable function.

    Uses ``sympy.Dummy`` symbols internally so that user-chosen symbol names
    (e.g. ``beta``) never collide with SciPy function names during lambdify.
    """
    all_symbols = list(ordered_symbols)
    if extra_symbols:
        all_symbols.extend(extra_symbols)

    # Replace every user symbol with a Dummy to avoid name collisions
    dummies = [sym.Dummy(s.name) for s in all_symbols]
    subs = list(zip(all_symbols, dummies))
    expr_safe = expr.subs(subs)

    return lambdify(dummies, expr_safe, modules=["scipy", "numpy"])


def _extract_neglog_sum_terms(score_expr):
    """If *score_expr* has the form ``-log(t1 + t2 + ... + tk)``, return the terms.

    This detects mixture-like NLL expressions that benefit from logsumexp
    evaluation.  Returns ``None`` if the expression does not match.
    """
    for log_term in score_expr.atoms(sym.log):
        arg = log_term.args[0]
        if isinstance(arg, sym.Add) and sym.simplify(score_expr + log_term) == 0:
            return list(arg.args)
    return None


def _try_analytical_fi(
    score_expr, params, y, grad_exprs, sympy_dist
):  # pylint: disable=R0912,R0914,R1702
    """Attempt to compute analytical Fisher Information.

    Strategy
    --------
    1. Compute the Hessian of the score w.r.t. the *raw* (natural) parameters.
    2. If all Hessian elements are free of ``y``, the FI in raw-parameter space
       equals the Hessian (since E[Hessian] = Hessian when it's constant in y).
    3. Transform to internal-parameter space via Jacobian:
       ``FI_internal[i,j] = J[i] * J[j] * FI_raw[i,j]``
       where ``J[k] = theta_k`` if log-transformed, else ``J[k] = 1``.
    4. If any Hessian element depends on ``y`` and ``sympy_dist`` is given,
       try ``symstats.E()``; fall back to None on failure.

    Returns
    -------
    list[list[sympy.Expr]] or None
    """
    n_params = len(params)
    param_symbols = [p for p, _ in params]

    try:
        # Quick check: if any gradient depends on y and we have no sympy_dist,
        # the Hessian will also depend on y and we can't compute E[] —
        # skip the expensive Hessian computation (falls back to MC metric).
        if sympy_dist is None and any(y in g.free_symbols for g in grad_exprs):
            return None

        # Compute raw-parameter Hessian: d²score / d(theta_i) d(theta_j)
        raw_grads = [sym.diff(score_expr, p) for p in param_symbols]
        raw_hessian = [[None] * n_params for _ in range(n_params)]

        all_y_free = True
        for i in range(n_params):
            for j in range(i, n_params):
                hij = sym.simplify(sym.diff(raw_grads[i], param_symbols[j]))
                raw_hessian[i][j] = hij
                raw_hessian[j][i] = hij
                if y in hij.free_symbols:
                    all_y_free = False

        if all_y_free:
            # FI_raw = Hessian (no expectation needed)
            # Transform: FI_internal[i,j] = J_i * J_j * FI_raw[i,j]
            fi_exprs = [[None] * n_params for _ in range(n_params)]
            for i in range(n_params):
                for j in range(n_params):
                    ji = params[i][0] if params[i][1] else sym.Integer(1)
                    jj = params[j][0] if params[j][1] else sym.Integer(1)
                    fi_exprs[i][j] = sym.simplify(ji * jj * raw_hessian[i][j])
            return fi_exprs

        # Some elements depend on y: try E[] with sympy_dist
        if sympy_dist is not None:
            fi_exprs = [[None] * n_params for _ in range(n_params)]
            for i in range(n_params):
                for j in range(i, n_params):
                    hij = raw_hessian[i][j]
                    if y not in hij.free_symbols:
                        eij = hij
                    else:
                        # Substitute the y symbol with the random variable,
                        # then take the unconditional expectation.
                        hij_rv = hij.subs(y, sympy_dist)
                        eij = symstats.E(hij_rv)
                        eij = sym.simplify(eij)
                        if eij.has(symstats.Expectation) or y in eij.free_symbols:
                            return None

                    ji = params[i][0] if params[i][1] else sym.Integer(1)
                    jj = params[j][0] if params[j][1] else sym.Integer(1)
                    fi_ij = sym.simplify(ji * jj * eij)
                    fi_exprs[i][j] = fi_ij
                    fi_exprs[j][i] = fi_ij
            return fi_exprs

        return None
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def make_sympy_log_score(  # pylint: disable=R0912,R0913,R0914,R0915,R0917
    params,
    y,
    score_expr=None,
    sympy_dist=None,
    extra_params=None,
    name=None,
):
    """Create a ``LogScore`` subclass from a symbolic score expression.

    Either ``score_expr`` or ``sympy_dist`` (or both) must be provided.
    When only ``sympy_dist`` is given the score is derived automatically
    as ``-log(density(sympy_dist)(y))``.

    Parameters
    ----------
    params : list[tuple[sympy.Symbol, bool]]
        Each entry is ``(symbol, log_transformed)``.  If ``log_transformed``
        is True the internal NGBoost parameter is ``log(symbol)`` and the
        chain-rule factor ``symbol * d/d(symbol)`` is applied automatically.
        If False the parameter is used with an identity link.
    y : sympy.Symbol
        The symbol representing the observed data in *score_expr*.
    score_expr : sympy.Expr or None
        The *negative log-likelihood* as a SymPy expression
        (i.e. ``-log p(y|theta)``).  If ``None``, it is derived
        automatically from ``sympy_dist``.
    sympy_dist : sympy.stats random variable or None
        A ``sympy.stats`` random variable.  Used to (a) derive
        ``score_expr`` when it is not given, and (b) compute the Fisher
        Information analytically when the Hessian depends on ``y``.
        Falls back to Monte-Carlo metric (inherited from ``LogScore``) on
        failure.
    extra_params : list[sympy.Symbol] or None
        Non-optimised parameters (e.g. ``n`` in BetaBinomial) that appear in
        the score expression but are *not* differentiated.
    name : str or None
        Class name for the generated ``LogScore`` subclass.

    Returns
    -------
    type
        A ``LogScore`` subclass with ``score``, ``d_score``, and ``metric``
        methods.
    """
    if score_expr is None and sympy_dist is None:
        raise ValueError("At least one of score_expr or sympy_dist must be provided")

    if score_expr is None:
        density = symstats.density(sympy_dist)(y)
        if isinstance(density, sym.Piecewise):
            density = _density_from_piecewise(density, y)
            if density is None:
                raise ValueError(
                    "Cannot auto-derive score from Piecewise density. "
                    "Provide score_expr explicitly."
                )
        score_expr = -sym.log(density)
        # Expand log(a^n * b^m / c) into n*log(a) + m*log(b) - log(c)
        # so that lambdify never evaluates y^(alpha-1) numerically
        # (which overflows when alpha<1 and y≈0).
        score_expr = sym.expand_log(score_expr, force=True)

    # ---- Detect logsumexp-amenable form BEFORE log-cleanup ----
    # For mixture-like scores of the form -log(t1 + t2 + ... + tk),
    # cancel() on sub-expressions creates cross-terms that overflow.
    # Detect this first so we can skip the problematic transformation.
    _lse_terms = _extract_neglog_sum_terms(score_expr)

    if _lse_terms is None:
        # Simplify fractions inside log() calls, e.g.
        # log(1 - a/(a+b)) → log(b/(a+b)) → log(b) - log(a+b).
        # Skip for logsumexp path (cancel would merge mixture terms).
        for log_term in list(score_expr.atoms(sym.log)):
            arg = log_term.args[0]
            if isinstance(arg, sym.Add):
                _, denom = arg.as_numer_denom()
                if denom != 1:
                    arg_simplified = sym.cancel(arg)
                    if arg_simplified != arg:
                        score_expr = score_expr.subs(
                            log_term,
                            sym.expand_log(sym.log(arg_simplified), force=True),
                        )

    # Replace log(beta(a,b)) → loggamma(a)+loggamma(b)-loggamma(a+b)
    # and log(gamma(x)) → loggamma(x) so lambdify uses scipy.special.gammaln
    # instead of computing beta()/gamma() first (which underflows to 0
    # for large parameters, causing log(0)).
    _w1 = sym.Wild("_w1")
    _w2 = sym.Wild("_w2")
    score_expr = score_expr.replace(
        sym.log(sym.beta(_w1, _w2)),
        sym.loggamma(_w1) + sym.loggamma(_w2) - sym.loggamma(_w1 + _w2),
    )
    score_expr = score_expr.replace(
        sym.log(sym.gamma(_w1)),
        sym.loggamma(_w1),
    )

    extra_params = extra_params or []
    param_symbols = [p for p, _ in params]
    n_params = len(params)

    _param_names = [str(p) for p, _ in params]
    _extra_names = [str(e) for e in extra_params]
    _EXP_CLIP_VAL = np.exp(_EXP_CLIP)  # max value for log-transformed params

    def _get_param_arrays(self_obj):
        param_arrays = []
        for name, (_, is_log) in zip(_param_names, params):
            val = np.asarray(getattr(self_obj, name), dtype=float)
            if is_log:
                # Log-transformed params must be positive and finite.
                # Lower bound prevents param^2 underflow and 1/param^2
                # overflow in lambdified expressions (see _LOG_PARAM_MIN).
                val = np.clip(val, _LOG_PARAM_MIN, _EXP_CLIP_VAL)
            param_arrays.append(val)
        extra_arrays = [getattr(self_obj, n) for n in _extra_names]
        return param_arrays, extra_arrays

    # ---- gradient expressions (needed for FI in all paths) ----
    grad_exprs = []
    for psym, is_log in params:
        raw_deriv = sym.diff(score_expr, psym)
        if is_log:
            grad_exprs.append(psym * raw_deriv)
        else:
            grad_exprs.append(raw_deriv)

    if _lse_terms is not None:
        # ---- Logsumexp path: numerically stable for mixture-like scores ----
        # Instead of evaluating -log(pdf1 + pdf2 + ...) directly (which
        # underflows to log(0) when individual PDFs are tiny), decompose
        # into log(pdf_k) per term and use scipy.special.logsumexp.
        # Gradients use the identity:
        #   d/dθ[-log Σ_k f_k] = -Σ_k (f_k/Σ_j f_j) · d(log f_k)/dθ
        #                       = -Σ_k softmax_k · d(log f_k)/dθ
        _K = len(_lse_terms)
        _log_term_exprs = [sym.expand_log(sym.log(t), force=True) for t in _lse_terms]
        _log_term_fns = [
            _build_lambdified(lt, [y], param_symbols + extra_params)
            for lt in _log_term_exprs
        ]

        # d(log f_k) / d(internal_param_j) for each term k and param j
        _d_log_term_fns = []
        for lt in _log_term_exprs:
            row = []
            for psym, is_log in params:
                raw = sym.diff(lt, psym)
                g = psym * raw if is_log else raw
                row.append(_build_lambdified(g, [y], param_symbols + extra_params))
            _d_log_term_fns.append(row)

        # Smallest positive float64, used to clamp Y > 0 for log-space
        # evaluation.  Prevents log(0) = -inf cascading to NaN.
        _TINY = np.finfo(float).tiny

        def _eval_log_terms(all_args):
            """Evaluate log(f_k) for each mixture term, clamping to avoid -inf."""
            log_vals = np.stack([fn(*all_args) for fn in _log_term_fns], axis=0)
            # Replace non-finite values BEFORE clipping.  np.clip does not
            # fix NaN, and NaN propagates through logsumexp → weights → D.
            # NaN arises when sigma^2 underflows to 0 (giving 0/0) or when
            # exp(logit) overflows (giving inf/inf).
            log_vals = np.nan_to_num(log_vals, nan=-1e15, posinf=700, neginf=-1e15)
            np.clip(log_vals, -1e15, None, out=log_vals)
            return log_vals

        def _safe_args(all_args):
            """Clamp Y and parameters to safe numerical ranges.

            Prevents exp() overflow inside lambdified expressions (e.g.
            ``exp(logit_w)`` when logit grows beyond ~710) and log(0)
            when Y underflows.
            """
            Y = np.asarray(all_args[0], dtype=float)
            Y_safe = np.clip(Y, _TINY, None)
            safe = [Y_safe]
            for arg in all_args[1:]:
                arr = np.asarray(arg, dtype=float)
                safe.append(np.clip(arr, -_EXP_CLIP, _EXP_CLIP))
            return safe

        def _score_fn(Y_arr, *all_params):
            all_args = _safe_args([Y_arr] + list(all_params))
            return -_scipy_logsumexp(_eval_log_terms(all_args), axis=0)

        def score_method(self, Y):
            param_arrays, extra_arrays = _get_param_arrays(self)
            return _score_fn(Y, *param_arrays, *extra_arrays)

        def d_score_method(self, Y):
            param_arrays, extra_arrays = _get_param_arrays(self)
            all_args = _safe_args([Y] + param_arrays + extra_arrays)
            log_vals = _eval_log_terms(all_args)
            lse = _scipy_logsumexp(log_vals, axis=0, keepdims=True)
            weights = np.exp(log_vals - lse)
            D = np.zeros((len(Y), n_params))
            for j in range(n_params):
                for k in range(_K):
                    grad_k = np.asarray(_d_log_term_fns[k][j](*all_args), dtype=float)
                    grad_k = np.where(np.isfinite(grad_k), grad_k, 0.0)
                    D[:, j] -= weights[k] * grad_k
            # Clamp extreme gradient values that arise when MC samples
            # land far outside all component supports.  These would
            # otherwise corrupt the Fisher Information estimate.
            np.clip(D, -1e6, 1e6, out=D)
            return D

    else:
        # ---- Standard path: direct lambdification ----
        _score_fn = _build_lambdified(score_expr, [y], param_symbols + extra_params)

        _grad_fns = [
            _build_lambdified(g, [y], param_symbols + extra_params) for g in grad_exprs
        ]

        def score_method(self, Y):
            param_arrays, extra_arrays = _get_param_arrays(self)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                vals = _score_fn(Y, *param_arrays, *extra_arrays)
                return np.where(np.isfinite(vals), vals, 1e8)

        def d_score_method(self, Y):
            param_arrays, extra_arrays = _get_param_arrays(self)
            D = np.zeros((len(Y), n_params))
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                for k, gfn in enumerate(_grad_fns):
                    vals = gfn(Y, *param_arrays, *extra_arrays)
                    D[:, k] = np.where(np.isfinite(vals), vals, 0.0)
            return D

    # ---- metric (Fisher Information) ----
    fi_fns = None
    fi_exprs = _try_analytical_fi(score_expr, params, y, grad_exprs, sympy_dist)
    if fi_exprs is not None:
        fi_fns = []
        for i in range(n_params):
            row = []
            for j in range(n_params):
                row.append(
                    _build_lambdified(
                        fi_exprs[i][j], param_symbols, extra_params or None
                    )
                )
            fi_fns.append(row)

    if fi_fns is not None:

        def metric_method(self):
            param_arrays, extra_arrays = _get_param_arrays(self)
            n = param_arrays[0].shape[0]
            FI = np.zeros((n, n_params, n_params))
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                for i in range(n_params):
                    for j in range(n_params):
                        vals = fi_fns[i][j](*param_arrays, *extra_arrays)
                        FI[:, i, j] = np.where(np.isfinite(vals), vals, 0.0)
            # Tiny ridge ensures positive definiteness when overflow
            # zeroed out entries that should be small but positive
            # (e.g. alpha*beta/(alpha+beta)^2 → 0/0 when both huge).
            FI[:, range(n_params), range(n_params)] += 1e-8
            return FI

    elif _lse_terms is not None:
        # Logsumexp path: MC metric with ridge regularization.
        # Mixture gradients can be very large for MC samples that land far
        # from all component modes, making the FI estimate ill-conditioned.
        # A small ridge on the diagonal ensures positive definiteness.

        def metric_method(self, n_mc_samples=100):
            FI = LogScore.metric(self, n_mc_samples)
            FI = np.where(np.isfinite(FI), FI, 0.0)
            FI[:, range(n_params), range(n_params)] += 1e-4
            return FI

    else:
        # Standard path MC fallback: wrap LogScore.metric with NaN
        # filtering and a small ridge for robustness (e.g. when
        # BetaBinomialEstN has n near y, producing NaN gradients).

        def metric_method(self, n_mc_samples=100):
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                FI = LogScore.metric(self, n_mc_samples)
            FI = np.where(np.isfinite(FI), FI, 0.0)
            FI[:, range(n_params), range(n_params)] += 1e-8
            return FI

    # ---- Build the class ----
    class_name = name or "SympyLogScore"

    attrs = {
        "score": score_method,
        "d_score": d_score_method,
    }
    if metric_method is not None:
        attrs["metric"] = metric_method

    cls = type(class_name, (LogScore,), attrs)
    cls._score_fn = _score_fn  # pylint: disable=protected-access
    return cls


def make_distribution(  # pylint: disable=R0912,R0913,R0914,R0915,R0917
    params,
    y,
    sympy_dist=None,
    score_expr=None,
    scipy_dist_cls=None,
    scipy_kwarg_map=None,
    extra_params=None,
    fit_fn=None,
    sample_fn=None,
    mean_fn=None,
    class_prob_exprs=None,
    name="SympyDistribution",
):
    """Create a complete NGBoost distribution from a SymPy definition.

    Combines ``make_sympy_log_score`` (for the score class) with an
    auto-generated distribution subclass (for ``__init__``, ``fit``,
    ``sample``, etc.), producing a class ready for
    ``NGBRegressor(Dist=...)`` or ``NGBClassifier(Dist=...)``.

    When ``class_prob_exprs`` is provided the result is a
    ``ClassificationDistn`` subclass with an auto-generated
    ``class_probs()`` method.  Otherwise the result is a
    ``RegressionDistn`` subclass with ``mean()``.

    Parameters
    ----------
    params : list[tuple[sympy.Symbol, bool]]
        Each entry is ``(symbol, log_transformed)``.
    y : sympy.Symbol
        The symbol representing observed data.
    sympy_dist : sympy.stats random variable or None
        Passed to ``make_sympy_log_score`` for auto-deriving the score
        and analytical Fisher Information.
    score_expr : sympy.Expr or None
        Manual negative log-likelihood.  If ``None``, derived from
        ``sympy_dist``.
    scipy_dist_cls : scipy.stats distribution class or None
        E.g. ``scipy.stats.beta``.  Enables auto-generated ``fit``,
        ``sample``, and ``mean`` (via ``__getattr__`` delegation).
        Mutually exclusive with ``class_prob_exprs``.
    scipy_kwarg_map : dict[str, sympy.Symbol] or None
        Maps scipy keyword argument names to sympy parameter symbols.
        E.g. ``{"a": alpha, "b": beta}`` for ``scipy.stats.beta(a=, b=)``.
        May reference extra_params symbols (e.g. ``{"n": n, ...}``).
    extra_params : list or None
        Non-optimised parameters (e.g. ``n`` in BetaBinomial).
        Each entry is either a bare ``sympy.Symbol`` (no default, for
        backward compatibility) or a ``(symbol, default_value)`` tuple.
        When a default is provided, the generated ``__init__`` accepts
        it as a keyword argument: ``Dist(params, n=20)``.
    fit_fn : callable or None
        Custom ``fit(Y) -> np.ndarray`` override.  Signature:
        ``fit(Y) -> np.array([internal_param_0, internal_param_1, ...])``.
    sample_fn : callable or None
        Custom ``sample(self, m) -> np.ndarray`` override.
    mean_fn : callable or None
        Custom ``mean(self) -> np.ndarray`` override.  When provided,
        takes precedence over scipy delegation and MC fallback.
    class_prob_exprs : list[sympy.Expr] or None
        SymPy expressions for class probabilities ``[P(Y=0), P(Y=1), ...]``.
        When provided, the result is a ``ClassificationDistn`` subclass
        with ``class_probs()`` and categorical ``sample()`` auto-generated.
    name : str
        Class name for the generated distribution.

    Returns
    -------
    type
        A ``RegressionDistn`` or ``ClassificationDistn`` subclass ready
        for NGBoost.

    Examples
    --------
    Regression (Beta distribution):

    >>> import sympy as sp, sympy.stats as symstats, scipy.stats
    >>> alpha, beta, y = sp.symbols("alpha beta y", positive=True)
    >>> Beta = make_distribution(
    ...     params=[(alpha, True), (beta, True)],
    ...     y=y,
    ...     sympy_dist=symstats.Beta("Y", alpha, beta),
    ...     scipy_dist_cls=scipy.stats.beta,
    ...     scipy_kwarg_map={"a": alpha, "b": beta},
    ...     name="Beta",
    ... )
    >>> from ngboost import NGBRegressor
    >>> ngb = NGBRegressor(Dist=Beta)

    Classification (Beta-Bernoulli):

    >>> p = alpha / (alpha + beta)
    >>> BetaBernoulli = make_distribution(
    ...     params=[(alpha, True), (beta, True)],
    ...     y=y,
    ...     sympy_dist=symstats.Bernoulli("Y", p),
    ...     class_prob_exprs=[1 - p, p],
    ...     name="BetaBernoulli",
    ... )
    >>> from ngboost import NGBClassifier
    >>> ngb = NGBClassifier(Dist=BetaBernoulli)
    """
    is_classification = class_prob_exprs is not None

    if is_classification and scipy_dist_cls is not None:
        raise ValueError("class_prob_exprs and scipy_dist_cls are mutually exclusive.")
    if is_classification and len(class_prob_exprs) < 2:
        raise ValueError(
            "class_prob_exprs must contain at least 2 expressions (one per class)."
        )

    _base_cls = ClassificationDistn if is_classification else RegressionDistn

    # ---- 0. Parse extra_params: accept [(sym, default)] or [sym] ----
    _extra_param_info = []  # [(symbol, str_name, default_or_None), ...]
    _extra_symbols = []  # just the symbols, for make_sympy_log_score
    if extra_params:
        for ep in extra_params:
            if isinstance(ep, (tuple, list)):
                sym_obj, default_val = ep
            else:
                sym_obj, default_val = ep, None
            _extra_param_info.append((sym_obj, str(sym_obj), default_val))
            _extra_symbols.append(sym_obj)

    # ---- 1. Create the LogScore subclass ----
    score_cls = make_sympy_log_score(
        params=params,
        y=y,
        score_expr=score_expr,
        sympy_dist=sympy_dist,
        extra_params=_extra_symbols or None,
        name=f"{name}LogScore",
    )

    # ---- 2. Precompute param metadata ----
    param_info = [(str(s), is_log) for s, is_log in params]
    _n_params = len(params)

    # Convert scipy_kwarg_map: {scipy_name: sympy_symbol} → {scipy_name: attr_name}
    _scipy_map = None
    if scipy_kwarg_map is not None:
        _scipy_map = {k: str(v) for k, v in scipy_kwarg_map.items()}

    # ---- 2b. Lambdify class probability expressions (classification) ----
    _class_prob_fns = None
    _n_classes = None
    if is_classification:
        _n_classes = len(class_prob_exprs)
        param_symbols = [s for s, _ in params]
        _extra_syms = _extra_symbols if _extra_symbols else None
        _class_prob_fns = [
            _build_lambdified(expr, param_symbols, _extra_syms)
            for expr in class_prob_exprs
        ]

    # ---- 3. __init__ ----
    def _init(self, internal_params, **kwargs):
        _base_cls.__init__(self, internal_params)
        for i, (pname, is_log) in enumerate(param_info):
            if is_log:
                val = np.exp(np.clip(internal_params[i], -150, _EXP_CLIP))
            else:
                val = internal_params[i]
            setattr(self, pname, val)
        # Set extra (non-optimised) params from kwargs or defaults
        for _, ep_name, ep_default in _extra_param_info:
            setattr(self, ep_name, kwargs.get(ep_name, ep_default))
        if scipy_dist_cls is not None and _scipy_map is not None:
            sc_kwargs = {k: getattr(self, v) for k, v in _scipy_map.items()}
            self.dist = scipy_dist_cls(**sc_kwargs)

    # ---- 4. fit ----
    if fit_fn is not None:
        _fit = staticmethod(fit_fn)
    elif scipy_dist_cls is not None and _scipy_map is not None:

        def _fit_auto(Y):
            # Determine which scipy params to fix
            fit_kw = {}
            if "loc" not in _scipy_map:
                fit_kw["floc"] = 0
            if "scale" not in _scipy_map:
                fit_kw["fscale"] = 1

            result = scipy_dist_cls.fit(Y, **fit_kw)

            # Parse: shape params come first, then loc, scale
            shapes_str = getattr(scipy_dist_cls, "shapes", None) or ""
            shape_names = [s.strip() for s in shapes_str.split(",") if s.strip()]

            fit_dict = {}
            for i, sn in enumerate(shape_names):
                fit_dict[sn] = result[i]
            if len(result) > len(shape_names):
                fit_dict["loc"] = result[len(shape_names)]
            if len(result) > len(shape_names) + 1:
                fit_dict["scale"] = result[len(shape_names) + 1]

            # Map back: for each of our params, find the scipy param
            reverse_map = {v: k for k, v in _scipy_map.items()}
            out = []
            for pname, is_log in param_info:
                scipy_name = reverse_map.get(pname)
                if scipy_name is None:
                    raise ValueError(
                        f"Parameter '{pname}' not found in scipy_kwarg_map"
                    )
                val = fit_dict[scipy_name]
                out.append(np.log(max(val, 1e-10)) if is_log else val)
            return np.array(out)

        _fit = staticmethod(_fit_auto)
    else:
        # Numerical MLE fallback: minimize mean NLL using the
        # already-lambdified score function.
        _score_fn_for_fit = score_cls._score_fn  # pylint: disable=protected-access

        def _fit_mle(Y):
            def nll(internal):
                natural = []
                for v, (_, is_log) in zip(internal, param_info):
                    natural.append(
                        np.exp(np.clip(v, -_EXP_CLIP, _EXP_CLIP)) if is_log else v
                    )
                return np.mean(_score_fn_for_fit(Y, *natural))

            x0 = np.zeros(_n_params)
            res = minimize(nll, x0, method="Nelder-Mead")
            return res.x

        _fit = staticmethod(_fit_mle)

    # ---- 5. sample ----
    if sample_fn is not None:
        _sample = sample_fn
    elif scipy_dist_cls is not None:

        def _sample(self, m):
            return np.array([self.dist.rvs() for _ in range(m)])

    elif is_classification:

        def _sample(self, m):
            probs = self.class_probs()  # (n_samples, K)
            p = np.squeeze(probs)
            if p.ndim == 1:
                return np.random.choice(_n_classes, size=m, p=p).astype(float)
            # Multi-instance: sample one label per instance, m times
            return np.array(
                [
                    [np.random.choice(_n_classes, p=p[i]) for i in range(len(p))]
                    for _ in range(m)
                ]
            ).astype(float)

    else:

        def _sample(self, m):
            raise NotImplementedError(
                f"{name}.sample() requires scipy_dist_cls or a custom sample_fn"
            )

    # ---- 6. params property ----
    @property
    def _params_prop(self):
        d = {pname: getattr(self, pname) for pname, _ in param_info}
        for _, ep_name, _ in _extra_param_info:
            d[ep_name] = getattr(self, ep_name, None)
        return d

    # ---- 6b. __getitem__ override to preserve extra params through slicing ----
    if _extra_param_info:

        def _getitem(self, key):
            obj = self.__class__(
                self._params[:, key]  # pylint: disable=protected-access
            )
            for _, ep_name, _ in _extra_param_info:
                setattr(obj, ep_name, getattr(self, ep_name))
            if scipy_dist_cls is not None and _scipy_map is not None:
                sc_kw = {k: getattr(obj, v) for k, v in _scipy_map.items()}
                obj.dist = scipy_dist_cls(**sc_kw)
            return obj

    else:
        _getitem = None

    # ---- 7. __getattr__ for scipy delegation ----
    def _getattr(self, attr_name):
        if "dist" in self.__dict__ and attr_name in dir(self.__dict__["dist"]):
            return getattr(self.__dict__["dist"], attr_name)
        return None

    # ---- 8. class_probs (classification) or mean() fallback (regression) ----
    _extra_names = [ep_name for _, ep_name, _ in _extra_param_info]

    if is_classification:

        def _class_probs(self):
            param_arrays = [
                np.asarray(getattr(self, pname), dtype=float) for pname, _ in param_info
            ]
            extra_arrays = [getattr(self, n) for n in _extra_names]
            cols = [
                np.asarray(fn(*param_arrays, *extra_arrays), dtype=float)
                for fn in _class_prob_fns
            ]
            return np.column_stack(cols)

    else:
        _class_probs = None

    if mean_fn is not None:
        _mean = mean_fn
    elif not is_classification and scipy_dist_cls is None:

        def _mean(self):
            return np.mean(self.sample(1000), axis=0)

    else:
        _mean = None

    # ---- 9. Build the class ----
    attrs = {
        "n_params": _n_params,
        "scores": [score_cls],
        "__init__": _init,
        "fit": _fit,
        "sample": _sample,
        "params": _params_prop,
    }
    if _class_probs is not None:
        attrs["class_probs"] = _class_probs
    if _mean is not None:
        attrs["mean"] = _mean
    if _getitem is not None:
        attrs["__getitem__"] = _getitem
    if scipy_dist_cls is not None:
        attrs["__getattr__"] = _getattr

    return type(name, (_base_cls,), attrs)
