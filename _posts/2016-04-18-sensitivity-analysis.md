---
layout: post
title:  "Analysis of sensitivity to outliers for two cost functions"
date:   2016-04-18
categories:
- mathematics
- machine learning
- data science
tags:
- cost functions
- heavy-tailed noise
- sensitivity to outliers
---

In [a previous post][heavytail], we demonstrated that using the usual
least-squares cost function can fail horribly in the presence
of heavy-tailed noise. We also demonstrated that using the $\ell^1$
cost (or a smoothed out version thereof) leads to much better results.

Intuitively, it is clear that $\ell^1$ is less sensitive to
outliers than squared $\ell^2$. But why do we need linear growth
of cost? Why is it not enough to have growth at, say, a power of
$\frac{3}{2}$? In this post, we attempt to go into
this in a little more depth.

First we'll state the general problem, then review the analysis
for least-squares regression, and finally look into the $\ell^1$
case.

* TOC
{:toc}




## Setup

Suppose we have an $n \times m$ matrix $X$ of training data (where $n$
is the number of samples and $m$ is the number of features), and a
target vector of $n$ observations $y$. We believe $y$ obeys

$$ y = X\beta^\ast + \xi \,, $$

where $\beta^\ast$ is an $m$-dimensional vector and $\xi$ is noise.

The goal is to find the *ground truth* $\beta^\ast$.

In this post, we will analyze the following ansatz:
following ansatz:

$$ \min_{\beta} \sum_{i=1}^n F( (X \beta - y)_i^2 ) \,, $$

where $F : \mathbb{R}_+ \to \mathbb{R}$ is everywhere differentiable
and strictly monotonically increasing: $F' > 0$.
(For least squares regression, we simply have $F(z) = z$. For $\ell^1$
minimization, $F(z) = \sqrt{z}$.)
Setting the gradient of the
above expression to zero and noting that

$$ r := X \beta - y = X(\beta - \beta^\ast) - \xi \,, $$

we arrive after a little rearranging at the equation

$$ \sum_{\ell=1}^m \left( \sum_{j=1}^n X_{jk} F'(r_j^2)  X_{j\ell} \right) (\beta - \beta^\ast)_\ell = \sum_{j=1}^n F'(r_j^2) X_{jk} \xi_j \,. \qquad (k = 1, \dotsc, m) \label{key-eq}
$$


## The squared $\ell^2$ cost function

Let us first consider least-squares regression, i.e. the case $F(z) = z$.
Then $F'(z) = 1$ and equation $\eqref{key-eq}$ simplifies to:

$$ X^T X (\beta - \beta^\ast) = X^T \xi \,. $$

To ease analysis of this equation,
we first note that, since $X^T X$ is [positive-definite][posdef],
we can use an orthogonal change of basis in the feature space
$\mathbb{R}^m$ to diagonalize it.

More explicitly, $X^T X$ has a [singular-value decomposition][svd]
of the form

$$X^T X = \mathcal{O} D \mathcal{O}^T$$

for some diagonal matrix $D$ and some orthogonal $m \times m$ matrix $\mathcal{O}$. Thus, replacing $X$ by $X \mathcal{O}$, we have

$$ X^T X = \textrm{diag}(\lambda_1^2, \dotsc, \lambda_m^2) $$

for some real values $\lambda_1, \dotsc, \lambda_m$,
and so $\eqref{key-eq}$ becomes simply

$$ \lambda_k^2 (\beta - \beta^\ast)_k = (X^T \xi)_k \,. \qquad (k = 1, \dotsc, m) $$

Assuming $\lambda_k^2 > 0$ for all $k$, we can further simplify.
Note that

$$ \lambda_k^2 = (X^T X)_{kk} = \sum_{j=1}^n |X_{jk}|^2 = ||X_{\cdot k}||_{\ell^2}^2 \,. $$

So if we $\ell^2$-normalize the $m$ feature vectors $X_{\cdot k}$,
we have the simple relationship

$$ \beta - \beta^\ast = X^T \xi \,. \label{mielc} $$

**Just to be clear:** This simple form is valid only after the following
two preprocessing steps we described above (and these steps do
not commute!):

1. Rotate feature space to make $X^T X$ diagonal.
2. $\ell^2$-normalize the columns of $X$.

In Python, these steps look something like this:

~~~ python
import numpy as np
from scipy import linalg

def normalize(X):
    # rotate feature space to diagonalize $X^T X$
    O, D, OT = linalg.svd(np.dot(X.T, X))
    X = np.dot(X, O)

    # $\ell^2$ normalize columns of $X$
    X = X / np.sqrt(np.sum(X * X, axis=0))

    return X
~~~

Equation $\eqref{mielc}$ relates the noise $\xi$ to how far
our recovered $\beta$ is from the ground truth $\beta^\ast$.
Note that if the components of $\xi$ are mean-zero random variables,
then taking the expectation of this equation gives $\beta = \beta^\ast$,
so the minimizer $\beta$ is an *unbiased* estimator.

Assume $n \geq m$. Then if $X^T X = \mathrm{Id}$, the matrix $X X^T$ is
simply the projection onto the linear space spanned by the columns.
Thus, we have

$$ || \beta - \beta^\ast ||_{\ell^2}^2 = \langle X X^T \xi, \xi \rangle \leq ||\xi||_{\ell^2}^2 \,, $$

with equality iff $\xi$ is in the column space of $X$.

This tells us how to *adversarially* pick $\xi$ so as to maximize the error
in recovery of $\beta$. This worst case is always a possibility.
However, probabilistically, we would typically expect the error
to be smaller. For example, if the entries of $\xi$ are i.i.d.
normal random variables with mean zero, then $\xi$ has no preferred direction.
If the variance of the random variables is $\sigma^2$, then the
expected value of the magnitude of the projection is just

$$ \mathbb{E} ||X X^T \xi||_{\ell^2}^2 = \frac{m}{n} \sigma^2 \,, $$

and hence

$$ \mathbb{E} ||\beta - \beta^\ast ||_{\ell^2} = \sqrt{\frac{m}{n}} \sigma \,.
$$

This is one version of the familiar result that the error decreases
like $\frac{1}{\sqrt{n}}$ as we collect more samples.

However, note that we assumed here finite variance: $\sigma^2 < \infty$.
For heavy-tailed noise all bets are off. As we saw
in [a previous post][heavytail], [Cauchy][cauchy] noise indeed wreaks
havoc with least-squares regression.


### Aside 1: Regularization and well-posedness

From the formula

$$ X^T X (\beta - \beta^\ast) = X^T \xi \label{jecio} $$

we have

$$ \beta - \beta^\ast = (X^T X)^+ X^T \xi$$

where $(\cdot)^+$ denotes the [pseudo-inverse][pinv] of the matrix.
(Theoretically, if $X^T X$ had strictly positive eigenvalues,
we could just take the regular inverse of the matrix.
Numerically, problems arise if the any of the $\lambda_k^2$'s is even
just close to zero.)

One purpose of regularization is to avoid the problem of small
eigenvalues. If we add a regularization term $C ||\beta||_{\ell^2}^2$
to our minimization problem, then equation $\eqref{jecio}$ instead
becomes

$$ (X^T X + C\ \mathrm{Id}) (\beta - \beta^\ast) = X^T \xi \,. $$

The eigenvalues of $X^T X + C\ \mathrm{Id}$ are going to be at least
$C > 0$, and so we can comfortably take the inverse:

$$ \beta - \beta^\ast = (X^T X + C\ \mathrm{Id})^{-1} X^T \xi \,. $$

Mathematically, the problem is *ill-posed* if any of the eigenvalues
of $X^T X$ vanishes. The pseudo-inverse gives us a unique inverse,
but in principle many other solutions exist, too. Adding some
regularization guarantees that all the relevant eigenvalues will be positive
and hence that the problem will be *well-posed*.


### Aside 2: Regularization and the bias-variance tradeoff

Let us again transform $X$ as we did for
equation $\eqref{mielc}$ above and then add a regularization term
$C ||\beta||_{\ell^2}^2$ to our minimization problem. Then
equation $\eqref{mielc}$ becomes

$$ \beta - \beta^\ast + C \beta = X^T \xi \,.$$

Subtracting $C \beta^\ast$ from both sides and dividing by $1+C$, this
says

$$ \beta - \beta^\ast = \underbrace{\frac{X^T \xi}{1 + C}}_{\mathrm{variance}} - \underbrace{\frac{C}{(1+C)} \beta^\ast}_{\mathrm{bias}} \,. $$

Dividing by the factor $1 + C$ reduces the variance for $C > 0$, but at the
expense of adding a bias.

## The $\ell^1$ cost function

Intuitively, it is clear that a linearly growing cost function should
be less sensitive to outliers than a quadratically growing one: outliers
literally *cost* much more. But how exactly does this work out and
why do we need linear growth as opposed to, say, just growth at a power
of $\frac{3}{2}$? I'll try to shed some light on this here.

First let us return briefly to the case of general $F$.
Recall equation $\eqref{key-eq}$, which tells us that the minimizer
$\beta$ satisfies

$$ \sum_{\ell=1}^m \left( \sum_{j=1}^n X_{jk} F'(r_j^2)  X_{j\ell} \right) (\beta - \beta^\ast)_\ell = \sum_{j=1}^n F'(r_j^2) X_{jk} \xi_j \,. \qquad (k = 1, \dotsc, m) \label{key-eq-restated}
 $$

In the least-squares case, we simplified the analysis by transforming $X$
so as to make $X^T X = \mathrm{Id}$ hold. In the more general case,
we note that the matrix

$$ A_{k\ell} = \sum_{j=1}^n X_{jk} F'(r_j^2) X_{j\ell} $$

is still positive-definite due to our assumption
that $F$ is increasing, i.e. $F' > 0$. So in principle we could do similar
transformations as before to turn $A$ into the identity matrix.
Unfortunately, these transformations now depend on the residues

$$ r = X(\beta - \beta^\ast) - \xi \,, $$

which makes the analysis less useful.

Instead, let us focus for now just on the right-hand side of equation
$\eqref{key-eq-restated}$. In the case of the $\ell^1$ cost, we have
$F(z) = 2 \sqrt{z}$ and hence $F'(z) = \frac{1}{\sqrt{z}}$. Thus,
the right-hand side is

$$ \sum_{j=1}^n \frac{X_{jk} \xi_j}{|r_j|} \,.
$$

Here finally we can see why the $\ell^1$ cost function deals better
with outliers: When $|\xi_j|$ is very large, we expect $|r_j| \sim |\xi_j|$,
and hence

$$ \frac{X_{jk} \xi_j}{|r_j|}  \sim \pm X_{jk} \,,
$$

which is much smaller than the potential contribution $X_{jk} \xi_j$ we
would have seen for the least-squares case. The cancellation is not exact,
but in practice it is usually enough to make $\ell^1$ regression work
in the presence of severe outliers.

Note that we just needed that $|F'(r^2)| \sim \frac{1}{|r|}$
for large $|r|$. This is true for smoothed-out versions of the $\ell^1$
cost function as well. All we require is that the cost function
be asymptotically linear.

Finally, note that for cost functions that asymptotically grow as some
power $p > 1$, we have $|F'(r^2)| \sim |r|^{p - 2}$, and hence
for a large outlier $\xi_j$, we would have a term like

$$ X_{jk} \xi_j |r_j|^{p-2} \sim \pm X_{jk} |\xi_j|^{p-1}
$$

in the right-hand side of equation $\eqref{key-eq-restated}$.
Compared
to least squares, the influence of the outlier is certainly reduced for
$1 < p < 2$, but only at $p = 1$ is its influence (approximately) bounded.

While this analysis is not completely rigorous, I hope it gives a better
sense of just why the $\ell^1$ cost works so well for heavy-tailed
noise and extreme outliers.


[pinv]: https://en.wikipedia.org/wiki/Moore–Penrose_pseudoinverse
[cauchy]: https://en.wikipedia.org/wiki/Cauchy_distribution
[heavytail]: http://mkliegl.github.io/heavy-tailed-noise
[posdef]: https://en.wikipedia.org/wiki/Positive-definite_matrix
[svd]: https://en.wikipedia.org/wiki/Singular_value_decomposition
