Martingale Matching
===================
The core idea of martingale matching is that instead of learning the score $\nabla_x \log p(x)$, and thus finding the "reverse SDE", we instead learn the generator of the SDE directly. As a simple consecuence of Itô's formula, we have that

$$ M = f(X_t) - f(X_0) - \int_0^t \mathcal{L}_s f(X_s) ds $$

is a martingale, where $f$ is a test function and $\mathcal{L}_s$ is the generator. This is very useful to us, since M being a martingale implies that $\mathbb{E}[M] = 0$. Hence, we can set this as our loss, and thereby learn the generator, which completely characterises our SDE. 

A lot of the supporting code, such as sampling and plotting, was taken from Holderrieth & Erives https://github.com/eje24/iap-diffusion-labs/tree/2026 (MIT License).

