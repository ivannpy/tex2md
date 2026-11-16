---
title: "Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions"
authors: ["Emiel Hoogeboom", "Didrik Nielsen", "Priyank Jaini", "Patrick Forré", "Max Welling"]
year: 2021
arxiv: "2102.05379"
arxiv_version: "v3"
marginnote:
  - bookmd5: d1e35a9be9b32c370d9fea2bedc22433c0b552e7619cfa1733b49e3a4d51721f
    path: "PCIC/Tesis/core/Argmax Flows and Multinomial Diffusion- Learning Categorical Distributions.pdf"
source: "arXiv LaTeX, tex2md 0.1"
converted: 2026-11-16
---

# Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions

**Autores:** Emiel Hoogeboom; Didrik Nielsen; Priyank Jaini; Patrick Forré; Max Welling

> arXiv:2102.05379v3 · https://arxiv.org/abs/2102.05379
> Convertido desde la fuente LaTeX de arXiv con tex2md (pandoc). Numeración de secciones, figuras, tablas y ecuaciones tomada de la compilación del artículo; las figuras se reducen a su leyenda. Ante cualquier duda, el PDF es la referencia.

## Abstract

Generative flows and diffusion models have been predominantly trained on ordinal data, for example natural images. This paper introduces two extensions of flows and diffusion for *categorical* data such as language or image segmentation: *Argmax Flows* and *Multinomial Diffusion*. Argmax Flows are defined by a composition of a continuous distribution (such as a normalizing flow), and an argmax function. To optimize this model, we learn a probabilistic inverse for the argmax that lifts the categorical data to a continuous space. Multinomial Diffusion gradually adds categorical noise in a diffusion process, for which the generative denoising process is learned. We demonstrate that our method outperforms existing dequantization approaches on text modelling and modelling on image segmentation maps in log-likelihood.

# 1 Introduction

**Figura 1.** Overview of generative models.

Many sources of high-dimensional data are categorical, for example language and image segmentation. Although natural images have been studied to a large extent with generative flows and diffusion models, categorical data has not had the same extensive treatment. Currently they are primarily modelled by autoregressive models, which are expensive to sample from (Cooijmans et al., 2017; Dai et al., 2019).

Normalizing flows are attractive because they can be designed to be fast both in the evaluation and sampling direction. Typically, normalizing flows model continuous distributions. As a result, directly optimizing a flow on discrete data may lead to arbitrarily high likelihoods. In literature this problem is resolved for ordinal data by adding noise in a unit interval around the discrete value (Uria et al., 2013; Theis et al., 2016; Ho et al., 2019). However, because these methods have been designed for ordinal data, they do not work well on categorical data.

Other attractive generative models are diffusion models (Sohl-Dickstein et al., 2015), which are fast to train due to an objective that decomposes over time steps (Ho et al., 2020). Diffusion models typically have a fixed diffusion process that gradually adds noise. This process is complemented by a learnable generative process that denoises the signal. Song et al. (2020); Nichol and Dhariwal (2021) have shown that diffusion models can also be designed for fast sampling. Thus far, diffusion models have been primarily trained to learn ordinal data distributions, such as natural images.

**Tabla 1.** Surjective flow layers for applying continuous flow models to discrete data. The layers are deterministic in the generative direction, but stochastic in the inference direction. Rounding corresponds to the commonly-used dequantization for ordinal data.

| **Layer** | **Generation** | **Inference** | **Applications** |
|:---|:--:|:--:|:---|
| Rounding | ${\boldsymbol{x}}= \lfloor {\boldsymbol{v}}\rfloor$ | ${\boldsymbol{v}}\sim q({\boldsymbol{v}}|{\boldsymbol{x}})$ with support | Ordinal Data |
|  |  | $\mathcal{S}({\boldsymbol{x}}) = \{{\boldsymbol{v}}| {\boldsymbol{x}}= \lfloor {\boldsymbol{v}}\rfloor \}$ | e.g. images, audio |
| Argmax | ${\boldsymbol{x}}= \mathop{\mathrm{arg\,max}}{\boldsymbol{v}}$ | ${\boldsymbol{v}}\sim q({\boldsymbol{v}}|{\boldsymbol{x}})$ with support | Categorical Data |
|  |  | $\mathcal{S}({\boldsymbol{x}}) = \{{\boldsymbol{v}}| {\boldsymbol{x}}= \mathop{\mathrm{arg\,max}}{\boldsymbol{v}}\}$ | e.g. text, segmentation |

<span id="tab:surjective" data-label="tab:surjective"></span>

Therefore, in this paper we introduce extensions of flows and diffusion models for categorical variables (depicted in Figure 1): *i)* Argmax Flows bridge the gap between categorical data and continuous normalizing flows using an argmax transformation and a corresponding family of probabilistic inverses for the argmax. In addition *ii)* we introduce Multinomial Diffusion, which is a diffusion model directly defined on categorical variables. Opposed to normalizing flows, defining diffusion for discrete variables directly does not require gradient approximations, because the diffusion trajectory is fixed. As a result of our work, generative normalizing flows and diffusion models can directly learn categorical data.

# 2 Background

#### Normalizing Flows

Given ${\mathcal{V}}= \mathbb{R}^d$ and ${\mathcal{Z}}= \mathbb{R}^d$ with densities $p_{V}$ and $p_{Z}$ respectively, normalizing flows (Rezende and Mohamed, 2015) learn a bijective and differentiable transformation $g : {\mathcal{Z}}\to {\mathcal{V}}$ such that the change-of-variables formula gives the density at any point ${\boldsymbol{v}}\in {\mathcal{V}}$: $$\begin{equation}
\tag{1}
    p_{V}({\boldsymbol{v}}) = p_{Z}({\boldsymbol{z}})\cdot \left|\det \frac{\mathrm{d}{\boldsymbol{z}}}{\mathrm{d}{\boldsymbol{v}}} \right|,  \qquad {\boldsymbol{v}}= g({\boldsymbol{z}}),
\end{equation}$$ where $p_{Z}$ can be any density (usually chosen as a standard Gaussian). Thus, normalizing flows provide a powerful framework to learn *exact* density functions. However, Eq. (1) is restricted to continuous densities.

To learn densities on ordinal discrete data (such as natural images), typically dequantization noise is added (Uria et al., 2013; Theis et al., 2016; Ho et al., 2019). Nielsen et al. (2020) reinterpreted dequantization as a surjective flow layer ${\boldsymbol{v}}\mapsto {\boldsymbol{x}}$ that is deterministic in one direction (${\boldsymbol{x}}= \mathsf{round}({\boldsymbol{v}})$) and stochastic in the other (${\boldsymbol{v}}= {\boldsymbol{x}}+ {\boldsymbol{u}}$ where ${\boldsymbol{u}}\sim q({\boldsymbol{u}}|{\boldsymbol{x}})$). Using this interpretation, dequantization can be seen as a probabilistic right-inverse for the rounding operation in the latent variable model given by: $$\begin{equation*}
\small
{P({\boldsymbol{x}}) =\! \int \! P({\boldsymbol{x}}|{\boldsymbol{v}}) p({\boldsymbol{v}})\mathop{}\!\mathrm{d}{\boldsymbol{v}}, \quad P({\boldsymbol{x}}|{\boldsymbol{v}}) \!=\! \delta\big({\boldsymbol{x}}\!=\! \mathsf{round}({\boldsymbol{v}})\big)},
\end{equation*}$$ where $\mathsf{round}$ is applied elementwise. In this case, the density model $p({\boldsymbol{v}})$ is modeled using a normalizing flow. Learning proceeds by introducing the variational distribution $q({\boldsymbol{v}}|{\boldsymbol{x}})$ that models the probabilistic right-inverse for the rounding surjection and optimizing the evidence lower bound (ELBO): $$\begin{align}
\log P({\boldsymbol{x}}) &\geq \mathbb{E}_{{\boldsymbol{v}}\sim q({\boldsymbol{v}}|{\boldsymbol{x}})} \left[ \log P({\boldsymbol{x}}| {\boldsymbol{v}}) + \log p({\boldsymbol{v}}) - \log q({\boldsymbol{v}}|{\boldsymbol{x}}) \right] = \mathbb{E}_{{\boldsymbol{v}}\sim q({\boldsymbol{v}}|{\boldsymbol{x}})} \left[ \log p({\boldsymbol{v}}) - \log q({\boldsymbol{v}}|{\boldsymbol{x}}) \right].
\tag{2}
\end{align}$$ The last equality holds under the constraint that the support of $q({\boldsymbol{v}}|{\boldsymbol{x}})$ is enforced to be only over the region $\mathcal{S} = \{ {\boldsymbol{v}}\in \mathbb{R}^d : {\boldsymbol{x}}= \mathsf{round}({\boldsymbol{v}}) \}$ which ensures that $P({\boldsymbol{x}}|{\boldsymbol{v}}) = 1$.

#### Diffusion Models

Given data ${\boldsymbol{x}}_0$, a diffusion model (Sohl-Dickstein et al., 2015) consists of predefined variational distributions $q({\boldsymbol{x}}_t | {\boldsymbol{x}}_{t-1})$ that gradually add noise over time steps $t \in \{1, \ldots, T\}$. The diffusion trajectory is defined such that $q({\boldsymbol{x}}_{t} | {\boldsymbol{x}}_{t-1})$ adds a small amount of noise around ${\boldsymbol{x}}_{t-1}$. This way, information is gradually destroyed such that at the final time step, ${\boldsymbol{x}}_T$ carries almost no information about ${\boldsymbol{x}}_0$. Their generative counterparts consists of learnable distributions $p({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_{t})$ that learn to denoise the data. When the diffusion process adds sufficiently small amounts of noise, it suffices to define the denoising trajectory using distributions that are factorized (without correlation) over the dimension axis. The distribution $p({\boldsymbol{x}}_T)$ is chosen to be similar to the distribution that the diffusion trajectory approaches. Diffusion models can be optimized using variational inference: $$\begin{equation*}
    \log P({\boldsymbol{x}}_0) \geq \mathbb{E}_{x_1, \ldots x_T \sim q} 
    \Big{[} \log p({\boldsymbol{x}}_T) + \sum_{t=1}^T \log \frac{p({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_t)}{q({\boldsymbol{x}}_{t} | {\boldsymbol{x}}_{t-1})} \Big{]}.
\end{equation*}$$

An important insight in diffusion is that by conditioning on ${\boldsymbol{x}}_0$, the posterior probability $q({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_{t}, {\boldsymbol{x}}_0) = q({\boldsymbol{x}}_{t} | {\boldsymbol{x}}_{t-1})q({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_0) / q({\boldsymbol{x}}_t | {\boldsymbol{x}}_0)$ is tractable and straightforward to compute, permitting a reformulation in terms of KL divergences that has lower variance (Sohl-Dickstein et al., 2015). Note that $\mathrm{KL} \big{(} q({\boldsymbol{x}}_T | {\boldsymbol{x}}_0) | p({\boldsymbol{x}}_T)\big{)} \approx 0$ if the diffusion trajectory $q$ is defined well: $$\begin{align}
\begin{split}
    &\log P({\boldsymbol{x}}_0) \geq \mathbb{E}_{q} 
    \Big{[}\log p({\boldsymbol{x}}_0 | {\boldsymbol{x}}_1) - \mathrm{KL} \big{(}  q({\boldsymbol{x}}_T | {\boldsymbol{x}}_0) | p({\boldsymbol{x}}_T) \big{)} - \sum_{t=2}^T \mathrm{KL} \big{(}q({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_t, {\boldsymbol{x}}_0) | p({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_t)\big{)} \Big{]}
\end{split}
\tag{3}
\end{align}$$

# 3 Argmax Flows

Argmax flows define discrete distributions using 1) a density model $p({\boldsymbol{v}})$, such as a normalizing flow, and 2) an argmax layer that maps the continuous ${\boldsymbol{v}}\in \mathbb{R}^{D \times K}$ to a discrete ${\boldsymbol{x}}\in \{1,2,...,K\}^D$ using $$\begin{equation}
\small
    {\boldsymbol{x}}= \mathop{\mathrm{arg\,max}}{\boldsymbol{v}}\quad \text{ where } \quad x_d = \mathop{\mathrm{arg\,max}}_k v_{dk}.
\end{equation}$$ This is a natural choice to model categorical variables, because it divides the entire continuous space of ${\boldsymbol{v}}$ into symmetric partitions corresponding to categories in ${\boldsymbol{x}}$. To sample from an argmax flow sample ${\boldsymbol{v}}\sim p({\boldsymbol{v}})$ and compute ${\boldsymbol{x}}= \mathop{\mathrm{arg\,max}}{\boldsymbol{v}}$ (Algorithm 1). To generate reasonable samples, it is up to the density model $p({\boldsymbol{v}})$ to capture any complicated dependencies between the different dimensions. While sampling from an argmax flow is straightforward, the main difficulty lies in *optimizing* this generative model. To compute the likelihood of a datapoint ${\boldsymbol{x}}$, we have to compute $$\begin{equation}
\small
    P({\boldsymbol{x}}) =\! \int P({\boldsymbol{x}}| {\boldsymbol{v}}) p({\boldsymbol{v}})d{\boldsymbol{v}}, ~~P({\boldsymbol{x}}|{\boldsymbol{v}}) \!=\! \delta\big({\boldsymbol{x}}\!=\! \mathop{\mathrm{arg\,max}}({\boldsymbol{v}})\big),
\end{equation}$$ which is intractable. Consequently, we resort to variational inference and specify a variational distribution $q({\boldsymbol{v}}|{\boldsymbol{x}})$. We note that naïvely choosing any variational distribution may lead to samples ${\boldsymbol{v}}\sim q({\boldsymbol{v}}| {\boldsymbol{x}})$ where $\delta({\boldsymbol{x}}= \mathop{\mathrm{arg\,max}}{\boldsymbol{v}}) = 0$, which yields an ELBO of negative infinity. To avoid this, we need a variational distribution $q({\boldsymbol{v}}|{\boldsymbol{x}})$ that satisfies what we term the *argmax constraint*: $$\begin{equation*}
\small
    {\boldsymbol{x}}= \mathop{\mathrm{arg\,max}}{\boldsymbol{v}}\quad \text{for all} \quad {\boldsymbol{v}}\sim q({\boldsymbol{v}}|{\boldsymbol{x}}).
\end{equation*}$$ That is, the variational distribution $q({\boldsymbol{v}}|{\boldsymbol{x}})$ should have support limited to $\mathcal{S}({\boldsymbol{x}}) = \{{\boldsymbol{v}}\in \mathbb{R}^{D\times K} : {\boldsymbol{x}}= \mathop{\mathrm{arg\,max}}{\boldsymbol{v}}\}$. Recall that under this condition, the ELBO simplifies to $\mathbb{E}_{{\boldsymbol{v}}\sim q({\boldsymbol{v}}|{\boldsymbol{x}})} \left[ \log p({\boldsymbol{v}}) - \log q({\boldsymbol{v}}|{\boldsymbol{x}}) \right]$, as shown in Algorithm 2. For an illustration of the method see Figure 1a.

**Tabla.** Sampling from Argmax Flows

<div class="algorithm">

<span id="alg:sample_argmax" data-label="alg:sample_argmax"></span>

<div class="algorithmic">

**Input:** $p({\boldsymbol{v}})$

**Output:** Sample ${\boldsymbol{x}}$

Sample ${\boldsymbol{v}}\sim p({\boldsymbol{v}})$

Compute ${\boldsymbol{x}}= \mathop{\mathrm{arg\,max}}{\boldsymbol{v}}$

</div>

</div>

<div class="algorithm">

**Algoritmo 2.** Optimizing Argmax Flows

<span id="alg:optimize_argmax" data-label="alg:optimize_argmax"></span>

<div class="algorithmic">

**Input:** ${\boldsymbol{x}}$, $p({\boldsymbol{v}})$, $q({\boldsymbol{v}}|{\boldsymbol{x}})$

**Output:** ELBO $\mathcal{L}$

Sample ${\boldsymbol{v}}\sim q({\boldsymbol{v}}|{\boldsymbol{x}})$

Compute $\mathcal{L} = \log p({\boldsymbol{v}}) - \log q({\boldsymbol{v}}|{\boldsymbol{x}})$

</div>

</div>

## 3.1 Probabilistic Inverse

The argmax layer may be viewed as a surjective flow layer (Nielsen et al., 2020). With this view, the variational distribution $q({\boldsymbol{v}}|{\boldsymbol{x}})$ specifies a distribution over the possible right-inverses of the argmax function, also known as a *stochastic inverse* or *probabilistic inverse*. Recall that the commonly-used dequantization layer for ordinal data corresponds to the probabilistic inverse of a rounding operation. As summarized in Table 1, this layer may thus be viewed as analogous to the argmax layer, where the round is for ordinal data while the argmax is for categorical data.

We are free to specify any variational distribution $q({\boldsymbol{v}}|{\boldsymbol{x}})$ that satisfies the argmax constraint. In the next paragraphs we outline three possible approaches. Since operations are performed independently across dimensions, we omit the dimension axis and let ${\boldsymbol{v}}\in \mathbb{R}^K$ and $x \in \{1, \ldots, K\}$.

#### Thresholding (Alg. 3).

A straightforward method to construct a distribution $q({\boldsymbol{v}}|x)$ satisfying the argmax constraint is to use thresholding. That is, we first sample an unbounded variable ${\boldsymbol{u}}\in \mathbb{R}^{K}$ from $q({\boldsymbol{u}}| x)$, which can be for example a conditional Gaussian or normalizing flow. Next, we map ${\boldsymbol{u}}$ to ${\boldsymbol{v}}$ such that element $x$ is the largest: $$\begin{equation}
    v_x = u_x \quad \text{and} \quad {\boldsymbol{v}}_{-x} = \mathrm{threshold}_T({\boldsymbol{u}}_{-x})
\end{equation}$$ where the thresholding is applied elementwise with threshold value $T=v_x$. This ensures that element $v_x$ is the largest, and consequently that $q({\boldsymbol{v}}|x)$ satisfies the argmax constraint. Note that we require the threshold function to be bijective, $\mathrm{threshold}_T: \mathbb{R}\rightarrow (-\infty,T)$, so that we can use the change-of-variables formula to compute $\log q({\boldsymbol{v}}|x)$. In our implementation, thresholding is implemented using a softplus such that all values are mapped below a limit $T$: $$\begin{equation}
\small
   v = \mathrm{threshold}_T(u) = T-\mathrm{softplus}(T-u),
\end{equation}$$ where $\mathrm{softplus}(z) = \log(1 + e^z)$ and for which it is guaranteed that $v \in (-\infty, T)$.

**Tabla.** Thresholding-based $q({\boldsymbol{v}}|{\boldsymbol{x}})$

<div class="algorithm">

<span id="alg:thresholding" data-label="alg:thresholding"></span>

<div class="algorithmic">

**Input:** ${\boldsymbol{x}}$, $q({\boldsymbol{u}}|{\boldsymbol{x}})$

**Output:** ${\boldsymbol{v}}$, $\log q({\boldsymbol{v}}|{\boldsymbol{x}})$

${\boldsymbol{u}}\sim q({\boldsymbol{u}}|{\boldsymbol{x}})$

${\boldsymbol{v}}_{{\boldsymbol{x}}} = {\boldsymbol{u}}_{{\boldsymbol{x}}}$

${\boldsymbol{v}}_{-{\boldsymbol{x}}} = \mathrm{threshold}({\boldsymbol{u}}_{-{\boldsymbol{x}}}, {\boldsymbol{x}})$

$\log q({\boldsymbol{v}}|{\boldsymbol{x}}) = \log q({\boldsymbol{u}}|{\boldsymbol{x}}) - \log |\det \mathop{}\!\mathrm{d}{\boldsymbol{v}}/ \mathop{}\!\mathrm{d}{\boldsymbol{u}}|$

</div>

</div>

<div class="algorithm">

**Algoritmo 4.** Gumbel-based $q({\boldsymbol{v}}|{\boldsymbol{x}})$

<span id="alg:gumbel" data-label="alg:gumbel"></span>

<div class="algorithmic">

**Input:** ${\boldsymbol{x}}$, $\boldsymbol{\phi}$

**Output:** ${\boldsymbol{v}}$, $\log q({\boldsymbol{v}}|{\boldsymbol{x}})$

$\phi_{\max} = \log \sum_i \exp \phi_i$

${\boldsymbol{v}}_{{\boldsymbol{x}}} \sim \mathrm{Gumbel}(\phi_{\max})$

${\boldsymbol{v}}_{-{\boldsymbol{x}}} \sim \mathrm{TruncGumbel}(\boldsymbol{\phi}_{-{\boldsymbol{x}}}, {\boldsymbol{v}}_{{\boldsymbol{x}}})$

$\log q({\boldsymbol{v}}|{\boldsymbol{x}}) = \log \mathrm{Gumbel}({\boldsymbol{v}}_{\boldsymbol{x}}| \phi_{\max})$

$+ \log \mathrm{TruncGumbel}({\boldsymbol{v}}_{-{\boldsymbol{x}}} | \boldsymbol{\phi}_{-{\boldsymbol{x}}}, {\boldsymbol{v}}_{\boldsymbol{x}})$

</div>

</div>

#### Gumbel (Alg. 4).

An alternative approach is to let $q({\boldsymbol{v}}|x) = \mathrm{Gumbel}({\boldsymbol{v}}|\boldsymbol{\phi})$ *restricted* to $\mathop{\mathrm{arg\,max}}{\boldsymbol{v}}= x$, where the location parameters $\boldsymbol{\phi}\leftarrow \mathrm{NN}(x)$ are predicted using a neural network $\mathrm{NN}$. The Gumbel distribution has favourable properties: The $\mathop{\mathrm{arg\,max}}$ and $\max$ are independent and the $\max$ is also distributed as a Gumbel: $$\begin{equation}
\small
\tag{8}
    \max_i v_i \sim \mathrm{Gumbel}(\phi_{\max}),
\end{equation}$$ where $\phi_{\max} = \log \sum_i \exp \phi_i$. For a more extensive introduction see (Maddison et al., 2014; Kool et al., 2019). To sample ${\boldsymbol{v}}\sim q({\boldsymbol{v}}|x)$, we thus first sample the maximum $v_x$ according to Eq. 8. Next, given the sample $v_x$, the remaining values can be sampled using *truncated* Gumbel distributions: $$\begin{equation}
\small
    v_{i} \sim \mathrm{TruncGumbel}(\phi_i; T) \text{ where } i \not= x
    \tag{9}
\end{equation}$$ where the truncation value $T$ is given by $v_x$ which ensures that the argmax constraint $v_x > v_i$ for $i\not=x$ is satisfied. Recall that to optimize Eq. 2, $\log q({\boldsymbol{v}}| {\boldsymbol{x}})$ is also required, which can be computed using the closed-form expressions for the log density functions (see Table 5). Another property of Gumbel distributions is that $$\begin{equation}
\small
    P(\mathop{\mathrm{arg\,max}}{\boldsymbol{v}}= i) = \exp \phi_i / \sum_i \exp \phi_i,
\end{equation}$$ which we use to initialize the location parameters $\boldsymbol{\phi}$ to match the empirical distribution of the first minibatch of the data.

#### Gumbel Thresholding.

This method unifies the methods from the previous two sections: Gumbel distributions and thresholding. The key insight is that the Gumbel sampling procedures as defined above can be seen as a reparametrization of a uniform noise distribution $\mathcal{U}(0, 1)^K$ which is put through the inverse CDF of the Gumbel distributions (see Table 5). From the perspective of change-of-variables, the log likelihood denotes the log volume change of this transformation. To increase expressitivity the uniform distribution can be replaced by a normalizing flow $q({\boldsymbol{u}}| x)$ that has support on the interval $(0, 1)^K$, which can be enforced using a sigmoid transformation. This section shows that a large collection of thresholding functions can be found by studying (truncated) inverse CDFs. In practice we find that performance is reasonably similar as long as the underlying noise ${\boldsymbol{u}}$ is learned.

#### Behavior of the Variational Posterior

Although several methods to learn $q$ have been proposed, it is unclear what expressitivity is required. In the following, the interactions between $q({\boldsymbol{v}}| {\boldsymbol{x}})$ and the density model $p({\boldsymbol{v}})$ are discussed. Recall that the variational bound that is optimized under expectation of a data distribution $\mathcal{D}$ can be seen as minimizing the $\operatorname{KL}$ distance between the aggregated posterior $q({\boldsymbol{v}}) = \mathbb{E}_{{\boldsymbol{x}}\sim \mathcal{D}}q({\boldsymbol{v}}| {\boldsymbol{x}})$ and the density model $p({\boldsymbol{v}})$, so $\operatorname{KL}(q({\boldsymbol{v}})|p({\boldsymbol{v}}))$. There are two distinct reasons which can cause this distance to be large: Firstly, the density model $p({\boldsymbol{v}})$ may not have the right probability mass in each argmax region. These desired probabilities solely depend on the data distribution $\mathcal{D}$. Secondly, the variational posterior $q({\boldsymbol{v}}| {\boldsymbol{x}})$ may not have the correct shape compared to $p({\boldsymbol{v}})$, *within* an argmax region. At initialization, the thresholding within $q$ can create low density regions at argmax boundaries.

In theory, if $p({\boldsymbol{v}})$ is a universal density approximator, then the model can be fitted for any well-behaved $q({\boldsymbol{v}}|{\boldsymbol{x}})$. Then $p({\boldsymbol{v}})$ can even fit the low density regions in the boundaries. This argument is trivial, as one can simply set $p({\boldsymbol{v}})$ to $q({\boldsymbol{v}}) = \mathbb{E}_{{\boldsymbol{x}}\sim \mathcal{D}}q({\boldsymbol{v}}| {\boldsymbol{x}})$. In practice, over training steps we find that $q$ does smooth out these boundary artifacts, and counteracts the thresholding so that the aggregated posterior becomes smoother.

## 3.2 Cartesian Products of Argmax Flows

In the current description, Argmax Flows require the same number of dimensions in ${\boldsymbol{v}}$ as there are classes in ${\boldsymbol{x}}$. To alleviate this constraint we introduce Cartesian products of Argmax Flows. To illustrate our method, consider a 256 class problem. One class can be represented using a single number in $\{1, \ldots, 256\}$, but also using two hexadecimal numbers $\{1, \ldots, 16\}^2$ or alternatively using eight binary numbers. Specifically, any base $K$ variable ${\boldsymbol{x}}^{(K)} \in \{1, \ldots, K\}^D$ can be converted to a base $M$ variable ${\boldsymbol{x}}^{(M)} \in \{1, \ldots, M\}^{d_m \times D}$ where $d_m = \lceil \log_M K \rceil$. Then the variable ${\boldsymbol{x}}^{(M)}$ with dimensionality $M \cdot d_m \cdot D$ represents the variable ${\boldsymbol{x}}^{(K)}$ with dimensionality $K \cdot D$, trading off symmetry for dimensionality. Even though this may lead to some unused additional classes, the ELBO objective in Equation 2 can still be optimized using an $M$-categorical Argmax Flow. Finally, note that Cartesian products of binary spaces are a special case where the variable can be encoded symmetrically into a single dimension to the positive and negative part using binary dequantization (Winkler et al., 2019). In this case, by trading-off symmetry the dimensionality increases only proportional to $\log_2 K$ .

# 4 Multinomial Diffusion

In this section we introduce an alternative likelihood-based model for categorical data: Multinomial Diffusion. In contrast with previous sections, ${\boldsymbol{x}}_t$ will be represented in one-hot encoded format ${\boldsymbol{x}}_t \in \{0, 1\}^K$. Specifically, for category $k$, $x_{k} = 1$ and $x_{j} = 0$ for $j \not= k$. Note that again the dimension axis is omitted for clarity as all distributions are independent over the dimension axis. We define the multinomial diffusion process using a categorical distribution that has a $\beta_t$ chance of resampling a category uniformly: $$\begin{equation}
    q({\boldsymbol{x}}_t | {\boldsymbol{x}}_{t-1}) =  \mathcal{C}({\boldsymbol{x}}_t | (1 - \beta_t) {\boldsymbol{x}}_{t-1} + \beta_t / K ),
    \tag{11}
\end{equation}$$ where $\mathcal{C}$ denotes a categorical distribution with probability parameters after $|$. Further addition (and subtraction) between scalars and vectors is done elementwise. This convention kept throughout this section. Since these distributions form a Markov chain, we can express the probability of any ${\boldsymbol{x}}_t$ given ${\boldsymbol{x}}_0$ as: $$\begin{equation}
    q({\boldsymbol{x}}_t | {\boldsymbol{x}}_{0}) =  \mathcal{C}({\boldsymbol{x}}_t | \bar{\alpha}_t {\boldsymbol{x}}_{0} + (1 - \bar{\alpha}_t) / K )
    \tag{12}
\end{equation}$$ where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{\tau=1}^t \alpha_\tau$. Intuïtively, for each next timestep, a little amount of uniform noise $\beta_t$ over the $K$ classes is introduced, and with a large probability $(1 - \beta_t)$ the previous value ${\boldsymbol{x}}_{t-1}$ is sampled. Using Equation 11 and 12 the categorical posterior $q({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_{t}, {\boldsymbol{x}}_0)$ can be computed in closed-form: $$\begin{align}
\begin{split}
    q({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_{t}, {\boldsymbol{x}}_0) &= \mathcal{C}({\boldsymbol{x}}_{t-1} | {\boldsymbol{\theta}}_{\mathrm{post}}({\boldsymbol{x}}_t, {\boldsymbol{x}}_0)), \,\, \text{ where } \,\, {\boldsymbol{\theta}}_{\mathrm{post}}({\boldsymbol{x}}_t, {\boldsymbol{x}}_0) = \tilde{{\boldsymbol{\theta}}} / \sum_{k=1}^K \tilde{\theta}_k
     \\
    \text{and } \,\tilde{{\boldsymbol{\theta}}} &= [\alpha_t {\boldsymbol{x}}_t + (1 - \alpha_t) / K] \odot [\bar{\alpha}_{t-1} \boldsymbol{x}_0 + (1 - \bar{\alpha}_{t-1}) / K ].
    \tag{13}
\end{split}
\end{align}$$

**Figura 2.** Overview of multinomial diffusion. A generative model $p({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_{t})$ learns to gradually denoise a signal from left to right. An inference diffusion process $q({\boldsymbol{x}}_{t} | {\boldsymbol{x}}_{t-1})$ gradually adds noise form right to left.

One of the innovations in (Ho et al., 2020) was the insight to not predict the parameters for the generative trajectory directly, but rather to predict the noise using the posterior equation for $q$. Although predicting the noise is difficult for discrete data, we predict a probability vector for $\hat{{\boldsymbol{x}}}_0$ from ${\boldsymbol{x}}_t$ and subsequently parametrize $p({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_{t})$ using the probability vector from $q({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_t, \hat{{\boldsymbol{x}}}_0)$, where ${\boldsymbol{x}}_0$ is approximated using a neural network $\hat{{\boldsymbol{x}}}_0 = \mu({\boldsymbol{x}}_t, t)$. Equation 13 will produce valid probability vectors that are non-negative and sums to one under the condition that the prediction $\hat{{\boldsymbol{x}}}_0$ is non-negative and sums to one, which is ensured with a softmax function in $\mu$. To summarize: $$\begin{align}
    \begin{split}
    p({\boldsymbol{x}}_{0} | {\boldsymbol{x}}_{1}) = \mathcal{C}({\boldsymbol{x}}_{0} | \hat{{\boldsymbol{x}}}_0) \, \text{ and } \,
    p({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_{t}) = \mathcal{C}({\boldsymbol{x}}_{t-1} | \boldsymbol{\theta}_{\mathrm{post}}({\boldsymbol{x}}_t, 
    \hat{{\boldsymbol{x}}}_0)) \, 
    \text{ where }\, \hat{{\boldsymbol{x}}}_0 = \mu({\boldsymbol{x}}_t, t)
    \end{split}
    \tag{14}
\end{align}$$ The KL terms in Equation 3 can be simply computed by enumerating the probabilities in Equation 13 and 14 and computing the KL divergence for discrete distributions in $L_{t-1}$ with $t \geq 2$: $$\begin{align}
\begin{split}
    \mathrm{KL} \big{(}q({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_t, {\boldsymbol{x}}_0) | p({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_t)\big{)}  &= \mathrm{KL} \big{(} \mathcal{C}(\boldsymbol{\theta}_{\mathrm{post}}({\boldsymbol{x}}_t, {\boldsymbol{x}}_0)) | \mathcal{C}(\boldsymbol{\theta}_{\mathrm{post}}({\boldsymbol{x}}_t, 
    \hat{{\boldsymbol{x}}}_0))\big{)},
\end{split}
\end{align}$$ which can be computed using $\sum_k \boldsymbol{\theta}_{\mathrm{post}}({\boldsymbol{x}}_t, {\boldsymbol{x}}_0))_k \cdot \log \frac{\boldsymbol{\theta}_{\mathrm{post}}({\boldsymbol{x}}_t, {\boldsymbol{x}}_0))_k}{\boldsymbol{\theta}_{\mathrm{post}}({\boldsymbol{x}}_t, 
\hat{{\boldsymbol{x}}}_0))_k}$. Furtermore, to compute $\log p({\boldsymbol{x}}_0 | {\boldsymbol{x}}_1)$ use that ${\boldsymbol{x}}_0$ is onehot: $$\begin{equation}
    \log p({\boldsymbol{x}}_0 | {\boldsymbol{x}}_1) = \sum_k {\boldsymbol{x}}_{0,k} \log  \hat{{\boldsymbol{x}}}_{0,k}
\end{equation}$$

# 5 Related Work

Deep generative models broadly fall into the categories autoregressive models ARMs (Germain et al., 2015), Variational Autoencoders (VAEs) (Kingma and Welling, 2014; Rezende et al., 2014), Adversarial Network (GANs) (Goodfellow et al., 2014), Normalizing Flows (Rezende and Mohamed, 2015), Energy-Based Models (EBMs) and Diffusion Models (Sohl-Dickstein et al., 2015).

Normalizing Flows typically learn a continuous distribution and dequantization is required to train these methods on ordinal data such as images. A large body of work is dedicated to building more expressive continuous normalizing flows (Dinh et al., 2017; Germain et al., 2015; Kingma et al., 2016; Papamakarios et al., 2017; Chen et al., 2018; Song et al., 2019; Perugachi-Diaz et al., 2020). To learn ordinal discrete distributions with normalizing flows, adding uniform noise in-between ordinal classes was proposed in (Uria et al., 2013) and later theoretically justified in (Theis et al., 2016). An extension for more powerful dequantization based on variational inference was proposed in (Ho et al., 2019), and connected to autoregressive models in (Nielsen and Winther, 2020). Dequantization for binary variables was proposed in (Winkler et al., 2019). Tran et al. (2019) propose invertible transformations for categorical variables directly. However, these methods can be difficult to train because of gradient bias and results on images have thus far not been demonstrated. In addition flows for ordinal discrete data (integers) have been explored in (Hoogeboom et al., 2019; van den Berg et al., 2020). In other works, VAEs have been adapted to learn a normalizing flow for the latent space (Ziegler and Rush, 2019; Lippe and Gavves, 2020). However, these approaches typically still utilize an argmax heuristic to sample, even though this is not the distribution specified during training.

Diffusion models were first introduced in (Sohl-Dickstein et al., 2015), who developed diffusion for Gaussian and Bernoulli distributions. Recently, Denoising Diffusion models (Ho et al., 2020) have been shown capable of generating high-dimensional images by architectural improvements and reparametrization of the predictions. Diffusion models are relatively fast to train, but slow to sample from as they require iterations over the many timesteps in the chain. Song et al. (2020); Nichol and Dhariwal (2021) showed that in practice samples can be generated using significantly fewer steps. Nichol and Dhariwal (2021) demonstrated that importance-weighting the objective components greatly improves log-likelihood performance. In (Song et al., 2020) a continuous-time extension of denoising diffusion models was proposed. After initial release of this paper we discovered that Song et al. (2020) concurrently also describe a framework for discrete diffusion, but without empirical evaluation.

# 6 Experiments

In our experiments we compare the performance of our methods on language modelling tasks and learning image segmentation maps unconditionally.

**Tabla 2.** Comparison of a coupling and autoregressive generative flows with uniform (Uria et al., 2013) and variational (Ho et al., 2019) dequantization and our proposed Argmax flows.

| Dequantization | Flow type | text8 (bpc) | enwik8 (bits per raw byte) |  |  |
|:---|:---|:--:|:--:|:--:|:--:|
| Uniform dequantization | Autoregressive | 1.90 | 2.14 |  |  |
| Variational dequantization |  | 1.43 | 1.44 |  |  |
| Argmax Flow (ours) |  | **1.38** | **1.42** |  |  |
| Uniform dequantization | Coupling | 2.01 | 2.33 |  |  |
| Variational dequantization |  | 2.08 | 2.28 |  |  |
| Argmax Flow (ours) |  | **1.82** | **1.93** |  |  |

**Tabla 3.** Comparison of different methods on `text8` and `enwik8`. Results are reported in negative log-likelihood with units bits per character (bpc) for `text8` and bits per raw byte (bpb) for `enwik8`.

| Model type |  | Model | text8 (bpc) | enwik8 (bpb) |  |
|:---|:---|:---|:--:|:--:|:--:|
| ARM |  | 64 Layer Transformer (Al-Rfou et al., 2019) | 1.13 | 1.06 |  |
|  |  | TransformerXL (Dai et al., 2019) | 1.08 | 0.99 |  |
| VAE |  | AF/AF$^\star$ (AR) (Ziegler and Rush, 2019) | 1.62 | 1.72 |  |
|  |  | IAF / SCF$^\star$ (Ziegler and Rush, 2019) | 1.88 | 2.03 |  |
|  |  | CategoricalNF (AR) (Lippe and Gavves, 2020) | 1.45 | \- |  |
| Generative Flow |  | Argmax Flow, AR (ours) | 1.39 | 1.42 |  |
|  |  | Argmax Coupling Flow (ours) | 1.82 | 1.93 |  |
| Diffusion |  | Multinomial Text Diffusion (ours) | 1.72 | 1.75 |  |

<div class="flushleft">

$\star$ Results obtained by running code from the official repository for the `text8` and `enwik8` datasets.\

</div>

## 6.1 Language data

In this section we compare our methods on two language datasets, `text8` and `enwik8`. `text8` contains 27 categories (‘a’ through ‘z’ and ‘ ’) and for `enwik8` the bytes are directly modelled which results in 256 categories.

#### Model description

Two versions of generative argmax flows are tested: using an autoregressive (AR) flow and a coupling-based flow for $p({\boldsymbol{v}})$. In these experiments the probabilistic inverse is based on the thresholding approach. Specifically, a conditional diagonal Gaussian $q({\boldsymbol{u}}| {\boldsymbol{x}})$ is trained and thresholded which gives the distribution $q({\boldsymbol{v}}| {\boldsymbol{x}})$. The argmax flow is defined on binary Cartesian products. This means that for $K=27$, a $5$-dimensional binary space is used and for $K=256$ an $8$-dimensional binary space. The argmax flow is compared to the current standard of training generative flows directly on discrete data: dequantization. We compare to both uniform and variational dequantization, where noise on a (0, 1) interval is added to the onehot representation of the categorical data. The autoregressive density model is based on the model proposed in (Lippe and Gavves, 2020). The coupling density model consists of 8 flow layers where each layer consists of a $1$ $\times$ $1$ convolution and mixture of logistics transformations (Ho et al., 2019). In the multinomial text diffusion model, the $\mu$ network is modeled by a 12-layer Transformer. For more extensive details about the experiment setup see Appendix B.

#### Comparison with Generative Flows

Firstly we compare the performance of generative flows directly trained on language data (Table 2). These experiments are using the same underlying normalizing flow: either a coupling-based flow or an autoregressive flow. Note that Argmax Flows consistently outperform both uniform and variational dequantization. This indicates that it is easier for a generative flow to learn the lifted continuous distribution using an argmax flow. An advantage of Argmax flows that may explain this difference is that they lift the variables into the entire Euclidean space, whereas traditional dequantization only introduce probability density on $(0, 1)$ intervals, leaving gaps with no probability density. The performance improvements of Argmax flows are even more pronounced when comparing coupling-based approaches. Also note that coupling flows have worse performance than autoregressive flows, with a difference that is generally smaller for images. This indicates that designing more expressive coupling layers for text is an interesting future research direction.

#### Comparison with other generative models

The performance compared to models in literature is presented in Table 3 alongside the performance of our Argmax Flows and Multinomial Diffusion. The latent variable approaches containing autoregressive components are marked using (AR). Although autoregressive flows still have the same disadvantages as ARMs, they provide perspective on where performance deficiencies are coming from. We find that our autoregressive Argmax Flows achieve better performance than the VAE approaches, they outperform AF/AF (Ziegler and Rush, 2019) and CategoricalNF (Lippe and Gavves, 2020).

**Figura 3.** Samples from models, `text8`.

**Figura 4.** Samples from models, cityscapes.

When comparing non-autoregressive models, Argmax Flows also outperforms the method that lifts the categorical space to a continuous space: IAF / SCF (Ziegler and Rush, 2019). Interestingly, the multinomial text diffusion is a non-autoregressive model that performs even better than the argmax coupling flow, but performs worse than the autoregressive version. For this model it is possible that different diffusion trajectories for $q$ would result in even better performance, because in the current form the denoising model has to be very robust to input noise. These experiments also highlight that there is still a distinct performance gap between standard ARMs and (autoregressive) continuous density model on text, possibly related to the dequantization gap (Nielsen and Winther, 2020). Samples from different models trained on `text8` are depicted in Figure 3. Because of difficulties in reproducing results from Discrete Flows, a comparison and analysis of discrete flows are left out of this section. Instead they are extensively discussed in Appendix C. For additional experiments regarding Cartesian products and sampling time see Appendix D.

**Figura 5.** Spell checking with Multinomial Text Diffusion.

#### Unsupervised spell-checking

An interesting by-product of the text diffusion model is that it can be used to spell-check text using a single forward pass. To demonstrate this, a sentence taken from the test data is corrupted by changing a few characters. This corrupted sequence is given as ${\boldsymbol{x}}_1$ to the generative denoising model, which is close to the data at step $0$. Then the denoising model predicts $p({\boldsymbol{x}}_0 | {\boldsymbol{x}}_1)$ and the most-likely ${\boldsymbol{x}}_0$ can be suggested. Note that this model only works for character-level corruption, not insertions. An example is depicted in Figure 5. Since the model chooses the most-likely matching word, larger corruptions will at some point lead to word changes.

## 6.2 Segmentation maps

For image-type data, we introduce a categorical image dataset: the cityscapes dataset is repurposed for *unconditional* image segmentation learning. In contrast with the standard setting, the distribution over the segmentation targets needs to be learned *without* conditioning on the photograph. To reduce computational cost, we rescale the segmentation maps from cityscapes to $32 \times 64$ images using nearest neighbour interpolation. We utilize the global categories as prediction targets which results in an 8-class problem.

**Tabla 4.** Performance of different dequantization methods on squares and cityscapes dataset, in bits per pixel, lower is better.

<table id="tab:results_cityscapes">
<thead>
<tr>
<th style="text-align: left;">Cityscapes</th>
<th style="text-align: left;">ELBO</th>
<th style="text-align: right;">IWBO</th>
<th style="text-align: right;"></th>
<th style="text-align: right;"></th>
<th style="text-align: right;"></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Round / Unif. (Uria et al., 2013)</td>
<td style="text-align: left;">1.010</td>
<td style="text-align: right;">0.930</td>
<td style="text-align: right;"></td>
<td style="text-align: right;"></td>
<td style="text-align: right;"></td>
</tr>
<tr>
<td style="text-align: left;">Round / Var. (Ho et al., 2019)</td>
<td style="text-align: left;">0.334</td>
<td style="text-align: right;">0.315</td>
<td style="text-align: right;"></td>
<td style="text-align: right;"></td>
<td style="text-align: right;"></td>
</tr>
<tr>
<td style="text-align: left;">Argmax / Softplus thres. (ours)</td>
<td style="text-align: left;"><strong>0.303</strong></td>
<td style="text-align: right;"><strong>0.290</strong></td>
<td style="text-align: right;"></td>
<td style="text-align: right;"></td>
<td style="text-align: right;"></td>
</tr>
<tr>
<td style="text-align: left;">Argmax / Gumbel dist. (ours)</td>
<td style="text-align: left;">0.365</td>
<td style="text-align: right;">0.341</td>
<td style="text-align: right;"></td>
<td style="text-align: right;"></td>
<td style="text-align: right;"></td>
</tr>
<tr>
<td style="text-align: left;">Argmax / Gumbel thres. (ours)</td>
<td style="text-align: left;"><strong>0.307</strong></td>
<td style="text-align: right;"><strong>0.287</strong></td>
<td style="text-align: right;"></td>
<td style="text-align: right;"></td>
<td style="text-align: right;"></td>
</tr>
<tr>
<td style="text-align: left;">Multinomial Diffusion (ours)</td>
<td colspan="2" style="text-align: center;">0.305</td>
<td style="text-align: right;"></td>
<td style="text-align: right;"></td>
<td style="text-align: right;"></td>
</tr>
</tbody>
</table>

#### Model description

The Argmax Flows are defined directly on the $K=8$ categorical space. The density model $p({\boldsymbol{v}})$ is defined using affine coupling layers parametrized by DenseNets (Huang et al., 2017). For the probabilistic inverse we learn a conditional flow $q({\boldsymbol{u}}| {\boldsymbol{x}})$ which is also based on the affine coupling structure. Depending on the method, either softplus or Gumbel thresholding is applied to obtain ${\boldsymbol{v}}$. Recall that for our first Gumbel approach it is equivalent to set $q({\boldsymbol{u}}| {\boldsymbol{x}})$ to the unit uniform distribution, whereas $q({\boldsymbol{u}}| {\boldsymbol{x}})$ is learned for Gumbel thresholding. We compare to existing dequantization strategies in literature: uniform (Uria et al., 2013) and variational dequantization (Ho et al., 2019) which are applied on the onehot representation. All models utilize the same underlying flow architectures and thus the number of parameters is roughly the same. The exception are uniform dequantization and the Gumbel distribution, since no additional variational flow distribution is needed. For more extensive details see Appendix B.

#### Comparison

The results of this experiment are shown in Table 4 in terms of ELBO and if available the IWBO (importance weighted bound) (Burda et al., 2016) with $1000$ samples measured in bits per pixel. Consistent with the language experiments, the traditional dequantization approaches (uniform / variational) are outperformed by Argmax Flows. Interestingly, although argmax flows with softplus thresholding achieves the best ELBO, the argmax flow with Gumbel thresholding approach achieves a better IWBO. The Multinomial Diffusion model performs somewhat worse with 0.37 bpp on test whereas it scored 0.33 bpp on train. Interestingly, this the only model where overfitting was an issue and data augmentation was required, which may explain this portion of the performance difference. For all other models training performance was comparable to test and validation performance. Samples from the different models trained on cityscapes are depicted in Figure 4. Another interesting point is that coupling flows had difficulty producing coherent text samples (Figure 3) but do not suffer from this problem on the cityscapes data which is more image-like. As coupling layers where initially designed for images (Dinh et al., 2015), they may require adjustments to increase their expressiveness on text.

# 7 Social Impact and Conclusion

#### Social Impact

The methods described in this paper can be used to learn categorical distributions. For that reason, they can potentially be used to generate high-dimensional categorical data, such as text or image segmentation maps, faster than iterative approaches. Possibly negative influences are the generation of fake media in the form of text, or very unhelpful automated chat bots for customer service. Our work could positively influence new methods for text generation, or improved segmentation for self-driving cars. In addition, our work may also be used for outlier detection to flag fake content. Also, we believe the method in its current form is still distant from direct applications as the ones mentioned above.

#### Conclusion

In this paper we propose two extensions for Normalizing Flows and Diffusion models to learn categorical data: Argmax Flows and Multinomial Diffusion. Our experiments show that our methods outperform comparable models in terms of negative log-likelihood. In addition, our experiments highlight distinct performance gaps in the field: Between standard ARMs, continuous autoregressive models and non-autoregressive continuous models. This indicates that future work could focus on two sources of decreased performance: 1) when discrete variables are lifted to a continuous space and further 2) when removing autoregressive components.

**Funding Disclosure**\
There are no additional sources of funding to disclose, beyond the affiliations of the authors.

# A Numerically stable Multinomial Diffusion in log space

In this section we explain how Multinomial Diffusion models can be implemented in a numerically safe manner in log-space. Note that in addition to this appendix with pseudo-code, the actual source code will also be released. First we define a few helper functions:

    def log_add_exp(a, b):
        maximum = max(a, b)
        return maximum + log(exp(a - maximum) + exp(b - maximum))
        
    def log_sum_exp(x):
        maximum = max(x, dim=1, keepdim=True)
        return maximum + log(exp(x - maximum).sum(dim=1))

    def index_to_log_onehot(x, num_classes):
        # Assume that onehot axis is inserted at dimension 1
        x_onehot = one_hot(x, num_classes)
        
        # Compute in log-space, extreme low values are later
        # filtered out by log sum exp calls.
        log_x = log(x_onehot.clamp(min=1e-40))
        return log_x

    def log_onehot_to_index(log_x):
        return log_x.argmax(1)
        
    def log_1_min_a(a):
        return log(1 - a.exp() + 1e-40)

Then we can initialize the variables we are planning to utilize for the multinomial diffusion model. This is done with float64 variables to limit the precision loss in the `log_1_min_a` computation. Since these are precomputed and later converted to float32, there is no meaningful increase in computation time.

    alphas = init_alphas() 
    log_alpha = np.log(alphas)
    log_cumprod_alpha = np.cumsum(log_alpha)

    log_1_min_alpha = log_1_min_a(log_alpha)
    log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

Then we can define the functions that we utilize to compute the log probabilities of the categorical distributions of the forward process. The functions below compute the probability vectors for $q({\boldsymbol{x}}_t | {\boldsymbol{x}}_{t-1})$, $q({\boldsymbol{x}}_t | {\boldsymbol{x}}_{0})$ and $q({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_t, {\boldsymbol{x}}_{0})$.

    def q_pred_one_timestep(log_x_t, t):
        # Computing alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha[t],
            log_1_min_alpha[t] - log(num_classes)
        )
        return log_probs

    def q_pred(log_x0, t):
        log_probs = log_add_exp(
            log_x0 + log_cumprod_alpha[t],
            log_1_min_cumprod_alpha[t] - log(num_classes)
        )
        return log_probs

    def q_posterior(log_x0, log_x_t, t):
        # Kronecker delta peak for q(x0 | x1, x0).
        if t == 0:
            log_probs_xtmin = log_x0
        else:
            log_probs_xtmin = q_pred(log_x0, t - 1)
        
        # Note log_x_t is used not x_tmin, subtle and not straightforward
        # why this is true. Corresponds to Algorithm 1.
        unnormed_logprobs = log_probs_xtmin + q_pred_one_timestep(log_x_t, t) 
            
        log_probs_posterior = unnormed_logprobs - log_sum_exp(unnormed_logprobs)
        return log_probs_posterior

Some magic is happening in `q_pred_one_timestep`. Recall that at some point we need to compute $\mathcal{C}({\boldsymbol{x}}_t | (1 - \beta_t) {\boldsymbol{x}}_{t-1} + \beta_t / K )$ for different values of ${\boldsymbol{x}}_t$, which when treated as a function outputs $(1 - \beta_t) + \beta_t / K$ if ${\boldsymbol{x}}_t = {\boldsymbol{x}}_{t-1}$ and $\beta_t / K$ otherwise. This function is symmetric, meaning that $\mathcal{C}({\boldsymbol{x}}_t | (1 - \beta_t) {\boldsymbol{x}}_{t-1} + \beta_t / K ) = \mathcal{C}({\boldsymbol{x}}_{t-1} | (1 - \beta_t) {\boldsymbol{x}}_{t} + \beta_t / K )$. This is why we can switch the conditioning and immediately return the different probability vectors for ${\boldsymbol{x}}_t$. This also corresponds to Equation 13.

Then using the `q_posterior` function as parametrization we predict the probability vector for $p({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_t)$ using a neural network.

    def p_pred(log_x_t, t):
        x_t = log_onehot_to_index(log_x_t)
        log_x_recon = logsoftmax(neuralnet(x_t, t))
        log_model_pred = q_posterior(log_x_recon, log_x_t, t)
        return log_model_pred

And then finally we can compute the loss term $L_t$ using the KL divergence for categorical distributions:

    def categorical_kl(log_prob_a, log_prob_b):
        kl = (log_prob_a.exp() * (log_prob_a - log_prob_b)).sum(dim=1)
        return kl

    def compute_Lt(log_x0, log_x_t, t):
        log_true_prob = q_posterior(log_x0, log_x_t, t)
        log_model_prob = p_pred(log_x_t, t)
        kl = categorical_kl(log_true_prob, log_model_prob)
        loss = sum_except_batch(kl)
        return loss

Coincidentally this code even works for $L_0$ because ${\boldsymbol{x}}_0$ is onehot and then: $$-\log \mathcal{C}({\boldsymbol{x}}_0 | \hat{{\boldsymbol{x}}}_{0}) - \sum_k {\boldsymbol{x}}_{0,k} \log  \hat{{\boldsymbol{x}}}_{0,k} = \sum_k {\boldsymbol{x}}_{0,k} [\underbrace{\log {\boldsymbol{x}}_{0,k}}_{0 \text{ or } \log 0} - \log  \hat{{\boldsymbol{x}}}_{0,k}] = \mathrm{KL}(\mathcal{C}({\boldsymbol{x}}_{0})||\mathcal{C}(\hat{{\boldsymbol{x}}}_{0})),$$ where in the last term ${\boldsymbol{x}}_{0}$ and $\hat{{\boldsymbol{x}}}_{0}$ are probability vectors and $0 \log 0$ is defined to be $0$.

# B Experimental details

This section gives details on experimental setup, architectures and optimization hyperparameters. In addition, the code to reproduce experiments will be released publicly.

#### Diffusion settings

For diffusion we use the cosine schedule for $\{\alpha_t\}$ from (Nichol and Dhariwal, 2021) with the difference that what was previously $\sqrt{\bar{\alpha}_t}$ is now $\bar{\alpha}_t$, so that their factor $\sqrt{\bar{\alpha}_t}$ for the Gaussian mean is equal to our factor $\bar{\alpha}_t$ for categorical parameters. Specifically, our $\bar{\alpha}_t$ are defined using: $$\begin{equation*}
    \bar{\alpha}_t =\frac{f(t)}{f(0)} \quad f(t) = \cos \left( \frac{t/T + s}{1  + s} \cdot \frac{\pi}{2} \right), \quad s = 0.008,
\end{equation*}$$ where $T$ is the total number of diffusion steps. Nichol and Dhariwal (2021) show that instead of sampling $t$ uniformly, variance is reduced when $t$ is importance-sampled with $q(t) \propto \sqrt{\mathbb{E}[L_t^2]}$, which is estimated using training statistics, and we use their approach. The objective can be summarized as: $$\begin{equation}
    \log P({\boldsymbol{x}}_0) \geq \mathbb{E}_{t \sim q(t), {\boldsymbol{x}}_t \sim q({\boldsymbol{x}}_t | {\boldsymbol{x}}_0)} 
    \left[ - \frac{1}{q(t)} \mathrm{KL} \big{(}q({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_t, {\boldsymbol{x}}_0) | p({\boldsymbol{x}}_{t-1} | {\boldsymbol{x}}_t) \right].
\end{equation}$$

#### Gumbel properties

In Table 5 a useful overview of Gumbel properties are given. These equations can be used to sample and compute the likelihood of the (truncated) Gumbel distributions. For a more extensive treatment see (Maddison et al., 2014; Kool et al., 2019).

**Tabla 5.** Summary of Gumbel properties.

<table id="tab:gumbel">
<thead>
<tr>
<th style="text-align: left;">Description</th>
<th style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mrow><mi mathvariant="normal">log</mi><mo>&#8289;</mo></mrow><mi>p</mi></mrow><annotation encoding="application/x-tex">\log p</annotation></semantics></math></th>
<th style="text-align: left;">Sample</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mrow><mi mathvariant="normal">G</mi><mi mathvariant="normal">u</mi><mi mathvariant="normal">m</mi><mi mathvariant="normal">b</mi><mi mathvariant="normal">e</mi><mi mathvariant="normal">l</mi></mrow><mo stretchy="false" form="prefix">(</mo><mi>g</mi><mo stretchy="false" form="prefix">|</mo><mi>ϕ</mi><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">\mathrm{Gumbel}(g | \phi)</annotation></semantics></math></td>
<td style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>ϕ</mi><mo>−</mo><mi>g</mi><mo>−</mo><mrow><mi mathvariant="normal">exp</mi><mo>&#8289;</mo></mrow><mo stretchy="false" form="prefix">(</mo><mi>ϕ</mi><mo>−</mo><mi>g</mi><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">\phi - g - \exp(\phi - g)</annotation></semantics></math></td>
<td style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>g</mi><mo>=</mo><mi>−</mi><mrow><mi mathvariant="normal">log</mi><mo>&#8289;</mo></mrow><mo stretchy="false" form="prefix">(</mo><mi>−</mi><mrow><mi mathvariant="normal">log</mi><mo>&#8289;</mo></mrow><mo stretchy="false" form="prefix">(</mo><mi>u</mi><mo stretchy="false" form="postfix">)</mo><mo stretchy="false" form="postfix">)</mo><mo>+</mo><mi>ϕ</mi></mrow><annotation encoding="application/x-tex">g = -\log (-\log(u)) + \phi</annotation></semantics></math><br />
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>u</mi><mo>∼</mo><mi mathvariant="script">𝒰</mi><mo stretchy="false" form="prefix">(</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">u \sim \mathcal{U}(0, 1)</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mrow><mi mathvariant="normal">max</mi><mo>&#8289;</mo></mrow><mi>i</mi></msub><mrow><mi mathvariant="normal">G</mi><mi mathvariant="normal">u</mi><mi mathvariant="normal">m</mi><mi mathvariant="normal">b</mi><mi mathvariant="normal">e</mi><mi mathvariant="normal">l</mi></mrow><mo stretchy="false" form="prefix">(</mo><msub><mi>g</mi><mi>i</mi></msub><mo stretchy="false" form="prefix">|</mo><mi>ϕ</mi><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">\max_i \mathrm{Gumbel}(g_i | \phi)</annotation></semantics></math></td>
<td style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mrow><mi mathvariant="normal">log</mi><mo>&#8289;</mo></mrow><mrow><mi mathvariant="normal">G</mi><mi mathvariant="normal">u</mi><mi mathvariant="normal">m</mi><mi mathvariant="normal">b</mi><mi mathvariant="normal">e</mi><mi mathvariant="normal">l</mi></mrow><mo stretchy="false" form="prefix">(</mo><msub><mi>g</mi><mi mathvariant="normal">max</mi></msub><mo stretchy="false" form="prefix">|</mo><msub><mi>ϕ</mi><mi mathvariant="normal">max</mi></msub><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">\log \mathrm{Gumbel}(g_{\max} | \phi_{\max})</annotation></semantics></math><br />
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>ϕ</mi><mi mathvariant="normal">max</mi></msub><mo>=</mo><mrow><mi mathvariant="normal">log</mi><mo>&#8289;</mo></mrow><msub><mo>∑</mo><mi>i</mi></msub><mrow><mi mathvariant="normal">exp</mi><mo>&#8289;</mo></mrow><msub><mi>ϕ</mi><mi>i</mi></msub></mrow><annotation encoding="application/x-tex">\phi_{\max} = \log \sum_i \exp \phi_i</annotation></semantics></math></td>
<td style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>g</mi><mi mathvariant="normal">max</mi></msub><mo>∼</mo><mrow><mi mathvariant="normal">G</mi><mi mathvariant="normal">u</mi><mi mathvariant="normal">m</mi><mi mathvariant="normal">b</mi><mi mathvariant="normal">e</mi><mi mathvariant="normal">l</mi></mrow><mo stretchy="false" form="prefix">(</mo><msub><mi>ϕ</mi><mi mathvariant="normal">max</mi></msub><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">g_{\max} \sim \mathrm{Gumbel}(\phi_{\max})</annotation></semantics></math><br />
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>ϕ</mi><mi mathvariant="normal">max</mi></msub><mo>=</mo><mrow><mi mathvariant="normal">log</mi><mo>&#8289;</mo></mrow><msub><mo>∑</mo><mi>i</mi></msub><mrow><mi mathvariant="normal">exp</mi><mo>&#8289;</mo></mrow><msub><mi>ϕ</mi><mi>i</mi></msub></mrow><annotation encoding="application/x-tex">\phi_{\max} = \log \sum_i \exp \phi_i</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mrow><mi mathvariant="normal">T</mi><mi mathvariant="normal">r</mi><mi mathvariant="normal">u</mi><mi mathvariant="normal">n</mi><mi mathvariant="normal">c</mi><mi mathvariant="normal">G</mi><mi mathvariant="normal">u</mi><mi mathvariant="normal">m</mi><mi mathvariant="normal">b</mi><mi mathvariant="normal">e</mi><mi mathvariant="normal">l</mi></mrow><mo stretchy="false" form="prefix">(</mo><mi>g</mi><mo stretchy="false" form="prefix">|</mo><mi>ϕ</mi><mo>,</mo><mi>T</mi><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">\mathrm{TruncGumbel}(g| \phi, T)</annotation></semantics></math></td>
<td style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>ϕ</mi><mo>−</mo><mi>g</mi><mo>−</mo><mrow><mi mathvariant="normal">exp</mi><mo>&#8289;</mo></mrow><mo stretchy="false" form="prefix">(</mo><mi>ϕ</mi><mo>−</mo><mi>g</mi><mo stretchy="false" form="postfix">)</mo><mo>+</mo><mrow><mi mathvariant="normal">exp</mi><mo>&#8289;</mo></mrow><mo stretchy="false" form="prefix">(</mo><mi>ϕ</mi><mo>−</mo><mi>T</mi><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">\phi - g - \exp(\phi - g) + \exp(\phi - T)</annotation></semantics></math><br />
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>g</mi><mo>&lt;</mo><mi>T</mi></mrow><annotation encoding="application/x-tex">g &lt; T</annotation></semantics></math> <math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mi>∞</mi></mrow><annotation encoding="application/x-tex">-\infty</annotation></semantics></math></td>
<td style="text-align: left;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>g</mi><mo>=</mo><mi>ϕ</mi><mo>−</mo><mrow><mi mathvariant="normal">log</mi><mo>&#8289;</mo></mrow><mo stretchy="false" form="prefix">(</mo><mrow><mi mathvariant="normal">exp</mi><mo>&#8289;</mo></mrow><mo stretchy="false" form="prefix">(</mo><mi>ϕ</mi><mo>−</mo><mi>T</mi><mo stretchy="false" form="postfix">)</mo><mo>−</mo><mrow><mi mathvariant="normal">log</mi><mo>&#8289;</mo></mrow><mi>u</mi><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">g = \phi - \log(\exp(\phi - T) - \log u)</annotation></semantics></math><br />
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>u</mi><mo>∼</mo><mi mathvariant="script">𝒰</mi><mo stretchy="false" form="prefix">(</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">u \sim \mathcal{U}(0, 1)</annotation></semantics></math></td>
</tr>
</tbody>
</table>

## B.1 Language Modelling

For the language modelling experiments we utilize the standard `text8` dataset with sequence length $256$ and `enwik8` dataset with sequence length $320$. The train/val/test splits are 90000000/5000000/5000000 for both `text8` and `enwik8`, as is standard in literature. The Multinomial Text Diffusion models are trained for $300$ epochs, whereas the Argmax Flows are trained for $40$ epochs, with the exception of the Argmax Coupling Flow on enwik8 which only needs to be trained for $20$ epochs. Further details are presented in Tables 6 and 7. In addition, the code to reproduce results will be publicly available. There are no known ethics issues with these datasets at the time of writing.

**Tabla 6.** Optimization details for text models.

| Model | batch size | lr | lr decay | optimizer | dropout |  |  |  |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Multinomial Text Diffusion (text8) | 32 | 0.0001 | 0.99 | Adam | 0 |  |  |  |
| Multinomial Text Diffusion (enwik8) | 32 | 0.0001 | 0.99 | Adam | 0 |  |  |  |
| Argmax AR Flow (text8) | 64 | 0.001 | 0.995 | Adam | 0.25 |  |  |  |
| Argmax AR Flow (enwik8) | 64 | 0.001 | 0.995 | Adam | 0.25 |  |  |  |
| Argmax Coupling Flow (text8) | 16 | 0.001 | 0.995 | Adamax | 0.05 |  |  |  |
| Argmax Coupling Flow (enwik8) | 32 | 0.001 | 0.995 | Adamax | 0.1 |  |  |  |

**Tabla 7.** Architecture description for text models.

| Model | Architecture description |  |  |  |  |  |  |  |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Multinomial Text Diffusion (text8) | 12-layer transformer 8 global, 8 local heads / 1000 diffusion steps |  |  |  |  |  |  |  |
| Multinomial Text Diffusion (enwik8) | 12-layer transformer 8 global, 8 local heads / 4000 diffusion steps |  |  |  |  |  |  |  |
| Argmax AR Flow (text8) | 2-layer LSTM, 2048 hidden units |  |  |  |  |  |  |  |
| Argmax AR Flow (enwik8) | 2-layer LSTM, 2048 hidden units |  |  |  |  |  |  |  |
| Argmax Coupling Flow (text8) | 2-layer bi-directional LSTM, 512 hidden units |  |  |  |  |  |  |  |
| Argmax Coupling Flow (enwik8) | 2-layer bi-directional LSTM, 768 hidden units |  |  |  |  |  |  |  |

## B.2 Cityscapes

#### Preprocessing

The Cityscapes (Cordts et al., 2016) segmentation maps are re-sampled to a $32$ by $64$ pixel image using nearest neighbour interpolation. The original segmentation maps are downloaded from <https://www.cityscapes-dataset.com/downloads/> where all files are contained in `gtFine_trainvaltest.zip`. Note that we train on a $8$-class problem since we only consider what is called the `category_id` field in torchvision. We re-purpose the validation set as test set, containing $500$ maps. The original train set containing $2975$ maps is split into $2500$ maps for training and $475$ maps for validation. The original test set is not utilized. To aid reproducibility we will publish source code that includes the preprocessing and the dataloaders. There are no known ethics issues with the segmentation maps at the time of writing. License is located at <https://www.cityscapes-dataset.com/license/>.

#### Architectures

For Cityscapes all models utilize the same architectures, although they represent a different part for their respective model designs. The density model $p({\boldsymbol{v}})$ consist of $4$ levels with $10$ subflows each, separated by squeeze layers, where each subflow consists of a $1$ $\times$ $1$ convolution and an affine coupling layer. The coupling layers are parametrized by DenseNets (Huang et al., 2017). The same model is used for the latent distribution in the VAE (usually referred to as $p({\boldsymbol{z}})$ in literature). The probabilistic inverse $q({\boldsymbol{v}}| {\boldsymbol{x}})$ is modelled by a single level flow that has $8$ subflows, again consisting of affine coupling layers and $1$ $\times$ $1$ convolutions. To condition on ${\boldsymbol{x}}$ it is processed by a DenseNet which outputs a representation for the coupling layers that is concatenated to the original input. The same model is utilized to parametrize the VAE encoder (commonly referred to as $q({\boldsymbol{z}}| {\boldsymbol{x}})$). The VAE additionally has a model for the decoder $p({\boldsymbol{x}}| {\boldsymbol{z}})$ which is parametrized by a DenseNet which outputs the parameters for a categorical distribution. The models are optimized using the same settings, and no hyperparameter search was performed. Specifically, the models are optimized with minibatch size $64$ for $2000$ epochs with the Adamax optimizer with learning rate $0.001$ and a linear learning rate warmup of $10$ epochs and a decay factor of $0.995$.

## B.3 Range of considered hyperparameters

For Multinomial Text Diffusion we experimented with the depth of transformers $\{1, 2, 4, 8, 12, 16, 20\}$ and the hidden size $\{128, 256, 512, 1024\}$. We found that models with depth $12$ and $512$ could be trained in a reasonable amount of time while giving good performance. For the cityscapes experiments no hyperparameter search was performed.

## B.4 Details on latent normalizing flows for text8

We utilize the official code repository from (Ziegler and Rush, 2019) in here[^2]. The original code utilizes $10$ ELBO samples, which is relatively expensive. For that reason we instead opt for $1$ ELBO sample and find it gives similar results. The batch size is increased from $16$ to $32$. Additionally we reduce the KL scheduling from $4$ initial $10^{-5}$ epochs to only $2$ initial $10^{-5}$ epoch and we anneal linearly over the next $4$ epochs instead of over the next $10$ epochs. In total the models are optimized for $30$ epochs. We verify that the resulting models still achieve similar performance on the Penn Tree Bank experiment compared to the original paper in terms of ELBO values: Our hyperparameter setup for AF/AF achieves slightly better performance with 1.46 versus 1.47 bpc and for IAF/SCF achieves slightly worse 1.78 versus 1.76 bpc.

## B.5 Computing infrastructure

Experiments where run on NVIDIA-GTX 1080Ti GPUs, CUDA 10.1 with Python version 3.7.6 in Pytorch 1.5.1 or 1.7.1.

# C Reproducing Discrete Flows

In this section we detail our efforts to reproduce the results from discrete flows (Tran et al., 2019). Specifically, we are interested in the discrete flows models that map to *factorized* distributions, for instance the discrete bipartite (coupling) flow. We avoid situations where an autoregressive base distribution is used, it may be difficult to identify how much the flow is actually learning versus the ARM as base. For this paper an official implementation was released at <https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/> in the files `discrete_flows.py` and `utils.py`. However, this codebase contains only the high-level modules and code for the toy example, it does not contain the specific code related to the language experiments. These high-level modules and the toy problem were ported to PyTorch here: <https://github.com/TrentBrick/PyTorchDiscreteFlows>. Using this codebase, we were able to compare on the quantized eight Gaussians toy dataset, as depicted in Figure 6. In this experiment we clearly see that argmax flows outperform discrete flows both numerically (6.32 versus 7.0 nats) and visually by comparing the samples or probability mass function.

**Figura 6.** Reproduction of the quantized eight Gaussians experiment. Plots show either the probability mass function or weighted number of samples (which will tend towards the pmf).

Subsequent efforts by others to reproduce the language experiments failed (see <https://github.com/TrentBrick/PyTorchDiscreteFlows/issues/1>). In another work, Lippe and Gavves (2020) also noticed the difficulty of getting discrete flows to succesfully optimize, as detailed in the set shuffling/summation experiment corresponding to Table 5 in the paper.

For this paper we also tried to reproduce the language experiments. After verifying the correctness of the `one_hot_argmax`, `one_hot_minus` and `one_hot_add` functions in <https://github.com/TrentBrick/PyTorchDiscreteFlows>, we implemented an autoregressive discrete flow layer with an expressive network, in an effort to limit the accumulated gradient bias. Recall that an autoregressive layer is more expressive than a coupling layer as it has more dependencies between dimensions. As can be seen in Table 8 our re-implementation also performed considerably worse, matching the experience of the others described above.

**Tabla 8.** Discrete Flows on `text8`. Note that AR is more expressive than coupling.

| Model | text8 (bpc) |
|:---|:--:|
| Discrete Flows from paper (coupling, factorized base, without scale) | 1.29 |
| Discrete Flows from paper (coupling, factorized base, with scale) | 1.23 |
| Discrete Flows reimplementation (AR, factorized base, without scale) | 4.13 |
| Argmax Flow, AR (ours) | 1.38 |
| Argmax Coupling Flow (ours) | 1.80 |

#### Final remarks

We have had extensive contact with the authors of (Tran et al., 2019) to resolve this issue over the course of several months. Unfortunately it is not possible for them to share the code for the language flows due to internal dependencies. Also, we have not been able to find any implementation of discrete flows online that achieves the reported performance on text. The authors generously offered to look at our reimplementation, which we have shared with them. At the time of writing we have not yet heard anything back on the code. For the reasons described in this appendix, we currently assume that the language experiments in discrete flows are not reproducible.

# D Additional experiments

A comparison of the performance for Cartesian products with different bases is shown in Table 9. Note that this experiment was performed using a somewhat smaller architecture then in the main text. As can be seen, the performance difference between different Cartesian products is relatively small. The performance does decreases slightly over larger base numbers, indicating that it is better to choose a small base that results in fewer overall dimensions.

**Tabla 9.** Cartesian Products with different base numbers trained using a slightly smaller version of the Argmax AR Flow on `text8`.

| Model         | text8 (bpc) |
|:--------------|:-----------:|
| $d_m=1, M=27$ |    1.45     |
| $d_m=2, M=6$  |    1.44     |
| $d_m=3, M=3$  |    1.44     |
| $d_m=5, M=2$  |    1.44     |

A comparison of sampling time speeds are shown in Table 10. A couple of orders in magnitude difference can be seen comparing autoregressive versus non-autoregressive models. This highlights the importance of researching generative models that can be built from non-autoregressive components. The main source of difference between our coupling approach and IAF/SCF is that we utilize mixture of discretized logistics (Ho et al., 2019) as coupling transformation, which requires a iterative process to invert over 1 dimension. The multinomial diffusion takes in-between the time of autoregressive and coupling models. Also reducing steps reduces the required sampling time, as is expected.

**Tabla 10.** Comparison of different methods in terms of sample time. Sample time is measured by generating a single text sample of length 256 averaged over $10$ runs, unless specified otherwise.

| Model type |  | Model | Sample time (s) |  |  |
|:---|:---|:---|:--:|:--:|:--:|
| ARM |  | 64 Layer Transformer (Al-Rfou et al., 2019) | 35.5$^\dagger$ |  |  |
| VAE |  | AF/AF$^\star$ (AR) (Ziegler and Rush, 2019) | 156 $\pm 1.8$ |  |  |
|  |  | IAF / SCF$^\star$ (Ziegler and Rush, 2019) | 0.04 $\pm0.004$ |  |  |
| Generative Flow |  | Argmax Flow, AR (ours) | 115 $\pm 0.03$ |  |  |
|  |  | Argmax Coupling Flow (ours) | 0.40 $\pm0.03$ |  |  |
|  |  | Discrete Flow (Tran et al., 2019) | 0.16$^\dagger$ |  |  |
| Diffusion |  | Multinomial Text Diffusion (ours) | 26.6 $\pm2.2$$^{\ddagger}$ |  |  |
|  |  | Multinomial Text Diffusion, 100 steps (ours) | 2.4 $\pm0.16$ |  |  |

<div class="flushleft">

$\dagger$ Computed on a 288-length sequence instead of 256-length, taken from (Tran et al., 2019).\
$\ddagger$ This result is for the complete 1000 timesteps chain, improvements are possible by skipping steps.

</div>

Due to the computational cost of running normalizing flows, it is not possible for us to run every model many times. However, generally single-run results suffice, as the performance variance of these models is relatively small. In Table 11 the standard deviation and average performance for a selection of models is shown, taken over $3$ runs. Observe that these standard deviations are small compared to the reported differences between the models. Notice that standard deviations for coupling models are larger, but the performance difference between those types of models is also larger.

**Tabla 11.** Average and standard deviations of several models.

| Dequantization     | Flow type | Dataset | average |  stdev  |     |
|:-------------------|:---------:|:-------:|:-------:|:-------:|:---:|
| Argmax Flow (ours) |    AR     |  text8  |  1.38   | $0.001$ |     |
| Argmax Flow (ours) |    AR     | enwik8  |  1.42   | $0.008$ |     |
| Argmax Flow (ours) | Coupling  |  text8  |  1.82   | $0.017$ |     |
| Argmax Flow (ours) | Coupling  | enwik8  |  1.93   | $0.012$ |     |

Finally, we also compare argmax flows to a situation where its density model exactly matches the density model in (Lippe and Gavves, 2020) on `text8`. In this experiment Argmax Flows (1.43 bpc) outperform CategoricalNF (1.45 bpc) in an equal setting.

# E Samples from the text models

Samples from our proposed models are presented in Table 12 and a Multinomial Text Diffusion train is shown in Figure 7, these results were not cherry-picked.

**Tabla 12.** Samples from models trained on text8.

| Model | Nr | Text |
|:---|:---|:---|
|  | 1 | ` that the role of tellings not be required also action characters passed on constitution ahmad a nobilitis first be closest to the cope and dhur and nophosons she criticized itm specifically on august one three movement and a renouncing local party of exte` |
|  | 2 | `nt is in this meant the replicat today through the understanding element thinks the sometimes seven five his final form of contair you are lotur and me es to ultimately this work on the future all all machine the silon words thereis greatly usaged up not t` |
|  | 3 | `arity island louis has convinced privatist provinces the restrained marriage of his income ted guilds which in gulick performed in one nine six seven then sponly onward the bambat loving in separate including tichatta westell s doubled a bound of his futur` |
|  | 4 | `same early duration without education as a golden core power to the pirit of spain arriving wise speech art and r t plain firman q one five six the same as part of herald h rogenszers a art poetic of literature at shaft bressen three five three five eight ` |
|  | 1 | ` heartedness frege thematically infered by the famous existence of a function f from the laplace definition we can analyze a definition of binary operations with additional size so their functionality cannot be reviewed here there is no change because its ` |
|  | 2 | `otal cost of learning objects from language to platonic linguistics examines why animate to indicate wild amphibious substances animal and marine life constituents of animals and bird sciences medieval biology biology and central medicine full discovery re` |
|  | 3 | `o use language combined with any of its subsets evolved into the group containing the primary concepts of a daily line on off the road and the material emulation of welcomes and prospects of pleasure and exercise have been committed projects in the economy` |
|  | 4 | `en that are beginning to forge since october one nine five zero the mandate was planted at k nigsberg during the car horizon at first please refer to a small government situated as well as in all these countries finally giving birth to a band here he was a` |
|  | 1 | `ns fergenur d alpha and le heigu man notabhe leglon lm n two six a gg opa movement as sympathetic dutch the term bilirubhah acquired the bava rian cheeh segt thmamouinaire vhvinus lihnos ineoneartis or medical iod ine the rave wesp published harsy varb hhgh` |
|  | 2 | `and inequalities syllee mike jean demet in standard rather than fmxed liga and a piare nut is gruncionde aodadneveshiopyhabally uchc one viredtlty three ben yi agricultariis the only mefamantia or nuil and mid satio for kigou wore not on the war rits af` |
|  | 3 | `e g chain within the sale of cooperative oppine p nge tyae yarot bouatta real frequency one mbj or rorbepetam iw by someone c langt b kindoms is the single yenta valve nor eosed collagen surkeys in the goubark cuisine of animum and two trantual measurement` |
|  | 4 | `hilepuin the king pete was added to or who cefralded to kiark n and panhpur not souhhvestern bat batas mudtlu for this creatures chew palenque lii lasron gentla tzanemi derived from oo four issais nivissos with the name convertinus magaa named wes orieanr` |

**Figura 7.** Intermediate steps of the generation chain of the Multinomial Text Diffusion model trained on `text8`.

# References

Cooijmans et al. (2017). Cooijmans, T.; Ballas, N.; Laurent, C.; Gül, .; Courville, A. C. Recurrent Batch Normalization. 5th International Conference on Learning Representations, ICLR. 2017

Dai et al. (2019). Dai, Z.; Yang, Z.; Yang, Y.; Carbonell, J. G.; Le, Q. V.; Salakhutdinov, R. Transformer-XL: Attentive Language Models beyond a Fixed-Length Context. Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019. 2019

Uria et al. (2013). Uria, B.; Murray, I.; Larochelle, H. RNADE: The Real-valued Neural Autoregressive Density-estimator. Advances in Neural Information Processing Systems. 2013; pp 2175–2183

Theis et al. (2016). Theis, L.; van den Oord, A.; Bethge, M. A note on the evaluation of generative models. International Conference on Learning Representations. 2016

Ho et al. (2019). Ho, J.; Chen, X.; Srinivas, A.; Duan, Y.; Abbeel, P. Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design. 36th International Conference on Machine Learning 2019,

Sohl-Dickstein et al. (2015). Sohl-Dickstein, J.; Weiss, E. A.; Maheswaranathan, N.; Ganguli, S. Deep Unsupervised Learning using Nonequilibrium Thermodynamics. Proceedings of the 32nd International Conference on Machine Learning, ICML. 2015

Ho et al. (2020). Ho, J.; Jain, A.; Abbeel, P. Denoising Diffusion Probabilistic Models. CoRR 2020, abs/2006.11239

Song et al. (2020). Song, J.; Meng, C.; Ermon, S. Denoising Diffusion Implicit Models. CoRR 2020, abs/2010.02502

Nichol and Dhariwal (2021). Nichol, A. Q.; Dhariwal, P. Improved Denoising Diffusion Probabilistic Models. 2021; https://openreview.net/forum?id=-NEXDKk8gZ

Rezende and Mohamed (2015). Rezende, D.; Mohamed, S. Variational Inference with Normalizing Flows. Proceedings of the 32nd International Conference on Machine Learning. 2015; pp 1530–1538

Nielsen et al. (2020). Nielsen, D.; Jaini, P.; Hoogeboom, E.; Winther, O.; Welling, M. SurVAE Flows: Surjections to Bridge the Gap between VAEs and Flows. CoRR 2020, abs/2007.02731

Maddison et al. (2014). Maddison, C. J.; Tarlow, D.; Minka, T. A\* Sampling. Advances in Neural Information Processing Systems 27: Annual Conference on Neural Information Processing Systems. 2014

Kool et al. (2019). Kool, W.; Van Hoof, H.; Welling, M. Stochastic Beams and Where To Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement. Proceedings of the 36th International Conference on Machine Learning. 2019

Winkler et al. (2019). Winkler, C.; Worrall, D. E.; Hoogeboom, E.; Welling, M. Learning Likelihoods with Conditional Normalizing Flows. CoRR 2019, abs/1912.00042

Germain et al. (2015). Germain, M.; Gregor, K.; Murray, I.; Larochelle, H. Made: Masked autoencoder for distribution estimation. International Conference on Machine Learning. 2015; pp 881–889

Kingma and Welling (2014). Kingma, D. P.; Welling, M. Auto-Encoding Variational Bayes. Proceedings of the 2nd International Conference on Learning Representations. 2014

Rezende et al. (2014). Rezende, D. J.; Mohamed, S.; Wierstra, D. Stochastic Backpropagation and Approximate Inference in Deep Generative Models. Proceedings of the 31th International Conference on Machine Learning, ICML. 2014

Goodfellow et al. (2014). Goodfellow, I.; Pouget-Abadie, J.; Mirza, M.; Xu, B.; Warde-Farley, D.; Ozair, S.; Courville, A.; Bengio, Y. Generative adversarial nets. Advances in neural information processing systems. 2014; pp 2672–2680

Dinh et al. (2017). Dinh, L.; Sohl-Dickstein, J.; Bengio, S. Density estimation using Real NVP. 5th International Conference on Learning Representations, ICLR 2017,

Kingma et al. (2016). Kingma, D. P.; Salimans, T.; Jozefowicz, R.; Chen, X.; Sutskever, I.; Welling, M. Improved variational inference with inverse autoregressive flow. Advances in Neural Information Processing Systems. 2016; pp 4743–4751

Papamakarios et al. (2017). Papamakarios, G.; Murray, I.; Pavlakou, T. Masked autoregressive flow for density estimation. Advances in Neural Information Processing Systems. 2017; pp 2338–2347

Chen et al. (2018). Chen, T. Q.; Rubanova, Y.; Bettencourt, J.; Duvenaud, D. K. Neural ordinary differential equations. Advances in Neural Information Processing Systems. 2018; pp 6572–6583

Song et al. (2019). Song, Y.; Meng, C.; Ermon, S. MintNet: Building Invertible Neural Networks with Masked Convolutions. Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019. 2019

Perugachi-Diaz et al. (2020). Perugachi-Diaz, Y.; Tomczak, J. M.; Bhulai, S. Invertible DenseNets. CoRR 2020, abs/2010.02125

Nielsen and Winther (2020). Nielsen, D.; Winther, O. Closing the Dequantization Gap: PixelCNN as a Single-Layer Flow. Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS. 2020

Tran et al. (2019). Tran, D.; Vafa, K.; Agrawal, K.; Dinh, L.; Poole, B. Discrete Flows: Invertible Generative Models of Discrete Data. ICLR 2019 Workshop DeepGenStruct 2019,

Hoogeboom et al. (2019). Hoogeboom, E.; Peters, J. W. T.; van den Berg, R.; Welling, M. Integer Discrete Flows and Lossless Compression. Neural Information Processing Systems 2019, NeurIPS 2019. 2019; pp 12134–12144

van den Berg et al. (2020). van den Berg, R.; Gritsenko, A. A.; Dehghani, M.; Snderby, C. K.; Salimans, T. IDF++: Analyzing and Improving Integer Discrete Flows for Lossless Compression. CoRR 2020, abs/2006.12459

Ziegler and Rush (2019). Ziegler, Z. M.; Rush, A. M. Latent Normalizing Flows for Discrete Sequences. Proceedings of the 36th International Conference on Machine Learning, ICML. 2019

Lippe and Gavves (2020). Lippe, P.; Gavves, E. Categorical Normalizing Flows via Continuous Transformations. CoRR 2020, abs/2006.09790

Song et al. (2020). Song, Y.; Sohl-Dickstein, J.; Kingma, D. P.; Kumar, A.; Ermon, S.; Poole, B. Score-Based Generative Modeling through Stochastic Differential Equations. CoRR 2020, abs/2011.13456

Al-Rfou et al. (2019). Al-Rfou, R.; Choe, D.; Constant, N.; Guo, M.; Jones, L. Character-Level Language Modeling with Deeper Self-Attention. The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019. 2019

Huang et al. (2017). Huang, G.; Liu, Z.; Van Der Maaten, L.; Weinberger, K. Q. Densely connected convolutional networks. Proceedings of the IEEE conference on computer vision and pattern recognition. 2017; pp 4700–4708

Burda et al. (2016). Burda, Y.; Grosse, R. B.; Salakhutdinov, R. Importance Weighted Autoencoders. 4th International Conference on Learning Representations. 2016

Dinh et al. (2015). Dinh, L.; Krueger, D.; Bengio, Y. NICE: Non-linear independent components estimation. 3rd International Conference on Learning Representations, ICLR, Workshop Track Proceedings 2015,

Cordts et al. (2016). Cordts, M.; Omran, M.; Ramos, S.; Rehfeld, T.; Enzweiler, M.; Benenson, R.; Franke, U.; Roth, S.; Schiele, B. The Cityscapes Dataset for Semantic Urban Scene Understanding. 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR. 2016; pp 3213–3223mcitethebibliography

[^1]: Equal contribution.

[^2]: <https://github.com/harvardnlp/TextFlow>
