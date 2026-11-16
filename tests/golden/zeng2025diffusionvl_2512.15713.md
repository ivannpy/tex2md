---
title: "DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models"
authors: ["Lunbin Zeng", "Jingfeng Yao", "Bencheng Liao", "Hongyuan Tao", "Wenyu Liu", "Xinggang Wang"]
year: 2025
arxiv: "2512.15713"
arxiv_version: "v3"
marginnote:
  - bookmd5: 81bf80f4b35a99ada7ce7f12ebfea30675883c86855c1758e465420ba3d3b066
    path: "PCIC/Tesis/ar-diff/DiffusionVL- Translating Any Autoregressive Models into Diffusion Vision Language Models.pdf"
source: "arXiv LaTeX, tex2md 0.1"
converted: 2026-11-16
---

# DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models

**Autores:** Lunbin Zeng; Jingfeng Yao; Bencheng Liao; Hongyuan Tao; Wenyu Liu; Xinggang Wang

> arXiv:2512.15713v3 · https://arxiv.org/abs/2512.15713
> Convertido desde la fuente LaTeX de arXiv con tex2md (pandoc). Numeración de secciones, figuras, tablas y ecuaciones tomada de la compilación del artículo; las figuras se reducen a su leyenda. Ante cualquier duda, el PDF es la referencia.

## Abstract

Diffusion-based decoding has recently emerged as an appealing alternative to autoregressive (AR) generation, offering the potential to update multiple tokens in parallel and reduce latency. However, diffusion vision language models (dVLMs) still lag significantly behind mainstream autoregressive vision language models. This is due to the scarcity and weaker performance of base diffusion language models (dLLMs) compared with their autoregressive counterparts. This raises a natural question: Can we build high-performing dVLMs directly from existing powerful AR models, without relying on dLLMs? We propose **DiffusionVL**, a family of dVLMs obtained by translating pretrained AR models into the diffusion paradigm via an efficient diffusion finetuning procedure that changes the training objective and decoding process while keeping the backbone architecture intact. Through an efficient diffusion finetuning strategy, we successfully adapt AR pretrained models into the diffusion paradigm. This approach yields two key observations: (1) The paradigm shift from AR-based multimodal models to diffusion is remarkably effective. (2) Direct conversion of an AR language model to a dVLM is also feasible, achieving performance comparable to that of the same AR model finetuned with standard autoregressive visual instruction tuning. To enable practical open-ended generation, we further integrate block decoding, which supports arbitrary-length outputs and KV-cache reuse for faster inference. Our experiments demonstrate that despite training with **less than 5% of the data** required by prior methods, DiffusionVL achieves a comprehensive performance improvement, with **a 34.4% gain** on the MMMU-Pro (vision) benchmark and **37.5% gain** on the MME (Cog.) benchmark, alongside a **2× inference speedup**. The model and code are released at <https://github.com/hustvl/DiffusionVL>.

**Figura 1.** **Performance Comparison.** Our DiffusionVL achieves state-of-the-art (SOTA) performance among diffusion vision language models including (Li et al., 2025b; You et al., 2025) and competitive performance with Qwen2.5-VL (Qwen et al., 2025).

<span id="fig: benchmark" data-label="fig: benchmark"></span>

[^1]

# 1 Introduction

Vision language models (VLMs) (Bai et al., 2025; Liu et al., 2023; Chen et al., 2024b; Tao et al., 2025) have achieved significant success in multimodal understanding. Most of them are autoregressive (AR) models with the next-token prediction (NTP) paradigm. A key limitation of the NTP paradigm is its inability to inherently support parallel inference, which limits its applicability in real-time scenarios.

Recently, the diffusion paradigm (Li et al., 2025b; You et al., 2025; Yu et al., 2025) has emerged as a promising alternative, offering more efficient parallel decoding potential than the NTP paradigm. However, existing diffusion VLMs (dVLMs) still lag behind advanced AR-VLMs on standard multimodal benchmarks (Figure 1). A key bottleneck is that today’s diffusion language models (dLLMs) are generally less capable and less mature than strong AR language models (AR-LMs), which limits the ceiling of dVLMs built on top of dLLMs. For instance, LLaDA-8B lags behind Qwen2.5-7B by 42.0% on the code task HumanEval (Chen et al., 2021). Influenced by the conventional visual instruction tuning, existing attempts still assume that building dVLMs must rely on cross-modal training from a dLLM of the same paradigm. In fact, dVLMs and AR-VLMs are structurally identical; the only difference lies in their attention patterns and training/inference behaviors. Since the architecture does not dictate the paradigm, building dVLMs from dLLMs is not the only option. Given these challenges and findings, a compelling question emerges: *Can we build high-performing dVLMs *directly* from existing powerful AR models, without relying on dLLMs?*

To answer this question, we explore translating any pretrained autoregressive models into diffusion vision language models (see Figure 2). We propose DiffusionVL, whose core technical contribution is demonstrating that a simple diffusion finetuning approach can achieve this translation.

Specifically, we convert the next-token prediction paradigm of the original AR model into a diffusion paradigm, which we refer to as "diffusion finetuning". Its key advantage is that it enables the construction of DiffusionVL from any AR model without any architectural modifications. For AR-VLMs, since these models are already vision-language aligned, we directly apply full-parameter diffusion finetuning to convert AR-VLMs to dVLMs. For AR-LMs, we build the vision-language model in two stages. We first train only the connector during the pretraining stage to align the vision and text spaces, with the autoregressive paradigm ensuring training stability (Liu et al., 2023). We then conduct diffusion finetuning in the second stage to complete the paradigm conversion. Furthermore, we adopt the block decoding strategy consistent with (Arriola et al., 2025), which not only supports response generation of arbitrary lengths but also effectively reuses KV-cache for enhanced efficiency. Building on these technical designs, we introduce the DiffusionVL family, which can be translated from AR models of any scale and modality, while maintaining efficient inference speeds.

**Figura 2.** **Paradigm Shift and Modality Shift.** (a): translating AR-VLMs to dVLMs (paradigm shift only). (b): translating AR-LMs to dVLMs (modality shift and paradigm shift). We demonstrate that any autoregressive models with different modalities can be effectively translated to diffusion vision language models.

Our large amounts of experimental results validate the performance and efficiency of DiffusionVL. By seamlessly translating AR-VLMs into DiffusionVL, our method achieves state-of-the-art (SOTA) performance among current dVLMs. Remarkably, this is accomplished using less than 5% of the training data required by prior approaches, thereby significantly narrowing the performance gap with advanced, data-intensive AR-VLMs. Specifically, DiffusionVL achieves a comprehensive performance improvement, with a 34.4% gain on the MMMU-Pro (vision) benchmark and 37.5% gain on the MME (Cog.) benchmark. When translating AR-LMs to DiffusionVL, we have drawn the following conclusions through detailed controlled experiments: under the same training data and configuration, the DiffusionVL finetuned from AR-LM consistently outperforms its counterpart finetuned from dLLM on downstream multimodal benchmarks. Furthermore, even when compared to AR-VLMs with autoregressive finetuning of the same paradigm, our DiffusionVL can still achieve comparable performance on downstream benchmarks. For inference, equipped with the block decoding strategy, DiffusionVL achieves a $2.0 \times$ speedup over previous dVLMs, demonstrating great potential of our method.

In summary, our contributions are threefold.

- We validate the effectiveness and feasibility of translating any pretrained autoregressive models into diffusion vision language models, providing an efficient and low-cost approach for developing high-performance diffusion vision language models.

- We incorporate a block decoding strategy that enables arbitrary-length generation and efficient KV-cache reuse, addressing two key limitations of existing diffusion vision language models.

- Extensive experimental results demonstrate that our method yields a state-of-the-art DiffusionVL, which not only narrows the performance gap with advanced autoregressive vision language models but also achieves a $2.0\times$ speedup compared to existing diffusion vision language models.

# 2 Related Work

## 2.1 Masked Diffusion Models

Diffusion models have achieved great success in vision generation and other computer vision tasks (Peebles and Xie, 2023; Esser et al., 2024; Yao et al., 2025; Yao et al., 2024; Zou et al., 2025; Liao et al., 2025; Li et al., 2025c; Zhu et al., 2025). At the same time, recent work has begun to explore the potential of diffusion in text generation. Prior work (Gong et al., 2024; Austin et al., 2021; Lou et al., 2023) on masked discrete diffusion models (MDMs) has already validated the effectiveness of this paradigm in small-scale text pretraining scenarios. These early studies demonstrated that MDMs can achieve perplexity levels comparable to those of AR-LMs while enabling parallel inference. Building on this basis, recent advanced models such as LLaDA (Nie et al., 2025) and Dream (Ye et al., 2025) have further validated the effectiveness and scalability of MDMs in modern large language model (LLM) settings.

In the visual domain, diffusion vision language models (dVLMs) are typically built from a pre-trained diffusion language model and converted through visual instruction finetuning. Dimple (Yu et al., 2025) designs a novel training paradigm that combines an autoregressive phase with a subsequent diffusion phase. LLaDA-V (You et al., 2025) directly explores the effectiveness of large-scale visual finetuning under the diffusion paradigm based on LLaDA. LaViDa (Li et al., 2025b) further introduces complementary masking and prefix cache to improve training and inference efficiency. These models follow the LLaVA (Liu et al., 2023) paradigm and have explored the potential of the mask diffusion paradigm in the visual and multimodal domains. However, they are limited by their inability to perform variable-length generation and to reuse the KV-cache efficiently. A notable performance gap also remains compared with advanced AR-VLMs. Beyond multimodal understanding, recent work extends the masked diffusion paradigm to unified frameworks: MMaDA (Yang et al., 2025) and LaViDa-O (Li et al., 2025a) demonstrate that the masked diffusion paradigm, when combined with finetuning and reinforcement learning, can support both image and text generation and editing within a single model.

## 2.2 Interpolation between AR and Diffusion

Due to limitations of the MDMs in KV-cache reuse and arbitrary-length generation, several works have explored how to combine the autoregressive (AR) and masked diffusion paradigms for text generation. SSD-LM (Han et al., 2022) introduced a block formulation of Gaussian text diffusion. AR-Diffusion (Wu et al., 2023) further extended SSD-LM by incorporating a left-to-right noise schedule. BD3-LM (Arriola et al., 2025) interpolates between discrete masked diffusion and autoregressive models by using inner-block diffusion and inter-block autoregressive decoding, and has served as a foundation for follow-up research. However, these early block diffusion methods are limited to small-scale text pre-training; their scalability to large language models and extension to the vision-language domain remained unproven.

Recently, some works have attempted to use autoregressive models as initialization and convert them into block diffusion paradigm. To scale this block diffusion paradigm to large text models, SDAR (Cheng et al., 2025), Fast-dLLM-V2 (Wu et al., 2025a) verified that finetuning from AR models can build efficient block diffusion language models. SDLM (Liu et al., 2025) further extended the block diffusion paradigm to next-sequence prediction. More recently, LLaDA 2.0 (Bie et al., 2025) scales block diffusion to 100B parameters and optimizes inference speed, achieving performance competitive with advanced open-source AR-LMs while delivering significantly faster inference. However, such AR-diffusion interpolation remains underexplored in vision language models; our work addresses this gap by translating any autoregressive models into dVLMs through a simple diffusion finetuning.

# 3 Method

## 3.1 Preliminaries

In the autoregressive paradigm, text generation is modeled as next-token prediction. Given an input sequence $\{x^1,\dots,x^L\}$, the training objective is to minimize the cross-entropy loss. 

$$\begin{equation}
\tag{1}
\mathcal{L}_{\text{AR}}(x; \theta) = -\mathbb{E}_x \left[ \sum_{i=1}^L \log P_\theta(x^i \mid x^{<i}) \right],
\end{equation}$$

where the model $P_\theta(\cdot |x^{<i})$ aims to maximize the conditional probability of the current position by using the preceding context $x^{<i}=x^1,\dots,x^{i-1}$.

By contrast, masked diffusion models can be subdivided into two paradigms: full diffusion and block diffusion. The full diffusion paradigm adds and removes noise simultaneously throughout the entire sequence. For a time $t \in(0,1)$, the input sequence is masked with a probability of $t$, yielding a noisy sequence $x_t$. The model $P_\theta(\cdot |x_t)$ is trained to minimize the expected masked position prediction loss. 

$$\begin{equation}
\tag{2}
\mathcal{L}_{\text{DM}}(x; \theta) = -\mathbb{E}_{t,x_0,x_t} \left[ \frac{1}{t} \sum_{i \in \mathcal{M}_t} \log P_\theta(x_0^i \mid x_t) \right],
\end{equation}$$

where $t \sim \mathcal{U}(0, 1)$ and $\mathcal{M}_t$ denotes the set of masked positions.

The block diffusion paradigm divides the entire sequence into several blocks of equal size and performs block-wise noise addition and denoising within each block. 

$$\begin{equation}
\tag{3}
\mathcal{L}_{\text{BDM}}(x; \theta) = -\mathbb{E}_{t,x_0,x_t} \left[ \sum_{i \in \mathcal{M}_t} \alpha \,\log P_\theta(x_0^i \mid \mathcal{C}_i) \right],
\end{equation}$$

where $\mathcal{C}_i=(x_{<i},x_{D(i)})$, $x_{<i}$ are clean contexts from earlier blocks, $x_{D(i)}$ denotes all contexts in the block containing position $i$, and $\alpha=\alpha_t'/(1-\alpha_t)$ with $\alpha_t'$ the instantaneous rate of change of $\alpha_t$ in continuous time.

## 3.2 Modalities and Paradigms

**Figura 3.** **The diffusion finetuning framework of DiffusionVL.** The noisy sequence is concatenated with the original clean sequence along the sequence dimension. Each noisy block attends to all preceding clean blocks (inter-block causal) and all positions within its own block (intra-block bidirectional). The cross-entropy loss is computed only at masked positions.

Architectures and paradigms are largely decoupled: A single Transformer architecture (Vaswani et al., 2017) can yield diverse models by applying different paradigms. Based on this observation, our model retains the same network architecture as existing autoregressive models and is initialized from them; only the training/inference behavior and the attention pattern differ. Accordingly, we design different diffusion finetuning pipelines for AR-VLMs and AR-LMs.

First, we describe the process of finetuning an AR-VLM into a dVLM as a paradigm shift. We demonstrate that this direct paradigm shift is an efficient way to build dVLMs, which aligns with the findings of (Cheng et al., 2025; Wu et al., 2025a) in the text domain. Since our model has already been vision-language aligned, we directly finetune the entire AR-VLM into a dVLM in an end-to-end manner using Eq. (3).

Furthermore, we extend the approach to build a dVLM based on an AR-LM, and describe it as a process of modality shift and paradigm shift. We adopt the traditional two-stage training approach similar to LLaVA (Liu et al., 2023). In the pretraining stage, we only train the connector to align the vision embedding space and text embedding space. Since this connector is randomly initialized to align vision and text embeddings, we use the standard autoregressive objective Eq. (1) to stabilize the modality shift process. In the finetuning stage, all components are jointly trained end-to-end using the diffusion finetuning strategy Eq. (3) to achieve modality shift and paradigm shift at the same time.

## 3.3 Diffusion Finetuning

In this section, we elaborate on the training framework of our diffusion finetuning, as illustrated in Figure 3.

The image is processed through the vision encoder and projector, and the resulting vision embeddings are concatenated with text embeddings. Each sequence is padded with `<EOS>` tokens to a length divisible by the block size $D$ and split into $B=L/D$ non-overlapping blocks $\{\mathbf{x}_{(1)}, \ldots, \mathbf{x}_{(B)}\}$. Unlike the sequence-level noise in prior dVLMs (You et al., 2025; Li et al., 2025b; Yu et al., 2025), we adopt a block-wise noise schedule: For each block $b$ containing response or padding tokens, a noise level $t_b \sim \mathcal{U}(0,1)$ is sampled and each token $x^i$ within it is independently masked as 

$$\begin{equation}
\tag{4}
\tilde{x}^i=
\begin{cases}
\texttt{[MASK]}, & p=t_b, \\
x^i, & p=1-t_b,
\end{cases}
\end{equation}$$

where $p$ is the noise probability, while blocks containing image or prompt tokens remain unmasked ($t_b = 0$). The noised sequence $\tilde{\mathbf{x}}$ and clean sequence $\mathbf{x}$ are concatenated along the sequence dimension, with an attention mask $\mathbf{M}$ enforcing intra-block bidirectional and inter-block causal attention. For position $i$ in noised block $b$, $\mathbf{M}_{ij}$ indicates whether $i$ may attend to $j$ ($1$) or not ($0$): 

$$\begin{equation}
\tag{5}
    \mathbf{M}_{ij} = 
    \begin{cases}
        1, & \text{if } j \text{ is in the same noised block as } i, \\
        1, & \text{if } j \text{ is in a clean block } b' \text{ with } b' < b \text{,} \\
        0, & \text{otherwise}.
    \end{cases}
\end{equation}$$

The model is trained to minimize the cross-entropy loss at masked positions using Eq. (3).

## 3.4 Diffusion Inference

<div class="algorithm">

**Algoritmo 1.** Diffusion Inference of DiffusionVL.

<span id="alg:inference" data-label="alg:inference"></span> $\mathbf{C}_0\gets \texttt{cat}\bigl(\texttt{proj}(\texttt{ve}(\mathbf{I})),\texttt{embed}(\mathbf{T})\bigr);\;\mathbf{Y}\gets\varnothing$

</div>

During inference, our model combines inter-block autoregressive generation with intra-block parallel diffusion decoding, naturally supporting KV-cache reuse and arbitrary-length generation. The full procedure is summarized in Algorithm 1.

Given an input image and a text prompt, we first encode them using the vision encoder, projector, and text embedding layer, and concatenate the resulting embeddings to initialize the context cache $\mathbf{C}_0$. For each subsequent block $m$, we initialize all $D$ positions as mask tokens and perform up to $S$ iterative denoising steps.

At each step $s$, let $\mathcal{M}^{(s-1)}$ denote the set of indices still masked in $\mathbf{x}_m^{(s-1)}$. For each $i \in \mathcal{M}^{(s-1)}$, we sample the prediction conditioned on the current block state and preceding context $\mathbf{C}_{m-1}$ via KV-cache reuse (following the hybrid attention pattern in Eq. (5)), and record the confidence $c_{m, i}^{(s)}$. A subset $\mathcal{U}^{(s)}$ is then chosen and unmasked based on $c_{m,i}^{(s)}$.

We adopt two remasking strategies following (Cheng et al., 2025) to determine $\mathcal{U}^{(s)}$: the static low-confidence remasking strategy unmasks the $k$ highest-confidence positions per step; the dynamic low-confidence remasking strategy further augments this set by additionally unmasking all positions whose confidence exceeds a threshold $\tau$, enabling faster decoding on simpler content. Once a block is fully denoised, it is appended to the context cache, and generation proceeds to block $m+1$ until an $\langle\text{EOS}\rangle$ token is produced.

# 4 Experiment

**Tabla 1.** **Benchmark Performance Comparison (Part 1)**. The top-2 results are highlighted separately for AR and Diffusion models. Best results are in **bold**, second-best are <u>underlined</u>.

<table id="tab:table1_part1_colored">
<thead>
<tr>
<th style="text-align: left;"><strong>Model</strong></th>
<th style="text-align: center;"><strong>Size</strong></th>
<th style="text-align: center;"><strong>Type</strong></th>
<th style="text-align: center;"><strong>Samples</strong></th>
<th style="text-align: center;"><strong>MMBench <span>[en-dev]</span></strong></th>
<th style="text-align: center;"><strong>MMMU <span>[val]</span></strong></th>
<th style="text-align: center;"><strong>MMMU-Pro <span>[std.]</span></strong></th>
<th style="text-align: center;"><strong>MMMU-Pro <span>[vision]</span></strong></th>
<th style="text-align: center;"><strong>MMStar <span>[test]</span></strong></th>
<th style="text-align: center;"><strong>MME <span>[cog.]</span></strong></th>
<th style="text-align: center;"><strong>MME <span>[perp.]</span></strong></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="11" style="text-align: center;"><strong><em>AutoRegressive Vision Language Models</em></strong></td>
</tr>
<tr>
<td style="text-align: left;"><strong>LLaVA</strong></td>
<td style="text-align: center;">7B</td>
<td style="text-align: center;">AR</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">38.7</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">809</td>
</tr>
<tr>
<td style="text-align: left;"><strong>LLaVA-1.5</strong></td>
<td style="text-align: center;">7B</td>
<td style="text-align: center;">AR</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">64.3</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">1510</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Cambrian-1</strong></td>
<td style="text-align: center;">8B</td>
<td style="text-align: center;">AR</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">75.9</td>
<td style="text-align: center;">42.7</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">1547</td>
</tr>
<tr>
<td style="text-align: left;"><strong>LLaVA-OV</strong></td>
<td style="text-align: center;">7B</td>
<td style="text-align: center;">AR</td>
<td style="text-align: center;">7.8M</td>
<td style="text-align: center;"><u><span style="color: arcolor">80.8</span></u></td>
<td style="text-align: center;"><u><span style="color: arcolor">48.8</span></u></td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;"><u><span style="color: arcolor">61.7</span></u></td>
<td style="text-align: center;">418</td>
<td style="text-align: center;"><u><span style="color: arcolor">1580</span></u></td>
</tr>
<tr>
<td style="text-align: left;"><strong>Qwen2.5VL</strong></td>
<td style="text-align: center;">3B</td>
<td style="text-align: center;">AR</td>
<td style="text-align: center;">&gt;9M</td>
<td style="text-align: center;">79.1</td>
<td style="text-align: center;">46.8</td>
<td style="text-align: center;"><u><span style="color: arcolor">31.2</span></u></td>
<td style="text-align: center;"><u><span style="color: arcolor">22.1</span></u></td>
<td style="text-align: center;">55.9</td>
<td style="text-align: center;"><u><span style="color: arcolor">620</span></u></td>
<td style="text-align: center;">1533</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;">7B</td>
<td style="text-align: center;">AR</td>
<td style="text-align: center;">&gt;9M</td>
<td style="text-align: center;"><strong><span style="color: arcolor">83.5</span></strong></td>
<td style="text-align: center;"><strong><span style="color: arcolor">51.1</span></strong></td>
<td style="text-align: center;"><strong><span style="color: arcolor">36.7</span></strong></td>
<td style="text-align: center;"><strong><span style="color: arcolor">33.4</span></strong></td>
<td style="text-align: center;"><strong><span style="color: arcolor">63.9</span></strong></td>
<td style="text-align: center;"><strong><span style="color: arcolor">646</span></strong></td>
<td style="text-align: center;"><strong><span style="color: arcolor">1680</span></strong></td>
</tr>
<tr>
<td colspan="11" style="text-align: center;"><strong><em>Diffusion Vision Language Models</em></strong></td>
</tr>
<tr>
<td style="text-align: left;"><strong>LaViDa-L</strong></td>
<td style="text-align: center;">8B</td>
<td style="text-align: center;">Diff.</td>
<td style="text-align: center;">1.6M</td>
<td style="text-align: center;">70.5</td>
<td style="text-align: center;">43.3</td>
<td style="text-align: center;">28.6</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">341</td>
<td style="text-align: center;">1365</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Dimple</strong></td>
<td style="text-align: center;">7B</td>
<td style="text-align: center;">Diff.</td>
<td style="text-align: center;">1.3M</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">45.2</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">432</td>
<td style="text-align: center;">1514</td>
</tr>
<tr>
<td style="text-align: left;"><strong>LLaDA-V</strong></td>
<td style="text-align: center;">8B</td>
<td style="text-align: center;">Diff.</td>
<td style="text-align: center;">16.5M</td>
<td style="text-align: center;"><u><span style="color: diffcolor">82.9</span></u></td>
<td style="text-align: center;">48.6</td>
<td style="text-align: center;"><u><span style="color: diffcolor">35.2</span></u></td>
<td style="text-align: center;">18.6</td>
<td style="text-align: center;"><u><span style="color: diffcolor">60.1</span></u></td>
<td style="text-align: center;">491</td>
<td style="text-align: center;">1507</td>
</tr>
<tr>
<td style="text-align: left;"><strong>DiffusionVL</strong></td>
<td style="text-align: center;">3B</td>
<td style="text-align: center;">Diff.</td>
<td style="text-align: center;">738K</td>
<td style="text-align: center;">80.1</td>
<td style="text-align: center;">47.2</td>
<td style="text-align: center;">31.2</td>
<td style="text-align: center;"><u><span style="color: diffcolor">20.2</span></u></td>
<td style="text-align: center;">55.9</td>
<td style="text-align: center;"><u><span style="color: diffcolor">594</span></u></td>
<td style="text-align: center;"><strong><span style="color: diffcolor">1539</span></strong></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;">7B</td>
<td style="text-align: center;">Diff.</td>
<td style="text-align: center;">738K</td>
<td style="text-align: center;"><strong><span style="color: diffcolor">83.5</span></strong></td>
<td style="text-align: center;"><strong><span style="color: diffcolor">49.3</span></strong></td>
<td style="text-align: center;"><strong><span style="color: diffcolor">36.9</span></strong></td>
<td style="text-align: center;"><strong><span style="color: diffcolor">25.0</span></strong></td>
<td style="text-align: center;"><strong><span style="color: diffcolor">63.2</span></strong></td>
<td style="text-align: center;"><strong><span style="color: diffcolor">675</span></strong></td>
<td style="text-align: center;"><u><span style="color: diffcolor">1519</span></u></td>
</tr>
</tbody>
</table>

**Tabla 2.** **Benchmark Performance Comparison (Part 2)**. DiffusionVL-7B achieves top-tier performance among Diffusion models and closes the gap with top AR models. Best in **bold**, second-best <u>underlined</u>.

<table id="tab:table1_part2_colored">
<thead>
<tr>
<th style="text-align: left;"><strong>Model</strong></th>
<th style="text-align: center;"><strong>Size</strong></th>
<th style="text-align: center;"><strong>Type</strong></th>
<th style="text-align: center;"><strong>Samples</strong></th>
<th style="text-align: center;"><strong>SeedBench <span>[img]</span></strong></th>
<th style="text-align: center;"><strong>SeedBench <span>[vid]</span></strong></th>
<th style="text-align: center;"><strong>AI2D</strong></th>
<th style="text-align: center;"><strong>ChartQA</strong></th>
<th style="text-align: center;"><strong>Realworld QA</strong></th>
<th style="text-align: center;"><strong>MuirBench</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="10" style="text-align: center;"><strong><em>AutoRegressive Vision Language Models</em></strong></td>
</tr>
<tr>
<td style="text-align: left;"><strong>LLaVA</strong></td>
<td style="text-align: center;">7B</td>
<td style="text-align: center;">AR</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">37.0</td>
<td style="text-align: center;">23.8</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
</tr>
<tr>
<td style="text-align: left;"><strong>LLaVA-1.5</strong></td>
<td style="text-align: center;">7B</td>
<td style="text-align: center;">AR</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">66.1</td>
<td style="text-align: center;">37.3</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Cambrian-1</strong></td>
<td style="text-align: center;">8B</td>
<td style="text-align: center;">AR</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">74.7</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">73.0</td>
<td style="text-align: center;">73.3</td>
<td style="text-align: center;">64.2</td>
<td style="text-align: center;">-</td>
</tr>
<tr>
<td style="text-align: left;"><strong>LLaVA-OV</strong></td>
<td style="text-align: center;">7B</td>
<td style="text-align: center;">AR</td>
<td style="text-align: center;">7.8M</td>
<td style="text-align: center;"><u><span style="color: arcolor">75.4</span></u></td>
<td style="text-align: center;"><u><span style="color: arcolor">56.9</span></u></td>
<td style="text-align: center;">81.4</td>
<td style="text-align: center;">80.0</td>
<td style="text-align: center;"><u><span style="color: arcolor">66.3</span></u></td>
<td style="text-align: center;">41.8</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Qwen2.5VL</strong></td>
<td style="text-align: center;">3B</td>
<td style="text-align: center;">AR</td>
<td style="text-align: center;">&gt;9M</td>
<td style="text-align: center;">74.8</td>
<td style="text-align: center;">55.1</td>
<td style="text-align: center;"><u><span style="color: arcolor">81.6</span></u></td>
<td style="text-align: center;"><u><span style="color: arcolor">84.0</span></u></td>
<td style="text-align: center;">65.4</td>
<td style="text-align: center;"><u><span style="color: arcolor">47.7</span></u></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;">7B</td>
<td style="text-align: center;">AR</td>
<td style="text-align: center;">&gt;9M</td>
<td style="text-align: center;"><strong><span style="color: arcolor">77.5</span></strong></td>
<td style="text-align: center;"><strong><span style="color: arcolor">61.3</span></strong></td>
<td style="text-align: center;"><strong><span style="color: arcolor">83.9</span></strong></td>
<td style="text-align: center;"><strong><span style="color: arcolor">87.3</span></strong></td>
<td style="text-align: center;"><strong><span style="color: arcolor">68.5</span></strong></td>
<td style="text-align: center;"><strong><span style="color: arcolor">59.6</span></strong></td>
</tr>
<tr>
<td colspan="10" style="text-align: center;"><strong><em>Diffusion Vision Language Models</em></strong></td>
</tr>
<tr>
<td style="text-align: left;"><strong>LaViDa-L</strong></td>
<td style="text-align: center;">8B</td>
<td style="text-align: center;">Diff.</td>
<td style="text-align: center;">1.6M</td>
<td style="text-align: center;">66.5</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">70.0</td>
<td style="text-align: center;">64.6</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Dimple</strong></td>
<td style="text-align: center;">7B</td>
<td style="text-align: center;">Diff.</td>
<td style="text-align: center;">1.3M</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">74.4</td>
<td style="text-align: center;">63.4</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
</tr>
<tr>
<td style="text-align: left;"><strong>LLaDA-V</strong></td>
<td style="text-align: center;">8B</td>
<td style="text-align: center;">Diff.</td>
<td style="text-align: center;">16.5M</td>
<td style="text-align: center;"><u><span style="color: diffcolor">74.8</span></u></td>
<td style="text-align: center;"><u><span style="color: diffcolor">53.7</span></u></td>
<td style="text-align: center;">77.8</td>
<td style="text-align: center;">78.3</td>
<td style="text-align: center;"><u><span style="color: diffcolor">63.2</span></u></td>
<td style="text-align: center;"><strong><span style="color: diffcolor">48.3</span></strong></td>
</tr>
<tr>
<td style="text-align: left;"><strong>DiffusionVL</strong></td>
<td style="text-align: center;">3B</td>
<td style="text-align: center;">Diff.</td>
<td style="text-align: center;">738K</td>
<td style="text-align: center;">73.9</td>
<td style="text-align: center;">52.2</td>
<td style="text-align: center;"><u><span style="color: diffcolor">78.4</span></u></td>
<td style="text-align: center;"><u><span style="color: diffcolor">79.9</span></u></td>
<td style="text-align: center;">61.6</td>
<td style="text-align: center;"><u><span style="color: diffcolor">47.2</span></u></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;">7B</td>
<td style="text-align: center;">Diff.</td>
<td style="text-align: center;">738K</td>
<td style="text-align: center;"><strong><span style="color: diffcolor">75.5</span></strong></td>
<td style="text-align: center;"><strong><span style="color: diffcolor">54.4</span></strong></td>
<td style="text-align: center;"><strong><span style="color: diffcolor">82.2</span></strong></td>
<td style="text-align: center;"><strong><span style="color: diffcolor">84.2</span></strong></td>
<td style="text-align: center;"><strong><span style="color: diffcolor">68.0</span></strong></td>
<td style="text-align: center;">44.8</td>
</tr>
</tbody>
</table>

**Tabla 3.** **A comparison of dVLM construction using different models and paradigms.** We compare dVLMs converted from AR-LMs versus those from dLLMs. The Base LLM column indicates MMLU scores of the base language models. The best results across all models are highlighted in **bold**, and the second-best are <u>underlined</u>. The AR, Block-Diff, and Full-Diff paradigms represent the different paradigms we discussed in Section 3.1.

<table id="tab: comparison of dVLM">
<thead>
<tr>
<th style="text-align: left;"><strong>Model</strong></th>
<th style="text-align: center;"><strong>Paradigm</strong></th>
<th style="text-align: center;"><strong>Base LLM (MMLU)</strong></th>
<th style="text-align: center;"><strong>MMMU (val)</strong></th>
<th style="text-align: center;"><strong>MMMU-Pro (std.)</strong></th>
<th style="text-align: center;"><strong>ChartQA</strong></th>
<th style="text-align: center;"><strong>RealWorld QA</strong></th>
<th style="text-align: center;"><strong>MME (cog.)</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="8" style="text-align: center;"><strong><em>Conversion from LLaDA-8B (dLLM base)</em></strong></td>
</tr>
<tr>
<td style="text-align: left;"><strong>LLaDA-V</strong></td>
<td style="text-align: center;">Full-Diff</td>
<td style="text-align: center;">65.9</td>
<td style="text-align: center;">42.4</td>
<td style="text-align: center;">26.0</td>
<td style="text-align: center;">20.2</td>
<td style="text-align: center;">60.1</td>
<td style="text-align: center;">342</td>
</tr>
<tr>
<td style="text-align: left;"><strong>LLaDA-V</strong></td>
<td style="text-align: center;">Block-Diff.</td>
<td style="text-align: center;"></td>
<td style="text-align: center;">32.6</td>
<td style="text-align: center;">17.1</td>
<td style="text-align: center;">29.8</td>
<td style="text-align: center;">44.2</td>
<td style="text-align: center;">261</td>
</tr>
<tr>
<td colspan="8" style="text-align: center;"><strong><em>Conversion from Qwen2.5-7B (AR-LM base)</em></strong></td>
</tr>
<tr>
<td style="text-align: left;"><strong>LLaVA</strong></td>
<td style="text-align: center;">AR</td>
<td style="text-align: center;">71.9</td>
<td style="text-align: center;"><strong>45.4</strong></td>
<td style="text-align: center;"><u>28.1</u></td>
<td style="text-align: center;"><u>52.8</u></td>
<td style="text-align: center;"><u>60.4</u></td>
<td style="text-align: center;"><u>356</u></td>
</tr>
<tr>
<td style="text-align: left;"><strong>DiffusionVL</strong></td>
<td style="text-align: center;">Block-Diff.</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"><u>43.7</u></td>
<td style="text-align: center;"><strong>28.4</strong></td>
<td style="text-align: center;"><strong>53.6</strong></td>
<td style="text-align: center;"><strong>60.7</strong></td>
<td style="text-align: center;"><strong>371</strong></td>
</tr>
</tbody>
</table>

## 4.1 Implementation Details

**Model architecture.** For building dVLMs from AR-VLMs, we use Qwen2.5-VL-3B-Instruct and Qwen2.5-VL-7B-Instruct (Bai et al., 2025) as base models. For finetuning from AR-LMs and dLLMs, we select Qwen2.5-7B-Instruct (Qwen et al., 2025) and LLaDA-8B-Instruct (Nie et al., 2025), respectively. The vision encoder is SigLip2-400M (Tschannen et al., 2025). The projector is a randomly initialized two-layer MLP.

**Training data.** For building dVLMs from AR-VLMs, we use only the 738K instruction-following samples from LLaVA-Next (Li et al., 2024) for end-to-end finetuning. For building dVLMs and AR-VLMs from AR-LMs and for building dVLMs from dLLMs, we adopt a two-stage data setup: the 580K-sample pretraining dataset from LLaVA-Pretrain (Liu et al., 2023) in the pretraining stage, and the 738K instruction-following samples from LLaVA-Next in the finetuning stage.

**Hyperparameters.** We use AdamW with a cosine decay scheduler. The pretraining stage uses a learning rate of $1\times10^{-3}$; the finetuning stage uses $1\times10^{-5}$ for the LLM and projector and $2\times10^{-6}$ for the vision encoder. The default training block size is 8. During inference, we adopt static low-confidence remasking with the number of denoising steps equal to the block size.

**Evaluation benchmarks.** We evaluate on a diverse set of vision-language benchmarks using the LMMS-Eval (Zhang et al., 2024a) library with its default prompts:

- *General knowledge:* MMMU (Yue et al., 2024a), MMMU-Pro (Yue et al., 2024b), MMStar (Chen et al., 2024a), MME (Fu et al., 2025), SeedBench (Li et al., 2023), MMBench (Liu et al., 2024), RealworldQA (xAI, 2024).

- *Chart and document understanding:* AI2D (Kembhavi et al., 2016), ChartQA (Masry et al., 2022).

- *Mathematical reasoning:* MathVista (Lu et al., 2023), MathVerse (Zhang et al., 2024b), MathVision (Ahmad et al., 2024).

- *Detail image captioning:* DetailCaps (Dong et al., 2024).

- *Multi-image understanding:* MuirBench (Wang et al., 2024).

## 4.2 Efficient dVLM Construction from AR-VLMs

Table 1 and Table 2 present the downstream multimodal benchmark results of DiffusionVL-3B and DiffusionVL-7B, which are derived from diffusion finetuning of Qwen2.5VL-3B-Instruct and Qwen2.5VL-7B-Instruct. We compare our models’ results with representative AR-VLMs and open-source dVLMs.

As shown in Table 1 and Table 2, DiffusionVL-7B achieves comprehensive performance that surpasses existing open-source dVLMs LaViDa-L (Li et al., 2025b), Dimple (Yu et al., 2025), and LLaDA-V (You et al., 2025) on multimodal benchmarks. This is achieved despite finetuning with only 738K samples, i.e., less than 5% of the data used for LLaDA-V, demonstrating strong vision-language capability and high training efficiency. Furthermore, leveraging the strong pretrained foundation of AR-VLMs, DiffusionVL-3B even outperforms the larger LaViDa-L-8B and Dimple-7B with less training data, further validating the effectiveness of our proposed method. Moreover, our DiffusionVL significantly narrows the gap between existing dVLMs and advanced AR-VLMs, demonstrating an efficient approach to building dVLMs.

## 4.3 Feasible dVLM Conversion from AR-LMs

In this section, we conduct multiple groups of experiments on different base language models and various finetuning paradigms to investigate the feasibility of finetuning dVLMs from AR-LMs. We perform autoregressive finetuning, block diffusion finetuning, and full diffusion finetuning based on the AR-LM Qwen2.5-7B-Instruct and the dLLM LLaDA-8B-Instruct, respectively, and report the differences in their performance on the downstream multimodal benchmarks under the same training settings and data.

As shown in Table 3, DiffusionVL significantly outperforms LLaDA-V (finetuned from LLaDA using block diffusion and full diffusion paradigms). This indicates that building a high-performance dVLM does not require a pre-existing dLLM; we can fully leverage existing powerful AR-LMs to develop such dVLMs. Notably, DiffusionVL exhibits negligible differences from LLaVA on downstream benchmarks. This further confirms that constructing dVLMs from AR-LMs is both feasible and effective.

It is worth noting that the performance difference between the DiffusionVL here and DiffusionVL in Section 4.2 does not imply that finetuning from AR-VLMs to dVLMs is the best approach. The DiffusionVL in Section 4.2 benefits more from the fact that its base model has already undergone extensive, high-quality vision-language alignment training. We believe that AR-LMs, with longer and higher-quality visual finetuning, also have the potential to build dVLMs achieving performance comparable to dVLMs built from AR-VLMs on downstream benchmarks.

## 4.4 Inference Speed and Quality

**Figura 4.** **Speed and quality trade-offs on detailed captioning and math reasoning.** We define the parallelism factor for dVLMs as the average number of tokens generated simultaneously throughout the sequence (e.g., $1\times$ corresponds to single-token sampling). Speed metrics were collected on 8 GPUs and reported as the average per device.

**Tabla 4.** **Comprehensive comparison on math reasoning benchmarks.** We report both accuracy (Acc.) and inference throughput (TPS). Best results are in **bold** and second-best are <u>underlined</u>.

<table id="tab:comprehensive_comparison">
<thead>
<tr>
<th style="text-align: left;"><strong>Model</strong></th>
<th colspan="2" style="text-align: center;"><strong>MathVista</strong></th>
<th colspan="2" style="text-align: center;"><strong>MathVerse</strong></th>
<th colspan="2" style="text-align: center;"><strong>MathVision</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><span>2-3</span> (lr)<span>4-5</span> (lr)<span>6-7</span></td>
<td style="text-align: center;">Acc.<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo>↑</mo><annotation encoding="application/x-tex">\uparrow</annotation></semantics></math></td>
<td style="text-align: center;">TPS<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo>↑</mo><annotation encoding="application/x-tex">\uparrow</annotation></semantics></math></td>
<td style="text-align: center;">Acc.<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo>↑</mo><annotation encoding="application/x-tex">\uparrow</annotation></semantics></math></td>
<td style="text-align: center;">TPS<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo>↑</mo><annotation encoding="application/x-tex">\uparrow</annotation></semantics></math></td>
<td style="text-align: center;">Acc.<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo>↑</mo><annotation encoding="application/x-tex">\uparrow</annotation></semantics></math></td>
<td style="text-align: center;">TPS<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo>↑</mo><annotation encoding="application/x-tex">\uparrow</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">DiffusionVL-7B</td>
<td style="text-align: center;"><strong>64.30</strong></td>
<td style="text-align: center;">5.21</td>
<td style="text-align: center;"><strong>36.29</strong></td>
<td style="text-align: center;">14.38</td>
<td style="text-align: center;"><strong>19.08</strong></td>
<td style="text-align: center;">17.74</td>
</tr>
<tr>
<td style="text-align: left;">LLaDA-V-8B</td>
<td style="text-align: center;">55.10</td>
<td style="text-align: center;"><u>23.07</u></td>
<td style="text-align: center;">31.35</td>
<td style="text-align: center;"><u>23.18</u></td>
<td style="text-align: center;">17.11</td>
<td style="text-align: center;"><u>22.75</u></td>
</tr>
<tr>
<td style="text-align: left;">DiffusionVL-CoT-7B</td>
<td style="text-align: center;"><u>63.70</u></td>
<td style="text-align: center;"><strong>39.47</strong></td>
<td style="text-align: center;"><u>32.74</u></td>
<td style="text-align: center;"><strong>42.31</strong></td>
<td style="text-align: center;"><u>18.75</u></td>
<td style="text-align: center;"><strong>39.26</strong></td>
</tr>
</tbody>
</table>

To evaluate inference speed and its trade-off with quality, we evaluate DiffusionVL on two representative tasks: detailed image captioning on DetailCaps (Dong et al., 2024) and mathematical reasoning on multiple benchmarks, including MathVista (Lu et al., 2023), MathVision (Ahmad et al., 2024), and MathVerse (Zhang et al., 2024b). During inference, we employ the static low-confidence remasking strategy to control the number of decoded tokens at each denoising step. For both tasks, we compare against LLaDA-V-8B (You et al., 2025) under its recommended fast-dLLM caching (Wu et al., 2025b) with recomputation every 32 steps. Figure 4 presents the speed-quality trade-offs on DetailCaps and MathVista, while Table 4 summarizes the full results across all math benchmarks.

We first examine image captioning on DetailCaps, limiting responses to 512 tokens and evaluating against ground-truth captions using BERTScore (Zhang\* et al., 2020). As shown in Figure 4(a), our DiffusionVL-7B achieves a BERTScore $2.02\times$ higher than LLaDA-V-8B while enjoying $2.0\times$ faster inference under the same parallelism. Moreover, we observe a clear test-time compute scaling law: increasing the number of denoising steps improves descriptive performance at the cost of inference speed, highlighting a tunable trade-off for DiffusionVL.

Turning to mathematical reasoning, we encounter a different challenge. While our DiffusionVL achieves competitive accuracy, its TPS lags behind LLaDA-V. This limitation, also observed in LaViDA (Li et al., 2025b), stems from the absence of chain-of-thought (CoT) data in our training corpus. To address this, we randomly sample 100K CoT examples from OpenMMReasoner-SFT (Zhang et al., 2025) and finetune the model with the original dataset, yielding DiffusionVL-CoT. For LLaDA-V, we enable fast-dLLM caching with a generation length of 64 (longer lengths degrade accuracy), while DiffusionVL-CoT uses a maximum length of 1024 to accommodate longer reasoning chains. As Figure 4(b) illustrates, DiffusionVL-CoT-7B outperforms LLaDA-V-8B by $8.6$ points on MathVista while achieving $1.7\times$ faster inference.  Table 4 further confirms these gains across all three math benchmarks.

## 4.5 Ablation Study

**Different block size performance.** To further explore the impact of different block sizes on the performance of diffusion finetuning, we conduct four groups of diffusion finetuning ablation experiments based on the Qwen2.5VL-3B-Instruct, setting the training block size from 1 to 16. During inference, we set the number of denoising steps equal to the block size to ensure models with different block sizes use the same number of denoising steps when generating the same tokens.

As shown in Table 5, smaller block sizes yield marginally better accuracy, while larger ones offer greater parallel potential and less computational overhead, leading to substantially higher throughput ($1.55\times$ from block size 1 to 16). Block size 8 achieves the best DetailCaps BERTScore and offers a good speed–quality balance, so we adopt it as our default configuration.

**Different noise performance.** To validate the effectiveness of our block-level noise scheduling strategy, we compare it against the conventional sequence-level noise used in prior dVLMs. In sequence-level noise, masking is applied uniformly across the entire sequence. In contrast, our block-level noise applies a consistent noise ratio to each block independently. This design better aligns with the block decoding process during inference. As shown in Table 6, block-level noise generally outperforms sequence-level noise across multiple benchmarks.

**Different diffusion paradigms.** To determine the optimal diffusion paradigm for VLMs, we perform block diffusion finetuning and full diffusion finetuning from the same Qwen2.5VL-7B-Instruct. Table 6 demonstrates that block diffusion yields superior performance on multimodal benchmarks. Furthermore, Table 4 shows that DiffusionVL-CoT (block diffusion) consistently achieves higher throughput than LLaDA-V (full diffusion), highlighting the inference efficiency advantage of the block diffusion paradigm. These results indicate that block diffusion strikes a superior balance between quality and efficiency, making it the preferred paradigm for building dVLMs from AR models.

**Different thresholds for dynamic remasking.** To explore more extreme acceleration, we adopt the dynamic low-confidence remasking strategy and conduct an ablation study on the relationship between the BERTScore / tokens per second (TPS) on DetailCaps and different thresholds under the condition of a fixed denoising step of 8. The choice of setting the denoising step to 8 is intended to minimize the number of tokens denoised in each fixed-schedule denoising step, which allows for a more intuitive demonstration of the differences between the dynamic remasking strategy and the static remasking strategy.

Table 7 shows that smaller dynamic thresholds allow the model to decode all tokens meeting the threshold condition at each denoising step, thereby achieving more significant acceleration. However, such acceleration comes at the cost of a certain degree of performance degradation for the model.

**Tabla 5.** **Ablation on training block size.** Effect of training block size on benchmark scores and throughput (TPS). The TPS throughput row is shaded in light gray. $^\dagger$  denotes our default block size. Best per row in **bold**. TPS measured per GPU on DetailCaps.

<table id="tab:ablation1_block">
<thead>
<tr>
<th colspan="5" style="text-align: center;"><em>Training block size</em></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><strong>Benchmark</strong></td>
<td style="text-align: center;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>b</mi><mspace width="-0.167em"></mspace><mo>=</mo><mspace width="-0.167em"></mspace><mn>1</mn></mrow><annotation encoding="application/x-tex">b\!=\!1</annotation></semantics></math></td>
<td style="text-align: center;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>b</mi><mspace width="-0.167em"></mspace><mo>=</mo><mspace width="-0.167em"></mspace><mn>4</mn></mrow><annotation encoding="application/x-tex">b\!=\!4</annotation></semantics></math></td>
<td style="text-align: center;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>b</mi><mspace width="-0.167em"></mspace><mo>=</mo><mspace width="-0.167em"></mspace><msup><mn>8</mn><mo>†</mo></msup></mrow><annotation encoding="application/x-tex">b\!=\!8^\dagger</annotation></semantics></math></td>
<td style="text-align: center;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>b</mi><mspace width="-0.167em"></mspace><mo>=</mo><mspace width="-0.167em"></mspace><mn>16</mn></mrow><annotation encoding="application/x-tex">b\!=\!16</annotation></semantics></math></td>
</tr>
<tr>
<td style="text-align: left;">ChartQA</td>
<td style="text-align: center;"><strong>82.8</strong></td>
<td style="text-align: center;">80.9</td>
<td style="text-align: center;">79.9</td>
<td style="text-align: center;">77.4</td>
</tr>
<tr>
<td style="text-align: left;">MMMU-Pro (std.)</td>
<td style="text-align: center;"><strong>32.1</strong></td>
<td style="text-align: center;">31.4</td>
<td style="text-align: center;">31.2</td>
<td style="text-align: center;">31.0</td>
</tr>
<tr>
<td style="text-align: left;">MMMU-Pro (vis.)</td>
<td style="text-align: center;"><strong>22.2</strong></td>
<td style="text-align: center;">19.9</td>
<td style="text-align: center;">20.2</td>
<td style="text-align: center;">18.1</td>
</tr>
<tr>
<td style="text-align: left;">MMStar</td>
<td style="text-align: center;"><strong>57.8</strong></td>
<td style="text-align: center;">56.5</td>
<td style="text-align: center;">55.9</td>
<td style="text-align: center;">55.2</td>
</tr>
<tr>
<td style="text-align: left;">DetailCaps</td>
<td style="text-align: center;">29.8</td>
<td style="text-align: center;">29.9</td>
<td style="text-align: center;"><strong>30.9</strong></td>
<td style="text-align: center;">29.9</td>
</tr>
<tr>
<td style="text-align: left;">TPS (tok/s)</td>
<td style="text-align: center;">26.9</td>
<td style="text-align: center;">38.8</td>
<td style="text-align: center;">39.0</td>
<td style="text-align: center;"><strong>41.6</strong></td>
</tr>
</tbody>
</table>

**Tabla 6.** **Ablation on noise strategy and diffusion paradigm** (Qwen2.5VL-7B-Instruct). *Block$^\dagger$* is the default block-wise setting; *Sequence* and *Full* are sequence-level noise and full diffusion, respectively. $^\dagger$  denotes our default. Best in each pairwise comparison against Block is in **bold**.

| **Metric**      | **Sequence** | **Full** | **Block$^\dagger$** |
|:----------------|:------------:|:--------:|:-------------------:|
| MMMU-Pro (std.) |    36.30     |  36.47   |      **36.94**      |
| MMMU-Pro (vis.) |    24.86     |  19.88   |      **24.97**      |
| MMStar          |    61.86     |  62.08   |      **63.18**      |
| ChartQA         |    83.88     |  52.00   |      **84.20**      |
| MME (cog.)      |    670.4     |  660.7   |      **675.4**      |
| MME (perp.)     |   **1527**   | **1607** |        1519         |

**Tabla 7.** **Ablation on dynamic remasking.** Dynamic low-confidence remasking at varying thresholds with fixed 8 denoising steps. $^\dagger$  denotes our default. Best per group in bold. TPS measured per GPU on DetailCaps.

<table id="tab:ablation1_remask">
<thead>
<tr>
<th colspan="6" style="text-align: center;"><em>Dynamic remasking</em></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><strong><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>τ</mi><annotation encoding="application/x-tex">\tau</annotation></semantics></math></strong></td>
<td style="text-align: center;">1.0<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msup><mi></mi><mo>†</mo></msup><annotation encoding="application/x-tex">^\dagger</annotation></semantics></math></td>
<td style="text-align: center;">0.8</td>
<td style="text-align: center;">0.6</td>
<td style="text-align: center;">0.4</td>
<td style="text-align: center;">0.2</td>
</tr>
<tr>
<td style="text-align: left;"><strong>TPS</strong><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo>↑</mo><annotation encoding="application/x-tex">\uparrow</annotation></semantics></math></td>
<td style="text-align: center;">31.5</td>
<td style="text-align: center;">34.6</td>
<td style="text-align: center;">37.8</td>
<td style="text-align: center;">46.6</td>
<td style="text-align: center;"><strong>71.8</strong></td>
</tr>
<tr>
<td style="text-align: left;"><strong>BERT</strong><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo>↑</mo><annotation encoding="application/x-tex">\uparrow</annotation></semantics></math></td>
<td style="text-align: center;"><strong>30.2</strong></td>
<td style="text-align: center;">30.1</td>
<td style="text-align: center;">29.7</td>
<td style="text-align: center;">27.4</td>
<td style="text-align: center;">18.6</td>
</tr>
</tbody>
</table>

# 5 Conclusion and Future Work

**Conclusion.** This paper focuses on a novel problem in the building of dVLMs: Is it possible to construct dVLMs based on existing powerful autoregressive models? In response, we propose DiffusionVL, a dVLM family that translates from any powerful autoregressive models. We obtain two key observations: (1) The paradigm shift from AR-VLMs to dVLMs is remarkably effective. (2) Direct conversion of an AR-LM to a dVLM is also feasible. Furthermore, we introduce a block decoding strategy into dVLMs that supports arbitrary-length generation and KV-cache reuse. With this integrated design, despite training with less than 5% of the data required by prior methods, DiffusionVL translated from AR-VLMs achieves state-of-the-art performance among existing dVLMs, alongside a $2.0\times$ inference speedup. DiffusionVL translated from AR-LMs not only outperforms the dVLMs built from dLLMs but also achieves performance competitive with AR-VLMs finetuned under the same autoregressive paradigm.

**Future Work.** During our research on DiffusionVL, we observe that the AR-to-diffusion paradigm conversion is remarkably straightforward. We plan to explore whether this conversion paradigm can extend beyond finetuning to other training stages, including pretraining and reinforcement learning. Moreover, our inference experiments reveal that the diffusion paradigm yields significant efficiency gains in long chain-of-thought (CoT) scenarios, which motivates future work on the throughput benefits that DiffusionVL could bring to reinforcement learning.

# References

Ahmad et al. (2024). Muhammad Awais Ahmad, Tauqir Ahmed, Muhammad Aslam, Amjad Rehman, Faten S Alamri, Saeed Ali Bahaj, and Tanzila Saba. Mathvision: An accessible intelligent agent for visually impaired people to understand mathematical equations. IEEE Access, 13:0 6155–6165, 2024.

Arriola et al. (2025). Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. arXiv preprint arXiv:2503.09573, 2025.

Austin et al. (2021). Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:0 17981–17993, 2021.

Bai et al. (2025). Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Bie et al. (2025). Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, Chengxi Li, Chongxuan Li, Jianguo Li, Zehuan Li, Huabin Liu, Lin Liu, Guoshan Lu, Xiaocheng Lu, Yuxin Ma, Jianfeng Tan, Lanning Wei, Ji-Rong Wen, Yipeng Xing, Xiaolu Zhang, Junbo Zhao, Da Zheng, Jun Zhou, Junlin Zhou, Zhanchao Zhou, Liwang Zhu, and Yihong Zhuang. Llada2.0: Scaling up diffusion language models to 100b, 2025.

Chen et al. (2024a). Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:0 27056–27087, 2024a.

Chen et al. (2021). Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021.

Chen et al. (2024b). Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024b.

Cheng et al. (2025). Shuang Cheng, Yihan Bian, Dawei Liu, Yuhua Jiang, Yihao Liu, Linfeng Zhang, Wenghai Wang, Qipeng Guo, Kai Chen, Biqing Qi\*, and Bowen Zhou. Sdar: A synergistic diffusion–autoregression paradigm for scalable sequence generation, 2025.

Dong et al. (2024). Hongyuan Dong, Jiawen Li, Bohong Wu, Jiacong Wang, Yuan Zhang, and Haoyuan Guo. Benchmarking and improving detail image caption. arXiv preprint arXiv:2405.19092, 2024.

Esser et al. (2024). Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Fu et al. (2025). Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji, Caifeng Shan, and Ran He. Mme: A comprehensive evaluation benchmark for multimodal large language models, 2025.

Gong et al. (2024). Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, et al. Scaling diffusion language models via adaptation from autoregressive models. arXiv preprint arXiv:2410.17891, 2024.

Han et al. (2022). Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. Ssd-lm: Semi-autoregressive simplex-based diffusion language model for text generation and modular control. arXiv preprint arXiv:2210.17432, 2022.

Kembhavi et al. (2016). Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European conference on computer vision, pages 235–251. Springer, 2016.

Li et al. (2023). Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.

Li et al. (2024). Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li. Llava-next: Stronger llms supercharge multimodal capabilities in the wild, 2024.

Li et al. (2025a). Shufan Li, Jiuxiang Gu, Kangning Liu, Zhe Lin, Zijun Wei, Aditya Grover, and Jason Kuen. Lavida-o: Elastic large masked diffusion models for unified multimodal understanding and generation, 2025a.

Li et al. (2025b). Shufan Li, Konstantinos Kallidromitis, Hritik Bansal, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Jason Kuen, Zhe Lin, Kai-Wei Chang, and Aditya Grover. Lavida: A large diffusion language model for multimodal understanding. arXiv preprint arXiv:2505.16839, 2025b.

Li et al. (2025c). Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, et al. Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052, 2025c.

Liao et al. (2025). Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12037–12047, 2025.

Liu et al. (2023). Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:0 34892–34916, 2023.

Liu et al. (2024). Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.

Liu et al. (2025). Yangzhou Liu, Yue Cao, Hao Li, Gen Luo, Zhe Chen, Weiyun Wang, Xiaobo Liang, Biqing Qi, Lijun Wu, Changyao Tian, et al. Sequential diffusion language models. arXiv preprint arXiv:2509.24007, 2025.

Lou et al. (2023). Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023.

Lu et al. (2023). Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

Masry et al. (2022). Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.

Nie et al. (2025). Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.

Peebles and Xie (2023). William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

Qwen et al. (2025). Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025.

Tao et al. (2025). Hongyuan Tao, Bencheng Liao, Shaoyu Chen, Haoran Yin, Qian Zhang, Wenyu Liu, and Xinggang Wang. Infinitevl: Synergizing linear and sparse attention for highly-efficient, unlimited-input vision-language models, 2025.

Tschannen et al. (2025). Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

Vaswani et al. (2017). Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, ukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Wang et al. (2024). Fei Wang, Xingyu Fu, James Y Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, et al. Muirbench: A comprehensive benchmark for robust multi-image understanding. arXiv preprint arXiv:2406.09411, 2024.

Wu et al. (2025a). Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, and Enze Xie. Fast-dllm v2: Efficient block-diffusion llm. arXiv preprint arXiv:2509.26328, 2025a.

Wu et al. (2025b). Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025b.

Wu et al. (2023). Tong Wu, Zhihao Fan, Xiao Liu, Hai-Tao Zheng, Yeyun Gong, Jian Jiao, Juntao Li, Jian Guo, Nan Duan, Weizhu Chen, et al. Ar-diffusion: Auto-regressive diffusion model for text generation. Advances in Neural Information Processing Systems, 36:0 39957–39974, 2023.

xAI (2024). xAI. Grok-1.5 vision preview: Connecting the digital and physical worlds with our first multimodal model, 2024.

Yang et al. (2025). Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models. arXiv preprint arXiv:2505.15809, 2025.

Yao et al. (2024). Jingfeng Yao, Cheng Wang, Wenyu Liu, and Xinggang Wang. Fasterdit: Towards faster diffusion transformers training without architecture modification. Advances in Neural Information Processing Systems, 37:0 56166–56189, 2024.

Yao et al. (2025). Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15703–15712, 2025.

Ye et al. (2025). Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025.

You et al. (2025). Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large language diffusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025.

Yu et al. (2025). Runpeng Yu, Xinyin Ma, and Xinchao Wang. Dimple: Discrete diffusion multimodal large language model with parallel decoding. arXiv preprint arXiv:2505.16990, 2025.

Yue et al. (2024a). Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024a.

Yue et al. (2024b). Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024b.

Zhang et al. (2024a). Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Reality check on the evaluation of large multimodal models, 2024a.

Zhang et al. (2025). Kaichen Zhang, Keming Wu, Zuhao Yang, Bo Li, Kairui Hu, Bin Wang, Ziwei Liu, Xingxuan Li, and Lidong Bing. Openmmreasoner: Pushing the frontiers for multimodal reasoning with an open and general recipe. arXiv preprint arXiv:2511.16334, 2025.

Zhang et al. (2024b). Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024b.

Zhang\* et al. (2020). Tianyi Zhang\*, Varsha Kishore\*, Felix Wu\*, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations, 2020.

Zhu et al. (2025). Lianghui Zhu, Zilong Huang, Bencheng Liao, Jun Hao Liew, Hanshu Yan, Jiashi Feng, and Xinggang Wang. Dig: Scalable and efficient diffusion models with gated linear attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7664–7674, 2025.

Zou et al. (2025). Ya Zou, Jingfeng Yao, Siyuan Yu, Shuai Zhang, Wenyu Liu, and Xinggang Wang. Turbo-vaed: Fast and stable transfer of video-vaes to mobile devices. arXiv preprint arXiv:2508.09136, 2025.

[^1]: $^*$ Equal Contribution; $^\dagger$ Corresponding author: Xinggang Wang (<span style="color: blue">`xgwang@hust.edu.cn`</span>).
