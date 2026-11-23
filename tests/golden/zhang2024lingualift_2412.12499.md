---
title: "LinguaLIFT: An Effective Two-stage Instruction Tuning Framework for Low-Resource Language Reasoning"
authors: ["Hongbin Zhang", "Kehai Chen", "Xuefeng Bai", "Yang Xiang", "Min Zhang"]
year: 2024
arxiv: "2412.12499"
arxiv_version: "v2"
marginnote:
  - bookmd5: 4a439691831e2e7c25e5fc9a4890c0f65a811a3fa054dee5b968def668c3bae5
    path: "PCIC/Tesis/low-resource/LinguaLIFT- An Effective Two-stage Instruction Tuning Framework for Low-Resource Language Reasoning.pdf"
source: "arXiv LaTeX, tex2md 0.1"
converted: 2026-11-22
---

# LinguaLIFT: An Effective Two-stage Instruction Tuning Framework for Low-Resource Language Reasoning

**Autores:** Hongbin Zhang; Kehai Chen; Xuefeng Bai; Yang Xiang; Min Zhang

> arXiv:2412.12499v2 · https://arxiv.org/abs/2412.12499
> Convertido desde la fuente LaTeX de arXiv con tex2md (pandoc). Numeración de secciones, figuras, tablas y ecuaciones tomada de la compilación del artículo; las figuras se reducen a su leyenda. Ante cualquier duda, el PDF es la referencia.

## Abstract

Large language models (LLMs) have exhibited impressive multilingual reasoning capabilities, driven by extensive multilingual pre-training corpora and instruction fine-tuning data. However, a performance gap exists between high- and low-resource language reasoning tasks due to the language imbalance in the pre-training corpus, which is exacerbated by evaluation bias in existing reasoning benchmarks lacking low-resource language coverage. To alleviate this issue, we propose LinguaLIFT, a two-stage instruction tuning framework for advancing low-resource language reasoning. LinguaLIFT employs a language alignment layer to capture multilingual alignment in a code-switched tuning way without requiring multilingual instruction or parallel data, thereby transferring the cross-lingual reasoning capabilities to low-resource languages through English-only instruction tuning data. To comprehensively evaluate the multilingual reasoning capabilities, we introduce the Multilingual Math World Problem (MMWP) benchmark, which spans 21 low-resource, 17 medium-resource, and 10 high-resource languages. Experimental results show that LinguaLIFT outperforms several competitive baselines across MMWP and four widely used benchmarks. [^3]

# 1 Introduction

Large Language Models (LLMs) have demonstrated impressive reasoning capabilities, opening up a new paradigm for artificial intelligence (Wei et al., 2022; Zhao et al., 2023). These capabilities arise from self-supervised pre-training on large-scale multilingual data, followed by supervised fine-tuning on multilingual instruction data (Ouyang et al., 2022; Muennighoff et al., 2023; Chung et al., 2024). As a result, LLMs have achieved remarkable performance across various natural language processing tasks (Chen et al., 2024a; Chen et al., 2024b; Zhang et al., 2024), especially high-resource language mathematical reasoning tasks (Shi et al., 2023; She et al., 2024).

**Figura 1.** Examples from the MGSM (Shi et al., 2023) dataset, where the mathematical problems share the same meaning across languages, but LLMs generate different answers. The red text marks the erroneous reasoning in the responses. Translations of the responses are provided in dashed boxes.

However, a substantial performance gap still exists between high- and low-resource language reasoning (Shi et al., 2023; Zhu et al., 2024), primarily due to the language imbalance within the large-scale multilingual corpora during pre-training, where high-resource languages dominate (Touvron et al., 2023; Lu et al., 2024). This gap is further exacerbated by evaluation bias inherent in existing multilingual reasoning benchmarks, which exhibit limited coverage of low-resource languages (Shi et al., 2023; Chen et al., 2023). As illustrated in Figure  1, while LLMs can answer correctly to the query in English, they struggle with the same query in low-resource languages (e.g., Bengali (Bn), Swahili (Sw)). These discrepancies emphasize LLMs’ fundamental limitation in low-resource language reasoning.

To alleviate this issue, this study proposes a two-stage instruction tuning framework (LinguaLIFT). The first stage introduces a language alignment layer to adapt a pre-trained multilingual encoder, utilizing code-switched translation data to enhance multilingual alignment. Building on this strengthened alignment, the second stage fine-tunes the LLM on English instruction data while keeping the alignment layer frozen, thereby transferring reasoning capabilities from English to low-resource languages. This study builds a new benchmark named Multilingual Math World Problem (MMWP), spanning 21 low-resource, 17 medium-resource, and 10 high-resource languages. Experiments demonstrate that LinguaLIFT significantly outperforms several strong competitive methods on the MMWP and four widely used benchmarks, such as MGSM (Shi et al., 2023), MSVAMP (Chen et al., 2023), XNLI (Conneau et al., 2018) and X-CSQA (Lin et al., 2021).

To summarize, our primary contributions are:

- The proposed LinguaLIFT significantly enhances low-resource language reasoning without requiring multilingual or parallel data.

- This study introduces MMWP, a new mathematical reasoning benchmark covering languages at varying resource levels, to evaluate multilingual reasoning tasks comprehensively.

- Experiments across mathematical and commonsense reasoning demonstrate the effectiveness and generalizability of LinguaLIFT in advancing low-resource language reasoning.

# 2 Related Work

#### Leveraging cross-lingual transfer within multilingual models.

While recent research highlights multilingual models’ robust cross-lingual transfer capabilities (Conneau et al., 2020; FitzGerald et al., 2023; Shaham et al., 2024)—achieved through alignment improvements via multilingual instruction or parallel data (Pfeiffer et al., 2020; Pan et al., 2021; Feng et al., 2022; Li et al., 2023a; Chen et al., 2024c)—such approaches remain impractical in low-resource scenarios due to data scarcity. To address this limitation, LinguaLIFT advances code-switched tuning, demonstrating success in cross-lingual transfer (Yang et al., 2020; Li et al., 2024), to enhance multilingual alignment without requiring multilingual instructions or parallel corpora.

#### Improving multilingual mathematical reasoning tasks.

Recent efforts to improve multilingual mathematical reasoning for LLMs can be categorized into three ways: 1) **Prompting close-source LLMs:** Qin et al. (2023), Shi et al. (2023), and Huang et al. (2023) designed prompts for closed-source LLMs like ChatGPT, translating non-English contexts into English for reasoning. This approach is limited by translation quality and does not improve multilingual understanding or works well for open-source LLMs (Zhu et al., 2024). 2) **Instruction-tuning open-source LLMs:** *MathOctopus* (Chen et al., 2023), *xCoT* (Chai et al., 2024), and *mCoT* (Lai and Nissim, 2024) adopted a translate-training method, translating English reasoning datasets into non-English and instruction-tuning LLMs. *QAlign* (Zhu et al., 2024) and *MAPO* (She et al., 2024) proposed approaches to transfer mathematical reasoning capabilities from English to non-English. While these methods improve multilingual reasoning, they incur high translation costs and errors, making them impractical for low-resource languages. 3) **Bridging existing skilled LLMs to multilingualism:** *Langbridge* (Yoon et al., 2024) combines pre-trained multilingual models with skilled reasoning LLMs, but a performance gap persists between low-resource and high-resource languages. While *MindMerger* (Huang et al., 2024) shows significant improvement, it relies heavily on parallel and multilingual reasoning data.

**Figura 2.** Overview of the proposed LinguaLIFT approach. **Stage-I (Language Align)**: A language alignment layer is introduced into the LLM to adapt the pre-trained multilingual encoder, thereby enhancing multilingual alignment through code-switched tuning. **Stage-II (Task Transfer)**: The LLM is fine-tuned on high-quality, English-only instruction data with the language alignment layer frozen, allowing the LLM to transfer reasoning capabilities learned from English to low-resource languages.

Unlike prior approaches that depend on costly translated data or impractical parallel corpora in low-resource settings, LinguaLIFT advances low-resource language reasoning without requiring translated multilingual or parallel datasets.

# 3 Methodology

To harness cross-lingual transfer for low-resource languages reasoning, this study proposes leveraging the language alignment established by code-switched tuning (Section §3.1) to transfer reasoning capabilities from English to low-resource languages (Section §3.2), as illustrated in Figure 2. Formally, given an input sequence $x$ with $l_x$ tokens, a multilingual pre-trained encoder generates a language-agnostic representation $X\in \mathbb{R}^{l_x \times d_1}$, which is subsequently aligned to the input embedding space of the LLM via a multi-layer perceptron alignment layer, yielding $\hat{X}\in \mathbb{R}^{l_x \times d_2}$. To leverage the built-in capabilities of the LLM for instruction following, the instruction context $q$ (with $l_q$ tokens) is mapped into $Q\in \mathbb{R}^{l_q \times d_2}$ via the LLM’s embedding layer. The combined input sequence is constructed by concatenating the start token \<bos\>, $Q$, alignment boundary tokens \<enc_start\> and \<enc_end\>, and $\hat{X}$, forming:

$$\begin{equation}
    (Q, \hat{X}) = [\text{<bos>}; Q; \text{<enc\_start>}; \hat{X}; \text{<enc\_end>}].
\tag{1}
\end{equation}$$

This structured input is then processed by the LLM to generate the final response.

## 3.1 Language Align Stage

Due to the scarcity of parallel corpora in low-resource languages, this study proposes enhancing the model’s multilingual alignment through code-switched tuning. This involves two key parts: (i) multilingual alignment lexicon construction and (ii) code-switched translation instruction tuning.

#### Multilingual alignment lexicon construction.

An unsupervised word translation method MUSE (Lample et al., 2018) is adopted to construct the multilingual alignment lexicons without parallel data. Specifically, a set of unique words $\mathcal{W}=\{w\}_i^{N}$ is extracted from English monolingual corpus $\mathcal{D}$, where $N$ is the number of unique words, excluding named entities and stop words. Word translations are obtained by retrieving the top nearest neighbor via projection of word embeddings into the target space. Further implementation details and validation steps in Appendix A.1 ensure the robustness and reliability of our constructed lexicons.

#### Code-switched translation instruction tuning.

The constructed multilingual alignment lexicons are used to generate code-switched inputs for instruction tuning. Formally, our training objective is to predict the English translation response tokens $y$ given the instruction context $q$ and code-switched input tokens $x_{l}$ in language $l$. The input embeddings for this task, represented as $(Q, \hat{X_l})$ from Equation (1), serve as the LLMs’ input. The optimization objective is defined as follows:

$$\begin{equation}
    -\underset{\theta}{\arg \min } \sum_{l \in L} \sum_{i=1}^{T} \log p_{\theta,\psi ,\phi} (y_{i} \mid (Q,\hat{X}_{l}), y_{<i}),
\tag{2}
\end{equation}$$

where $T$ is the target sequence length, $y_{<i}$ represents the preceding tokens, $y_i$ denotes the $i$-th token, and $L$ is the set of target languages. Trainable parameters include $\theta$ (language alignment layer and boundary tokens), while $\psi$ (multilingual encoder) and $\phi$ (LLM) remain frozen.

## 3.2 Task Transfer Stage

Following the language alignment established in the *Language Align Stage*, the LLM is fine-tuned using high-quality English reasoning instruction data while keeping the language alignment layer frozen. This strategy preserves the acquired multilingual alignment and cross-lingual transfer capabilities, unlocking the potential of LLM to transfer strong reasoning capabilities learned from English to low-resource languages. The training minimizes the negative log-likelihood of the response $y$, given the combined instruction context and query input $(Q, \hat{X}_{en})$:

$$\begin{equation}
    -\underset{\phi}{\arg \min }  \sum_{i=1}^{T} \log p_{\tilde{\theta},\psi ,\phi}\left(y_{i} \mid (Q,\hat{X}_{en}), y_{<i}\right),
\tag{3}
\end{equation}$$

where $\phi$ denotes the trainable LLM parameter, and $\tilde{\theta}$ is initialized from the checkpoint of $\theta$ trained at the first stage. The multilingual encoder parameters $\psi$ and $\tilde{\theta}$ are frozen throughout the training.

# 4 The MMWP Benchmark

Existing multilingual mathematical reasoning benchmarks (Shi et al., 2023; Chen et al., 2023) predominantly focus on 7 high-resource languages and 3 low-resource languages, leaving a significant gap in coverage for low-resource languages. This imbalance introduces evaluation bias, as models are optimized for high-resource languages while their performance on low-resource languages remains underexplored, hindering the comprehensive development of multilingual models. To fill this gap, we build a new Multilingual Math World Problem (MMWP) benchmark and describe its collection process in this section.

#### Source data.

We use AsDiV (Miao et al., 2020) and MAWPS (Koncel-Kedziorski et al., 2016) as base datasets. From the AsDiv official test set, we randomly select 500 examples and another 500 from MAWPS, where all the problems require multiple steps to solve, as described by Miao et al. (2020). We filter out duplicates and problems with non-numeric answers, resulting in 811 examples.

#### Target language selection.

Our study employs a typologically diverse sample of 48 languages representing 12 language families and 12 writing systems, encompassing low-, medium-, and high-resource levels. The selection comprises 21 low-resource, 17 medium-resource, and 10 high-resource languages. Language categorization list and selection criteria are detailed in Appendices B.1 and B.2, respectively.

#### Translation and post-calibration.

English questions are translated into target languages using Google Translate. Translation quality is verified through combined automated and human evaluations. Five annotators then refine inaccuracies via GPT-4-guided post-calibration, followed by reviews and adjustments from two language experts prominent in minor languages. Detailed accuracy assurance procedures are provided in Appendix B.3.

# 5 Experiments

## 5.1 Datasets

#### Evaluation dataset.

We use the MMWP and the latest multilingual benchmarks, **MGSM** (Shi et al., 2023) and **MSVAMP** (Chen et al., 2023) to evaluate the performance of LLMs in multilingual mathematical reasoning through zero-shot chain-of-thought reasoning (Wei et al., 2022) setting. To further assess the task generalization of LinguaLIFT, we incorporate several challenging multilingual datasets, including **X-CSQA** (Lin et al., 2021) for commonsense reasoning and **XNLI** (Conneau et al., 2018) for natural language inference.

#### Training dataset.

We utilize English-only instruction data following prior work (Lu et al., 2024; Huang et al., 2024), which include **MetaMathQA** (Yu et al., 2024) for mathematical reasoning, **MultiNLI** (Williams et al., 2018) for natural language inference, and a set of unified commonsense reasoning tasks, comprising the **X-CSQA**, **OpenBookQA** (Mihaylov et al., 2018), **ARC** (Clark et al., 2018), and **QASC** (Khot et al., 2020) datasets. Additionally, to further explore the potential of the proposed method, we incorporate the recent advanced mathematical reasoning instruction dataset, **OpenMathInstruct-2** (Toshniwal et al., 2024). Statistics of the datasets involved are presented in Appendix A.2, and the prompts for each task are given in Appendix A.3.

## 5.2 Baselines

Three categories of baselines are considered: (1) **Mono-SFT**: *MAmmoTH* (Yue et al., 2024), *WizardMath* (Luo et al., 2023), *MetaMath* (Yu et al., 2024), and *OpenMath2* (Toshniwal et al., 2024) fine-tune on respective English instruction dataset. (2) **Multi-SFT**: *MathOctopus* (Chen et al., 2023), *QAlign* (Zhu et al., 2024), *MAPO* (She et al., 2024) fine-tune on translated multilingual instruction datasets (e.g., MGSM8K-Instruct (Chen et al., 2023)) or parallel corpus. (3) **Leveraging External Tools or Models**: *Translate-En* (Shi et al., 2023) use SOTA translation models (Team et al., 2022) to translate the query into English; *LangBridge* (Yoon et al., 2024) and *MindMerger* (Huang et al., 2024) replace the input of LLMs with the hidden states output by mT5 (Xue et al., 2021). To ensure a fair comparison with MindMerger (the previous SOTA), we further consider continuing training (*Add Low-Resource*) or retraining (*Low-Resource Retrain*) with open-source low-resource parallel data. More training details are presented in Appendix A.4.

## 5.3 Experimental Results

**Tabla 1.** Results on the MMWP benchmark, where “LR.”, “MR.”, and “HR.” denote mean accuracy for low-, medium-, and high-resource languages, respectively. “Use Multi.” denotes whether multilingual or parallel corpus is utilized. Bold values indicate the highest score across systems. $\dagger$ marks results obtained using officially released models, $\Diamond$ denotes the LoRA (Hu et al., 2022) implementation, and $\ddagger$ signifies results from our local implementation. The LLM and extra data used in the third baseline category are specified in brackets.

| **LLaMA-2-7B as base model** | **Use** | **LR.** | **MR.** | **HR.** | **Avg.** |
|:---|:--:|:--:|:--:|:--:|:--:|
|  | **Multi.** |  |  |  |  |
| baseline |  |  |  |  |  |
| ***Mono-SFT*** |  |  |  |  |  |
| MAmmoTH^†^ | ✗ | 6.36 | 20.4 | 26.1 | 15.5 |
| WizardMath^†^ | ✗ | 10.1 | 28.0 | 32.2 | 21.0 |
| MetaMath^†^ | ✗ | 13.6 | 37.2 | 41.8 | 27.8 |
| OpenMath2^‡^ | ✗ | 19.3 | 55.5 | **62.7** | 41.1 |
| ***Multi-SFT*** |  |  |  |  |  |
| MathOctopus-Parallel^†^ | ✓ | 11.0 | 20.8 | 25.2 | 17.4 |
| MathOctopus-MAPO-DPO^†^ | ✓ | 18.6 | 29.8 | 33.1 | 25.6 |
| MetaMathOctopus-MAPO-DPO^†^ | ✓ | 16.4 | 36.7 | 43.9 | 29.3 |
| QAlign-MetaMathQA^†^ | ✓ | 17.1 | 39.0 | 44.5 | 30.6 |
| ***Leveraging External Tools or Models*** |  |  |  |  |  |
| Translate-En^‡^ (MetaMath) | ✗ | 27.6 | 36.4 | 40.6 | 33.4 |
| LangBridge^†^ (MetaMath) | ✗ | 33.4 | 36.8 | 39.4 | 35.9 |
| MindMerger-Soft^†^ (MetaMath) | ✓ | 36.6 | 40.7 | 43.2 | 39.4 |
| (Add Low-Resource)^‡^ | ✓ | 36.7 | 39.8 | 41.5 | 38.8 |
| (Low-Resource Retrain)^‡^ | ✓ | 33.4 | 36.8 | 39.4 | 35.9 |
| Translate-En^‡^ (OpenMath2) | ✗ | 43.1 | 54.7 | 61.5 | 51.0 |
| LangBridge^‡^ (OpenMath2) | ✗ | 47.9 | 53.8 | 59.2 | 52.3 |
| MindMerger-Soft^‡^ (OpenMath2) | ✓ | 49.5 | 54.8 | 61.2 | 53.8 |
| our methods |  |  |  |  |  |
| **LinguaLIFT (MetaMathQA)** | ✗ | 41.2 | 44.3 | 45.5 | 43.2 |
| **LinguaLIFT^$\Diamond$^ (MetaMathQA)** | ✗ | 38.6 | 42.7 | 44.1 | 41.2 |
| **LinguaLIFT (OpenMathInstruct-2)** | ✗ | **55.4** | **61.2** | 62.5 | **58.9** |
| **LinguaLIFT^$\Diamond$^ (OpenMathInstruct-2)** | ✗ | 53.5 | 59.0 | 60.8 | 57.0 |

#### LinguaLIFT demonstrates significant improvements across low-resource languages.

Table 1 presents experimental results on the MMWP test set, grouped by language resource levels. We highlight seven key findings: (1) Mono-SFT models exhibit substantial performance drops in low-resource languages. (2) Multi-SFT models outperform monolingual counterparts in low-resource settings but retain a notable performance gap between low- and high-resource languages. (3) Models using external translation systems or pre-trained multilingual models generalize poorly to unseen languages and underperform in low-resource contexts. (4) Even with supplementary low-resource data, MindMerger fails to improve low-resource language reasoning performance. (5) LinguaLIFT substantially improves low-resource reasoning accuracy, surpassing all baselines without requiring multilingual or parallel data. (6) Integrating advanced English reasoning datasets (e.g., OpenMathInstruct-2) further enhances reasoning performance, underscoring the need to adapt to evolving high-quality English reasoning data.

#### Robust generalization and task versatility.

**Tabla 2.** Results on MGSM and MSVAMP datasets. The usage of abbreviation, symbol and boldface are the same in Table 1.

|  | **MGSM** |  |  | **MSVAMP** |  |  |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **LLaMA-2-7B as base model** | **LR.** | **HR.** | **Avg.** | **LR.** | **HR.** | **Avg.** |
| baseline |  |  |  |  |  |  |
| ***Mono-SFT*** |  |  |  |  |  |  |
| MAmmoTH^†^ | 3.40 | 32.4 | 21.9 | 6.57 | 40.1 | 30.1 |
| WizardMath^†^ | 4.00 | 37.7 | 25.5 | 15.7 | 48.5 | 38.7 |
| MetaMath^†^ | 4.60 | 51.7 | 34.6 | 15.2 | 61.2 | 47.4 |
| OpenMath2^‡^ | 5.60 | 60.2 | 40.4 | 18.8 | 70.5 | 55.0 |
| ***Multi-SFT*** |  |  |  |  |  |  |
| MathOctopus-Parallel^†^ | 28.0 | 42.4 | 37.2 | 33.6 | 43.8 | 40.8 |
| MathOctopus-MAPO-DPO^†^ | 30.6 | 43.4 | 38.8 | 52.5 | 58.8 | 56.9 |
| MetaMathOctopus-MAPO-DPO^†^ | 31.0 | 55.6 | 46.7 | 57.8 | 67.4 | 64.5 |
| QAlign-MetaMathQA^†^ | 26.3 | 55.9 | 45.2 | 48.4 | 61.5 | 57.6 |
| ***Leveraging External Tools or Models*** |  |  |  |  |  |  |
| Translate-En^‡^ (MetaMath) | 39.6 | 55.1 | 49.4 | 47.4 | 52.3 | 50.9 |
| LangBridge^†^ (MetaMath) | 38.4 | 51.9 | 47.0 | 43.9 | 54.5 | 51.4 |
| MindMerger-Soft^†^ (MetaMath) | 53.1 | 57.9 | 56.2 | 52.7 | 60.6 | 58.2 |
| Translate-En^‡^ (OpenMath2) | 41.3 | 63.8 | 55.6 | 51.9 | 62.2 | 59.1 |
| LangBridge^‡^ (OpenMath2) | 42.6 | 60.8 | 54.2 | 47.9 | 64.4 | 59.5 |
| MindMerger-Soft^‡^ (OpenMath2) | 60.5 | **67.5** | 65.0 | 63.4 | 74.2 | 70.9 |
| our methods |  |  |  |  |  |  |
| **LinguaLIFT (MetaMathQA)** | 55.4 | 58.8 | 57.6 | 56.1 | 60.6 | 59.3 |
| **LinguaLIFT^$\Diamond$^ (MetaMathQA)** | 51.2 | 54.7 | 53.5 | 51.5 | 56.1 | 54.7 |
| **LinguaLIFT (OpenMathInstruct-2)** | **63.8** | 66.5 | **65.5** | **67.2** | **74.3** | **72.2** |
| **LinguaLIFT^$\Diamond$^ (OpenMathInstruct-2)** | 59.9 | 62.9 | 61.8 | 63.5 | 70.5 | 68.4 |

Experimental results across four multilingual reasoning benchmarks (Tables 2 and 3) demonstrate LinguaLIFT’s robust generalization capabilities in out-of-domain scenarios. The framework achieves significant improvements over existing baselines in mathematical reasoning tasks (MGSM, MSVAMP) for low-resource languages while exhibiting strong adaptability to cross-domain commonsense reasoning challenges (XNLI, X-CSQA). By excelling in numerical computation and understanding semantics, LinguaLIFT demonstrates its task-agnostic generalization ability across various tasks, effectively transferring learned patterns between different reasoning paradigms and thus enhancing performance in low-resource language reasoning. Furthermore, the model enhances reasoning accuracy even in high-resource languages, underscoring its dual capacity to generalize and adapt across tasks and resource levels.

**Tabla 3.** Results on XNLI and X-CSQA datasets. The usage of abbreviation is the same in Table 2. The asterisk symbol (∗) indicates results are taken from the published results of Huang et al. (2024).

|  | **XNLI** |  |  | **X-CSQA** |  |  |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **LLaMA-2-7B as base model** | **LR.** | **HR.** | **Avg.** | **LR.** | **HR.** | **Avg.** |
| baselines |  |  |  |  |  |  |
| ***Mono-SFT(English-only Task Data)*** |  |  |  |  |  |  |
| Mono-SFT^∗^ | 58.7 | 80.1 | 68.7 | 28.6 | 58.6 | 51.3 |
| ***Multi-SFT(Multilingual Task Data with Query Translation)*** |  |  |  |  |  |  |
| Multi-SFT^∗^ | 63.6 | 81.5 | 71.9 | 29.4 | 48.6 | 43.8 |
| QAlign^∗^ | 67.1 | 80.9 | 73.5 | 35.5 | 57.9 | 52.3 |
| ***Freezing LLM with External Tools or Models*** |  |  |  |  |  |  |
| LangBridge^∗^ | 73.9 | 79.4 | 76.5 | 30.9 | 37.8 | 36.1 |
| Translate-En^∗^ | 71.1 | 79.6 | 75.1 | 42.7 | 55.5 | 52.3 |
| MindMerger-Soft^∗^ | 74.4 | 83.1 | 78.4 | 47.9 | 65.4 | 61.0 |
| our methods |  |  |  |  |  |  |
| **LinguaLIFT** | **77.6** | **83.3** | **80.3** | **49.6** | **65.5** | **61.5** |

Overall, LinguaLIFT established itself as a universally applicable and strong potential method for LLMs to advance low-resource language performance across various reasoning tasks. More experimental results are provided in Appendix C.

# 6 Analysis

## 6.1 Two-stage Training Ablation

**Figura 3.** Ablation study of two-stage training on MGSM, showing average accuracy for low-resource (LR.), high-resource (HR.), and all languages (Avg.).

As shown in Figure 3, ablating the *Language Align Stage* results in a significant performance decline for low-resource languages, empirically validating its role in cross-lingual knowledge transfer. Similarly, removing the *Task Transfer Stage* degrades performance across all languages. These results underscore both stages’ mutually dependent contributions to enhancing reasoning tasks. Further details are provided in Appendix D.1.

## 6.2 Trainable Modules Ablation

**Figura 4.** Trainable modules ablation on MGSM. The usage of abbreviation is the same in Figure 3.

As shown in Figure 4, training the LLM during the Language Align stage impairs high-resource language reasoning, likely due to catastrophic forgetting of acquired knowledge. Freezing the LLM during the Task Transfer stage similarly degrades performance across all languages. Notably, training the language alignment layer in this stage undermines low-resource reasoning, as it disrupts the pre-established multilingual alignment. These findings necessitate sequential training: first optimizing the alignment layer, followed by LLM fine-tuning. Further details are available in Appendix D.2.

## 6.3 Multilingual Encoder Ablation

**Figura 5.** Accuracy (%) of LinguaLIFT across encoders’ scales and types on MGSM’s low-resource reasoning.

Figure 5 evaluates low-resource language reasoning on the MGSM test set using the mT5 series (Xue et al., 2021) across various scales and four pre-trained multilingual encoders: mBERT (Devlin et al., 2019), XLM (Lample and Conneau, 2019), XLM-R (Conneau et al., 2020), and LaBSE (Feng et al., 2022). Performance improved significantly as mT5 scaled from 270M to 2.2B parameters but stagnated beyond this threshold, indicating a plateau at larger scales. Notably, LaBSE, leveraging parallel corpora for robust cross-lingual alignment, outperformed other encoders. More detailed information is provided in Appendix D.3.

## 6.4 Generalizes Well to Various LLMs

**Tabla 4.** Tesults on MGSM across various LLMs. Abbreviation and symbol are the same in Table 1.

|  | **Mistral-7B** |  |  | **Llama-2-13B** |  |  |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Methods** | **LR.** | **HR.** | **Avg.** | **LR.** | **HR.** | **Avg.** |
| baseline |  |  |  |  |  |  |
| ***Mono-SFT(English-only Task Data)*** |  |  |  |  |  |  |
| MetaMath^∗^ | 30.0 | 66.5 | 55.6 | 8.53 | 59.7 | 44.4 |
| ***Multi-SFT(Multilingual Task Data with Query Translation)*** |  |  |  |  |  |  |
| MathOctopus-Parallel^∗^ | 48.1 | 51.1 | 50.2 | 41.6 | 47.7 | 45.8 |
| QAlign-MetaMathQA^∗^ | 50.7 | 59.3 | 56.7 | 44.7 | 62.4 | 57.1 |
| ***Freezing LLM with External Tools or Models*** |  |  |  |  |  |  |
| Translate-En^∗^ | 53.7 | 60.1 | 58.2 | 44.4 | 55.5 | 52.2 |
| LangBridge^∗^ | 52.4 | 65.8 | 61.8 | 41.3 | 52.9 | 49.5 |
| MindMerger-Soft^∗^ | 56.8 | 69.1 | 65.4 | 57.1 | 65.1 | 62.7 |
| our methods |  |  |  |  |  |  |
| **LinguaLIFT (MetaMathQA)** | 57.7 | 68.8 | 65.5 | 58.7 | 64.7 | 62.9 |

We extend the evaluations in Section 5 to diverse LLMs and scales, including Llama-2-13B (Touvron et al., 2023) and Mistral-7B (Jiang et al., 2023). Table 4 demonstrates LinguaLIFT’s broader applicability across LLM scales and architectures, with full results in Appendix D.4.

## 6.5 Analysis of Language Transferability

**Figura 6.** Reasoning accuracy on MMWP group by language families and writing systems.

As illustrated in Figure 6, LinguaLIFT outperformed all competitive baselines, achieving state-of-the-art transferability across language families and writing systems, particularly in languages sharing English’s syntax or script (e.g., Germanic, Latin). These findings highlight LinguaLIFT as a robust cross-lingual transfer framework, leveraging shared linguistic features to enhance low-resource language reasoning tasks. More details are provided in Appendix D.5.

## 6.6 Analysis of Code-Switched Ratio

**Figura 7.** Ablation experiments on the impact of varying code-switch ratios on multilingual alignment and reasoning performance.

We assess language alignment and mathematical reasoning via top-1 retrieval accuracy on Tatoeba (Artetxe and Schwenk, 2019) and MGSM accuracy, respectively (Figure 7). Higher code-switching ratios enhance multilingual alignment, especially for low-resource languages, with proportional gains in reasoning accuracy. Notably, at an 80% ratio, the model matches the performance of parallel corpus-trained systems, indicating that word-level code-switching suffices for robust alignment. This demonstrates that explicit parallel corpora are unnecessary, as code-switching inherently promotes multilingual understanding and reasoning in low-resource scenarios.

## 6.7 Analysis of Code-Switched Part-of-Speech

**Figura 8.** The reasoning performance on MGSM group by the combinations of POS: **(1) individual POS categories:**Verb (V), Adposition (ADP), Pronoun (Pron); **(2) syntactic function combinations:**Verb+Adverb (V+Adv), Adjective+Adverb (Adj+Adv), Pronoun+Auxiliary+Conjunction (Pron + Aux + Conj); and **(3) key syntactic structures:**Subject-Verb (N+V) and Prepositional Phrases (N+ADP).

**Figura 9.** T-SNE visualization in the spaces of the Mono-SFT input embeddings, MindMerger mapping layer outputs, and LinguaLIFT language alignment layer outputs.

As illustrated in Figure 8, noun substitutions significantly impair the model’s reasoning capabilities among part-of-speech combinations. Syntactic structures involving subject-verb pairs and prepositional phrases demonstrate superior performance compared to other combinations, underscoring the importance of core arguments (subjects, verbs) and their syntactic relations (prepositions) in capturing essential relational information for cross-linguistic reasoning generalization. Conversely, adjective-adverb and auxiliary-conjunction combinations exhibit the lowest performance, indicating that modifiers play a less critical role in reasoning. Further details are available in Appendix D.6.

## 6.8 Incorporating Multilingual Data

**Figura 10.** Accuracy (%) on MGSM when LinguaLIFT leverages multilingual instruction and parallel data.

Beyond using code-switched translation data, LinguaLIFT can adapt to the training setting of previous works by integrating real multilingual data, further advancing reasoning performance. As shown in Figure 10, Multi-LinguaLIFT achieves a higher ceiling for low-resource language reasoning, with full experimental details in Appendix D.7.

## 6.9 Alignment Representation Visualization

For each low-resource language in the MMWP benchmark, we extracted 100 semantically equivalent texts from the Flores-101 dataset (Goyal et al., 2022). We derived mean pooling representations using multiple methods and visualized them via T-SNE (van der Maaten and Hinton, 2008). As shown in Appendix E.1, cross-lingual alignment positively correlates with reasoning transfer: representations closer to English exhibit more substantial reasoning capability generalization. Figure 9 reveals that the *LLaMA2-7B*’s low-resource language representations are scattered and divergent from English, indicating the challenges in knowledge transfer. *MindMerger* achieves partial alignment, with some low-resource language representations overlapping English representations; however, others remain isolated, reflecting its instability in diverse low-resource scenarios. In contrast, *LinguaLIFT* tightly aligns low-resource languages with English, enabling effective reasoning transfer from English instruction data and yielding superior performance.

## 6.10 Supplementary Experiments

More supplementary experiments comprising: (1) quantitative correlations between multilingual alignment and reasoning performance (Appendix E.1), (2) language alignment layer selection (Appendix E.2), and (3) zero-shot CoT examples in mathematical reasoning (Appendix E.3).

# 7 Conclusion

This paper proposes LinguaLIFT, a novel two-stage instruction tuning framework that enhances low-resource language reasoning without relying on parallel corpora or multilingual instruction data. Additionally, we introduce MMWP, a multilingual benchmark spanning 21 low-resource, 17 medium-resource, and 10 high-resource languages, to comprehensively evaluate multilingual mathematical reasoning tasks. Experiments on the MMWP and other widely used benchmarks demonstrate its effectiveness in advancing low-resource language reasoning and further alleviating the performance gap between high-resource and low-resource language reasoning in LLMs.

# Limitations

While our experimental results demonstrate that leveraging code-switched tuning and cross-lingual transfer can help improve low-resource language reasoning, its implementation entails computational considerations requiring a moderately-size multilingual encoder for cross-lingual alignment, which introduces notable resource demands. Although we employ LoRA for parameter efficiency, this adaptation may underperform full-parameter tuning in specific scenarios. Future work should prioritize developing lightweight adaptation strategies that preserve cross-lingual transfer capabilities while optimizing computational efficiency.

# A Implementation Details

## A.1 Collection of the Alignment Lexicons

To better construct multilingual alignment lexicons for low-resource language reasoning, we leveraged the Unsupervised Bilingual Lexicon Induction (UBLI) (Zhang et al., 2017; Lample et al., 2018; Dou et al., 2018; Artetxe et al., 2019; Li et al., 2023b), which has been proven effective in inducing word translation pairs by aligning independently trained word embeddings in two languages.

Initially, we tokenized the text using the Spacy library’s tokenization function[^4]. The resultant word set comprised all words, barring specific named entities, numbers, and date tokens. This step is crucial in ensuring our focus on frequent and pertinent terms that are likely to hold significance in reasoning domains.

Following prior outstanding work MUSE[^5] (Lample et al., 2018), we construct multilingual lexicons using adversarial training to establish a linear mapping between source and target spaces without relying on cross-lingual supervision. This process involves training the model to align the word embeddings of the source and target languages in a shared semantic space. For each word in the source language, we identify the most relevant translations in the target language by projecting the source word embeddings into the target space and retrieving the top nearest neighbor words based on cosine similarity. These translated words form the bilingual lexicons, which are essential for enhancing multilingual understanding. Table 5 provides examples of word translations derived from the multilingual alignment lexicons we constructed, illustrating the effectiveness of this approach.

**Tabla 5.** Word translations examples for English to several low-resource languages.

The accuracy of the word mappings in our code-switch approach is a critical consideration, especially in the absence of parallel data. To demonstrate the feasibility and effectiveness of our unsupervised method, we address this important question from multiple perspectives:

$\bullet$ *Methodology Validation*: The unsupervised word translation method we adopted is based on MUSE (Lample et al., 2018), which has been thoroughly validated in previous research. As demonstrated in MUSE (Section 4.2), this approach achieves remarkable performance without any parallel data - up to 80% accuracy for high-resource languages and around 50% for low-resource languages when translating to English, which is competitive with supervised methods. Constructing a lexicon in such settings is inherently challenging, yet our method achieves remarkable performance, demonstrating the novelty and effectiveness of our approach.

$\bullet$ *Further evaluation using GPT-4o*: To evaluate the accuracy of the lexicons constructed in Stage 1, we employed GPT-4o to evaluate the quality of the generated word mappings. The evaluation, using P@5 accuracy, yielded impressive results: (i) High-resource languages: 87.4% accuracy (P@5) and (ii) Low-resource languages: 61.9% accuracy (P@5).

These results further validate the quality of the unsupervised word mappings produced by our method, underscoring the strength of our approach even in low-resource settings.

This unsupervised approach allowed us to obtain diverse translations while preserving the general meaning of the words, which is crucial for capturing cross-lingual alignment. Moreover, by avoiding context-specific translations, we ensured that the generated translations had broad applicability across multiple tasks and languages. This strategy reduced the reliance on large-scale parallel corpora or pre-existing bilingual dictionaries and provided a robust and scalable solution for building multilingual lexicons in an unsupervised manner.

## A.2 Dataset Statistics

In this study, we utilize a variety of datasets for training and evaluation purposes, as detailed in Table 6. The datasets are divided into two main categories: **Training Datasets** and **Evaluation Datasets**.

The **Training Datasets** include data from different sources and cover a range of languages and sample sizes. These datasets are primarily used for training the models:

- **MetaMathQA** (Yu et al., 2024): A dataset with 395,000 samples in a single language, primarily used for mathematical question-answering tasks.

- **OpenMathInstruct-2** (Toshniwal et al., 2024): This dataset contains 1,000,000 samples and is also in a single language, focusing on instructional tasks related to mathematics.

- **MGSM8KInstruct** (Chen et al., 2023): A multilingual dataset involving 10 languages and 73,599 samples, designed for general-purpose instruction-based tasks.

- **Multi-NLI** (Williams et al., 2018): With 392,702 samples, this dataset is in a single language and focuses on natural language inference tasks.

- **QASC** (Khot et al., 2020): A dataset with 8,134 samples used for question-answering tasks based on commonsense reasoning in a single language.

- **ARC** (Clark et al., 2018): The ARC dataset contains 3,370 samples for answering elementary science questions in a single language.

- **OpenBookQA** (Mihaylov et al., 2018): This dataset consists of 4,957 samples specifically designed for open-book question answering in a single language.

- **X-CSQA(En)** (Lin et al., 2021): A dataset of 8,888 samples for cross-lingual commonsense question answering in English.

## Evaluation Datasets

The **Evaluation Datasets** are used to assess the performance of the trained models across different tasks and languages:

- **MMWP**: A multilingual dataset with 38,400 samples covering 48 languages. It is used for evaluating mathematical word problems.

- **MGSM** (Shi et al., 2023): With 11 languages and 2,750 samples, this dataset is used for evaluating general language model performance across multiple languages.

- **MSVAMP** (Chen et al., 2023): A dataset for evaluating model performance on visual question-answering tasks, containing 10,000 samples across 10 languages.

- **XNLI** (Conneau et al., 2018): This evaluation dataset covers 15 languages and contains 75,150 samples used for cross-lingual natural language inference tasks.

- **X-CSQA** (Lin et al., 2021): A dataset with 16,000 samples, spanning 16 languages, designed for cross-lingual commonsense question answering.

The **number of languages** (#Lang) and **number of samples** (#Samples) provided in each dataset enable a comprehensive evaluation of models across various linguistic and task-specific domains.

**Tabla 6.** Statistics of involved datasets. "#Langs" denotes the number of languages covered by the dataset, and "#Sample" refers to the total number of samples it contains. "Scenario" denotes the usage scenario of the specific dataset discussed in the methodology section.

|       **Dataset**       | **\# Lang** | **\# Samples** |
|:-----------------------:|:-----------:|:--------------:|
|   **Traing Datasets**   |             |                |
|       MetaMathQA        |      1      |    395,000     |
|   OpenMathInstruct-2    |      1      |   1,000,000    |
|     MGSM8KInstruct      |     10      |     73,599     |
|       MetaMathQA        |      1      |    395,000     |
|        Multi-NLI        |      1      |    392,702     |
|          QASC           |      1      |     8,134      |
|           ARC           |      1      |     3,370      |
|       OpenBookQA        |      1      |     4,957      |
|       X-CSQA(En)        |      1      |     8,888      |
| **Evaluation Datasets** |             |                |
|          MMWP           |     48      |     38,928     |
|          MGSM           |     11      |     2,750      |
|         MSVAMP          |     10      |     10,000     |
|          XNLI           |     15      |     75,150     |
|         X-CSQA          |     16      |     16,000     |

## A.3 Instruction Tuning Prompts

The prompt for code-switched tuning is adapted from Zhang et al. (2023), where the source language, source sentence, and target language are replaced with the relevant translations.\
The prompts for mathematical reasoning tasks, natural language inference, and commonsense question answering are modified from Toshniwal et al. (2024) and Lu et al. (2024), with the general instruction being replaced by the specific problems from the training data.

Translate the following code-switched sentence from `{source_lang}` to pure `{target_lang}`:

`{source_lang}`: `{source_sentence}`\
`{target_lang}`:

Solve the following math problem. Make sure to put the answer (and only the answer) inside boxed{}.

`{instruction}`

I will give you a premise and a hypothesis. Choose the most appropriate relationship from the following options: Entailment, Neutral, Contradiction.

\### Premise:

`{premise}`

\### Hypothesis:

`{hypo}`

\### Answer:

Question:

`{question}`\
Choices:

A. `{choice_A}`

B. `{choice_B}`

C. `{choice_C}`

D. `{choice_D}`

E. `{choice_E}`\
Answer:

## A.4 Training Details

We use mT5 (Xue et al., 2021) as the pre-trained multilingual encoder illustrated in Section §3. We extend LlamaFactory[^6] (Zheng et al., 2024) as the training codebase for our experiments. In the first stage, we train only the additional alignment modules (e.g., the language alignment layer and the boundary tokens) for 3 epochs, with a constant learning rate of 6e-4 and a batch size of 256. In the second stage, we fine-tune all parameters of the LLM for 3 epochs. The learning rate is set to 2e-5, with a warm-up ratio of 0.05 and a cosine learning rate scheduler. We also apply a weight decay of 1e-2 and use a batch size of 128. All experiments are conducted on eight NVIDIA A100 GPUs for a day.

# B Details of Constructing the MMWP Benchmark

## B.1 Language Categorization List.

- **Low-resource languages**: Afrikaans (af), Arabic (ar), Belarusian (be), Bengali (bn), Basque (eu), Gujarati (gu), Hausa (ha), Hindi (hi), Armenian (hy), Icelandic (is), Kannada (kn), Luxembourgish (lb), Macedonian (mk), Malayalam (ml), Marathi (mr), Nepali (ne), Slovak (sk), Swahili (sw), Tamil (ta), Telugu (te), Thai (th).

- **Medium-resource languages**: Bulgarian (bg), Catalan (ca), Czech (cs), Danish (da), Finnish (fi), Croatian (hr), Hungarian (hu), Indonesian (id), Korean (ko), Norwegian Bokmål (nb), Polish (pl), Portuguese (pt), Romanian (ro), Slovenian (sl), Serbian (sr), Ukrainian (uk), Vietnamese (vi).

- **High-resource languages**: German (de), English (en), Spanish (es), French (fr), Italian (it), Japanese (ja), Dutch (nl), Russian (ru), Swedish (sv), Chinese (zh).

## B.2 Selection of Target Languages and Categorization of Resource-level.

In our study, we employ a unique categorization method for selecting target languages, which deviates somewhat from the conventional definitions based on the abundance or scarcity of linguistic resources. This deviation is primarily due to our focus on LLMs. Rather than adhering to traditional classifications, we opt to categorize languages based on their language distribution in the pre-training corpus used for the LLMs.

Guided by the LLaMA2 technical report (Touvron et al., 2023), we define low-resource languages as those that constitute less than 0.005% of the available multilingual datasets. On the other hand, we categorize languages with a representation percentage between 0.005% and 0.1% as medium-resource languages. This categorization method allows us to incorporate a diverse set of languages into our study, thereby enabling a more effective and wide-ranging assessment of the multilingual reasoning capabilities of our model.

Furthermore, we believe that the comprehensive evaluation of multilingual tasks necessitates the inclusion of languages from various families and scripts. This diversity is crucial in understanding the robustness and versatility of our model, as it allows us to evaluate its performance and generalization capabilities across different linguistic contexts and resource levels. By incorporating languages from various families, scripts, and resource categories, we can ensure more comprehensive coverage in our multilingual settings evaluation.

This unique approach to language selection and categorization provides a more nuanced understanding of language resource levels in LLMs while also ensuring a broader and more diverse evaluation of multilingual tasks for LLMs. It also underscores the importance of considering language families and scripts for the assessment of LLMs, thereby contributing to a more comprehensive and inclusive approach to language model development and evaluation.

## B.3 Evaluation and post-calibration process proves the reliability of the MMWP benchmark.

To ensure the quality of the proposed MMWP benchmark, we conducted both human and automatic quality estimation evaluations and further post-calibration processes.

For human evaluation, we employed a total of five annotators for all languages, each responsible for assessing the quality of the translated dataset and performing post-editing and calibration on the translations to ensure consistency and quality across the dataset. These annotators, native Chinese speakers with professional English proficiency and extensive translation and linguistic research experience, reviewed translations for grammatical correctness, fluency, and semantic accuracy. To ensure the reliability of the annotations, we hired two experts who are from Lan-bridge, a qualified institution for translation services[^7], recognized by the ISO[^8], to serve as instructors to assess the reliability of the annotation. They evaluated the randomly selected translations and reannotated those with significant deviations. The evaluation process also included back-translation using GPT-4, which enabled annotators to compare the translated versions with the original texts. Any substantial deviations were addressed through post-editing to align the translations with the original intent and maintain native-like fluency. We carefully measured inter-annotator agreement (IAA) to ensure consistency and reliability. We found substantial agreement among annotators, with a Cohen Kappa score of 0.67 on randomly selected 50 instances, indicating substantial consistency in evaluating the translations. In cases where errors were identified, the annotators worked together to correct them, with an improved IAA of 0.80. The two experts’ determinations are consistent, with an IAA of 0.78.

Complementing the human evaluation, we conducted an automatic evaluation to assess the MMWP dataset’s translation quality quantitatively. In this process, we adopted a back-translation strategy, translating the MMWP problems into the target languages and then back into English. Subsequently, we used widely-accepted automatic evaluation metrics such as BLEU (Papineni et al., 2002), chrF (Popović, 2015), and TER (Olive et al., 2011) to compare the back-translated problems with the original MMWP dataset. These metrics provided a quantitative measure of the overlap between the original and back-translated versions, thereby offering a quantifiable estimation of the translation fidelity. The results of this quality estimation, presented in Table 7, attest to the reliable quality of the translations.

**Tabla 7.** Automatic quality estimation of MMWP using back-translations, showing average results across all languages.

|         | BLEU$\uparrow$ | chrF$\uparrow$ | TER$\downarrow$ |
|:-------:|:--------------:|:--------------:|:---------------:|
| Average |     78.71      |     86.08      |      1.09       |

The results from both the human and automatic evaluations indicate that the dataset is properly constructed and adequately reflects the multilingual nature of the tasks. The human post-editing process ensured that the translated problems maintained their semantic integrity, while the automatic evaluation confirmed that the translations preserved the necessary linguistic structure for multilingual reasoning. Overall, the quality assurance measures we implemented to guarantee that the MMWP benchmark is reliable and effective in evaluating multilingual mathematical reasoning across diverse languages and resource levels.

The prompts for assisting the post-editing process and back-translating the low-resource language texts are presented below.

You are provided with the following:

1.  A low-resource source language text.

2.  The back-translated English text (BT).

3.  The reference English translation (REF).

Your task is to evaluate the quality of the back-translated English text (BT) based on three criteria: **grammatical correctness**, **fluency**, and **semantic accuracy**. Then, propose **three alternative revisions** to improve the BT. For each revision, explain why it was made and how it improves the translation.

1.  **Grammatical Correctness**:\
    Does the back-translated text adhere to standard English grammar rules (e.g., subject-verb agreement, punctuation, tense consistency)?

2.  **Fluency**:\
    Is the back-translated text natural and smooth? Does it sound like it was written by a native speaker?

3.  **Semantic Accuracy**:\
    Does the back-translated text accurately reflect the meaning of the source language text? Are there any discrepancies in the interpretation of the source?

**Provide three revision suggestions for improving the BT:**

1.  Each revision should aim to enhance either grammatical correctness, fluency, or semantic accuracy.

2.  For each suggestion, explain:

    - Why the revision is necessary.

    - Which aspect of translation quality (grammar, fluency, or accuracy) it improves.

You are a translation assistant. Directly translate the mathematical problems from `{source_lang}` to English without additional explanations.

`{source_sentence}`

# C Complete Experimental Results

## C.1 Evaluation Results on MMWP

The complete experimental results on MMWP are shown in Table 8, 9 and 10. These tables present Mono-SFT, Multi-SFT, and Leveraging External Tools or Models category comparison baselines.

**Tabla 8.** Detailed results(Accuracy) of Mono-SFT baselines on the MMWP benchmark across 48 languages.

| **Resource Level** | **Language** | **MAmmoTH-7B** | **WizardMath-7B** | **MetaMath-7B** | **OpenMath2** |
|:--:|:--:|:--:|:--:|:--:|:--:|
| **Low-Resource** | **af** | 20.0 | 25.3 | 32.4 | 48.1 |
|  | **ar** | 14.8 | 18.6 | 23.6 | 33.1 |
|  | **be** | 7.6 | 12.5 | 17.9 | 30.3 |
|  | **bn** | 4.8 | 7.9 | 9.9 | 15.7 |
|  | **eu** | 2.2 | 4.2 | 6.3 | 6.5 |
|  | **gu** | 2.5 | 3.6 | 4.8 | 4.0 |
|  | **ha** | 2.8 | 4.4 | 5.9 | 6.4 |
|  | **hi** | 7.6 | 17.0 | 24.5 | 34.4 |
|  | **hy** | 3.2 | 3.7 | 5.8 | 6.2 |
|  | **is** | 6.5 | 10.1 | 14.3 | 24.3 |
|  | **kn** | 2.1 | 3.6 | 6.4 | 4.8 |
|  | **lb** | 6.8 | 10.9 | 14.6 | 25.7 |
|  | **mk** | 15.9 | 20.1 | 30.1 | 45.6 |
|  | **ml** | 2.5 | 4.2 | 5.2 | 5.7 |
|  | **mr** | 3.0 | 8.4 | 9.6 | 14.9 |
|  | **ne** | 2.0 | 5.8 | 8.4 | 12.3 |
|  | **sk** | 15.0 | 23.7 | 32.3 | 47.5 |
|  | **sw** | 3.1 | 6.5 | 7.5 | 8.4 |
|  | **ta** | 2.6 | 6.0 | 6.3 | 5.4 |
|  | **te** | 2.5 | 4.9 | 6.2 | 4.4 |
|  | **th** | 6.0 | 11.6 | 13.4 | 21.5 |
| **Average** |  | 6.4 | 10.1 | 13.6 | 19.3 |
| **Medium-Resource** | **bg** | 23.2 | 29.1 | 38.7 | 59.7 |
|  | **ca** | 23.9 | 31.4 | 38.8 | 59.1 |
|  | **cs** | 19.7 | 28.7 | 37.5 | 56.7 |
|  | **da** | 23.4 | 30.3 | 40.3 | 55.7 |
|  | **fi** | 17.3 | 22.9 | 32.6 | 51.1 |
|  | **hr** | 19.9 | 24.8 | 35.1 | 54.5 |
|  | **hu** | 13.4 | 24.3 | 31.0 | 47.2 |
|  | **id** | 22.2 | 30.0 | 38.4 | 57.6 |
|  | **ko** | 14.4 | 25.9 | 35.0 | 47.4 |
|  | **nb** | 23.2 | 29.4 | 37.9 | 56.5 |
|  | **pl** | 21.2 | 28.0 | 37.9 | 57.3 |
|  | **pt** | 25.5 | 36.3 | 42.9 | 65.0 |
|  | **ro** | 20.4 | 27.1 | 39.5 | 56.1 |
|  | **sl** | 18.3 | 24.5 | 33.1 | 51.7 |
|  | **sr** | 18.4 | 26.8 | 36.1 | 54.0 |
|  | **uk** | 20.7 | 27.6 | 40.3 | 58.8 |
|  | **vi** | 21.8 | 28.4 | 37.0 | 54.8 |
| **Average** |  | 20.4 | 28.0 | 37.2 | 55.5 |
| **High-resource** | **de** | 25.2 | 32.3 | 41.9 | 62.8 |
|  | **en** | 36.6 | 37.6 | 42.7 | 75.0 |
|  | **es** | 30.2 | 33.8 | 43.4 | 66.7 |
|  | **fr** | 28.7 | 30.8 | 43.5 | 64.4 |
|  | **it** | 28.1 | 34.7 | 42.9 | 63.1 |
|  | **ja** | 19.7 | 28.6 | 40.8 | 56.4 |
|  | **nl** | 23.9 | 31.2 | 41.2 | 58.9 |
|  | **ru** | 23.9 | 30.3 | 41.6 | 60.8 |
|  | **sv** | 24.2 | 30.3 | 42.5 | 59.7 |
|  | **zh** | 20.7 | 32.1 | 37.9 | 59.2 |
| **Average** |  | 26.1 | 32.2 | 41.8 | 62.7 |

**Tabla 9.** Detailed results of Multi-SFT on the MMWP benchmark across 48 languages.

| **Resource Level** | **Language** | **MathOctopus-Parallel-7B** | **MathOctopus-MAPO-DPO-7B** | **MetaMathOctopus-MAPO-DPO-7B** | **QAlign-MetaMathQA-7B** |
|:--:|:--:|:--:|:--:|:--:|:--:|
| **Low-Resource** | **af** | 19.1 | 27.5 | 32.2 | 37.4 |
|  | **ar** | 8.6 | 17.1 | 20.4 | 22.4 |
|  | **be** | 14.6 | 23.2 | 21.8 | 18.1 |
|  | **bn** | 20.1 | 28.1 | 28.4 | 26.6 |
|  | **eu** | 5.6 | 11.2 | 9.4 | 7.5 |
|  | **gu** | 6.2 | 10.9 | 5.3 | 6.3 |
|  | **ha** | 5.6 | 10.6 | 8.8 | 6.3 |
|  | **hi** | 11.1 | 24.3 | 14.7 | 22.2 |
|  | **hy** | 6.0 | 8.4 | 6.0 | 6.5 |
|  | **is** | 10.5 | 18.9 | 15.7 | 17.4 |
|  | **kn** | 5.4 | 13.9 | 5.2 | 5.2 |
|  | **lb** | 8.8 | 15.5 | 18.0 | 14.7 |
|  | **mk** | 15.9 | 26.5 | 32.2 | 31.3 |
|  | **ml** | 5.9 | 9.9 | 7.3 | 6.7 |
|  | **mr** | 6.8 | 15.2 | 5.4 | 11.0 |
|  | **ne** | 6.5 | 16.3 | 9.1 | 10.2 |
|  | **sk** | 16.3 | 25.8 | 33.2 | 31.6 |
|  | **sw** | 21.7 | 26.9 | 30.7 | 30.2 |
|  | **ta** | 4.0 | 11.0 | 4.1 | 6.8 |
|  | **te** | 10.2 | 17.9 | 4.0 | 5.1 |
|  | **th** | 21.3 | 31.1 | 33.8 | 35.4 |
| **Average** |  | 11.0 | 18.6 | 16.4 | 17.1 |
| **Medium-Resource** | **bg** | 20.6 | 30.1 | 36.4 | 39.2 |
|  | **ca** | 22.8 | 32.9 | 41.8 | 41.2 |
|  | **cs** | 18.9 | 28.0 | 36.3 | 39.1 |
|  | **da** | 22.1 | 28.7 | 39.5 | 40.7 |
|  | **fi** | 19.2 | 27.1 | 35.5 | 36.7 |
|  | **hr** | 21.5 | 28.7 | 33.1 | 36.4 |
|  | **hu** | 19.1 | 30.0 | 31.7 | 35.4 |
|  | **id** | 20.8 | 29.1 | 36.5 | 39.5 |
|  | **ko** | 18.0 | 29.4 | 33.4 | 38.5 |
|  | **nb** | 22.6 | 28.5 | 40.3 | 40.3 |
|  | **pl** | 21.0 | 31.3 | 38.1 | 39.2 |
|  | **pt** | 24.3 | 32.4 | 44.1 | 43.8 |
|  | **ro** | 19.6 | 30.7 | 36.4 | 38.8 |
|  | **sl** | 19.2 | 30.2 | 32.6 | 36.3 |
|  | **sr** | 18.5 | 29.4 | 33.3 | 37.2 |
|  | **uk** | 24.8 | 31.2 | 37.7 | 41.1 |
|  | **vi** | 20.4 | 28.9 | 36.6 | 39.3 |
| **Average** |  | 20.8 | 29.8 | 36.7 | 39.0 |
| **High-resource** | **de** | 27.7 | 33.5 | 41.3 | 45.1 |
|  | **en** | 29.6 | 36.0 | 53.5 | 52.9 |
|  | **es** | 25.8 | 33.7 | 46.4 | 46.0 |
|  | **fr** | 24.8 | 32.4 | 44.1 | 43.4 |
|  | **it** | 24.0 | 31.6 | 44.8 | 43.4 |
|  | **ja** | 25.4 | 33.5 | 40.6 | 43.9 |
|  | **nl** | 21.7 | 30.7 | 39.6 | 42.2 |
|  | **ru** | 26.5 | 33.8 | 43.5 | 44.1 |
|  | **sv** | 21.1 | 30.5 | 40.9 | 42.2 |
|  | **zh** | 25.8 | 35.0 | 44.8 | 41.7 |
| **Average** |  | 25.2 | 33.1 | 43.9 | 44.5 |

**Tabla 10.** Detailed results(Accuracy) of Leveraging External Tools or Models baseline and LinguaLIFT on the MMWP benchmark across 48 languages.

| **Resource Level** | **Language** | **Translate-En** | **Langbridge** | **MindMerger** | **AlignIFT-MetaMath** |
|:--:|:--:|:--:|:--:|:--:|:--:|
| **Low-Resource** | **af** | 33.0 | 40.7 | 40.2 | 46.4 |
|  | **ar** | 34.9 | 33.4 | 35.8 | 42.5 |
|  | **be** | 31.2 | 35.1 | 37.1 | 39.6 |
|  | **bn** | 27.7 | 31.9 | 37.6 | 39.5 |
|  | **eu** | 18.8 | 30.2 | 35.4 | 39.6 |
|  | **gu** | 21.6 | 31.8 | 35.8 | 41.0 |
|  | **ha** | 19.9 | 28.0 | 29.1 | 34.4 |
|  | **hi** | 27.4 | 35.3 | 39.4 | 41.7 |
|  | **hy** | 31.5 | 33.3 | 34.7 | 39.7 |
|  | **is** | 31.5 | 34.7 | 35.6 | 41.9 |
|  | **kn** | 26.7 | 29.7 | 33.8 | 37.4 |
|  | **lb** | 29.2 | 36.0 | 37.6 | 43.4 |
|  | **mk** | 35.9 | 40.8 | 40.7 | 43.9 |
|  | **ml** | 19.3 | 28.4 | 37.4 | 40.7 |
|  | **mr** | 25.6 | 30.5 | 34.3 | 37.6 |
|  | **ne** | 25.9 | 31.8 | 37.9 | 40.6 |
|  | **sk** | 29.2 | 36.4 | 38.6 | 42.5 |
|  | **sw** | 27.3 | 41.2 | 36.4 | 47.6 |
|  | **ta** | 24.4 | 27.1 | 35.0 | 40.5 |
|  | **te** | 28.5 | 29.7 | 37.1 | 41.9 |
|  | **th** | 30.3 | 34.9 | 38.7 | 41.7 |
| **Average** |  | 27.6 | 33.4 | 36.6 | 41.2 |
| **Medium-Resource** | **bg** | 38.4 | 40.3 | 40.6 | 46.5 |
|  | **ca** | 35.3 | 37.0 | 41.7 | 45.4 |
|  | **cs** | 37.7 | 37.4 | 44.1 | 47.4 |
|  | **da** | 39.4 | 39.5 | 42.1 | 46.0 |
|  | **fi** | 34.1 | 33.4 | 37.7 | 41.3 |
|  | **hr** | 31.0 | 33.7 | 37.4 | 40.2 |
|  | **hu** | 36.1 | 33.9 | 37.1 | 41.4 |
|  | **id** | 38.0 | 36.9 | 39.8 | 43.0 |
|  | **ko** | 30.7 | 33.1 | 39.5 | 42.6 |
|  | **nb** | 38.2 | 40.0 | 40.9 | 48.1 |
|  | **pl** | 38.8 | 38.5 | 40.6 | 43.8 |
|  | **pt** | 42.4 | 41.7 | 42.9 | 49.1 |
|  | **ro** | 38.0 | 37.5 | 42.3 | 43.7 |
|  | **sl** | 34.4 | 34.8 | 40.4 | 44.1 |
|  | **sr** | 38.1 | 36.7 | 42.4 | 44.8 |
|  | **uk** | 36.3 | 37.9 | 41.9 | 43.7 |
|  | **vi** | 31.6 | 33.9 | 39.8 | 42.0 |
| **Average** |  | 36.4 | 36.8 | 40.7 | 44.3 |
| **High-resource** | **de** | 41.0 | 40.7 | 43.9 | 44.6 |
|  | **en** | 42.5 | 47.1 | 47.8 | 49.7 |
|  | **es** | 40.6 | 39.6 | 43.9 | 49.0 |
|  | **fr** | 40.2 | 39.3 | 42.5 | 45.6 |
|  | **it** | 42.6 | 41.9 | 42.7 | 47.7 |
|  | **ja** | 39.6 | 37.0 | 40.2 | 40.4 |
|  | **nl** | 39.8 | 41.9 | 43.5 | 47.5 |
|  | **ru** | 38.9 | 40.4 | 42.4 | 46.5 |
|  | **sv** | 38.8 | 31.0 | 42.2 | 41.4 |
|  | **zh** | 41.5 | 35.1 | 43.3 | 42.1 |
| **Average** |  | 40.6 | 39.4 | 43.2 | 45.5 |

## C.2 Evaluation Results on MGSM and MMWP

The complete experimental results on MGSM and MSVAMP are shown in Table 11, and 12.

**Tabla 11.** Experimental Results on the MGSM Dataset. "LR." "HR." and "Avg." represent the average performance for low-resource languages, high-resource languages, and all languages, respectively. Following prior work (Yoon et al., 2024), we classify Bn, Te, Th, and Sw as low-resource languages, while the remaining languages are categorized as high-resource. The dagger symbol (†) indicates results obtained using officially released models, the diamond symbol ($\Diamond$) signifies the LoRA implementation, and the double dagger symbol (‡) denotes results from our local implementation.

| **LLaMA-2-7B as base model** | **Bn** | **De** | **En** | **Es** | **Fr** | **Ja** | **Ru** | **Sw** | **Te** | **Th** | **Zh** | **LR.** | **HR.** | **Avg.** |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Baseline |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| *Mono-SFT* |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MAmmoTH^†^ | 3.20 | 33.2 | 47.2 | 34.4 | 32.4 | 23.2 | 30.4 | 2.00 | 1.60 | 6.80 | 26.0 | 3.40 | 32.4 | 21.9 |
| WizardMath^†^ | 4.40 | 37.2 | 51.6 | 43.2 | 39.2 | 27.2 | 36.8 | 4.40 | 1.60 | 5.60 | 28.8 | 4.00 | 37.7 | 25.5 |
| MetaMath^†^ | 5.60 | 56.4 | 68.0 | 54.4 | 56.0 | 34.8 | 52.8 | 5.20 | 2.00 | 5.60 | 39.6 | 4.60 | 51.7 | 34.6 |
| OpenMath2^‡^ | 6.80 | 65.6 | 79.2 | 66.0 | 62.4 | 42.4 | 57.6 | 6.00 | 2.80 | 6.80 | 48.4 | 5.60 | 60.2 | 40.4 |
| *Multi-SFT* |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MathOctopus-Parallel^†^ | 31.2 | 46.0 | 51.6 | 42.8 | 43.6 | 34.0 | 40.0 | 36.4 | 10.8 | 33.6 | 38.8 | 28.0 | 42.4 | 37.2 |
| MathOctopus-MAPO-DPO^†^ | 33.2 | 47.2 | 46.8 | 43.2 | 39.6 | 41.6 | 40.8 | 37.2 | 13.6 | 38.4 | 44.8 | 30.6 | 43.4 | 38.8 |
| MetaMathOctopus-MAPO-DPO^†^ | 35.6 | 52.4 | 70.0 | 58.0 | 51.2 | 46.4 | 56.0 | 42.4 | 2.00 | 44.0 | 55.2 | 31.0 | 55.6 | 46.7 |
| QAlign-MetaMathQA^†^ | 29.6 | 54.0 | 68.4 | 57.6 | 59.2 | 45.2 | 58.4 | 35.6 | 2.40 | 37.6 | 48.8 | 26.3 | 55.9 | 45.2 |
| *Leveraging External Tools or Models* |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Translate-En-MetaMath^‡^ | 49.0 | 59.6 | 65.6 | 59.8 | 56.2 | 49.0 | 48.4 | 37.4 | 34.6 | 37.2 | 47.0 | 39.6 | 55.1 | 49.4 |
| LangBridge-MetaMath^†^ | 41.2 | 53.2 | 62.4 | 58.0 | 51.6 | 39.6 | 55.2 | 39.6 | 28.0 | 44.8 | 43.2 | 38.4 | 51.9 | 47.0 |
| MindMerger-Soft-MetaMath^†^ | 50.4 | 59.6 | 67.2 | 58.4 | 55.6 | 50.0 | 61.6 | 55.2 | 52.8 | 54.2 | 53.2 | 53.1 | 57.9 | 56.2 |
| Translate-En-OpenMath2^‡^ | 52.1 | 69.8 | 76.9 | 71.6 | 63.4 | 57.4 | 54.2 | 38.4 | 35.6 | 39.2 | 53.2 | 41.3 | 63.8 | 55.6 |
| LangBridge-OpenMath2^‡^ | 43.2 | 63.0 | 73.8 | 69.8 | 58.9 | 47.6 | 60.2 | 41.0 | 39.8 | 46.2 | 52.2 | 42.6 | 60.8 | 54.2 |
| MindMerger-Soft-OpenMath2^‡^ | 60.0 | 69.4 | 77.8 | 74.2 | 63.4 | 57.8 | 71.8 | 61.2 | 57.2 | 63.6 | 58.4 | 60.5 | 67.5 | 65.0 |
| Our Methods |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **LinguaLIFT-MetaMathQA** | 54.4 | 62.0 | 64.8 | 63.6 | 56.8 | 50.0 | 60.4 | 55.6 | 54.0 | 57.6 | 54.0 | 55.4 | 58.8 | 57.6 |
| **LinguaLIFT-MetaMathQA^$\Diamond$^** | 50.2 | 57.2 | 61.4 | 60.2 | 52.6 | 45.8 | 55.4 | 51.2 | 49.8 | 53.6 | 50.6 | 51.2 | 54.7 | 53.5 |
| **LinguaLIFT-OpenMathInstruct-2** | 63.0 | 69.6 | 76.0 | 75.2 | 62.8 | 52.0 | 71.6 | 64.2 | 61.2 | 66.6 | 58.0 | 63.8 | 66.5 | 65.5 |
| **LinguaLIFT-OpenMathInstruct-2^$\Diamond$^** | 59.2 | 65.0 | 72.8 | 71.6 | 59.4 | 48.6 | 67.2 | 60.0 | 57.4 | 62.8 | 55.4 | 59.9 | 62.9 | 61.8 |

**Tabla 12.** Experimental Results on the MSVAMP Dataset. "LR." "HR." and "Avg." represent the average performance for low-resource languages, high-resource languages, and all languages, respectively. Following prior work (Yoon et al., 2024), we classify Bn, Th, and Sw as low-resource languages, while the remaining languages are categorized as high-resource. The dagger symbol (†) indicates results obtained using officially released models, while the double dagger symbol (‡) denotes results from our local implementation.

| **LLaMA-2-7B as base model** | **Bn** | **De** | **En** | **Es** | **Fr** | **Ja** | **Ru** | **Sw** | **Th** | **Zh** | **LR.** | **HR.** | **Avg.** |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| baselines |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ***Mono-SFT*** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MAmmoTH^†^ | 6.2 | 44.1 | 39.5 | 45.4 | 42.3 | 34.1 | 38.1 | 5.1 | 8.4 | 37.5 | 6.57 | 40.1 | 30.1 |
| WizardMath^†^ | 16.4 | 49.1 | 56.1 | 50.5 | 50.8 | 45.4 | 44.8 | 13.4 | 17.2 | 43.1 | 15.7 | 48.5 | 38.7 |
| MetaMath^†^ | 12.5 | 63.5 | 67.2 | 64.7 | 64.9 | 54.2 | 58.2 | 16.7 | 16.5 | 55.5 | 15.2 | 61.2 | 47.4 |
| OpenMath2^‡^ | 19.9 | 72.7 | 78.6 | 72.5 | 72.9 | 64.6 | 68.5 | 13.1 | 23.5 | 63.8 | 18.8 | 70.5 | 55.0 |
| ***Multi-SFT*** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MathOctopus-Parallel^†^ | 27.8 | 43.8 | 46.1 | 45.4 | 44.9 | 41.4 | 43.8 | 38.1 | 34.9 | 41.4 | 33.6 | 43.8 | 40.8 |
| MathOctopus-MAPO-DPO^†^ | 48.3 | 57.6 | 59.0 | 59.4 | 60.0 | 58.7 | 58.0 | 54.6 | 54.7 | 59.1 | 52.5 | 58.8 | 56.9 |
| MetaMathOctopus-MAPO-DPO^†^ | 50.3 | 67.7 | 71.5 | 68.7 | 67.5 | 65.4 | 64.7 | 61.3 | 61.7 | 66.0 | 57.8 | 67.4 | 64.5 |
| QAlign-MetaMathQA^†^ | 41.9 | 63.5 | 65.5 | 63.0 | 62.8 | 58.0 | 61.5 | 53.6 | 49.7 | 56.2 | 48.4 | 61.5 | 57.6 |
| ***Leveraging External Tools or Models*** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Translate-En-MetaMath^‡^ | 47.9 | 43.9 | 60.6 | 51.4 | 50.9 | 50.4 | 53.4 | 43.1 | 51.3 | 55.8 | 47.4 | 52.3 | 50.9 |
| LangBridge-MetaMath^†^ | 39.6 | 58.8 | 60.1 | 56.8 | 57.9 | 45.2 | 53.6 | 45.8 | 46.3 | 49.4 | 43.9 | 54.5 | 51.4 |
| MindMerger-Soft-MetaMath^†^ | 52.0 | 61.1 | 64.5 | 62.9 | 60.8 | 59.0 | 58.6 | 54.0 | 52.1 | 57.3 | 52.7 | 60.6 | 58.2 |
| Translate-En-OpenMath2^‡^ | 56.3 | 54.1 | 72.0 | 59.5 | 59.3 | 60.9 | 64.6 | 40.2 | 59.2 | 64.9 | 51.9 | 62.2 | 59.1 |
| LangBridge-OpenMath2^‡^ | 48.0 | 68.6 | 71.9 | 64.8 | 66.6 | 56.5 | 64.4 | 42.3 | 53.5 | 58.3 | 47.9 | 64.4 | 59.5 |
| MindMerger-Soft-OpenMath2^‡^ | 61.2 | 77.2 | 77.1 | 73.6 | 73.5 | 74.1 | 70.8 | 64.8 | 64.1 | 72.8 | 63.4 | 74.2 | 70.9 |
| our methods |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **LinguaLIFT-MetaMathQA** | 54.3 | 62.8 | 65.2 | 59.9 | 62.7 | 57.9 | 59.7 | 56.8 | 57.2 | 56.0 | 56.1 | 60.6 | 59.3 |
| **LinguaLIFT-MetaMathQA^$\Diamond$^** | 49.8 | 57.6 | 61.1 | 55.4 | 58.2 | 53.5 | 55.3 | 52.1 | 52.7 | 51.4 | 51.5 | 56.1 | 54.7 |
| **LinguaLIFT-OpenMathInstruct-2** | 65.2 | 76.4 | 77.9 | 74.2 | 74.5 | 73.5 | 71.4 | 68.6 | 67.9 | 72.2 | 67.2 | 74.3 | 72.2 |
| **LinguaLIFT-OpenMathInstruct-2^$\Diamond$^** | 61.3 | 72.1 | 74.2 | 70.5 | 70.8 | 69.7 | 67.5 | 64.9 | 64.2 | 68.5 | 63.5 | 70/5 | 68.4 |

## C.3 Evaluation Results on XNLI and X-CSQA

The complete experimental results on XNLI and X-CSQA are shown in Table 13, and 14.

**Tabla 13.** Experimental Results on the XNLI Dataset. "LR." "HR." and "Avg." represent the average performance for low-resource languages, high-resource languages, and all languages, respectively. The asterisk symbol (∗) indicates results obtained directly from the published results (Huang et al., 2024).

| **LLaMA-2-7B as base model** | **Ar** | **Bg** | **Sw** | **Th** | **Tr** | **Ur** | **El** | **Hi** | **Zh** | **Ru** | **Vi** | **De** | **Fr** | **Es** | **En** | **LR.** | **HR.** | **Avg.** |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| baselines |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ***Mono-SFT*** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mono-SFT^∗^ | 60.9 | 76.7 | 45.9 | 55.4 | 61.9 | 49.2 | 63.7 | 55.7 | 74.7 | 77.6 | 73.7 | 80.6 | 82.2 | 82.2 | 90.0 | 58.7 | 80.1 | 68.7 |
| ***Multi-SFT*** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Multi-SFT^∗^ | 61.7 | 78.7 | 56.3 | 60.1 | 65.6 | 57.5 | 67.0 | 61.7 | 79.1 | 79.7 | 73.7 | 82.3 | 82.9 | 83.9 | 88.8 | 63.6 | 81.5 | 71.9 |
| QAlign^∗^ | 67.0 | 79.4 | 65.2 | 65.2 | 67.9 | 62.2 | 66.5 | 63.3 | 76.6 | 79.2 | 73.7 | 80.9 | 83.1 | 83.8 | 89.1 | 67.1 | 80.9 | 73.5 |
| ***Leveraging External Tools or Models*** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LangBridge^∗^ | 75.2 | 79.6 | 71.7 | 72.4 | 74.8 | 66.9 | 79.1 | 71.1 | 77.4 | 77.4 | 78.5 | 78.8 | 79.9 | 80.5 | 83.4 | 73.9 | 79.4 | 76.5 |
| Translate-En^∗^ | 68.9 | 80.8 | 65.3 | 69.5 | 74.5 | 61.6 | 79.3 | 68.7 | 74.8 | 76 | 76.7 | 80.6 | 80.4 | 81.4 | 87.4 | 71.1 | 79.6 | 75.1 |
| MindMerger-Soft^∗^ | 76.2 | 82.4 | 66.6 | 71.8 | 75.7 | 69.4 | 78.5 | 74.7 | 80.0 | 80.7 | 80.3 | 83.5 | 83.9 | 84.4 | 88.7 | 74.4 | 83.1 | 78.4 |
| our methods |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **LinguaLIFT** | 78.4 | 83.5 | 75.1 | 77.0 | 77.1 | 72.4 | 82.2 | 75.1 | 80.2 | 80.5 | 80.8 | 83.6 | 84.0 | 84.4 | 89.5 | 77.6 | 83.3 | 80.3 |

**Tabla 14.** Experimental Results on the X-CSQA Dataset. "LR." "HR." and "Avg." represent the average performance for details of the low-resource, high-resource, and all languages, respectively. The asterisk symbol (∗) indicates results obtained directly from the published results (Huang et al., 2024).

| **LLaMA-2-7B as base model** | **Ar** | **De** | **En** | **Es** | **Fr** | **Hi** | **It** | **Ja** | **Nl** | **Pl** | **Pt** | **Ru** | **Sw** | **Ur** | **Vi** | **Zh** | **LR.** | **HR.** | **Avg.** |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---:|---:|:--:|
| baselines |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ***Mono-SFT*** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mono-SFT^∗^ | 32.3 | 61.2 | 76.3 | 64.0 | 63.5 | 32.9 | 56.0 | 49.1 | 57.5 | 50.6 | 61.7 | 56.0 | 24.2 | 25.1 | 50.9 | 56.5 | 28.6 | 58.6 | 51.3 |
| ***Multi-SFT*** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Multi-SFT^∗^ | 28.7 | 49.1 | 67.2 | 54.3 | 52.1 | 32.0 | 50.2 | 38.7 | 45.9 | 45.5 | 51.2 | 46.5 | 27.6 | 29.2 | 38.8 | 43.8 | 29.4 | 48.6 | 43.8 |
| QAlign^∗^ | 36.3 | 58.8 | 75.7 | 63.1 | 60.3 | 37.8 | 58.3 | 49.2 | 56.3 | 51.3 | 59.8 | 56.3 | 35.1 | 32.6 | 50.5 | 54.8 | 35.5 | 57.9 | 52.3 |
| ***Leveraging External Tools or Models*** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| LangBridge^∗^ | 30.6 | 37.4 | 44.4 | 38.4 | 38.2 | 30.6 | 39.1 | 33.9 | 38.4 | 39.8 | 36.3 | 35.1 | 31.8 | 30.5 | 33.3 | 39.8 | 30.9 | 37.8 | 36.1 |
| Translate-En^∗^ | 44.6 | 57.3 | 71.3 | 55.5 | 57.2 | 48.4 | 56.3 | 47.1 | 55.0 | 53.3 | 54.7 | 54.4 | 36.5 | 41.3 | 51.8 | 51.5 | 42.7 | 55.5 | 52.3 |
| MindMerger-Soft^∗^ | 51.4 | 67.0 | 78.1 | 69.1 | 68.1 | 48.4 | 66.8 | 53.9 | 63.8 | 63.3 | 67.1 | 63.7 | 45.5 | 46.2 | 60.6 | 62.9 | 47.9 | 65.4 | 61.0 |
| our methods |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **LinguaLIFT** | 54.1 | 68.7 | 78.3 | 67.9 | 68.7 | 50.6 | 65.7 | 53.8 | 63.5 | 62.9 | 68.2 | 63.5 | 46.7 | 46.8 | 60.4 | 64.1 | 49.6 | 65.5 | 61.5 |

# D Analysis Experiments Details

This section outlines the experimental implementation for the various analysis experiments presented in the main text.

## D.1 Ablation of Two-Stage Training

To demonstrate the necessity of the proposed two-stage instruction tuning approach, we perform an ablation study on the distinct components of the two-stage process: the Language-Align stage and the Task-Transfer stage. This study is conducted on the mathematical reasoning datasets MGSM and MSVAMP to assess the impact of each stage’s removal on reasoning performance across all languages. The detailed results are presented in Tables 15 and 16 for MGSM and MSVAMP, respectively, where “w/o” indicates the absence of the specific stage.

When the Language-Align stage was removed, we fine-tuned both the language-alignment layer and the LLM using only English reasoning data. This resulted in a noticeable decline in performance in low-resource languages, such as a 10.2% performance drop in Bengali (Bn), 13.2% performance drop in Thai (Th), 21.2% performance drop in Telugu (Te), and 7.6% performance drop in Swahili (Sw) on the MGSM dataset and a 14.4% performance drop in Bengali (Bn), 15.7% performance drop in Swahili (Sw) and 19.8% performance drop in Thai (Th) on the MSVAMP dataset. This decline highlights the importance of using code-switched multilingual input as a warm-up strategy before directly applying English-only instruction data. By incorporating this stage, LinguaLIFT can better leverage the multilingual model’s representation space, which contains valuable information from low-resource languages.

When the Task-Transfer stage was removed, we fine-tuned the language alignment layer using only code-switched translation data without updating the LLM’s parameters. After ablating the Task-Transfer stage, the models couldn’t complete reasoning tasks across all languages, resulting in an average accuracy of just 8.49% on the MGSM dataset and 9.74% on the MSVAMP dataset. This demonstrates that reasoning instruction tuning is critical for enabling the LLM to perform reasoning operations effectively.

These results underscore the complementary nature of the two stages: the Language-Align stage enhances the cross-lingual transferability within models, which allows the models to better transfer task-solving capabilities to low-resource languages in the Task-Transfer stage.

**Tabla 15.** Two-stage Training Ablation Experimental Results on the MGSM Dataset. "LR." "HR." and "Avg." represent the average performance for low-resource languages, high-resource languages, and all languages, respectively. Following prior work (Huang et al., 2024), we classify Bn, Te, Th, and Sw as low-resource languages, while the remaining languages are categorized as high-resource.

| **MGSM** | **Bn** | **De** | **En** | **Es** | **Fr** | **Ja** | **Ru** | **Sw** | **Te** | **Th** | **Zh** | **LR.** | **HR.** | **Avg.** |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **LinguaLIFT** | 52.8 | 62.8 | 64.8 | 65.2 | 55.6 | 48.4 | 61.2 | 53.2 | 50.4 | 56.0 | 54.8 | 54.0 | 59.0 | 57.5 |
| *w/o language-align* | 41.6 | 56.8 | 67.2 | 60.4 | 58.8 | 38.4 | 56.4 | 45.6 | 29.2 | 42.8 | 46.4 | 39.8 | 54.9 | 49.4 |
| *w/o task-transfer* | 3.20 | 12.8 | 14.6 | 12.4 | 12.0 | 9.20 | 11.2 | 2.40 | 1.60 | 3.60 | 10.8 | 2.60 | 11.9 | 8.49 |
| Mono-SFT | 6.00 | 58.0 | 64.8 | 55.6 | 55.6 | 37.6 | 52.4 | 4.40 | 0.00 | 4.80 | 40.80 | 3.80 | 52.1 | 34.6 |

**Tabla 16.** Two-stage Training Ablation Experimental Results on the MSVAMP Dataset. "LR." "HR." and "Avg." represent the average performance for low-resource languages, high-resource languages, and all languages, respectively. Following prior work (Huang et al., 2024), we classify Bn, Te, Th, and Sw as low-resource languages, while the remaining languages are categorized as high-resource.

| **MSVAMP** | **Bn** | **De** | **En** | **Es** | **Fr** | **Ja** | **Ru** | **Sw** | **Th** | **Zh** | **LR.** | **HR.** | **Avg.** |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **LinguaLIFT** | 51.3 | 62.8 | 67.2 | 59.9 | 62.7 | 54.3 | 59.7 | 56.5 | 55.2 | 56.0 | 54.3 | 60.4 | 58.6 |
| *w/o language-align* | 36.9 | 58.7 | 63.6 | 58.8 | 62.1 | 53.5 | 55.3 | 40.8 | 35.4 | 53.1 | 37.7 | 57.9 | 51.8 |
| *w/o task-transfer* | 3.8 | 10.4 | 16.5 | 11.7 | 15.3 | 7.7 | 12.0 | 2.2 | 4.1 | 13.7 | 3.37 | 12.5 | 9.74 |
| Mono-SFT | 15.2 | 59.5 | 64.9 | 62.9 | 61.7 | 52.4 | 58.3 | 15.2 | 18.3 | 53.7 | 16.2 | 59.1 | 46.2 |

## D.2 Ablation of Trainable Modules

To further demonstrate the essential design of the two-stage training approach, we conduct an ablation study on the trainable modules during each stage. This study is performed on the mathematical reasoning datasets MGSM and MSVAMP to assess the impact of training each module at the appropriate stage. The detailed results are presented in Tables 17 and 18 for MGSM and MSVAMP, respectively, where “w/o” indicates the absence of a specific operation.

The results reveal that performance on high-resource languages suffers when the LLM is frozen during the Language-Align stage. We attribute this decline to catastrophic forgetting, where the initial training on low-resource languages interferes with the model’s ability to retain its high-resource language capabilities. Specifically, the average performance on high-resource languages decreases by 3.5% on the MGSM dataset and 2.6% on the MSVAMP dataset.

When the LLM training is ablated in the Task-Transfer stage, performance significantly deteriorates across all languages. This indicates that relying solely on the language alignment layer is insufficient for learning high-level reasoning tasks and effectively transferring knowledge to low-resource languages. The average performance across all languages drops to 35.4% on the MGSM dataset and 37.8% on the MSVAMP dataset.

Furthermore, freezing the language alignment layer during the Task-Transfer stage results in a decline in performance on low-resource languages. This suggests that the alignment learned in the language alignment layer is disrupted during continuous training, as the average performance on low-resource languages drops by 9.1% on the MGSM dataset and 5.3% on the MSVAMP dataset.

These findings emphasize the benefit of training the language alignment layer first, followed by the LLM training, to improve performance on both low-resource and high-resource languages.

**Tabla 17.** Trainable Modules Ablation Experimental Results on the MGSM Dataset. "LR." "HR." and "Avg." represent the average performance for low-resource languages, high-resource languages, and all languages, respectively. Following prior work (Huang et al., 2024), we classify Bn, Te, Th, and Sw as low-resource languages, while the remaining languages are categorized as high-resource.

| **MGSM** | **Bn** | **De** | **En** | **Es** | **Fr** | **Ja** | **Ru** | **Sw** | **Te** | **Th** | **Zh** | **LR.** | **HR.** | **Avg.** |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **LinguaLIFT** | 52.8 | 62.8 | 64.8 | 65.2 | 55.6 | 48.4 | 61.2 | 53.2 | 50.4 | 56.0 | 54.8 | 54.0 | 59.0 | 57.5 |
| *w/o freezing LLM in Language-Align* | 50.6 | 59.3 | 62.0 | 60.1 | 52.7 | 45.0 | 58.2 | 52.6 | 48.2 | 54.5 | 51.5 | 51.5 | 55.5 | 54.1 |
| *w/o training LLM in Task-Transfer* | 26.4 | 42.4 | 47.6 | 45.6 | 40.4 | 30.4 | 41.2 | 26.8 | 24.8 | 28.4 | 35.6 | 26.6 | 40.5 | 35.4 |
| *w/o freezing Language Alignment Layer in Task-Transfer* | 46.8 | 60.2 | 66.4 | 59.2 | 57.8 | 46.8 | 58.8 | 45.2 | 39.6 | 48.0 | 49.6 | 44.9 | 57.0 | 52.6 |

**Tabla 18.** Trainable Modules Ablation Experimental Results on the MSVAMP Dataset. "LR." "HR." and "Avg." represent the average performance for low-resource languages, high-resource languages, and all languages, respectively. Following prior work (Huang et al., 2024), we classify Bn, Te, Th, and Sw as low-resource languages, while the remaining languages are categorized as high-resource.

| **MSVAMP** | **Bn** | **De** | **En** | **Es** | **Fr** | **Ja** | **Ru** | **Sw** | **Th** | **Zh** | **LR.** | **HR.** | **Avg.** |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **LinguaLIFT** | 51.3 | 62.8 | 67.2 | 59.9 | 62.7 | 54.3 | 59.7 | 56.5 | 55.2 | 56.0 | 54.3 | 60.4 | 58.6 |
| *w/o freezing LLM in Language-Align* | 50.7 | 59.9 | 65.4 | 57.4 | 55.6 | 50.1 | 57.5 | 55.6 | 54.9 | 53.0 | 53.7 | 57.0 | 56.0 |
| *w/o training LLM in Task-Transfer* | 29.3 | 49.7 | 51.3 | 45.6 | 47.4 | 28.8 | 35.8 | 27.1 | 24.8 | 38.3 | 27.1 | 42.4 | 37.8 |
| *w/o freezing Language Alignment Layer in Task-Transfer* | 42.4 | 60.4 | 65.6 | 60.2 | 60.8 | 53.3 | 58.5 | 53.1 | 51.6 | 50.8 | 49.0 | 58.8 | 55.9 |

## D.3 Impact of Multilingual Encoder Size on Language Alignment

We experimented with various multilingual encoders regarding types and sizes and evaluated the corresponding reasoning performance on the XNLI dataset. Figure 5 illustrates the XNLI performance across five different sizes of the mT5 encoder: 270M (Small), 470M (Base), 820M (Large), 2.2B (XL), and 6.7B (XXL). Additionally, we evaluate five types of multilingual encoders: mT5, mBERT, XLM, XLM-R, and LaBSE. Our findings indicate that performance improves significantly as the encoder size increases from 270M to 2.2B for the mT5 encoder, with diminishing returns observed as the model scales beyond 2.2B to 6.7B. These results suggest that while enlarging the encoder size leads to better performance, there is a point beyond which further scaling offers limited improvements. In terms of encoder types, we observe that encoders with stronger language alignment capabilities yield better performance. Specifically, more powerful multilingual models like LaBSE, which is fine-tuned with parallel corpus, are more effective at enhancing the capabilities of low-resource languages, demonstrating that greater language alignment leads to better transfer and generalization across diverse languages.

## D.4 Adapting to different types and scales of LLMs

LinguaLIFT can be flexibly adapted to different LLMs. To validate this, we conducted experiments on Mistral-7B (Jiang et al., 2023) and the larger Llama-2-13B (Touvron et al., 2023). As shown in Tables 19 and 20, LinguaLIFT outperforms various baselines, with average accuracy improvements of at least 1.6% and 0.9% for low-resource languages based on Llama-2-13B and Mistral-7B, respectively. These results demonstrate LinguaLIFT’s potential for broader applicability across LLMs.

**Tabla 19.** Experimental Results on the MGSM Dataset. "LR." "HR." and "Avg." represent the average performance for low-resource languages, high-resource languages, and all languages, respectively. Following prior work (citation: LLaMA2), we classify Bn, Te, Th, and Sw as low-resource languages, while the remaining languages are categorized as high-resource. The asterisk symbol (∗) denotes results taken directly from the published results of Zhu et al. (2024); Huang et al. (2024).

| **Llama-2-13B as base model** | **Bn** | **De** | **En** | **Es** | **Fr** | **Ja** | **Ru** | **Sw** | **Th** | **Zh** | **LR.** | **HR.** | **Avg.** |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| baseline |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ***Mono-SFT*** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MetaMath^∗^ | 11.6 | 64.8 | 67.2 | 65.2 | 65.2 | 42.8 | 63.6 | 7.60 | 6.40 | 49.2 | 8.53 | 59.7 | 44.4 |
| ***Multi-SFT*** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MathOctopus-Parallel^∗^ | 35.2 | 44.4 | 53.2 | 48.0 | 48.4 | 43.2 | 47.6 | 42.8 | 46.8 | 48.8 | 41.6 | 47.7 | 45.8 |
| QAlign-MetaMathQA^∗^ | 38.4 | 62.0 | 69.2 | 67.2 | 62.4 | 52.4 | 64.4 | 46.0 | 49.6 | 59.2 | 44.7 | 62.4 | 57.1 |
| ***Leveraging External Tools or Models*** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Translate-En^∗^ | 34.8 | 53.6 | 70.8 | 62.4 | 54.0 | 44.4 | 45.6 | 44.4 | 54.0 | 58.0 | 44.4 | 55.5 | 52.2 |
| LangBridge^∗^ | 39.2 | 55.2 | 65.2 | 60.8 | 54.8 | 33.6 | 58.8 | 42.0 | 42.8 | 42.0 | 41.3 | 52.9 | 49.4 |
| MindMerger-Soft^∗^ | 55.2 | 65.2 | 68.8 | 69.6 | 63.6 | 60.0 | 68.0 | 56.4 | 59.6 | 60.4 | 57.1 | 65.1 | 62.7 |
| our methods |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **LinguaLIFT** | 57.6 | 64.4 | 67.8 | 70.4 | 64.0 | 59.8 | 67.2 | 57.8 | 60.8 | 59.4 | 58.7 | 64.7 | 62.9 |

**Tabla 20.** Experimental Results on the MGSM Dataset. "LR." "HR." and "Avg." represent the average performance for low-resource languages, high-resource languages, and all languages, respectively. Following prior work (Huang et al., 2024), we classify Bn, Te, Th, and Sw as low-resource languages, while the remaining languages are categorized as high-resource. The asterisk symbol (∗) denotes results taken directly from the published results of Zhu et al. (2024); Huang et al. (2024).

| **Mistral-7B as base model** | **Bn** | **De** | **En** | **Es** | **Fr** | **Ja** | **Ru** | **Sw** | **Th** | **Zh** | **LR.** | **HR.** | **Avg.** |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| baseline |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ***Mono-SFT*** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MetaMath^∗^ | 38.4 | 70.4 | 78.0 | 71.2 | 70.8 | 50.8 | 67.2 | 16.8 | 34.8 | 57.2 | 30.0 | 66.5 | 55.6 |
| ***Multi-SFT*** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MathOctopus-Parallel^∗^ | 44.0 | 50.0 | 58.4 | 53.2 | 47.2 | 48.0 | 49.6 | 51.6 | 48.8 | 51.6 | 48.1 | 51.1 | 50.2 |
| QAlign-MetaMathQA^∗^ | 45.6 | 59.2 | 65.8 | 63.6 | 59.8 | 49.4 | 60.2 | 55.2 | 51.2 | 57.2 | 50.7 | 59.3 | 56.7 |
| ***Leveraging External Tools or Models*** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Translate-En^∗^ | 54.6 | 50.4 | 69.7 | 58.6 | 56.7 | 57.2 | 64.9 | 47.7 | 58.7 | 63.1 | 53.7 | 60.1 | 58.2 |
| LangBridge^∗^ | 50.0 | 68.4 | 65.6 | 65.6 | 68.8 | 58.4 | 68.4 | 47.2 | 60.0 | 65.6 | 52.4 | 65.8 | 61.8 |
| MindMerger-Soft^∗^ | 57.6 | 69.2 | 79.2 | 71.6 | 69.6 | 57.2 | 68.4 | 53.2 | 59.6 | 68.8 | 56.8 | 69.1 | 65.4 |
| our methods |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **LinguaLIFT** | 58.4 | 67.2 | 77.6 | 72.1 | 70.6 | 56.4 | 66.8 | 58.4 | 63.2 | 68.2 | 60.0 | 68.4 | 65.9 |

## D.5 Language Transferability in Language Families and Writing Systems

**Tabla 21.** The detailed experimental results of LinguaLIFT on MMWP benchmark grouped by language family.

| **Language Family** | **Language** | **MetaMath-7B** | **MathOctopus-Parallel-7B** | **MindMerger** | **AlignIFT-MetaMath** |
|:--:|:--:|:--:|:--:|:--:|:--:|
| **Indo-European-Germanic** | af | 32.4 | 19.1 | 40.2 | 46.4 |
|  | is | 14.3 | 10.5 | 35.6 | 41.9 |
|  | lb | 14.6 | 8.75 | 37.6 | 43.4 |
|  | **Average** | 20.4 | 12.8 | 37.8 | 43.9 |
| **Indo-European-Slavic** | be | 17.9 | 14.6 | 37.1 | 39.6 |
|  | mk | 30.1 | 15.9 | 40.7 | 42.9 |
|  | sk | 32.3 | 16.3 | 38.6 | 42.5 |
|  | **Average** | 26.8 | 15.6 | 38.8 | 41.7 |
| **Indo-European-Indo-Iranian** | bn | 9.86 | 20.1 | 37.6 | 39.5 |
|  | gu | 4.81 | 6.17 | 35.8 | 39.0 |
|  | hi | 24.5 | 11.1 | 39.4 | 41.7 |
|  | mr | 9.62 | 6.78 | 34.3 | 32.6 |
|  | ne | 8.38 | 6.54 | 37.9 | 38.6 |
|  | **Average** | 11.4 | 10.1 | 37.0 | 38.3 |
| **Dravidian** | ta | 6.29 | 3.95 | 35.0 | 36.5 |
|  | te | 6.17 | 10.2 | 37.1 | 36.9 |
|  | kn | 6.41 | 5.43 | 33.8 | 33.4 |
|  | ml | 5.18 | 5.92 | 37.4 | 36.7 |
|  | **Average** | 6.01 | 6.38 | 35.8 | 35.9 |
| **Other** | ar | 23.6 | 8.63 | 40.2 | 42.5 |
|  | eu | 6.29 | 5.55 | 35.4 | 35.6 |
|  | ha | 5.92 | 5.55 | 29.1 | 33.4 |
|  | hy | 5.80 | 6.04 | 34.7 | 39.7 |
|  | sw | 7.52 | 21.7 | 36.4 | 47.6 |
|  | th | 13.4 | 21.3 | 38.7 | 41.7 |
|  | **Average** | 10.4 | 11.5 | 35.8 | 40.1 |

**Tabla 22.** The detailed experimental results of LinguaLIFT on MMWP benchmark grouped by writing systems.

| **Writing Scripts** | **Language** | **MetaMath-7B** | **MathOctopus-Parallel-7B** | **MingMerger** | **AlignIFT-MetaMath** |
|:--:|:--:|:--:|:--:|:--:|:--:|
| **Latin** | af | 32.4 | 19.1 | 40.2 | 46.4 |
|  | eu | 6.29 | 5.55 | 35.4 | 35.6 |
|  | is | 14.3 | 10.5 | 35.6 | 41.9 |
|  | lb | 14.6 | 8.8 | 37.6 | 43.4 |
|  | sw | 7.52 | 21.7 | 36.4 | 47.6 |
|  | sk | 32.3 | 16.3 | 38.6 | 42.5 |
|  | **Average** | 17.9 | 13.7 | 37.3 | 42.9 |
| **Arabic** | ar | 23.6 | 8.63 | 40.2 | 42.5 |
|  | ha | 5.92 | 5.55 | 29.1 | 33.4 |
|  | **Average** | 14.8 | 7.09 | 34.7 | 38.0 |
| **Cyrillic** | be | 17.8 | 14.6 | 37.1 | 39.6 |
|  | mk | 30.1 | 15.9 | 40.7 | 42.9 |
|  | **Average** | 24.0 | 15.3 | 38.9 | 41.3 |
| **Devanagari** | hi | 24.5 | 11.1 | 39.4 | 41.7 |
|  | mr | 9.6 | 6.8 | 34.3 | 32.6 |
|  | ne | 8.4 | 6.5 | 37.9 | 38.6 |
|  | **Average** | 14.2 | 8.1 | 37.2 | 37.6 |
| **Other** | ta | 6.29 | 3.95 | 35.0 | 36.5 |
|  | te | 6.17 | 10.2 | 37.1 | 36.9 |
|  | ml | 5.18 | 5.92 | 37.4 | 36.7 |
|  | bn | 9.86 | 20.1 | 37.6 | 39.5 |
|  | hy | 5.8 | 6.04 | 34.7 | 39.7 |
|  | kn | 6.41 | 5.43 | 33.8 | 33.4 |
|  | gu | 4.81 | 6.17 | 35.8 | 39.0 |
|  | th | 13.4 | 21.3 | 38.7 | 41.7 |
|  | **Average** | 7.6 | 10.8 | 36.3 | 38.3 |

In our analysis, we considered the influence of language families and writing systems on LinguaLIFT’s performance. This approach allowed us to delve into the nuances of language transferability, particularly in low-resource languages that share linguistic characteristics with English.

We organized the MMWP benchmark results according to language family and writing system, following the categorization proposed by Zhang et al. (2023). As depicted in Figure 6, and detailed results of specific low-resource languages reasoning tasks presented in Table 21, LinguaLIFT demonstrates superior performance over all baseline models across different language families. Notably, it excels in languages from the Indo-European-Indo-Iranian, Indo-European-Germanic, and Dravidian families. The Indo-European-Germanic languages, which share a common lineage with English, register stronger performance, underscoring the potential benefits of shared ancestry in cross-lingual transfer. In contrast, languages that diverge significantly from English in terms of syntax, such as those from the Dravidian family, present a more challenging scenario, yet LinguaLIFT still yields impressive results.

In Figure 6 and in Table 22, we also observed similar trends in the context of writing system transfer. LinguaLIFT exhibits exceptional performance in languages that employ shared scripts, particularly those using Latin-based orthographies. This observation underscores the critical role of script similarity in facilitating the transfer of knowledge across languages. The shared orthographic features can provide an additional layer of shared structure that aids in the cross-lingual transfer process, enabling more effective learning and translation across related languages.

In summary, our findings underscore the impressive transferability of LinguaLIFT across different low-resource languages from various language families and writing systems. The method significantly enhances cross-lingual transfer, particularly when linguistic factors such as language family and writing system align with English ones. These results highlight the potential of LinguaLIFT as a robust approach for multilingual processing, capable of leveraging shared linguistic characteristics for improved performance, even in low-resource settings.

## D.6 Analysis of Code-Switch Tuning

In our experiment, we aimed to scrutinize the influence of the part-of-speech (POS) of the substitution words on the multilingual reasoning performance in the context of code-switched tuning. This analysis is essential as it provides insights into the role of various word categories and their combinations in the effectiveness of language model tuning and reasoning capabilities.

**Tabla 23.** Part-of-speech statistics results from the MGSM English test set.

| **Part-of-Speech** | **#Nums** |
|:------------------:|:---------:|
|        Noun        |   84726   |
|        Verb        |   42451   |
|     Adposition     |   37727   |
|      Pronoun       |   25012   |
|     Adjective      |   21790   |
|    Conjunction     |   20764   |
|     Auxiliary      |   16409   |
|       Adverb       |   9150    |

**Figura 11.** Effects of tuning LLM with mixed supervised data. Generally, incorporating multilingual supervised data into LinguaLIFT can achieve a higher ceiling for low-resource language reasoning performance.

Our initial step involved an examination of the distribution of part-of-speech (POS) tags in the reasoning queries, the results of which are tabulated in Table 23. This analysis facilitated a preliminary understanding of the prevalence of different word classes within the reasoning queries, setting the stage for further investigation into their impact on multilingual reasoning.

Subsequently, we categorized the POS combinations into three distinct groups for a more granular analysis. The first group comprised individual POS categories, specifically Verbs, Adpositions, and Pronouns. These categories were selected due to their fundamental role in sentence construction and their potential to affect the meaning and structure of statements significantly.

The second group consisted of syntactic function combinations, including Verb+Adverb, Adjective+Adverb, and Pronoun+Auxiliary+Conjunction combinations. These combinations were chosen based on their syntactic roles and capacity to influence sentence structure and meaning. They are integral to creating complex sentence structures and are often pivotal in conveying nuanced meanings.

The third group focused on key syntactic structures: Subject-Verb (Noun+Verb) and Prepositional Phrases (Adposition+Noun). These structures were selected due to their central role in sentence construction and their potential to encapsulate core semantic information. They form the backbone of many sentence structures and play a crucial role in interpreting a sentence’s meaning.

By analyzing the impact of these POS categories and combinations on the performance of code-switched tuning, we aim to provide a comprehensive understanding of the interplay between syntax and semantics in the context of multilingual reasoning. This analysis will shed light on the importance of different word classes and their combinations in improving the efficacy of multilingual language models.

Further experiments were conducted to investigate the effects of replacing different types of words (e.g., nouns, verbs, prepositions, etc.) on the multilingual reasoning capabilities of LinguaLIFT models. As indicated in Figure 8, we observed that replacing nouns significantly impacted the model’s reasoning performance within the individual POS categories. This finding leads to the hypothesis that reasoning in query sentences may be closely tied to noun-based understanding, possibly due to the central role nouns play in representing key entities and concepts in the task.

Additionally, we found that syntactic structures involving Subject-Verb combinations and Prepositional Phrases performed the best, surpassing all other POS combinations. This suggests that core arguments (subjects and verbs) and their relational elements (prepositions) are crucial for capturing the key relationships in reasoning tasks.

In contrast, the Adjective+Adverb combination showed the weakest performance, indicating that modifiers are less critical for reasoning tasks. This supports the rationale of previous work (Gaur and Saunshi, 2023; Mirzadeh et al., 2024) that proposed transforming mathematical word problems into symbolic reasoning tasks.

Overall, these findings highlight that, for reasoning tasks, word alignment—particularly involving nouns and verbs—is sufficient to enable the model to generalize reasoning abilities across multiple languages.

## D.7 Incorporating multilingual reasoning data in LinguaLIFT

In some settings, we observed that incorporating multilingual reasoning data during the first stage can further enhance task transfer performance in the second stage. This is evidenced by the improved performance compared to the pure English-only reasoning task training. In the mixed supervision setting, we first fine-tune the language alignment layer during the Language-Align stage using multilingual reasoning instruction data GSM8KINSTRUCT (Chen et al., 2023) and code-switching translation instruction data, followed by fine-tuning LLM with English-only instruction data, MetaMathQA. The experimental results on MGSM are presented in Figure 11. We find that incorporating additional multilingual supervision yields an average performance gain of 1.0% on low-resource reasoning tasks and allows the model to achieve comparable performance on high-resource reasoning tasks relative to the vanilla setting.

# E Supplementary Experiments

## E.1 Quantitative Analysis of the Correlation between Multilingual Alignment and Reasoning Performance

To evaluate the cross-lingual alignment capabilities of LLMs, we employ the multilingual sentence retrieval benchmark **Tatoeba** (Artetxe and Schwenk, 2019), a widely used dataset for assessing ML-LMs. The dataset includes up to 1,000 sentences per language, along with their English translations. In our study, we focus on a subset of the original benchmark that aligns with the language categories used in the reasoning test sets MGSM, ensuring consistency with our experimental design.

We adopt the evaluation procedure outlined in XTREME (Hu et al., 2020) to evaluate cross-lingual alignment. For each language pair, we compute the cosine similarity between sentences in the source language and their closest counterparts in the target language. Following the methods of Yang et al. (2021) and Xie et al. (2022), we use top-1 retrieval accuracy as a metric to quantify alignment between the two languages. A higher retrieval accuracy score indicates a stronger alignment.

*(a) MGSM*

*(b) MSVAMP*

**Figura 12.** Performances on MGSM and MSVAMP test set, where languages are sorted by decreasing retrieval accuracy. The trend indicates the mathemtical reasoning performance worsens along with the decreasing cross-lingual alignment degree.

**Tabla 24.** The ablation experiments of the selection of language alignment layer results on the MGSM Dataset. "LR." "HR." and "Avg." represent the average performance for low-resource languages, high-resource languages, and all languages, respectively. Following prior work (Huang et al., 2024), we classify Bn, Te, Th, and Sw as low-resource languages, while the remaining languages are categorized as high-resource.

| **Architecture** | **\# Params** | **Bn** | **De** | **En** | **Es** | **Fr** | **Ja** | **Ru** | **Sw** | **Te** | **Th** | **Zh** | **LR.** | **HR.** | **Avg.** |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Linear | 4 M | 53.1 | 61.6 | 63.9 | 63.8 | 56.6 | 46.0 | 56.6 | 49.0 | 44.4 | 50.6 | 52.2 | 49.3 | 57.2 | 54.4 |
| 2 layers MLP | 10 M | 54.4 | 62.0 | 64.8 | 63.6 | 56.8 | 50.0 | 60.4 | 55.6 | 54.0 | 57.6 | 54.0 | 55.4 | 58.8 | 57.6 |
| 3 layers MLP | 14 M | 53.8 | 62.8 | 66.6 | 56.6 | 58.8 | 50.0 | 59.4 | 53.8 | 53.6 | 58.2 | 53.4 | 54.9 | 58.2 | 57.0 |

**Tabla 25.** Spearman’s rank correlation coefficient ($\rho$) between retrieval accuracy (%) on Tatoeba and reasoning accuracy on MGSM. An asterisk (∗) signifies a statistically significant correlation (p-value \< 0.05).

|    Test set    |     MGSM     |               |    MSVAMP    |               |
|:--------------:|:------------:|:-------------:|:------------:|:-------------:|
| Resource-Level | Low-Resource | High-Resource | Low-Resource | High-Resource |
|     $\rho$     |   0.733^∗^   |   0.778^∗^    |   0.763^∗^   |   0.783^∗^    |

Figure 12 illustrates that the performance of low-resource languages lags significantly behind that of other languages in both mathematical reasoning tasks and the cross-lingual sentence retrieval task. It also shows that a higher alignment degree tends to correlate with improved mathematical reasoning performance. Additionally, we compute the Spearman’s rank correlation between the retrieval accuracy scores and reasoning performance, as presented in Table 25, which reveals a strong correlation between the two. Both the observed trend and the correlation coefficient confirm that the consistency of multilingual mathematical reasoning is closely tied to the cross-lingual alignment degree.

## E.2 Impact of Different Language Alignment Layers

In the main experiment, we utilize two layers of MLP as the language alignment layers to transfer multilingual alignment from the pre-trained multilingual encoder into the LLMs. We also conducted experiments to evaluate the performance of LinguaLIFT when ablating different mapping layers, as shown in Table 24. In contrast to the findings of Yoon et al. (2024), the two layers of MLP used in our main experiment achieved the best performance. In comparison, using a linear layer resulted in lower performance, likely due to its limited capacity to effectively adapt the pre-trained multilingual encoder to LLM, attributed to the smaller number of parameters.

## E.3 LinguaLIFT CoT Examples

We present three examples of CoT reasoning. These examples demonstrate LinguaLIFT’s ability to effectively understand low-resource languages and generate corresponding mathematical reasoning processes using the chain-of-thought reasoning strategy. Figure 13 illustrates a zero-shot CoT example from the MGSM Bengali dataset. The Vanilla Mono-SFT model produces incorrect reasoning, while the LinguaLIFT model generates a correct reasoning process, ultimately leading to the correct answer. Similar trends are observed in the Thai and Swahili, shown in Figures 14 and 15, where the LinguaLIFT model corrects the erroneous reasoning produced by the Mono-SFT model when presented with the same problems.

**Figura 13.** Example of zero-shot CoT generated by MetaMath-7B when prompted with an example from MGSM Bengali. Translation of the input and response is provided in brown text.

**Figura 14.** Example of zero-shot CoT generated by MetaMath-7B when prompted with an example from MGSM Thai. Translation of the input and response is provided in brown text.

**Figura 15.** Example of zero-shot CoT generated by MetaMath-7B when prompted with an example from MGSM Swahili. Translation of the input and response is provided in brown text.

# References

Artetxe et al. (2019). Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2019. Bilingual lexicon induction through unsupervised machine translation. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 5002–5007, Florence, Italy. Association for Computational Linguistics.

Artetxe and Schwenk (2019). Mikel Artetxe and Holger Schwenk. 2019. Massively multilingual sentence embeddings for zero-shot cross-lingual transfer and beyond. Transactions of the Association for Computational Linguistics, 7:597–610.

Chai et al. (2024). Linzheng Chai, Jian Yang, Tao Sun, Hongcheng Guo, Jiaheng Liu, Bing Wang, Xiannian Liang, Jiaqi Bai, Tongliang Li, Qiyao Peng, et al. 2024. xcot: Cross-lingual instruction tuning for cross-lingual chain-of-thought reasoning. arXiv preprint arXiv:2401.07037.

Chen et al. (2024a). Andong Chen, Kehai Chen, Yang Xiang, Xuefeng Bai, Muyun Yang, Tiejun Zhao, et al. 2024a. Llm-based translation inference with iterative bilingual understanding. arXiv preprint arXiv:2410.12543.

Chen et al. (2024b). Andong Chen, Lianzhang Lou, Kehai Chen, Xuefeng Bai, Yang Xiang, Muyun Yang, Tiejun Zhao, and Min Zhang. 2024b. DUAL-REFLECT: Enhancing large language models for reflective translation through dual learning feedback mechanisms. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 693–704, Bangkok, Thailand. Association for Computational Linguistics.

Chen et al. (2023). Nuo Chen, Zinan Zheng, Ning Wu, Linjun Shou, Ming Gong, Yangqiu Song, Dongmei Zhang, and Jia Li. 2023. Breaking language barriers in multilingual mathematical reasoning: Insights and observations. arXiv preprint arXiv:2310.20246.

Chen et al. (2024c). Pinzhen Chen, Shaoxiong Ji, Nikolay Bogoychev, Andrey Kutuzov, Barry Haddow, and Kenneth Heafield. 2024c. Monolingual or multilingual instruction tuning: Which makes a better alpaca. In Findings of the Association for Computational Linguistics: EACL 2024, pages 1347–1356, St. Julian’s, Malta. Association for Computational Linguistics.

Chung et al. (2024). Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2024. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53.

Clark et al. (2018). Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Conneau et al. (2020). Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440–8451, Online. Association for Computational Linguistics.

Conneau et al. (2018). Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. XNLI: Evaluating cross-lingual sentence representations. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2475–2485, Brussels, Belgium. Association for Computational Linguistics.

Devlin et al. (2019). Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Dou et al. (2018). Zi-Yi Dou, Zhi-Hao Zhou, and Shujian Huang. 2018. Unsupervised bilingual lexicon induction via latent variable models. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 621–626, Brussels, Belgium. Association for Computational Linguistics.

Feng et al. (2022). Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. 2022. Language-agnostic BERT sentence embedding. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 878–891, Dublin, Ireland. Association for Computational Linguistics.

FitzGerald et al. (2023). Jack FitzGerald, Christopher Hench, Charith Peris, Scott Mackie, Kay Rottmann, Ana Sanchez, Aaron Nash, Liam Urbach, Vishesh Kakarala, Richa Singh, Swetha Ranganath, Laurie Crist, Misha Britan, Wouter Leeuwis, Gokhan Tur, and Prem Natarajan. 2023. MASSIVE: A 1M-example multilingual natural language understanding dataset with 51 typologically-diverse languages. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4277–4302, Toronto, Canada. Association for Computational Linguistics.

Gaur and Saunshi (2023). Vedant Gaur and Nikunj Saunshi. 2023. Reasoning in large language models through symbolic math word problems. In Findings of the Association for Computational Linguistics: ACL 2023, pages 5889–5903, Toronto, Canada. Association for Computational Linguistics.

Goyal et al. (2022). Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio Ranzato, Francisco Guzmán, and Angela Fan. 2022. The Flores-101 evaluation benchmark for low-resource and multilingual machine translation. Transactions of the Association for Computational Linguistics, 10:522–538.

Hu et al. (2022). Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Hu et al. (2020). Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. 2020. XTREME: A massively multilingual multi-task benchmark for evaluating cross-lingual generalisation. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 4411–4421. PMLR.

Huang et al. (2023). Haoyang Huang, Tianyi Tang, Dongdong Zhang, Xin Zhao, Ting Song, Yan Xia, and Furu Wei. 2023. Not all languages are created equal in LLMs: Improving multilingual capability by cross-lingual-thought prompting. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 12365–12394, Singapore. Association for Computational Linguistics.

Huang et al. (2024). Zixian Huang, Wenhao Zhu, Gong Cheng, Lei Li, and Fei Yuan. 2024. Mindmerger: Efficiently boosting LLM reasoning in non-english languages. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Jiang et al. (2023). Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Khot et al. (2020). Tushar Khot, Peter Clark, Michal Guerquin, Peter Jansen, and Ashish Sabharwal. 2020. Qasc: A dataset for question answering via sentence composition. Proceedings of the AAAI Conference on Artificial Intelligence, 34(05):8082–8090.

Koncel-Kedziorski et al. (2016). Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. 2016. MAWPS: A math word problem repository. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1152–1157, San Diego, California. Association for Computational Linguistics.

Lai and Nissim (2024). Huiyuan Lai and Malvina Nissim. 2024. mCoT: Multilingual instruction tuning for reasoning consistency in language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12012–12026, Bangkok, Thailand. Association for Computational Linguistics.

Lample and Conneau (2019). Guillaume Lample and Alexis Conneau. 2019. Cross-lingual language model pretraining. Advances in Neural Information Processing Systems (NeurIPS).

Lample et al. (2018). Guillaume Lample, Alexis Conneau, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. 2018. Word translation without parallel data. In International Conference on Learning Representations.

Li et al. (2023a). Haonan Li, Fajri Koto, Minghao Wu, Alham Fikri Aji, and Timothy Baldwin. 2023a. Bactrian-x: Multilingual replicable instruction-following models with low-rank adaptation. arXiv preprint arXiv:2305.15011.

Li et al. (2024). Jiahuan Li, Shujian Huang, Aarron Ching, Xinyu Dai, and Jiajun Chen. 2024. PreAlign: Boosting cross-lingual transfer by early establishment of multilingual alignment. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 10246–10257, Miami, Florida, USA. Association for Computational Linguistics.

Li et al. (2023b). Yaoyiran Li, Anna Korhonen, and Ivan Vulić. 2023b. On bilingual lexicon induction with large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9577–9599, Singapore. Association for Computational Linguistics.

Lin et al. (2021). Bill Yuchen Lin, Seyeon Lee, Xiaoyang Qiao, and Xiang Ren. 2021. Common sense beyond English: Evaluating and improving multilingual language models for commonsense reasoning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1274–1287, Online. Association for Computational Linguistics.

Lu et al. (2024). Yinquan Lu, Wenhao Zhu, Lei Li, Yu Qiao, and Fei Yuan. 2024. LLaMAX: Scaling linguistic horizons of LLM by enhancing translation capabilities beyond 100 languages. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 10748–10772, Miami, Florida, USA. Association for Computational Linguistics.

Luo et al. (2023). Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. 2023. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583.

Miao et al. (2020). Shen-yun Miao, Chao-Chun Liang, and Keh-Yih Su. 2020. A diverse corpus for evaluating and developing english math word problem solvers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 975–984.

Mihaylov et al. (2018). Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2381–2391, Brussels, Belgium. Association for Computational Linguistics.

Mirzadeh et al. (2024). Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, and Mehrdad Farajtabar. 2024. Gsm-symbolic: Understanding the limitations of mathematical reasoning in large language models. Preprint, arXiv:2410.05229.

Muennighoff et al. (2023). Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng Xin Yong, Hailey Schoelkopf, Xiangru Tang, Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert Webson, Edward Raff, and Colin Raffel. 2023. Crosslingual generalization through multitask finetuning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15991–16111, Toronto, Canada. Association for Computational Linguistics.

Olive et al. (2011). Joseph Olive, Caitlin Christianson, and John McCary. 2011. Handbook of natural language processing and machine translation: DARPA global autonomous language exploitation. Springer Science & Business Media.

Ouyang et al. (2022). Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc.

Pan et al. (2021). Lin Pan, Chung-Wei Hang, Haode Qi, Abhishek Shah, Saloni Potdar, and Mo Yu. 2021. Multilingual BERT post-pretraining alignment. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 210–219, Online. Association for Computational Linguistics.

Papineni et al. (2002). Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Pfeiffer et al. (2020). Jonas Pfeiffer, Ivan Vulić, Iryna Gurevych, and Sebastian Ruder. 2020. MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7654–7673, Online. Association for Computational Linguistics.

Popović (2015). Maja Popović. 2015. chrF: character n-gram F-score for automatic MT evaluation. In Proceedings of the Tenth Workshop on Statistical Machine Translation, pages 392–395, Lisbon, Portugal. Association for Computational Linguistics.

Qin et al. (2023). Libo Qin, Qiguang Chen, Fuxuan Wei, Shijue Huang, and Wanxiang Che. 2023. Cross-lingual prompting: Improving zero-shot chain-of-thought reasoning across languages. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2695–2709, Singapore. Association for Computational Linguistics.

Shaham et al. (2024). Uri Shaham, Jonathan Herzig, Roee Aharoni, Idan Szpektor, Reut Tsarfaty, and Matan Eyal. 2024. Multilingual instruction tuning with just a pinch of multilinguality. In Findings of the Association for Computational Linguistics: ACL 2024, pages 2304–2317, Bangkok, Thailand. Association for Computational Linguistics.

She et al. (2024). Shuaijie She, Wei Zou, Shujian Huang, Wenhao Zhu, Xiang Liu, Xiang Geng, and Jiajun Chen. 2024. MAPO: Advancing multilingual reasoning through multilingual-alignment-as-preference optimization. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10015–10027, Bangkok, Thailand. Association for Computational Linguistics.

Shi et al. (2023). Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. 2023. Language models are multilingual chain-of-thought reasoners. In The Eleventh International Conference on Learning Representations.

Team et al. (2022). NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang. 2022. No language left behind: Scaling human-centered machine translation. Preprint, arXiv:2207.04672.

Toshniwal et al. (2024). Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, and Igor Gitman. 2024. Openmathinstruct-2: Accelerating ai for math with massive open-source instruction data. arXiv preprint arXiv:2410.01560.

Touvron et al. (2023). Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

van der Maaten and Hinton (2008). Laurens van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-sne. Journal of Machine Learning Research, 9(86):2579–2605.

Wei et al. (2022). Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837. Curran Associates, Inc.

Williams et al. (2018). Adina Williams, Nikita Nangia, and Samuel Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1112–1122, New Orleans, Louisiana. Association for Computational Linguistics.

Xie et al. (2022). Zhihui Xie, Handong Zhao, Tong Yu, and Shuai Li. 2022. Discovering low-rank subspaces for language-agnostic multilingual representations. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5617–5633, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Xue et al. (2021). Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. mT5: A massively multilingual pre-trained text-to-text transformer. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 483–498, Online. Association for Computational Linguistics.

Yang et al. (2020). Zhen Yang, Bojie Hu, Ambyera Han, Shen Huang, and Qi Ju. 2020. CSP:code-switching pre-training for neural machine translation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2624–2636, Online. Association for Computational Linguistics.

Yang et al. (2021). Ziyi Yang, Yinfei Yang, Daniel Cer, and Eric Darve. 2021. A simple and effective method to eliminate the self language bias in multilingual representations. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5825–5832, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Yoon et al. (2024). Dongkeun Yoon, Joel Jang, Sungdong Kim, Seungone Kim, Sheikh Shafayat, and Minjoon Seo. 2024. LangBridge: Multilingual reasoning without multilingual supervision. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7502–7522, Bangkok, Thailand. Association for Computational Linguistics.

Yu et al. (2024). Longhui Yu, Weisen Jiang, Han Shi, Jincheng YU, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2024. Metamath: Bootstrap your own mathematical questions for large language models. In The Twelfth International Conference on Learning Representations.

Yue et al. (2024). Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2024. MAmmoTH: Building math generalist models through hybrid instruction tuning. In The Twelfth International Conference on Learning Representations.

Zhang et al. (2024). Hongbin Zhang, Kehai Chen, Xuefeng Bai, Yang Xiang, and Min Zhang. 2024. Paying more attention to source context: Mitigating unfaithful translations from large language model. In Findings of the Association for Computational Linguistics: ACL 2024, pages 13816–13836, Bangkok, Thailand. Association for Computational Linguistics.

Zhang et al. (2017). Meng Zhang, Yang Liu, Huanbo Luan, and Maosong Sun. 2017. Adversarial training for unsupervised bilingual lexicon induction. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1959–1970, Vancouver, Canada. Association for Computational Linguistics.

Zhang et al. (2023). Ruochen Zhang, Samuel Cahyawijaya, Jan Christian Blaise Cruz, Genta Winata, and Alham Fikri Aji. 2023. Multilingual large language models are not (yet) code-switchers. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12567–12582, Singapore. Association for Computational Linguistics.

Zhao et al. (2023). Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223.

Zheng et al. (2024). Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. 2024. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand. Association for Computational Linguistics.

Zhu et al. (2024). Wenhao Zhu, Shujian Huang, Fei Yuan, Shuaijie She, Jiajun Chen, and Alexandra Birch. 2024. Question translation training for better multilingual reasoning. In Findings of the Association for Computational Linguistics: ACL 2024, pages 8411–8423, Bangkok, Thailand. Association for Computational Linguistics.

[^1]: Our code and data will be released once accepted.

[^2]: Corresponding Author

[^3]: Our code and data will be released once accepted.

[^4]: <https://spacy.io/>

[^5]: <https://github.com/facebookresearch/MUSE>

[^6]: <https://github.com/hiyouga/LLaMA-Factory>

[^7]: Requirements for translation services: <https://www.iso.org/standard/59149.html>.

[^8]: International Organization for Standardization: <https://www.iso.org/home.html>.
