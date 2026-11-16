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
converted: 2026-11-16
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
\end{equation}$$

where $T$ is the target sequence length, $y_{<i}$ represents the preceding tokens, $y_i$ denotes the $i$-th token, and $L$ is the set of target languages. Trainable parameters include $\theta$ (language alignment layer and boundary tokens), while $\psi$ (multilingual encoder) and $\phi$ (LLM) remain frozen.

## 3.2 Task Transfer Stage

Following the language alignment established in the *Language Align Stage*, the LLM is fine-tuned using high-quality English reasoning instruction data while keeping the language alignment layer frozen. This strategy preserves the acquired multilingual alignment and cross-lingual transfer capabilities, unlocking the potential of LLM to transfer strong reasoning capabilities learned from English to low-resource languages. The training minimizes the negative log-likelihood of the response $y$, given the combined instruction context and query input $(Q, \hat{X}_{en})$: 

$$\begin{equation}
    -\underset{\phi}{\arg \min }  \sum_{i=1}^{T} \log p_{\tilde{\theta},\psi ,\phi}\left(y_{i} \mid (Q,\hat{X}_{en}), y_{<i}\right),
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

|                               |            |          |          |          |          |
|:------------------------------|:----------:|:--------:|:--------:|:--------:|:--------:|
| **LLaMA-2-7B as base model**  |  **Use**   | **LR.**  | **MR.**  | **HR.**  | **Avg.** |
|                               | **Multi.** |          |          |          |          |
|                               |            |          |          |          |          |
|                               |            |          |          |          |          |
| MAmmoTH^†^                    |            |   6.36   |   20.4   |   26.1   |   15.5   |
| WizardMath^†^                 |            |   10.1   |   28.0   |   32.2   |   21.0   |
| MetaMath^†^                   |            |   13.6   |   37.2   |   41.8   |   27.8   |
| OpenMath2                     |            |   19.3   |   55.5   | **62.7** |   41.1   |
|                               |            |          |          |          |          |
| MathOctopus-Parallel^†^       |            |   11.0   |   20.8   |   25.2   |   17.4   |
| MathOctopus-MAPO-DPO^†^       |            |   18.6   |   29.8   |   33.1   |   25.6   |
| MetaMathOctopus-MAPO-DPO^†^   |            |   16.4   |   36.7   |   43.9   |   29.3   |
| QAlign-MetaMathQA^†^          |            |   17.1   |   39.0   |   44.5   |   30.6   |
|                               |            |          |          |          |          |
| Translate-En (MetaMath)       |            |   27.6   |   36.4   |   40.6   |   33.4   |
| LangBridge^†^ (MetaMath)      |            |   33.4   |   36.8   |   39.4   |   35.9   |
| MindMerger-Soft^†^ (MetaMath) |            |   36.6   |   40.7   |   43.2   |   39.4   |
|                               |            |   36.7   |   39.8   |   41.5   |   38.8   |
|                               |            |   33.4   |   36.8   |   39.4   |   35.9   |
| Translate-En (OpenMath2)      |            |   43.1   |   54.7   |   61.5   |   51.0   |
| LangBridge (OpenMath2)        |            |   47.9   |   53.8   |   59.2   |   52.3   |
| MindMerger-Soft (OpenMath2)   |            |   49.5   |   54.8   |   61.2   |   53.8   |
|                               |            |          |          |          |          |
|                               |            |   41.2   |   44.3   |   45.5   |   43.2   |
|                               |            |   38.6   |   42.7   |   44.1   |   41.2   |
|                               |            | **55.4** | **61.2** |   62.5   | **58.9** |
|                               |            |   53.5   |   59.0   |   60.8   |   57.0   |

#### LinguaLIFT demonstrates significant improvements across low-resource languages.

Table 1 presents experimental results on the <span class="smallcaps">MMWP</span> test set, grouped by language resource levels. We highlight seven key findings: (1) Mono-SFT models exhibit substantial performance drops in low-resource languages. (2) Multi-SFT models outperform monolingual counterparts in low-resource settings but retain a notable performance gap between low- and high-resource languages. (3) Models using external translation systems or pre-trained multilingual models generalize poorly to unseen languages and underperform in low-resource contexts. (4) Even with supplementary low-resource data, MindMerger fails to improve low-resource language reasoning performance. (5) LinguaLIFT substantially improves low-resource reasoning accuracy, surpassing all baselines without requiring multilingual or parallel data. (6) Integrating advanced English reasoning datasets (e.g., OpenMathInstruct-2) further enhances reasoning performance, underscoring the need to adapt to evolving high-quality English reasoning data.

#### Robust generalization and task versatility.

**Tabla 2.** Results on MGSM and MSVAMP datasets. The usage of abbreviation, symbol and boldface are the same in Table 1.

<table id="tab:abbr-mgsm-msvamp">
<tbody>
<tr>
<td style="text-align: center;"></td>
<td colspan="3" style="text-align: center;"><strong>MGSM</strong></td>
<td colspan="3" style="text-align: center;"><strong>MSVAMP</strong></td>
</tr>
<tr>
<td style="text-align: center;"><strong>LLaMA-2-7B as base model</strong></td>
<td style="text-align: center;"><strong>LR.</strong></td>
<td style="text-align: center;"><strong>HR.</strong></td>
<td style="text-align: center;"><strong>Avg.</strong></td>
<td style="text-align: center;"><strong>LR.</strong></td>
<td style="text-align: center;"><strong>HR.</strong></td>
<td style="text-align: center;"><strong>Avg.</strong></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">3.40</td>
<td style="text-align: center;">32.4</td>
<td style="text-align: center;">21.9</td>
<td style="text-align: center;">6.57</td>
<td style="text-align: center;">40.1</td>
<td style="text-align: center;">30.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">4.00</td>
<td style="text-align: center;">37.7</td>
<td style="text-align: center;">25.5</td>
<td style="text-align: center;">15.7</td>
<td style="text-align: center;">48.5</td>
<td style="text-align: center;">38.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">4.60</td>
<td style="text-align: center;">51.7</td>
<td style="text-align: center;">34.6</td>
<td style="text-align: center;">15.2</td>
<td style="text-align: center;">61.2</td>
<td style="text-align: center;">47.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">5.60</td>
<td style="text-align: center;">60.2</td>
<td style="text-align: center;">40.4</td>
<td style="text-align: center;">18.8</td>
<td style="text-align: center;">70.5</td>
<td style="text-align: center;">55.0</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">28.0</td>
<td style="text-align: center;">42.4</td>
<td style="text-align: center;">37.2</td>
<td style="text-align: center;">33.6</td>
<td style="text-align: center;">43.8</td>
<td style="text-align: center;">40.8</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">30.6</td>
<td style="text-align: center;">43.4</td>
<td style="text-align: center;">38.8</td>
<td style="text-align: center;">52.5</td>
<td style="text-align: center;">58.8</td>
<td style="text-align: center;">56.9</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">31.0</td>
<td style="text-align: center;">55.6</td>
<td style="text-align: center;">46.7</td>
<td style="text-align: center;">57.8</td>
<td style="text-align: center;">67.4</td>
<td style="text-align: center;">64.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">26.3</td>
<td style="text-align: center;">55.9</td>
<td style="text-align: center;">45.2</td>
<td style="text-align: center;">48.4</td>
<td style="text-align: center;">61.5</td>
<td style="text-align: center;">57.6</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">39.6</td>
<td style="text-align: center;">55.1</td>
<td style="text-align: center;">49.4</td>
<td style="text-align: center;">47.4</td>
<td style="text-align: center;">52.3</td>
<td style="text-align: center;">50.9</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">38.4</td>
<td style="text-align: center;">51.9</td>
<td style="text-align: center;">47.0</td>
<td style="text-align: center;">43.9</td>
<td style="text-align: center;">54.5</td>
<td style="text-align: center;">51.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">53.1</td>
<td style="text-align: center;">57.9</td>
<td style="text-align: center;">56.2</td>
<td style="text-align: center;">52.7</td>
<td style="text-align: center;">60.6</td>
<td style="text-align: center;">58.2</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">41.3</td>
<td style="text-align: center;">63.8</td>
<td style="text-align: center;">55.6</td>
<td style="text-align: center;">51.9</td>
<td style="text-align: center;">62.2</td>
<td style="text-align: center;">59.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">42.6</td>
<td style="text-align: center;">60.8</td>
<td style="text-align: center;">54.2</td>
<td style="text-align: center;">47.9</td>
<td style="text-align: center;">64.4</td>
<td style="text-align: center;">59.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">60.5</td>
<td style="text-align: center;"><strong>67.5</strong></td>
<td style="text-align: center;">65.0</td>
<td style="text-align: center;">63.4</td>
<td style="text-align: center;">74.2</td>
<td style="text-align: center;">70.9</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">55.4</td>
<td style="text-align: center;">58.8</td>
<td style="text-align: center;">57.6</td>
<td style="text-align: center;">56.1</td>
<td style="text-align: center;">60.6</td>
<td style="text-align: center;">59.3</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">51.2</td>
<td style="text-align: center;">54.7</td>
<td style="text-align: center;">53.5</td>
<td style="text-align: center;">51.5</td>
<td style="text-align: center;">56.1</td>
<td style="text-align: center;">54.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>63.8</strong></td>
<td style="text-align: center;">66.5</td>
<td style="text-align: center;"><strong>65.5</strong></td>
<td style="text-align: center;"><strong>67.2</strong></td>
<td style="text-align: center;"><strong>74.3</strong></td>
<td style="text-align: center;"><strong>72.2</strong></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">59.9</td>
<td style="text-align: center;">62.9</td>
<td style="text-align: center;">61.8</td>
<td style="text-align: center;">63.5</td>
<td style="text-align: center;">70.5</td>
<td style="text-align: center;">68.4</td>
</tr>
</tbody>
</table>

Experimental results across four multilingual reasoning benchmarks (Tables 2 and 3) demonstrate LinguaLIFT’s robust generalization capabilities in out-of-domain scenarios. The framework achieves significant improvements over existing baselines in mathematical reasoning tasks (MGSM, MSVAMP) for low-resource languages while exhibiting strong adaptability to cross-domain commonsense reasoning challenges (XNLI, X-CSQA). By excelling in numerical computation and understanding semantics, LinguaLIFT demonstrates its task-agnostic generalization ability across various tasks, effectively transferring learned patterns between different reasoning paradigms and thus enhancing performance in low-resource language reasoning. Furthermore, the model enhances reasoning accuracy even in high-resource languages, underscoring its dual capacity to generalize and adapt across tasks and resource levels.

**Tabla 3.** Results on XNLI and X-CSQA datasets. The usage of abbreviation is the same in Table 2. The asterisk symbol () indicates results are taken from the published results of Huang et al. (2024).

<table id="tab:abbr-xnl-xcsqa">
<tbody>
<tr>
<td style="text-align: center;"></td>
<td colspan="3" style="text-align: center;"><strong>XNLI</strong></td>
<td colspan="3" style="text-align: center;"><strong>X-CSQA</strong></td>
</tr>
<tr>
<td style="text-align: center;"><strong>LLaMA-2-7B as base model</strong></td>
<td style="text-align: center;"><strong>LR.</strong></td>
<td style="text-align: center;"><strong>HR.</strong></td>
<td style="text-align: center;"><strong>Avg.</strong></td>
<td style="text-align: center;"><strong>LR.</strong></td>
<td style="text-align: center;"><strong>HR.</strong></td>
<td style="text-align: center;"><strong>Avg.</strong></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">58.7</td>
<td style="text-align: center;">80.1</td>
<td style="text-align: center;">68.7</td>
<td style="text-align: center;">28.6</td>
<td style="text-align: center;">58.6</td>
<td style="text-align: center;">51.3</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">63.6</td>
<td style="text-align: center;">81.5</td>
<td style="text-align: center;">71.9</td>
<td style="text-align: center;">29.4</td>
<td style="text-align: center;">48.6</td>
<td style="text-align: center;">43.8</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">67.1</td>
<td style="text-align: center;">80.9</td>
<td style="text-align: center;">73.5</td>
<td style="text-align: center;">35.5</td>
<td style="text-align: center;">57.9</td>
<td style="text-align: center;">52.3</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">73.9</td>
<td style="text-align: center;">79.4</td>
<td style="text-align: center;">76.5</td>
<td style="text-align: center;">30.9</td>
<td style="text-align: center;">37.8</td>
<td style="text-align: center;">36.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">71.1</td>
<td style="text-align: center;">79.6</td>
<td style="text-align: center;">75.1</td>
<td style="text-align: center;">42.7</td>
<td style="text-align: center;">55.5</td>
<td style="text-align: center;">52.3</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">74.4</td>
<td style="text-align: center;">83.1</td>
<td style="text-align: center;">78.4</td>
<td style="text-align: center;">47.9</td>
<td style="text-align: center;">65.4</td>
<td style="text-align: center;">61.0</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>77.6</strong></td>
<td style="text-align: center;"><strong>83.3</strong></td>
<td style="text-align: center;"><strong>80.3</strong></td>
<td style="text-align: center;"><strong>49.6</strong></td>
<td style="text-align: center;"><strong>65.5</strong></td>
<td style="text-align: center;"><strong>61.5</strong></td>
</tr>
</tbody>
</table>

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

<table id="tab:abbr-various-llms">
<tbody>
<tr>
<td style="text-align: left;"><strong></strong></td>
<td colspan="3" style="text-align: center;"><strong>Mistral-7B</strong></td>
<td colspan="3" style="text-align: center;"><strong>Llama-2-13B</strong></td>
</tr>
<tr>
<td style="text-align: center;"><strong>Methods</strong></td>
<td style="text-align: center;"><strong>LR.</strong></td>
<td style="text-align: center;"><strong>HR.</strong></td>
<td style="text-align: center;"><strong>Avg.</strong></td>
<td style="text-align: center;"><strong>LR.</strong></td>
<td style="text-align: center;"><strong>HR.</strong></td>
<td style="text-align: center;"><strong>Avg.</strong></td>
</tr>
<tr>
<td colspan="7" style="text-align: center;"><span class="smallcaps">baseline</span></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">30.0</td>
<td style="text-align: center;">66.5</td>
<td style="text-align: center;">55.6</td>
<td style="text-align: center;">8.53</td>
<td style="text-align: center;">59.7</td>
<td style="text-align: center;">44.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">48.1</td>
<td style="text-align: center;">51.1</td>
<td style="text-align: center;">50.2</td>
<td style="text-align: center;">41.6</td>
<td style="text-align: center;">47.7</td>
<td style="text-align: center;">45.8</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">50.7</td>
<td style="text-align: center;">59.3</td>
<td style="text-align: center;">56.7</td>
<td style="text-align: center;">44.7</td>
<td style="text-align: center;">62.4</td>
<td style="text-align: center;">57.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">53.7</td>
<td style="text-align: center;">60.1</td>
<td style="text-align: center;">58.2</td>
<td style="text-align: center;">44.4</td>
<td style="text-align: center;">55.5</td>
<td style="text-align: center;">52.2</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">52.4</td>
<td style="text-align: center;">65.8</td>
<td style="text-align: center;">61.8</td>
<td style="text-align: center;">41.3</td>
<td style="text-align: center;">52.9</td>
<td style="text-align: center;">49.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;">56.8</td>
<td style="text-align: center;">69.1</td>
<td style="text-align: center;">65.4</td>
<td style="text-align: center;">57.1</td>
<td style="text-align: center;">65.1</td>
<td style="text-align: center;">62.7</td>
</tr>
<tr>
<td colspan="7" style="text-align: center;"><span class="smallcaps">our methods</span></td>
</tr>
<tr>
<td style="text-align: center;"><strong>LinguaLIFT (MetaMathQA)</strong></td>
<td style="text-align: center;">57.7</td>
<td style="text-align: center;">68.8</td>
<td style="text-align: center;">65.5</td>
<td style="text-align: center;">58.7</td>
<td style="text-align: center;">64.7</td>
<td style="text-align: center;">62.9</td>
</tr>
</tbody>
</table>

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

<table id="dataset-stats">
<thead>
<tr>
<th style="text-align: center;"><strong>Dataset</strong></th>
<th style="text-align: center;"><strong># Lang</strong></th>
<th style="text-align: center;"><strong># Samples</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="3" style="text-align: center;"><strong>Traing Datasets</strong></td>
</tr>
<tr>
<td style="text-align: center;">MetaMathQA</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">395,000</td>
</tr>
<tr>
<td style="text-align: center;">OpenMathInstruct-2</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">1,000,000</td>
</tr>
<tr>
<td style="text-align: center;">MGSM8KInstruct</td>
<td style="text-align: center;">10</td>
<td style="text-align: center;">73,599</td>
</tr>
<tr>
<td style="text-align: center;">MetaMathQA</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">395,000</td>
</tr>
<tr>
<td style="text-align: center;">Multi-NLI</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">392,702</td>
</tr>
<tr>
<td style="text-align: center;">QASC</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">8,134</td>
</tr>
<tr>
<td style="text-align: center;">ARC</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">3,370</td>
</tr>
<tr>
<td style="text-align: center;">OpenBookQA</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">4,957</td>
</tr>
<tr>
<td style="text-align: center;">X-CSQA(En)</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">8,888</td>
</tr>
<tr>
<td colspan="3" style="text-align: center;"><strong>Evaluation Datasets</strong></td>
</tr>
<tr>
<td style="text-align: center;">MMWP</td>
<td style="text-align: center;">48</td>
<td style="text-align: center;">38,928</td>
</tr>
<tr>
<td style="text-align: center;">MGSM</td>
<td style="text-align: center;">11</td>
<td style="text-align: center;">2,750</td>
</tr>
<tr>
<td style="text-align: center;">MSVAMP</td>
<td style="text-align: center;">10</td>
<td style="text-align: center;">10,000</td>
</tr>
<tr>
<td style="text-align: center;">XNLI</td>
<td style="text-align: center;">15</td>
<td style="text-align: center;">75,150</td>
</tr>
<tr>
<td style="text-align: center;">X-CSQA</td>
<td style="text-align: center;">16</td>
<td style="text-align: center;">16,000</td>
</tr>
</tbody>
</table>

## A.3 Instruction Tuning Prompts

The prompt for code-switched tuning is adapted from Zhang et al. (2023), where the source language, source sentence, and target language are replaced with the relevant translations.\
The prompts for mathematical reasoning tasks, natural language inference, and commonsense question answering are modified from Toshniwal et al. (2024) and Lu et al. (2024), with the general instruction being replaced by the specific problems from the training data.

<div class="tcolorbox">

Translate the following code-switched sentence from `{source_lang}` to pure `{target_lang}`:

`{source_lang}`: `{source_sentence}`\
`{target_lang}`:

</div>

<div class="tcolorbox">

Solve the following math problem. Make sure to put the answer (and only the answer) inside boxed{}.

`{instruction}`

</div>

<div class="tcolorbox">

I will give you a premise and a hypothesis. Choose the most appropriate relationship from the following options: Entailment, Neutral, Contradiction.

\### Premise:

`{premise}`

\### Hypothesis:

`{hypo}`

\### Answer:

</div>

<div class="tcolorbox">

Question:

`{question}`\
Choices:

A. `{choice_A}`

B. `{choice_B}`

C. `{choice_C}`

D. `{choice_D}`

E. `{choice_E}`\
Answer:

</div>

## A.4 Training Details

We use mT5 (Xue et al., 2021) as the pre-trained multilingual encoder illustrated in Section §3. We extend LlamaFactory[^6] (Zheng et al., 2024) as the training codebase for our experiments. In the first stage, we train only the additional alignment modules (e.g., the language alignment layer and the boundary tokens) for 3 epochs, with a constant learning rate of 6e-4 and a batch size of 256. In the second stage, we fine-tune all parameters of the LLM for 3 epochs. The learning rate is set to 2e-5, with a warm-up ratio of 0.05 and a cosine learning rate scheduler. We also apply a weight decay of 1e-2 and use a batch size of 128. All experiments are conducted on eight NVIDIA A100 GPUs for a day.

# B Details of Constructing the MMWP Benchmark

## B.1 Language Categorization List.

- **Low-resource languages**: Afrikaans (<span class="smallcaps">af</span>), Arabic (<span class="smallcaps">ar</span>), Belarusian (<span class="smallcaps">be</span>), Bengali (<span class="smallcaps">bn</span>), Basque (<span class="smallcaps">eu</span>), Gujarati (<span class="smallcaps">gu</span>), Hausa (<span class="smallcaps">ha</span>), Hindi (<span class="smallcaps">hi</span>), Armenian (<span class="smallcaps">hy</span>), Icelandic (<span class="smallcaps">is</span>), Kannada (<span class="smallcaps">kn</span>), Luxembourgish (<span class="smallcaps">lb</span>), Macedonian (<span class="smallcaps">mk</span>), Malayalam (<span class="smallcaps">ml</span>), Marathi (<span class="smallcaps">mr</span>), Nepali (<span class="smallcaps">ne</span>), Slovak (<span class="smallcaps">sk</span>), Swahili (<span class="smallcaps">sw</span>), Tamil (<span class="smallcaps">ta</span>), Telugu (<span class="smallcaps">te</span>), Thai (<span class="smallcaps">th</span>).

- **Medium-resource languages**: Bulgarian (<span class="smallcaps">bg</span>), Catalan (<span class="smallcaps">ca</span>), Czech (<span class="smallcaps">cs</span>), Danish (<span class="smallcaps">da</span>), Finnish (<span class="smallcaps">fi</span>), Croatian (<span class="smallcaps">hr</span>), Hungarian (<span class="smallcaps">hu</span>), Indonesian (<span class="smallcaps">id</span>), Korean (<span class="smallcaps">ko</span>), Norwegian Bokmål (<span class="smallcaps">nb</span>), Polish (<span class="smallcaps">pl</span>), Portuguese (<span class="smallcaps">pt</span>), Romanian (<span class="smallcaps">ro</span>), Slovenian (<span class="smallcaps">sl</span>), Serbian (<span class="smallcaps">sr</span>), Ukrainian (<span class="smallcaps">uk</span>), Vietnamese (<span class="smallcaps">vi</span>).

- **High-resource languages**: German (<span class="smallcaps">de</span>), English (<span class="smallcaps">en</span>), Spanish (<span class="smallcaps">es</span>), French (<span class="smallcaps">fr</span>), Italian (<span class="smallcaps">it</span>), Japanese (<span class="smallcaps">ja</span>), Dutch (<span class="smallcaps">nl</span>), Russian (<span class="smallcaps">ru</span>), Swedish (<span class="smallcaps">sv</span>), Chinese (<span class="smallcaps">zh</span>).

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

<div class="tcolorbox">

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

</div>

<div class="tcolorbox">

You are a translation assistant. Directly translate the mathematical problems from `{source_lang}` to English without additional explanations.

`{source_sentence}`

</div>

# C Complete Experimental Results

## C.1 Evaluation Results on MMWP

The complete experimental results on MMWP are shown in Table 8, 9 and 10. These tables present Mono-SFT, Multi-SFT, and Leveraging External Tools or Models category comparison baselines.

**Tabla 8.** Detailed results(Accuracy) of Mono-SFT baselines on the MMWP benchmark across 48 languages.

<table id="tab:MMWP-details-monosft">
<thead>
<tr>
<th style="text-align: center;"><strong>Resource Level</strong></th>
<th style="text-align: center;"><strong>Language</strong></th>
<th style="text-align: center;"><strong>MAmmoTH-7B</strong></th>
<th style="text-align: center;"><strong>WizardMath-7B</strong></th>
<th style="text-align: center;"><strong>MetaMath-7B</strong></th>
<th style="text-align: center;"><strong>OpenMath2</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;"><strong>Low-Resource</strong></td>
<td style="text-align: center;"><strong>af</strong></td>
<td style="text-align: center;">20.0</td>
<td style="text-align: center;">25.3</td>
<td style="text-align: center;">32.4</td>
<td style="text-align: center;">48.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ar</strong></td>
<td style="text-align: center;">14.8</td>
<td style="text-align: center;">18.6</td>
<td style="text-align: center;">23.6</td>
<td style="text-align: center;">33.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>be</strong></td>
<td style="text-align: center;">7.6</td>
<td style="text-align: center;">12.5</td>
<td style="text-align: center;">17.9</td>
<td style="text-align: center;">30.3</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>bn</strong></td>
<td style="text-align: center;">4.8</td>
<td style="text-align: center;">7.9</td>
<td style="text-align: center;">9.9</td>
<td style="text-align: center;">15.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>eu</strong></td>
<td style="text-align: center;">2.2</td>
<td style="text-align: center;">4.2</td>
<td style="text-align: center;">6.3</td>
<td style="text-align: center;">6.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>gu</strong></td>
<td style="text-align: center;">2.5</td>
<td style="text-align: center;">3.6</td>
<td style="text-align: center;">4.8</td>
<td style="text-align: center;">4.0</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ha</strong></td>
<td style="text-align: center;">2.8</td>
<td style="text-align: center;">4.4</td>
<td style="text-align: center;">5.9</td>
<td style="text-align: center;">6.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>hi</strong></td>
<td style="text-align: center;">7.6</td>
<td style="text-align: center;">17.0</td>
<td style="text-align: center;">24.5</td>
<td style="text-align: center;">34.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>hy</strong></td>
<td style="text-align: center;">3.2</td>
<td style="text-align: center;">3.7</td>
<td style="text-align: center;">5.8</td>
<td style="text-align: center;">6.2</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>is</strong></td>
<td style="text-align: center;">6.5</td>
<td style="text-align: center;">10.1</td>
<td style="text-align: center;">14.3</td>
<td style="text-align: center;">24.3</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>kn</strong></td>
<td style="text-align: center;">2.1</td>
<td style="text-align: center;">3.6</td>
<td style="text-align: center;">6.4</td>
<td style="text-align: center;">4.8</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>lb</strong></td>
<td style="text-align: center;">6.8</td>
<td style="text-align: center;">10.9</td>
<td style="text-align: center;">14.6</td>
<td style="text-align: center;">25.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>mk</strong></td>
<td style="text-align: center;">15.9</td>
<td style="text-align: center;">20.1</td>
<td style="text-align: center;">30.1</td>
<td style="text-align: center;">45.6</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ml</strong></td>
<td style="text-align: center;">2.5</td>
<td style="text-align: center;">4.2</td>
<td style="text-align: center;">5.2</td>
<td style="text-align: center;">5.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>mr</strong></td>
<td style="text-align: center;">3.0</td>
<td style="text-align: center;">8.4</td>
<td style="text-align: center;">9.6</td>
<td style="text-align: center;">14.9</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ne</strong></td>
<td style="text-align: center;">2.0</td>
<td style="text-align: center;">5.8</td>
<td style="text-align: center;">8.4</td>
<td style="text-align: center;">12.3</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>sk</strong></td>
<td style="text-align: center;">15.0</td>
<td style="text-align: center;">23.7</td>
<td style="text-align: center;">32.3</td>
<td style="text-align: center;">47.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>sw</strong></td>
<td style="text-align: center;">3.1</td>
<td style="text-align: center;">6.5</td>
<td style="text-align: center;">7.5</td>
<td style="text-align: center;">8.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ta</strong></td>
<td style="text-align: center;">2.6</td>
<td style="text-align: center;">6.0</td>
<td style="text-align: center;">6.3</td>
<td style="text-align: center;">5.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>te</strong></td>
<td style="text-align: center;">2.5</td>
<td style="text-align: center;">4.9</td>
<td style="text-align: center;">6.2</td>
<td style="text-align: center;">4.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>th</strong></td>
<td style="text-align: center;">6.0</td>
<td style="text-align: center;">11.6</td>
<td style="text-align: center;">13.4</td>
<td style="text-align: center;">21.5</td>
</tr>
<tr>
<td colspan="2" style="text-align: center;"><strong>Average</strong></td>
<td style="text-align: center;">6.4</td>
<td style="text-align: center;">10.1</td>
<td style="text-align: center;">13.6</td>
<td style="text-align: center;">19.3</td>
</tr>
<tr>
<td style="text-align: center;"><strong>Medium-Resource</strong></td>
<td style="text-align: center;"><strong>bg</strong></td>
<td style="text-align: center;">23.2</td>
<td style="text-align: center;">29.1</td>
<td style="text-align: center;">38.7</td>
<td style="text-align: center;">59.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ca</strong></td>
<td style="text-align: center;">23.9</td>
<td style="text-align: center;">31.4</td>
<td style="text-align: center;">38.8</td>
<td style="text-align: center;">59.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>cs</strong></td>
<td style="text-align: center;">19.7</td>
<td style="text-align: center;">28.7</td>
<td style="text-align: center;">37.5</td>
<td style="text-align: center;">56.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>da</strong></td>
<td style="text-align: center;">23.4</td>
<td style="text-align: center;">30.3</td>
<td style="text-align: center;">40.3</td>
<td style="text-align: center;">55.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>fi</strong></td>
<td style="text-align: center;">17.3</td>
<td style="text-align: center;">22.9</td>
<td style="text-align: center;">32.6</td>
<td style="text-align: center;">51.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>hr</strong></td>
<td style="text-align: center;">19.9</td>
<td style="text-align: center;">24.8</td>
<td style="text-align: center;">35.1</td>
<td style="text-align: center;">54.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>hu</strong></td>
<td style="text-align: center;">13.4</td>
<td style="text-align: center;">24.3</td>
<td style="text-align: center;">31.0</td>
<td style="text-align: center;">47.2</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>id</strong></td>
<td style="text-align: center;">22.2</td>
<td style="text-align: center;">30.0</td>
<td style="text-align: center;">38.4</td>
<td style="text-align: center;">57.6</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ko</strong></td>
<td style="text-align: center;">14.4</td>
<td style="text-align: center;">25.9</td>
<td style="text-align: center;">35.0</td>
<td style="text-align: center;">47.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>nb</strong></td>
<td style="text-align: center;">23.2</td>
<td style="text-align: center;">29.4</td>
<td style="text-align: center;">37.9</td>
<td style="text-align: center;">56.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>pl</strong></td>
<td style="text-align: center;">21.2</td>
<td style="text-align: center;">28.0</td>
<td style="text-align: center;">37.9</td>
<td style="text-align: center;">57.3</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>pt</strong></td>
<td style="text-align: center;">25.5</td>
<td style="text-align: center;">36.3</td>
<td style="text-align: center;">42.9</td>
<td style="text-align: center;">65.0</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ro</strong></td>
<td style="text-align: center;">20.4</td>
<td style="text-align: center;">27.1</td>
<td style="text-align: center;">39.5</td>
<td style="text-align: center;">56.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>sl</strong></td>
<td style="text-align: center;">18.3</td>
<td style="text-align: center;">24.5</td>
<td style="text-align: center;">33.1</td>
<td style="text-align: center;">51.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>sr</strong></td>
<td style="text-align: center;">18.4</td>
<td style="text-align: center;">26.8</td>
<td style="text-align: center;">36.1</td>
<td style="text-align: center;">54.0</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>uk</strong></td>
<td style="text-align: center;">20.7</td>
<td style="text-align: center;">27.6</td>
<td style="text-align: center;">40.3</td>
<td style="text-align: center;">58.8</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>vi</strong></td>
<td style="text-align: center;">21.8</td>
<td style="text-align: center;">28.4</td>
<td style="text-align: center;">37.0</td>
<td style="text-align: center;">54.8</td>
</tr>
<tr>
<td colspan="2" style="text-align: center;"><strong>Average</strong></td>
<td style="text-align: center;">20.4</td>
<td style="text-align: center;">28.0</td>
<td style="text-align: center;">37.2</td>
<td style="text-align: center;">55.5</td>
</tr>
<tr>
<td style="text-align: center;"><strong>High-resource</strong></td>
<td style="text-align: center;"><strong>de</strong></td>
<td style="text-align: center;">25.2</td>
<td style="text-align: center;">32.3</td>
<td style="text-align: center;">41.9</td>
<td style="text-align: center;">62.8</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>en</strong></td>
<td style="text-align: center;">36.6</td>
<td style="text-align: center;">37.6</td>
<td style="text-align: center;">42.7</td>
<td style="text-align: center;">75.0</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>es</strong></td>
<td style="text-align: center;">30.2</td>
<td style="text-align: center;">33.8</td>
<td style="text-align: center;">43.4</td>
<td style="text-align: center;">66.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>fr</strong></td>
<td style="text-align: center;">28.7</td>
<td style="text-align: center;">30.8</td>
<td style="text-align: center;">43.5</td>
<td style="text-align: center;">64.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>it</strong></td>
<td style="text-align: center;">28.1</td>
<td style="text-align: center;">34.7</td>
<td style="text-align: center;">42.9</td>
<td style="text-align: center;">63.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ja</strong></td>
<td style="text-align: center;">19.7</td>
<td style="text-align: center;">28.6</td>
<td style="text-align: center;">40.8</td>
<td style="text-align: center;">56.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>nl</strong></td>
<td style="text-align: center;">23.9</td>
<td style="text-align: center;">31.2</td>
<td style="text-align: center;">41.2</td>
<td style="text-align: center;">58.9</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ru</strong></td>
<td style="text-align: center;">23.9</td>
<td style="text-align: center;">30.3</td>
<td style="text-align: center;">41.6</td>
<td style="text-align: center;">60.8</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>sv</strong></td>
<td style="text-align: center;">24.2</td>
<td style="text-align: center;">30.3</td>
<td style="text-align: center;">42.5</td>
<td style="text-align: center;">59.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>zh</strong></td>
<td style="text-align: center;">20.7</td>
<td style="text-align: center;">32.1</td>
<td style="text-align: center;">37.9</td>
<td style="text-align: center;">59.2</td>
</tr>
<tr>
<td colspan="2" style="text-align: center;"><strong>Average</strong></td>
<td style="text-align: center;">26.1</td>
<td style="text-align: center;">32.2</td>
<td style="text-align: center;">41.8</td>
<td style="text-align: center;">62.7</td>
</tr>
</tbody>
</table>

**Tabla 9.** Detailed results of Multi-SFT on the MMWP benchmark across 48 languages.

<table id="tab:MMWP-details-multisft">
<thead>
<tr>
<th style="text-align: center;"><strong>Resource Level</strong></th>
<th style="text-align: center;"><strong>Language</strong></th>
<th style="text-align: center;"><strong>MathOctopus-Parallel-7B</strong></th>
<th style="text-align: center;"><strong>MathOctopus-MAPO-DPO-7B</strong></th>
<th style="text-align: center;"><strong>MetaMathOctopus-MAPO-DPO-7B</strong></th>
<th style="text-align: center;"><strong>QAlign-MetaMathQA-7B</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;"><strong>Low-Resource</strong></td>
<td style="text-align: center;"><strong>af</strong></td>
<td style="text-align: center;">19.1</td>
<td style="text-align: center;">27.5</td>
<td style="text-align: center;">32.2</td>
<td style="text-align: center;">37.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ar</strong></td>
<td style="text-align: center;">8.6</td>
<td style="text-align: center;">17.1</td>
<td style="text-align: center;">20.4</td>
<td style="text-align: center;">22.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>be</strong></td>
<td style="text-align: center;">14.6</td>
<td style="text-align: center;">23.2</td>
<td style="text-align: center;">21.8</td>
<td style="text-align: center;">18.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>bn</strong></td>
<td style="text-align: center;">20.1</td>
<td style="text-align: center;">28.1</td>
<td style="text-align: center;">28.4</td>
<td style="text-align: center;">26.6</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>eu</strong></td>
<td style="text-align: center;">5.6</td>
<td style="text-align: center;">11.2</td>
<td style="text-align: center;">9.4</td>
<td style="text-align: center;">7.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>gu</strong></td>
<td style="text-align: center;">6.2</td>
<td style="text-align: center;">10.9</td>
<td style="text-align: center;">5.3</td>
<td style="text-align: center;">6.3</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ha</strong></td>
<td style="text-align: center;">5.6</td>
<td style="text-align: center;">10.6</td>
<td style="text-align: center;">8.8</td>
<td style="text-align: center;">6.3</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>hi</strong></td>
<td style="text-align: center;">11.1</td>
<td style="text-align: center;">24.3</td>
<td style="text-align: center;">14.7</td>
<td style="text-align: center;">22.2</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>hy</strong></td>
<td style="text-align: center;">6.0</td>
<td style="text-align: center;">8.4</td>
<td style="text-align: center;">6.0</td>
<td style="text-align: center;">6.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>is</strong></td>
<td style="text-align: center;">10.5</td>
<td style="text-align: center;">18.9</td>
<td style="text-align: center;">15.7</td>
<td style="text-align: center;">17.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>kn</strong></td>
<td style="text-align: center;">5.4</td>
<td style="text-align: center;">13.9</td>
<td style="text-align: center;">5.2</td>
<td style="text-align: center;">5.2</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>lb</strong></td>
<td style="text-align: center;">8.8</td>
<td style="text-align: center;">15.5</td>
<td style="text-align: center;">18.0</td>
<td style="text-align: center;">14.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>mk</strong></td>
<td style="text-align: center;">15.9</td>
<td style="text-align: center;">26.5</td>
<td style="text-align: center;">32.2</td>
<td style="text-align: center;">31.3</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ml</strong></td>
<td style="text-align: center;">5.9</td>
<td style="text-align: center;">9.9</td>
<td style="text-align: center;">7.3</td>
<td style="text-align: center;">6.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>mr</strong></td>
<td style="text-align: center;">6.8</td>
<td style="text-align: center;">15.2</td>
<td style="text-align: center;">5.4</td>
<td style="text-align: center;">11.0</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ne</strong></td>
<td style="text-align: center;">6.5</td>
<td style="text-align: center;">16.3</td>
<td style="text-align: center;">9.1</td>
<td style="text-align: center;">10.2</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>sk</strong></td>
<td style="text-align: center;">16.3</td>
<td style="text-align: center;">25.8</td>
<td style="text-align: center;">33.2</td>
<td style="text-align: center;">31.6</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>sw</strong></td>
<td style="text-align: center;">21.7</td>
<td style="text-align: center;">26.9</td>
<td style="text-align: center;">30.7</td>
<td style="text-align: center;">30.2</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ta</strong></td>
<td style="text-align: center;">4.0</td>
<td style="text-align: center;">11.0</td>
<td style="text-align: center;">4.1</td>
<td style="text-align: center;">6.8</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>te</strong></td>
<td style="text-align: center;">10.2</td>
<td style="text-align: center;">17.9</td>
<td style="text-align: center;">4.0</td>
<td style="text-align: center;">5.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>th</strong></td>
<td style="text-align: center;">21.3</td>
<td style="text-align: center;">31.1</td>
<td style="text-align: center;">33.8</td>
<td style="text-align: center;">35.4</td>
</tr>
<tr>
<td colspan="2" style="text-align: center;"><strong>Average</strong></td>
<td style="text-align: center;">11.0</td>
<td style="text-align: center;">18.6</td>
<td style="text-align: center;">16.4</td>
<td style="text-align: center;">17.1</td>
</tr>
<tr>
<td style="text-align: center;"><strong>Medium-Resource</strong></td>
<td style="text-align: center;"><strong>bg</strong></td>
<td style="text-align: center;">20.6</td>
<td style="text-align: center;">30.1</td>
<td style="text-align: center;">36.4</td>
<td style="text-align: center;">39.2</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ca</strong></td>
<td style="text-align: center;">22.8</td>
<td style="text-align: center;">32.9</td>
<td style="text-align: center;">41.8</td>
<td style="text-align: center;">41.2</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>cs</strong></td>
<td style="text-align: center;">18.9</td>
<td style="text-align: center;">28.0</td>
<td style="text-align: center;">36.3</td>
<td style="text-align: center;">39.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>da</strong></td>
<td style="text-align: center;">22.1</td>
<td style="text-align: center;">28.7</td>
<td style="text-align: center;">39.5</td>
<td style="text-align: center;">40.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>fi</strong></td>
<td style="text-align: center;">19.2</td>
<td style="text-align: center;">27.1</td>
<td style="text-align: center;">35.5</td>
<td style="text-align: center;">36.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>hr</strong></td>
<td style="text-align: center;">21.5</td>
<td style="text-align: center;">28.7</td>
<td style="text-align: center;">33.1</td>
<td style="text-align: center;">36.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>hu</strong></td>
<td style="text-align: center;">19.1</td>
<td style="text-align: center;">30.0</td>
<td style="text-align: center;">31.7</td>
<td style="text-align: center;">35.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>id</strong></td>
<td style="text-align: center;">20.8</td>
<td style="text-align: center;">29.1</td>
<td style="text-align: center;">36.5</td>
<td style="text-align: center;">39.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ko</strong></td>
<td style="text-align: center;">18.0</td>
<td style="text-align: center;">29.4</td>
<td style="text-align: center;">33.4</td>
<td style="text-align: center;">38.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>nb</strong></td>
<td style="text-align: center;">22.6</td>
<td style="text-align: center;">28.5</td>
<td style="text-align: center;">40.3</td>
<td style="text-align: center;">40.3</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>pl</strong></td>
<td style="text-align: center;">21.0</td>
<td style="text-align: center;">31.3</td>
<td style="text-align: center;">38.1</td>
<td style="text-align: center;">39.2</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>pt</strong></td>
<td style="text-align: center;">24.3</td>
<td style="text-align: center;">32.4</td>
<td style="text-align: center;">44.1</td>
<td style="text-align: center;">43.8</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ro</strong></td>
<td style="text-align: center;">19.6</td>
<td style="text-align: center;">30.7</td>
<td style="text-align: center;">36.4</td>
<td style="text-align: center;">38.8</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>sl</strong></td>
<td style="text-align: center;">19.2</td>
<td style="text-align: center;">30.2</td>
<td style="text-align: center;">32.6</td>
<td style="text-align: center;">36.3</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>sr</strong></td>
<td style="text-align: center;">18.5</td>
<td style="text-align: center;">29.4</td>
<td style="text-align: center;">33.3</td>
<td style="text-align: center;">37.2</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>uk</strong></td>
<td style="text-align: center;">24.8</td>
<td style="text-align: center;">31.2</td>
<td style="text-align: center;">37.7</td>
<td style="text-align: center;">41.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>vi</strong></td>
<td style="text-align: center;">20.4</td>
<td style="text-align: center;">28.9</td>
<td style="text-align: center;">36.6</td>
<td style="text-align: center;">39.3</td>
</tr>
<tr>
<td colspan="2" style="text-align: center;"><strong>Average</strong></td>
<td style="text-align: center;">20.8</td>
<td style="text-align: center;">29.8</td>
<td style="text-align: center;">36.7</td>
<td style="text-align: center;">39.0</td>
</tr>
<tr>
<td style="text-align: center;"><strong>High-resource</strong></td>
<td style="text-align: center;"><strong>de</strong></td>
<td style="text-align: center;">27.7</td>
<td style="text-align: center;">33.5</td>
<td style="text-align: center;">41.3</td>
<td style="text-align: center;">45.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>en</strong></td>
<td style="text-align: center;">29.6</td>
<td style="text-align: center;">36.0</td>
<td style="text-align: center;">53.5</td>
<td style="text-align: center;">52.9</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>es</strong></td>
<td style="text-align: center;">25.8</td>
<td style="text-align: center;">33.7</td>
<td style="text-align: center;">46.4</td>
<td style="text-align: center;">46.0</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>fr</strong></td>
<td style="text-align: center;">24.8</td>
<td style="text-align: center;">32.4</td>
<td style="text-align: center;">44.1</td>
<td style="text-align: center;">43.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>it</strong></td>
<td style="text-align: center;">24.0</td>
<td style="text-align: center;">31.6</td>
<td style="text-align: center;">44.8</td>
<td style="text-align: center;">43.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ja</strong></td>
<td style="text-align: center;">25.4</td>
<td style="text-align: center;">33.5</td>
<td style="text-align: center;">40.6</td>
<td style="text-align: center;">43.9</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>nl</strong></td>
<td style="text-align: center;">21.7</td>
<td style="text-align: center;">30.7</td>
<td style="text-align: center;">39.6</td>
<td style="text-align: center;">42.2</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ru</strong></td>
<td style="text-align: center;">26.5</td>
<td style="text-align: center;">33.8</td>
<td style="text-align: center;">43.5</td>
<td style="text-align: center;">44.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>sv</strong></td>
<td style="text-align: center;">21.1</td>
<td style="text-align: center;">30.5</td>
<td style="text-align: center;">40.9</td>
<td style="text-align: center;">42.2</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>zh</strong></td>
<td style="text-align: center;">25.8</td>
<td style="text-align: center;">35.0</td>
<td style="text-align: center;">44.8</td>
<td style="text-align: center;">41.7</td>
</tr>
<tr>
<td colspan="2" style="text-align: center;"><strong>Average</strong></td>
<td style="text-align: center;">25.2</td>
<td style="text-align: center;">33.1</td>
<td style="text-align: center;">43.9</td>
<td style="text-align: center;">44.5</td>
</tr>
</tbody>
</table>

**Tabla 10.** Detailed results(Accuracy) of Leveraging External Tools or Models baseline and LinguaLIFT on the MMWP benchmark across 48 languages.

<table id="tab:MMWP-details-external">
<thead>
<tr>
<th style="text-align: center;"><strong>Resource Level</strong></th>
<th style="text-align: center;"><strong>Language</strong></th>
<th style="text-align: center;"><strong>Translate-En</strong></th>
<th style="text-align: center;"><strong>Langbridge</strong></th>
<th style="text-align: center;"><strong>MindMerger</strong></th>
<th style="text-align: center;"><strong>AlignIFT-MetaMath</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;"><strong>Low-Resource</strong></td>
<td style="text-align: center;"><strong>af</strong></td>
<td style="text-align: center;">33.0</td>
<td style="text-align: center;">40.7</td>
<td style="text-align: center;">40.2</td>
<td style="text-align: center;">46.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ar</strong></td>
<td style="text-align: center;">34.9</td>
<td style="text-align: center;">33.4</td>
<td style="text-align: center;">35.8</td>
<td style="text-align: center;">42.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>be</strong></td>
<td style="text-align: center;">31.2</td>
<td style="text-align: center;">35.1</td>
<td style="text-align: center;">37.1</td>
<td style="text-align: center;">39.6</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>bn</strong></td>
<td style="text-align: center;">27.7</td>
<td style="text-align: center;">31.9</td>
<td style="text-align: center;">37.6</td>
<td style="text-align: center;">39.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>eu</strong></td>
<td style="text-align: center;">18.8</td>
<td style="text-align: center;">30.2</td>
<td style="text-align: center;">35.4</td>
<td style="text-align: center;">39.6</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>gu</strong></td>
<td style="text-align: center;">21.6</td>
<td style="text-align: center;">31.8</td>
<td style="text-align: center;">35.8</td>
<td style="text-align: center;">41.0</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ha</strong></td>
<td style="text-align: center;">19.9</td>
<td style="text-align: center;">28.0</td>
<td style="text-align: center;">29.1</td>
<td style="text-align: center;">34.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>hi</strong></td>
<td style="text-align: center;">27.4</td>
<td style="text-align: center;">35.3</td>
<td style="text-align: center;">39.4</td>
<td style="text-align: center;">41.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>hy</strong></td>
<td style="text-align: center;">31.5</td>
<td style="text-align: center;">33.3</td>
<td style="text-align: center;">34.7</td>
<td style="text-align: center;">39.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>is</strong></td>
<td style="text-align: center;">31.5</td>
<td style="text-align: center;">34.7</td>
<td style="text-align: center;">35.6</td>
<td style="text-align: center;">41.9</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>kn</strong></td>
<td style="text-align: center;">26.7</td>
<td style="text-align: center;">29.7</td>
<td style="text-align: center;">33.8</td>
<td style="text-align: center;">37.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>lb</strong></td>
<td style="text-align: center;">29.2</td>
<td style="text-align: center;">36.0</td>
<td style="text-align: center;">37.6</td>
<td style="text-align: center;">43.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>mk</strong></td>
<td style="text-align: center;">35.9</td>
<td style="text-align: center;">40.8</td>
<td style="text-align: center;">40.7</td>
<td style="text-align: center;">43.9</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ml</strong></td>
<td style="text-align: center;">19.3</td>
<td style="text-align: center;">28.4</td>
<td style="text-align: center;">37.4</td>
<td style="text-align: center;">40.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>mr</strong></td>
<td style="text-align: center;">25.6</td>
<td style="text-align: center;">30.5</td>
<td style="text-align: center;">34.3</td>
<td style="text-align: center;">37.6</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ne</strong></td>
<td style="text-align: center;">25.9</td>
<td style="text-align: center;">31.8</td>
<td style="text-align: center;">37.9</td>
<td style="text-align: center;">40.6</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>sk</strong></td>
<td style="text-align: center;">29.2</td>
<td style="text-align: center;">36.4</td>
<td style="text-align: center;">38.6</td>
<td style="text-align: center;">42.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>sw</strong></td>
<td style="text-align: center;">27.3</td>
<td style="text-align: center;">41.2</td>
<td style="text-align: center;">36.4</td>
<td style="text-align: center;">47.6</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ta</strong></td>
<td style="text-align: center;">24.4</td>
<td style="text-align: center;">27.1</td>
<td style="text-align: center;">35.0</td>
<td style="text-align: center;">40.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>te</strong></td>
<td style="text-align: center;">28.5</td>
<td style="text-align: center;">29.7</td>
<td style="text-align: center;">37.1</td>
<td style="text-align: center;">41.9</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>th</strong></td>
<td style="text-align: center;">30.3</td>
<td style="text-align: center;">34.9</td>
<td style="text-align: center;">38.7</td>
<td style="text-align: center;">41.7</td>
</tr>
<tr>
<td colspan="2" style="text-align: center;"><strong>Average</strong></td>
<td style="text-align: center;">27.6</td>
<td style="text-align: center;">33.4</td>
<td style="text-align: center;">36.6</td>
<td style="text-align: center;">41.2</td>
</tr>
<tr>
<td style="text-align: center;"><strong>Medium-Resource</strong></td>
<td style="text-align: center;"><strong>bg</strong></td>
<td style="text-align: center;">38.4</td>
<td style="text-align: center;">40.3</td>
<td style="text-align: center;">40.6</td>
<td style="text-align: center;">46.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ca</strong></td>
<td style="text-align: center;">35.3</td>
<td style="text-align: center;">37.0</td>
<td style="text-align: center;">41.7</td>
<td style="text-align: center;">45.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>cs</strong></td>
<td style="text-align: center;">37.7</td>
<td style="text-align: center;">37.4</td>
<td style="text-align: center;">44.1</td>
<td style="text-align: center;">47.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>da</strong></td>
<td style="text-align: center;">39.4</td>
<td style="text-align: center;">39.5</td>
<td style="text-align: center;">42.1</td>
<td style="text-align: center;">46.0</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>fi</strong></td>
<td style="text-align: center;">34.1</td>
<td style="text-align: center;">33.4</td>
<td style="text-align: center;">37.7</td>
<td style="text-align: center;">41.3</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>hr</strong></td>
<td style="text-align: center;">31.0</td>
<td style="text-align: center;">33.7</td>
<td style="text-align: center;">37.4</td>
<td style="text-align: center;">40.2</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>hu</strong></td>
<td style="text-align: center;">36.1</td>
<td style="text-align: center;">33.9</td>
<td style="text-align: center;">37.1</td>
<td style="text-align: center;">41.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>id</strong></td>
<td style="text-align: center;">38.0</td>
<td style="text-align: center;">36.9</td>
<td style="text-align: center;">39.8</td>
<td style="text-align: center;">43.0</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ko</strong></td>
<td style="text-align: center;">30.7</td>
<td style="text-align: center;">33.1</td>
<td style="text-align: center;">39.5</td>
<td style="text-align: center;">42.6</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>nb</strong></td>
<td style="text-align: center;">38.2</td>
<td style="text-align: center;">40.0</td>
<td style="text-align: center;">40.9</td>
<td style="text-align: center;">48.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>pl</strong></td>
<td style="text-align: center;">38.8</td>
<td style="text-align: center;">38.5</td>
<td style="text-align: center;">40.6</td>
<td style="text-align: center;">43.8</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>pt</strong></td>
<td style="text-align: center;">42.4</td>
<td style="text-align: center;">41.7</td>
<td style="text-align: center;">42.9</td>
<td style="text-align: center;">49.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ro</strong></td>
<td style="text-align: center;">38.0</td>
<td style="text-align: center;">37.5</td>
<td style="text-align: center;">42.3</td>
<td style="text-align: center;">43.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>sl</strong></td>
<td style="text-align: center;">34.4</td>
<td style="text-align: center;">34.8</td>
<td style="text-align: center;">40.4</td>
<td style="text-align: center;">44.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>sr</strong></td>
<td style="text-align: center;">38.1</td>
<td style="text-align: center;">36.7</td>
<td style="text-align: center;">42.4</td>
<td style="text-align: center;">44.8</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>uk</strong></td>
<td style="text-align: center;">36.3</td>
<td style="text-align: center;">37.9</td>
<td style="text-align: center;">41.9</td>
<td style="text-align: center;">43.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>vi</strong></td>
<td style="text-align: center;">31.6</td>
<td style="text-align: center;">33.9</td>
<td style="text-align: center;">39.8</td>
<td style="text-align: center;">42.0</td>
</tr>
<tr>
<td colspan="2" style="text-align: center;"><strong>Average</strong></td>
<td style="text-align: center;">36.4</td>
<td style="text-align: center;">36.8</td>
<td style="text-align: center;">40.7</td>
<td style="text-align: center;">44.3</td>
</tr>
<tr>
<td style="text-align: center;"><strong>High-resource</strong></td>
<td style="text-align: center;"><strong>de</strong></td>
<td style="text-align: center;">41.0</td>
<td style="text-align: center;">40.7</td>
<td style="text-align: center;">43.9</td>
<td style="text-align: center;">44.6</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>en</strong></td>
<td style="text-align: center;">42.5</td>
<td style="text-align: center;">47.1</td>
<td style="text-align: center;">47.8</td>
<td style="text-align: center;">49.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>es</strong></td>
<td style="text-align: center;">40.6</td>
<td style="text-align: center;">39.6</td>
<td style="text-align: center;">43.9</td>
<td style="text-align: center;">49.0</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>fr</strong></td>
<td style="text-align: center;">40.2</td>
<td style="text-align: center;">39.3</td>
<td style="text-align: center;">42.5</td>
<td style="text-align: center;">45.6</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>it</strong></td>
<td style="text-align: center;">42.6</td>
<td style="text-align: center;">41.9</td>
<td style="text-align: center;">42.7</td>
<td style="text-align: center;">47.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ja</strong></td>
<td style="text-align: center;">39.6</td>
<td style="text-align: center;">37.0</td>
<td style="text-align: center;">40.2</td>
<td style="text-align: center;">40.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>nl</strong></td>
<td style="text-align: center;">39.8</td>
<td style="text-align: center;">41.9</td>
<td style="text-align: center;">43.5</td>
<td style="text-align: center;">47.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>ru</strong></td>
<td style="text-align: center;">38.9</td>
<td style="text-align: center;">40.4</td>
<td style="text-align: center;">42.4</td>
<td style="text-align: center;">46.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>sv</strong></td>
<td style="text-align: center;">38.8</td>
<td style="text-align: center;">31.0</td>
<td style="text-align: center;">42.2</td>
<td style="text-align: center;">41.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><strong>zh</strong></td>
<td style="text-align: center;">41.5</td>
<td style="text-align: center;">35.1</td>
<td style="text-align: center;">43.3</td>
<td style="text-align: center;">42.1</td>
</tr>
<tr>
<td colspan="2" style="text-align: center;"><strong>Average</strong></td>
<td style="text-align: center;">40.6</td>
<td style="text-align: center;">39.4</td>
<td style="text-align: center;">43.2</td>
<td style="text-align: center;">45.5</td>
</tr>
</tbody>
</table>

## C.2 Evaluation Results on MGSM and MMWP

The complete experimental results on MGSM and MSVAMP are shown in Table 11, and 12.

**Tabla 11.** Experimental Results on the MGSM Dataset. "LR." "HR." and "Avg." represent the average performance for low-resource languages, high-resource languages, and all languages, respectively. Following prior work (Yoon et al., 2024), we classify Bn, Te, Th, and Sw as low-resource languages, while the remaining languages are categorized as high-resource. The dagger symbol (†) indicates results obtained using officially released models, the diamond symbol ($\Diamond$) signifies the LoRA implementation, and the double dagger symbol () denotes results from our local implementation.

<table id="main-result-mgsm">
<thead>
<tr>
<th style="text-align: left;"><strong>LLaMA-2-7B as base model</strong></th>
<th style="text-align: center;"><strong>Bn</strong></th>
<th style="text-align: center;"><strong>De</strong></th>
<th style="text-align: center;"><strong>En</strong></th>
<th style="text-align: center;"><strong>Es</strong></th>
<th style="text-align: center;"><strong>Fr</strong></th>
<th style="text-align: center;"><strong>Ja</strong></th>
<th style="text-align: center;"><strong>Ru</strong></th>
<th style="text-align: center;"><strong>Sw</strong></th>
<th style="text-align: center;"><strong>Te</strong></th>
<th style="text-align: center;"><strong>Th</strong></th>
<th style="text-align: center;"><strong>Zh</strong></th>
<th style="text-align: center;"><strong>LR.</strong></th>
<th style="text-align: center;"><strong>HR.</strong></th>
<th style="text-align: center;"><strong>Avg.</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="15" style="text-align: center;">Baseline</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">MAmmoTH<sup>†</sup></td>
<td style="text-align: center;">3.20</td>
<td style="text-align: center;">33.2</td>
<td style="text-align: center;">47.2</td>
<td style="text-align: center;">34.4</td>
<td style="text-align: center;">32.4</td>
<td style="text-align: center;">23.2</td>
<td style="text-align: center;">30.4</td>
<td style="text-align: center;">2.00</td>
<td style="text-align: center;">1.60</td>
<td style="text-align: center;">6.80</td>
<td style="text-align: center;">26.0</td>
<td style="text-align: center;">3.40</td>
<td style="text-align: center;">32.4</td>
<td style="text-align: center;">21.9</td>
</tr>
<tr>
<td style="text-align: center;">WizardMath<sup>†</sup></td>
<td style="text-align: center;">4.40</td>
<td style="text-align: center;">37.2</td>
<td style="text-align: center;">51.6</td>
<td style="text-align: center;">43.2</td>
<td style="text-align: center;">39.2</td>
<td style="text-align: center;">27.2</td>
<td style="text-align: center;">36.8</td>
<td style="text-align: center;">4.40</td>
<td style="text-align: center;">1.60</td>
<td style="text-align: center;">5.60</td>
<td style="text-align: center;">28.8</td>
<td style="text-align: center;">4.00</td>
<td style="text-align: center;">37.7</td>
<td style="text-align: center;">25.5</td>
</tr>
<tr>
<td style="text-align: center;">MetaMath<sup>†</sup></td>
<td style="text-align: center;">5.60</td>
<td style="text-align: center;">56.4</td>
<td style="text-align: center;">68.0</td>
<td style="text-align: center;">54.4</td>
<td style="text-align: center;">56.0</td>
<td style="text-align: center;">34.8</td>
<td style="text-align: center;">52.8</td>
<td style="text-align: center;">5.20</td>
<td style="text-align: center;">2.00</td>
<td style="text-align: center;">5.60</td>
<td style="text-align: center;">39.6</td>
<td style="text-align: center;">4.60</td>
<td style="text-align: center;">51.7</td>
<td style="text-align: center;">34.6</td>
</tr>
<tr>
<td style="text-align: center;">OpenMath2<sup></sup></td>
<td style="text-align: center;">6.80</td>
<td style="text-align: center;">65.6</td>
<td style="text-align: center;">79.2</td>
<td style="text-align: center;">66.0</td>
<td style="text-align: center;">62.4</td>
<td style="text-align: center;">42.4</td>
<td style="text-align: center;">57.6</td>
<td style="text-align: center;">6.00</td>
<td style="text-align: center;">2.80</td>
<td style="text-align: center;">6.80</td>
<td style="text-align: center;">48.4</td>
<td style="text-align: center;">5.60</td>
<td style="text-align: center;">60.2</td>
<td style="text-align: center;">40.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">MathOctopus-Parallel<sup>†</sup></td>
<td style="text-align: center;">31.2</td>
<td style="text-align: center;">46.0</td>
<td style="text-align: center;">51.6</td>
<td style="text-align: center;">42.8</td>
<td style="text-align: center;">43.6</td>
<td style="text-align: center;">34.0</td>
<td style="text-align: center;">40.0</td>
<td style="text-align: center;">36.4</td>
<td style="text-align: center;">10.8</td>
<td style="text-align: center;">33.6</td>
<td style="text-align: center;">38.8</td>
<td style="text-align: center;">28.0</td>
<td style="text-align: center;">42.4</td>
<td style="text-align: center;">37.2</td>
</tr>
<tr>
<td style="text-align: center;">MathOctopus-MAPO-DPO<sup>†</sup></td>
<td style="text-align: center;">33.2</td>
<td style="text-align: center;">47.2</td>
<td style="text-align: center;">46.8</td>
<td style="text-align: center;">43.2</td>
<td style="text-align: center;">39.6</td>
<td style="text-align: center;">41.6</td>
<td style="text-align: center;">40.8</td>
<td style="text-align: center;">37.2</td>
<td style="text-align: center;">13.6</td>
<td style="text-align: center;">38.4</td>
<td style="text-align: center;">44.8</td>
<td style="text-align: center;">30.6</td>
<td style="text-align: center;">43.4</td>
<td style="text-align: center;">38.8</td>
</tr>
<tr>
<td style="text-align: center;">MetaMathOctopus-MAPO-DPO<sup>†</sup></td>
<td style="text-align: center;">35.6</td>
<td style="text-align: center;">52.4</td>
<td style="text-align: center;">70.0</td>
<td style="text-align: center;">58.0</td>
<td style="text-align: center;">51.2</td>
<td style="text-align: center;">46.4</td>
<td style="text-align: center;">56.0</td>
<td style="text-align: center;">42.4</td>
<td style="text-align: center;">2.00</td>
<td style="text-align: center;">44.0</td>
<td style="text-align: center;">55.2</td>
<td style="text-align: center;">31.0</td>
<td style="text-align: center;">55.6</td>
<td style="text-align: center;">46.7</td>
</tr>
<tr>
<td style="text-align: center;">QAlign-MetaMathQA<sup>†</sup></td>
<td style="text-align: center;">29.6</td>
<td style="text-align: center;">54.0</td>
<td style="text-align: center;">68.4</td>
<td style="text-align: center;">57.6</td>
<td style="text-align: center;">59.2</td>
<td style="text-align: center;">45.2</td>
<td style="text-align: center;">58.4</td>
<td style="text-align: center;">35.6</td>
<td style="text-align: center;">2.40</td>
<td style="text-align: center;">37.6</td>
<td style="text-align: center;">48.8</td>
<td style="text-align: center;">26.3</td>
<td style="text-align: center;">55.9</td>
<td style="text-align: center;">45.2</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">Translate-En-MetaMath<sup></sup></td>
<td style="text-align: center;">49.0</td>
<td style="text-align: center;">59.6</td>
<td style="text-align: center;">65.6</td>
<td style="text-align: center;">59.8</td>
<td style="text-align: center;">56.2</td>
<td style="text-align: center;">49.0</td>
<td style="text-align: center;">48.4</td>
<td style="text-align: center;">37.4</td>
<td style="text-align: center;">34.6</td>
<td style="text-align: center;">37.2</td>
<td style="text-align: center;">47.0</td>
<td style="text-align: center;">39.6</td>
<td style="text-align: center;">55.1</td>
<td style="text-align: center;">49.4</td>
</tr>
<tr>
<td style="text-align: center;">LangBridge-MetaMath<sup>†</sup></td>
<td style="text-align: center;">41.2</td>
<td style="text-align: center;">53.2</td>
<td style="text-align: center;">62.4</td>
<td style="text-align: center;">58.0</td>
<td style="text-align: center;">51.6</td>
<td style="text-align: center;">39.6</td>
<td style="text-align: center;">55.2</td>
<td style="text-align: center;">39.6</td>
<td style="text-align: center;">28.0</td>
<td style="text-align: center;">44.8</td>
<td style="text-align: center;">43.2</td>
<td style="text-align: center;">38.4</td>
<td style="text-align: center;">51.9</td>
<td style="text-align: center;">47.0</td>
</tr>
<tr>
<td style="text-align: center;">MindMerger-Soft-MetaMath<sup>†</sup></td>
<td style="text-align: center;">50.4</td>
<td style="text-align: center;">59.6</td>
<td style="text-align: center;">67.2</td>
<td style="text-align: center;">58.4</td>
<td style="text-align: center;">55.6</td>
<td style="text-align: center;">50.0</td>
<td style="text-align: center;">61.6</td>
<td style="text-align: center;">55.2</td>
<td style="text-align: center;">52.8</td>
<td style="text-align: center;">54.2</td>
<td style="text-align: center;">53.2</td>
<td style="text-align: center;">53.1</td>
<td style="text-align: center;">57.9</td>
<td style="text-align: center;">56.2</td>
</tr>
<tr>
<td style="text-align: center;">Translate-En-OpenMath2<sup></sup></td>
<td style="text-align: center;">52.1</td>
<td style="text-align: center;">69.8</td>
<td style="text-align: center;">76.9</td>
<td style="text-align: center;">71.6</td>
<td style="text-align: center;">63.4</td>
<td style="text-align: center;">57.4</td>
<td style="text-align: center;">54.2</td>
<td style="text-align: center;">38.4</td>
<td style="text-align: center;">35.6</td>
<td style="text-align: center;">39.2</td>
<td style="text-align: center;">53.2</td>
<td style="text-align: center;">41.3</td>
<td style="text-align: center;">63.8</td>
<td style="text-align: center;">55.6</td>
</tr>
<tr>
<td style="text-align: center;">LangBridge-OpenMath2<sup></sup></td>
<td style="text-align: center;">43.2</td>
<td style="text-align: center;">63.0</td>
<td style="text-align: center;">73.8</td>
<td style="text-align: center;">69.8</td>
<td style="text-align: center;">58.9</td>
<td style="text-align: center;">47.6</td>
<td style="text-align: center;">60.2</td>
<td style="text-align: center;">41.0</td>
<td style="text-align: center;">39.8</td>
<td style="text-align: center;">46.2</td>
<td style="text-align: center;">52.2</td>
<td style="text-align: center;">42.6</td>
<td style="text-align: center;">60.8</td>
<td style="text-align: center;">54.2</td>
</tr>
<tr>
<td style="text-align: center;">MindMerger-Soft-OpenMath2<sup></sup></td>
<td style="text-align: center;">60.0</td>
<td style="text-align: center;">69.4</td>
<td style="text-align: center;">77.8</td>
<td style="text-align: center;">74.2</td>
<td style="text-align: center;">63.4</td>
<td style="text-align: center;">57.8</td>
<td style="text-align: center;">71.8</td>
<td style="text-align: center;">61.2</td>
<td style="text-align: center;">57.2</td>
<td style="text-align: center;">63.6</td>
<td style="text-align: center;">58.4</td>
<td style="text-align: center;">60.5</td>
<td style="text-align: center;">67.5</td>
<td style="text-align: center;">65.0</td>
</tr>
<tr>
<td colspan="15" style="text-align: center;">Our Methods</td>
</tr>
<tr>
<td style="text-align: center;"><strong>LinguaLIFT-MetaMathQA</strong></td>
<td style="text-align: center;">54.4</td>
<td style="text-align: center;">62.0</td>
<td style="text-align: center;">64.8</td>
<td style="text-align: center;">63.6</td>
<td style="text-align: center;">56.8</td>
<td style="text-align: center;">50.0</td>
<td style="text-align: center;">60.4</td>
<td style="text-align: center;">55.6</td>
<td style="text-align: center;">54.0</td>
<td style="text-align: center;">57.6</td>
<td style="text-align: center;">54.0</td>
<td style="text-align: center;">55.4</td>
<td style="text-align: center;">58.8</td>
<td style="text-align: center;">57.6</td>
</tr>
<tr>
<td style="text-align: center;"><strong>LinguaLIFT-MetaMathQA<sup><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo>◇</mo><annotation encoding="application/x-tex">\Diamond</annotation></semantics></math></sup></strong></td>
<td style="text-align: center;">50.2</td>
<td style="text-align: center;">57.2</td>
<td style="text-align: center;">61.4</td>
<td style="text-align: center;">60.2</td>
<td style="text-align: center;">52.6</td>
<td style="text-align: center;">45.8</td>
<td style="text-align: center;">55.4</td>
<td style="text-align: center;">51.2</td>
<td style="text-align: center;">49.8</td>
<td style="text-align: center;">53.6</td>
<td style="text-align: center;">50.6</td>
<td style="text-align: center;">51.2</td>
<td style="text-align: center;">54.7</td>
<td style="text-align: center;">53.5</td>
</tr>
<tr>
<td style="text-align: center;"><strong>LinguaLIFT-OpenMathInstruct-2</strong></td>
<td style="text-align: center;">63.0</td>
<td style="text-align: center;">69.6</td>
<td style="text-align: center;">76.0</td>
<td style="text-align: center;">75.2</td>
<td style="text-align: center;">62.8</td>
<td style="text-align: center;">52.0</td>
<td style="text-align: center;">71.6</td>
<td style="text-align: center;">64.2</td>
<td style="text-align: center;">61.2</td>
<td style="text-align: center;">66.6</td>
<td style="text-align: center;">58.0</td>
<td style="text-align: center;">63.8</td>
<td style="text-align: center;">66.5</td>
<td style="text-align: center;">65.5</td>
</tr>
<tr>
<td style="text-align: center;"><strong>LinguaLIFT-OpenMathInstruct-2<sup><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo>◇</mo><annotation encoding="application/x-tex">\Diamond</annotation></semantics></math></sup></strong></td>
<td style="text-align: center;">59.2</td>
<td style="text-align: center;">65.0</td>
<td style="text-align: center;">72.8</td>
<td style="text-align: center;">71.6</td>
<td style="text-align: center;">59.4</td>
<td style="text-align: center;">48.6</td>
<td style="text-align: center;">67.2</td>
<td style="text-align: center;">60.0</td>
<td style="text-align: center;">57.4</td>
<td style="text-align: center;">62.8</td>
<td style="text-align: center;">55.4</td>
<td style="text-align: center;">59.9</td>
<td style="text-align: center;">62.9</td>
<td style="text-align: center;">61.8</td>
</tr>
</tbody>
</table>

**Tabla 12.** Experimental Results on the MSVAMP Dataset. "LR." "HR." and "Avg." represent the average performance for low-resource languages, high-resource languages, and all languages, respectively. Following prior work (Yoon et al., 2024), we classify Bn, Th, and Sw as low-resource languages, while the remaining languages are categorized as high-resource. The dagger symbol (†) indicates results obtained using officially released models, while the double dagger symbol () denotes results from our local implementation.

<table id="main-result-msvamp">
<thead>
<tr>
<th style="text-align: left;"><strong>LLaMA-2-7B as base model</strong></th>
<th style="text-align: center;"><strong>Bn</strong></th>
<th style="text-align: center;"><strong>De</strong></th>
<th style="text-align: center;"><strong>En</strong></th>
<th style="text-align: center;"><strong>Es</strong></th>
<th style="text-align: center;"><strong>Fr</strong></th>
<th style="text-align: center;"><strong>Ja</strong></th>
<th style="text-align: center;"><strong>Ru</strong></th>
<th style="text-align: center;"><strong>Sw</strong></th>
<th style="text-align: center;"><strong>Th</strong></th>
<th style="text-align: center;"><strong>Zh</strong></th>
<th style="text-align: center;"><strong>LR.</strong></th>
<th style="text-align: center;"><strong>HR.</strong></th>
<th style="text-align: center;"><strong>Avg.</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="14" style="text-align: center;"><span class="smallcaps">baselines</span></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">MAmmoTH<sup>†</sup></td>
<td style="text-align: center;">6.2</td>
<td style="text-align: center;">44.1</td>
<td style="text-align: center;">39.5</td>
<td style="text-align: center;">45.4</td>
<td style="text-align: center;">42.3</td>
<td style="text-align: center;">34.1</td>
<td style="text-align: center;">38.1</td>
<td style="text-align: center;">5.1</td>
<td style="text-align: center;">8.4</td>
<td style="text-align: center;">37.5</td>
<td style="text-align: center;">6.57</td>
<td style="text-align: center;">40.1</td>
<td style="text-align: center;">30.1</td>
</tr>
<tr>
<td style="text-align: center;">WizardMath<sup>†</sup></td>
<td style="text-align: center;">16.4</td>
<td style="text-align: center;">49.1</td>
<td style="text-align: center;">56.1</td>
<td style="text-align: center;">50.5</td>
<td style="text-align: center;">50.8</td>
<td style="text-align: center;">45.4</td>
<td style="text-align: center;">44.8</td>
<td style="text-align: center;">13.4</td>
<td style="text-align: center;">17.2</td>
<td style="text-align: center;">43.1</td>
<td style="text-align: center;">15.7</td>
<td style="text-align: center;">48.5</td>
<td style="text-align: center;">38.7</td>
</tr>
<tr>
<td style="text-align: center;">MetaMath<sup>†</sup></td>
<td style="text-align: center;">12.5</td>
<td style="text-align: center;">63.5</td>
<td style="text-align: center;">67.2</td>
<td style="text-align: center;">64.7</td>
<td style="text-align: center;">64.9</td>
<td style="text-align: center;">54.2</td>
<td style="text-align: center;">58.2</td>
<td style="text-align: center;">16.7</td>
<td style="text-align: center;">16.5</td>
<td style="text-align: center;">55.5</td>
<td style="text-align: center;">15.2</td>
<td style="text-align: center;">61.2</td>
<td style="text-align: center;">47.4</td>
</tr>
<tr>
<td style="text-align: center;">OpenMath2<sup></sup></td>
<td style="text-align: center;">19.9</td>
<td style="text-align: center;">72.7</td>
<td style="text-align: center;">78.6</td>
<td style="text-align: center;">72.5</td>
<td style="text-align: center;">72.9</td>
<td style="text-align: center;">64.6</td>
<td style="text-align: center;">68.5</td>
<td style="text-align: center;">13.1</td>
<td style="text-align: center;">23.5</td>
<td style="text-align: center;">63.8</td>
<td style="text-align: center;">18.8</td>
<td style="text-align: center;">70.5</td>
<td style="text-align: center;">55.0</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">MathOctopus-Parallel<sup>†</sup></td>
<td style="text-align: center;">27.8</td>
<td style="text-align: center;">43.8</td>
<td style="text-align: center;">46.1</td>
<td style="text-align: center;">45.4</td>
<td style="text-align: center;">44.9</td>
<td style="text-align: center;">41.4</td>
<td style="text-align: center;">43.8</td>
<td style="text-align: center;">38.1</td>
<td style="text-align: center;">34.9</td>
<td style="text-align: center;">41.4</td>
<td style="text-align: center;">33.6</td>
<td style="text-align: center;">43.8</td>
<td style="text-align: center;">40.8</td>
</tr>
<tr>
<td style="text-align: center;">MathOctopus-MAPO-DPO<sup>†</sup></td>
<td style="text-align: center;">48.3</td>
<td style="text-align: center;">57.6</td>
<td style="text-align: center;">59.0</td>
<td style="text-align: center;">59.4</td>
<td style="text-align: center;">60.0</td>
<td style="text-align: center;">58.7</td>
<td style="text-align: center;">58.0</td>
<td style="text-align: center;">54.6</td>
<td style="text-align: center;">54.7</td>
<td style="text-align: center;">59.1</td>
<td style="text-align: center;">52.5</td>
<td style="text-align: center;">58.8</td>
<td style="text-align: center;">56.9</td>
</tr>
<tr>
<td style="text-align: center;">MetaMathOctopus-MAPO-DPO<sup>†</sup></td>
<td style="text-align: center;">50.3</td>
<td style="text-align: center;">67.7</td>
<td style="text-align: center;">71.5</td>
<td style="text-align: center;">68.7</td>
<td style="text-align: center;">67.5</td>
<td style="text-align: center;">65.4</td>
<td style="text-align: center;">64.7</td>
<td style="text-align: center;">61.3</td>
<td style="text-align: center;">61.7</td>
<td style="text-align: center;">66.0</td>
<td style="text-align: center;">57.8</td>
<td style="text-align: center;">67.4</td>
<td style="text-align: center;">64.5</td>
</tr>
<tr>
<td style="text-align: center;">QAlign-MetaMathQA<sup>†</sup></td>
<td style="text-align: center;">41.9</td>
<td style="text-align: center;">63.5</td>
<td style="text-align: center;">65.5</td>
<td style="text-align: center;">63.0</td>
<td style="text-align: center;">62.8</td>
<td style="text-align: center;">58.0</td>
<td style="text-align: center;">61.5</td>
<td style="text-align: center;">53.6</td>
<td style="text-align: center;">49.7</td>
<td style="text-align: center;">56.2</td>
<td style="text-align: center;">48.4</td>
<td style="text-align: center;">61.5</td>
<td style="text-align: center;">57.6</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">Translate-En-MetaMath<sup></sup></td>
<td style="text-align: center;">47.9</td>
<td style="text-align: center;">43.9</td>
<td style="text-align: center;">60.6</td>
<td style="text-align: center;">51.4</td>
<td style="text-align: center;">50.9</td>
<td style="text-align: center;">50.4</td>
<td style="text-align: center;">53.4</td>
<td style="text-align: center;">43.1</td>
<td style="text-align: center;">51.3</td>
<td style="text-align: center;">55.8</td>
<td style="text-align: center;">47.4</td>
<td style="text-align: center;">52.3</td>
<td style="text-align: center;">50.9</td>
</tr>
<tr>
<td style="text-align: center;">LangBridge-MetaMath<sup>†</sup></td>
<td style="text-align: center;">39.6</td>
<td style="text-align: center;">58.8</td>
<td style="text-align: center;">60.1</td>
<td style="text-align: center;">56.8</td>
<td style="text-align: center;">57.9</td>
<td style="text-align: center;">45.2</td>
<td style="text-align: center;">53.6</td>
<td style="text-align: center;">45.8</td>
<td style="text-align: center;">46.3</td>
<td style="text-align: center;">49.4</td>
<td style="text-align: center;">43.9</td>
<td style="text-align: center;">54.5</td>
<td style="text-align: center;">51.4</td>
</tr>
<tr>
<td style="text-align: center;">MindMerger-Soft-MetaMath<sup>†</sup></td>
<td style="text-align: center;">52.0</td>
<td style="text-align: center;">61.1</td>
<td style="text-align: center;">64.5</td>
<td style="text-align: center;">62.9</td>
<td style="text-align: center;">60.8</td>
<td style="text-align: center;">59.0</td>
<td style="text-align: center;">58.6</td>
<td style="text-align: center;">54.0</td>
<td style="text-align: center;">52.1</td>
<td style="text-align: center;">57.3</td>
<td style="text-align: center;">52.7</td>
<td style="text-align: center;">60.6</td>
<td style="text-align: center;">58.2</td>
</tr>
<tr>
<td style="text-align: center;">Translate-En-OpenMath2<sup></sup></td>
<td style="text-align: center;">56.3</td>
<td style="text-align: center;">54.1</td>
<td style="text-align: center;">72.0</td>
<td style="text-align: center;">59.5</td>
<td style="text-align: center;">59.3</td>
<td style="text-align: center;">60.9</td>
<td style="text-align: center;">64.6</td>
<td style="text-align: center;">40.2</td>
<td style="text-align: center;">59.2</td>
<td style="text-align: center;">64.9</td>
<td style="text-align: center;">51.9</td>
<td style="text-align: center;">62.2</td>
<td style="text-align: center;">59.1</td>
</tr>
<tr>
<td style="text-align: center;">LangBridge-OpenMath2<sup></sup></td>
<td style="text-align: center;">48.0</td>
<td style="text-align: center;">68.6</td>
<td style="text-align: center;">71.9</td>
<td style="text-align: center;">64.8</td>
<td style="text-align: center;">66.6</td>
<td style="text-align: center;">56.5</td>
<td style="text-align: center;">64.4</td>
<td style="text-align: center;">42.3</td>
<td style="text-align: center;">53.5</td>
<td style="text-align: center;">58.3</td>
<td style="text-align: center;">47.9</td>
<td style="text-align: center;">64.4</td>
<td style="text-align: center;">59.5</td>
</tr>
<tr>
<td style="text-align: center;">MindMerger-Soft-OpenMath2<sup></sup></td>
<td style="text-align: center;">61.2</td>
<td style="text-align: center;">77.2</td>
<td style="text-align: center;">77.1</td>
<td style="text-align: center;">73.6</td>
<td style="text-align: center;">73.5</td>
<td style="text-align: center;">74.1</td>
<td style="text-align: center;">70.8</td>
<td style="text-align: center;">64.8</td>
<td style="text-align: center;">64.1</td>
<td style="text-align: center;">72.8</td>
<td style="text-align: center;">63.4</td>
<td style="text-align: center;">74.2</td>
<td style="text-align: center;">70.9</td>
</tr>
<tr>
<td colspan="14" style="text-align: center;"><span class="smallcaps">our methods</span></td>
</tr>
<tr>
<td style="text-align: center;"><strong>LinguaLIFT-MetaMathQA</strong></td>
<td style="text-align: center;">54.3</td>
<td style="text-align: center;">62.8</td>
<td style="text-align: center;">65.2</td>
<td style="text-align: center;">59.9</td>
<td style="text-align: center;">62.7</td>
<td style="text-align: center;">57.9</td>
<td style="text-align: center;">59.7</td>
<td style="text-align: center;">56.8</td>
<td style="text-align: center;">57.2</td>
<td style="text-align: center;">56.0</td>
<td style="text-align: center;">56.1</td>
<td style="text-align: center;">60.6</td>
<td style="text-align: center;">59.3</td>
</tr>
<tr>
<td style="text-align: center;"><strong>LinguaLIFT-MetaMathQA<sup><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo>◇</mo><annotation encoding="application/x-tex">\Diamond</annotation></semantics></math></sup></strong></td>
<td style="text-align: center;">49.8</td>
<td style="text-align: center;">57.6</td>
<td style="text-align: center;">61.1</td>
<td style="text-align: center;">55.4</td>
<td style="text-align: center;">58.2</td>
<td style="text-align: center;">53.5</td>
<td style="text-align: center;">55.3</td>
<td style="text-align: center;">52.1</td>
<td style="text-align: center;">52.7</td>
<td style="text-align: center;">51.4</td>
<td style="text-align: center;">51.5</td>
<td style="text-align: center;">56.1</td>
<td style="text-align: center;">54.7</td>
</tr>
<tr>
<td style="text-align: center;"><strong>LinguaLIFT-OpenMathInstruct-2</strong></td>
<td style="text-align: center;">65.2</td>
<td style="text-align: center;">76.4</td>
<td style="text-align: center;">77.9</td>
<td style="text-align: center;">74.2</td>
<td style="text-align: center;">74.5</td>
<td style="text-align: center;">73.5</td>
<td style="text-align: center;">71.4</td>
<td style="text-align: center;">68.6</td>
<td style="text-align: center;">67.9</td>
<td style="text-align: center;">72.2</td>
<td style="text-align: center;">67.2</td>
<td style="text-align: center;">74.3</td>
<td style="text-align: center;">72.2</td>
</tr>
<tr>
<td style="text-align: center;"><strong>LinguaLIFT-OpenMathInstruct-2<sup><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo>◇</mo><annotation encoding="application/x-tex">\Diamond</annotation></semantics></math></sup></strong></td>
<td style="text-align: center;">61.3</td>
<td style="text-align: center;">72.1</td>
<td style="text-align: center;">74.2</td>
<td style="text-align: center;">70.5</td>
<td style="text-align: center;">70.8</td>
<td style="text-align: center;">69.7</td>
<td style="text-align: center;">67.5</td>
<td style="text-align: center;">64.9</td>
<td style="text-align: center;">64.2</td>
<td style="text-align: center;">68.5</td>
<td style="text-align: center;">63.5</td>
<td style="text-align: center;">70/5</td>
<td style="text-align: center;">68.4</td>
</tr>
</tbody>
</table>

## C.3 Evaluation Results on XNLI and X-CSQA

The complete experimental results on XNLI and X-CSQA are shown in Table 13, and 14.

**Tabla 13.** Experimental Results on the XNLI Dataset. "LR." "HR." and "Avg." represent the average performance for low-resource languages, high-resource languages, and all languages, respectively. The asterisk symbol () indicates results obtained directly from the published results (Huang et al., 2024).

<table id="tab:xnli-results">
<thead>
<tr>
<th style="text-align: left;"><strong>LLaMA-2-7B as base model</strong></th>
<th style="text-align: center;"><strong>Ar</strong></th>
<th style="text-align: center;"><strong>Bg</strong></th>
<th style="text-align: center;"><strong>Sw</strong></th>
<th style="text-align: center;"><strong>Th</strong></th>
<th style="text-align: center;"><strong>Tr</strong></th>
<th style="text-align: center;"><strong>Ur</strong></th>
<th style="text-align: center;"><strong>El</strong></th>
<th style="text-align: center;"><strong>Hi</strong></th>
<th style="text-align: center;"><strong>Zh</strong></th>
<th style="text-align: center;"><strong>Ru</strong></th>
<th style="text-align: center;"><strong>Vi</strong></th>
<th style="text-align: center;"><strong>De</strong></th>
<th style="text-align: center;"><strong>Fr</strong></th>
<th style="text-align: center;"><strong>Es</strong></th>
<th style="text-align: center;"><strong>En</strong></th>
<th style="text-align: center;"><strong>LR.</strong></th>
<th style="text-align: center;"><strong>HR.</strong></th>
<th style="text-align: center;"><strong>Avg.</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="19" style="text-align: center;"><span class="smallcaps">baselines</span></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">Mono-SFT<sup></sup></td>
<td style="text-align: center;">60.9</td>
<td style="text-align: center;">76.7</td>
<td style="text-align: center;">45.9</td>
<td style="text-align: center;">55.4</td>
<td style="text-align: center;">61.9</td>
<td style="text-align: center;">49.2</td>
<td style="text-align: center;">63.7</td>
<td style="text-align: center;">55.7</td>
<td style="text-align: center;">74.7</td>
<td style="text-align: center;">77.6</td>
<td style="text-align: center;">73.7</td>
<td style="text-align: center;">80.6</td>
<td style="text-align: center;">82.2</td>
<td style="text-align: center;">82.2</td>
<td style="text-align: center;">90.0</td>
<td style="text-align: center;">58.7</td>
<td style="text-align: center;">80.1</td>
<td style="text-align: center;">68.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">Multi-SFT<sup></sup></td>
<td style="text-align: center;">61.7</td>
<td style="text-align: center;">78.7</td>
<td style="text-align: center;">56.3</td>
<td style="text-align: center;">60.1</td>
<td style="text-align: center;">65.6</td>
<td style="text-align: center;">57.5</td>
<td style="text-align: center;">67.0</td>
<td style="text-align: center;">61.7</td>
<td style="text-align: center;">79.1</td>
<td style="text-align: center;">79.7</td>
<td style="text-align: center;">73.7</td>
<td style="text-align: center;">82.3</td>
<td style="text-align: center;">82.9</td>
<td style="text-align: center;">83.9</td>
<td style="text-align: center;">88.8</td>
<td style="text-align: center;">63.6</td>
<td style="text-align: center;">81.5</td>
<td style="text-align: center;">71.9</td>
</tr>
<tr>
<td style="text-align: center;">QAlign<sup></sup></td>
<td style="text-align: center;">67.0</td>
<td style="text-align: center;">79.4</td>
<td style="text-align: center;">65.2</td>
<td style="text-align: center;">65.2</td>
<td style="text-align: center;">67.9</td>
<td style="text-align: center;">62.2</td>
<td style="text-align: center;">66.5</td>
<td style="text-align: center;">63.3</td>
<td style="text-align: center;">76.6</td>
<td style="text-align: center;">79.2</td>
<td style="text-align: center;">73.7</td>
<td style="text-align: center;">80.9</td>
<td style="text-align: center;">83.1</td>
<td style="text-align: center;">83.8</td>
<td style="text-align: center;">89.1</td>
<td style="text-align: center;">67.1</td>
<td style="text-align: center;">80.9</td>
<td style="text-align: center;">73.5</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">LangBridge<sup></sup></td>
<td style="text-align: center;">75.2</td>
<td style="text-align: center;">79.6</td>
<td style="text-align: center;">71.7</td>
<td style="text-align: center;">72.4</td>
<td style="text-align: center;">74.8</td>
<td style="text-align: center;">66.9</td>
<td style="text-align: center;">79.1</td>
<td style="text-align: center;">71.1</td>
<td style="text-align: center;">77.4</td>
<td style="text-align: center;">77.4</td>
<td style="text-align: center;">78.5</td>
<td style="text-align: center;">78.8</td>
<td style="text-align: center;">79.9</td>
<td style="text-align: center;">80.5</td>
<td style="text-align: center;">83.4</td>
<td style="text-align: center;">73.9</td>
<td style="text-align: center;">79.4</td>
<td style="text-align: center;">76.5</td>
</tr>
<tr>
<td style="text-align: center;">Translate-En<sup></sup></td>
<td style="text-align: center;">68.9</td>
<td style="text-align: center;">80.8</td>
<td style="text-align: center;">65.3</td>
<td style="text-align: center;">69.5</td>
<td style="text-align: center;">74.5</td>
<td style="text-align: center;">61.6</td>
<td style="text-align: center;">79.3</td>
<td style="text-align: center;">68.7</td>
<td style="text-align: center;">74.8</td>
<td style="text-align: center;">76</td>
<td style="text-align: center;">76.7</td>
<td style="text-align: center;">80.6</td>
<td style="text-align: center;">80.4</td>
<td style="text-align: center;">81.4</td>
<td style="text-align: center;">87.4</td>
<td style="text-align: center;">71.1</td>
<td style="text-align: center;">79.6</td>
<td style="text-align: center;">75.1</td>
</tr>
<tr>
<td style="text-align: center;">MindMerger-Soft<sup></sup></td>
<td style="text-align: center;">76.2</td>
<td style="text-align: center;">82.4</td>
<td style="text-align: center;">66.6</td>
<td style="text-align: center;">71.8</td>
<td style="text-align: center;">75.7</td>
<td style="text-align: center;">69.4</td>
<td style="text-align: center;">78.5</td>
<td style="text-align: center;">74.7</td>
<td style="text-align: center;">80.0</td>
<td style="text-align: center;">80.7</td>
<td style="text-align: center;">80.3</td>
<td style="text-align: center;">83.5</td>
<td style="text-align: center;">83.9</td>
<td style="text-align: center;">84.4</td>
<td style="text-align: center;">88.7</td>
<td style="text-align: center;">74.4</td>
<td style="text-align: center;">83.1</td>
<td style="text-align: center;">78.4</td>
</tr>
<tr>
<td colspan="19" style="text-align: center;"><span class="smallcaps">our methods</span></td>
</tr>
<tr>
<td style="text-align: center;"><strong>LinguaLIFT</strong></td>
<td style="text-align: center;">78.4</td>
<td style="text-align: center;">83.5</td>
<td style="text-align: center;">75.1</td>
<td style="text-align: center;">77.0</td>
<td style="text-align: center;">77.1</td>
<td style="text-align: center;">72.4</td>
<td style="text-align: center;">82.2</td>
<td style="text-align: center;">75.1</td>
<td style="text-align: center;">80.2</td>
<td style="text-align: center;">80.5</td>
<td style="text-align: center;">80.8</td>
<td style="text-align: center;">83.6</td>
<td style="text-align: center;">84.0</td>
<td style="text-align: center;">84.4</td>
<td style="text-align: center;">89.5</td>
<td style="text-align: center;">77.6</td>
<td style="text-align: center;">83.3</td>
<td style="text-align: center;">80.3</td>
</tr>
</tbody>
</table>

**Tabla 14.** Experimental Results on the X-CSQA Dataset. "LR." "HR." and "Avg." represent the average performance for details of the low-resource, high-resource, and all languages, respectively. The asterisk symbol () indicates results obtained directly from the published results (Huang et al., 2024).

<table id="tab:xcsqa-results">
<thead>
<tr>
<th style="text-align: left;"><strong>LLaMA-2-7B as base model</strong></th>
<th style="text-align: center;"><strong>Ar</strong></th>
<th style="text-align: center;"><strong>De</strong></th>
<th style="text-align: center;"><strong>En</strong></th>
<th style="text-align: center;"><strong>Es</strong></th>
<th style="text-align: center;"><strong>Fr</strong></th>
<th style="text-align: center;"><strong>Hi</strong></th>
<th style="text-align: center;"><strong>It</strong></th>
<th style="text-align: center;"><strong>Ja</strong></th>
<th style="text-align: center;"><strong>Nl</strong></th>
<th style="text-align: center;"><strong>Pl</strong></th>
<th style="text-align: center;"><strong>Pt</strong></th>
<th style="text-align: center;"><strong>Ru</strong></th>
<th style="text-align: center;"><strong>Sw</strong></th>
<th style="text-align: center;"><strong>Ur</strong></th>
<th style="text-align: center;"><strong>Vi</strong></th>
<th style="text-align: left;"><strong>Zh</strong></th>
<th style="text-align: center;"><strong>LR.</strong></th>
<th style="text-align: center;"><strong>HR.</strong></th>
<th style="text-align: center;"><strong>Avg.</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="20" style="text-align: center;"><span class="smallcaps">baselines</span></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: left;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: right;"></td>
<td style="text-align: right;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">Mono-SFT<sup></sup></td>
<td style="text-align: center;">32.3</td>
<td style="text-align: center;">61.2</td>
<td style="text-align: center;">76.3</td>
<td style="text-align: center;">64.0</td>
<td style="text-align: center;">63.5</td>
<td style="text-align: center;">32.9</td>
<td style="text-align: center;">56.0</td>
<td style="text-align: center;">49.1</td>
<td style="text-align: center;">57.5</td>
<td style="text-align: center;">50.6</td>
<td style="text-align: center;">61.7</td>
<td style="text-align: center;">56.0</td>
<td style="text-align: center;">24.2</td>
<td style="text-align: center;">25.1</td>
<td style="text-align: center;">50.9</td>
<td style="text-align: center;">56.5</td>
<td style="text-align: right;">28.6</td>
<td style="text-align: right;">58.6</td>
<td style="text-align: center;">51.3</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: left;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: right;"></td>
<td style="text-align: right;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">Multi-SFT<sup></sup></td>
<td style="text-align: center;">28.7</td>
<td style="text-align: center;">49.1</td>
<td style="text-align: center;">67.2</td>
<td style="text-align: center;">54.3</td>
<td style="text-align: center;">52.1</td>
<td style="text-align: center;">32.0</td>
<td style="text-align: center;">50.2</td>
<td style="text-align: center;">38.7</td>
<td style="text-align: center;">45.9</td>
<td style="text-align: center;">45.5</td>
<td style="text-align: center;">51.2</td>
<td style="text-align: center;">46.5</td>
<td style="text-align: center;">27.6</td>
<td style="text-align: center;">29.2</td>
<td style="text-align: center;">38.8</td>
<td style="text-align: center;">43.8</td>
<td style="text-align: right;">29.4</td>
<td style="text-align: right;">48.6</td>
<td style="text-align: center;">43.8</td>
</tr>
<tr>
<td style="text-align: center;">QAlign<sup></sup></td>
<td style="text-align: center;">36.3</td>
<td style="text-align: center;">58.8</td>
<td style="text-align: center;">75.7</td>
<td style="text-align: center;">63.1</td>
<td style="text-align: center;">60.3</td>
<td style="text-align: center;">37.8</td>
<td style="text-align: center;">58.3</td>
<td style="text-align: center;">49.2</td>
<td style="text-align: center;">56.3</td>
<td style="text-align: center;">51.3</td>
<td style="text-align: center;">59.8</td>
<td style="text-align: center;">56.3</td>
<td style="text-align: center;">35.1</td>
<td style="text-align: center;">32.6</td>
<td style="text-align: center;">50.5</td>
<td style="text-align: center;">54.8</td>
<td style="text-align: right;">35.5</td>
<td style="text-align: right;">57.9</td>
<td style="text-align: center;">52.3</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: left;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: right;"></td>
<td style="text-align: right;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">LangBridge<sup></sup></td>
<td style="text-align: center;">30.6</td>
<td style="text-align: center;">37.4</td>
<td style="text-align: center;">44.4</td>
<td style="text-align: center;">38.4</td>
<td style="text-align: center;">38.2</td>
<td style="text-align: center;">30.6</td>
<td style="text-align: center;">39.1</td>
<td style="text-align: center;">33.9</td>
<td style="text-align: center;">38.4</td>
<td style="text-align: center;">39.8</td>
<td style="text-align: center;">36.3</td>
<td style="text-align: center;">35.1</td>
<td style="text-align: center;">31.8</td>
<td style="text-align: center;">30.5</td>
<td style="text-align: center;">33.3</td>
<td style="text-align: center;">39.8</td>
<td style="text-align: right;">30.9</td>
<td style="text-align: right;">37.8</td>
<td style="text-align: center;">36.1</td>
</tr>
<tr>
<td style="text-align: center;">Translate-En<sup></sup></td>
<td style="text-align: center;">44.6</td>
<td style="text-align: center;">57.3</td>
<td style="text-align: center;">71.3</td>
<td style="text-align: center;">55.5</td>
<td style="text-align: center;">57.2</td>
<td style="text-align: center;">48.4</td>
<td style="text-align: center;">56.3</td>
<td style="text-align: center;">47.1</td>
<td style="text-align: center;">55.0</td>
<td style="text-align: center;">53.3</td>
<td style="text-align: center;">54.7</td>
<td style="text-align: center;">54.4</td>
<td style="text-align: center;">36.5</td>
<td style="text-align: center;">41.3</td>
<td style="text-align: center;">51.8</td>
<td style="text-align: center;">51.5</td>
<td style="text-align: right;">42.7</td>
<td style="text-align: right;">55.5</td>
<td style="text-align: center;">52.3</td>
</tr>
<tr>
<td style="text-align: center;">MindMerger-Soft<sup></sup></td>
<td style="text-align: center;">51.4</td>
<td style="text-align: center;">67.0</td>
<td style="text-align: center;">78.1</td>
<td style="text-align: center;">69.1</td>
<td style="text-align: center;">68.1</td>
<td style="text-align: center;">48.4</td>
<td style="text-align: center;">66.8</td>
<td style="text-align: center;">53.9</td>
<td style="text-align: center;">63.8</td>
<td style="text-align: center;">63.3</td>
<td style="text-align: center;">67.1</td>
<td style="text-align: center;">63.7</td>
<td style="text-align: center;">45.5</td>
<td style="text-align: center;">46.2</td>
<td style="text-align: center;">60.6</td>
<td style="text-align: center;">62.9</td>
<td style="text-align: right;">47.9</td>
<td style="text-align: right;">65.4</td>
<td style="text-align: center;">61.0</td>
</tr>
<tr>
<td colspan="20" style="text-align: center;"><span class="smallcaps">our methods</span></td>
</tr>
<tr>
<td style="text-align: center;"><strong>LinguaLIFT</strong></td>
<td style="text-align: right;">54.1</td>
<td style="text-align: right;">68.7</td>
<td style="text-align: right;">78.3</td>
<td style="text-align: right;">67.9</td>
<td style="text-align: right;">68.7</td>
<td style="text-align: right;">50.6</td>
<td style="text-align: right;">65.7</td>
<td style="text-align: right;">53.8</td>
<td style="text-align: right;">63.5</td>
<td style="text-align: right;">62.9</td>
<td style="text-align: right;">68.2</td>
<td style="text-align: right;">63.5</td>
<td style="text-align: right;">46.7</td>
<td style="text-align: right;">46.8</td>
<td style="text-align: right;">60.4</td>
<td style="text-align: right;">64.1</td>
<td style="text-align: center;">49.6</td>
<td style="text-align: center;">65.5</td>
<td style="text-align: center;">61.5</td>
</tr>
</tbody>
</table>

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

<span id="apdx:trainable-ablation" data-label="apdx:trainable-ablation"></span>

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

**Tabla 19.** Experimental Results on the MGSM Dataset. "LR." "HR." and "Avg." represent the average performance for low-resource languages, high-resource languages, and all languages, respectively. Following prior work (citation: LLaMA2), we classify Bn, Te, Th, and Sw as low-resource languages, while the remaining languages are categorized as high-resource. The asterisk symbol () denotes results taken directly from the published results of Zhu et al. (2024); Huang et al. (2024).

<table id="apdx-llama2-13b-mgsm">
<thead>
<tr>
<th style="text-align: left;"><strong>Llama-2-13B as base model</strong></th>
<th style="text-align: center;"><strong>Bn</strong></th>
<th style="text-align: center;"><strong>De</strong></th>
<th style="text-align: center;"><strong>En</strong></th>
<th style="text-align: center;"><strong>Es</strong></th>
<th style="text-align: center;"><strong>Fr</strong></th>
<th style="text-align: center;"><strong>Ja</strong></th>
<th style="text-align: center;"><strong>Ru</strong></th>
<th style="text-align: center;"><strong>Sw</strong></th>
<th style="text-align: center;"><strong>Th</strong></th>
<th style="text-align: center;"><strong>Zh</strong></th>
<th style="text-align: center;"><strong>LR.</strong></th>
<th style="text-align: center;"><strong>HR.</strong></th>
<th style="text-align: center;"><strong>Avg.</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="14" style="text-align: center;"><span class="smallcaps">baseline</span></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">MetaMath<sup></sup></td>
<td style="text-align: center;">11.6</td>
<td style="text-align: center;">64.8</td>
<td style="text-align: center;">67.2</td>
<td style="text-align: center;">65.2</td>
<td style="text-align: center;">65.2</td>
<td style="text-align: center;">42.8</td>
<td style="text-align: center;">63.6</td>
<td style="text-align: center;">7.60</td>
<td style="text-align: center;">6.40</td>
<td style="text-align: center;">49.2</td>
<td style="text-align: center;">8.53</td>
<td style="text-align: center;">59.7</td>
<td style="text-align: center;">44.4</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">MathOctopus-Parallel<sup></sup></td>
<td style="text-align: center;">35.2</td>
<td style="text-align: center;">44.4</td>
<td style="text-align: center;">53.2</td>
<td style="text-align: center;">48.0</td>
<td style="text-align: center;">48.4</td>
<td style="text-align: center;">43.2</td>
<td style="text-align: center;">47.6</td>
<td style="text-align: center;">42.8</td>
<td style="text-align: center;">46.8</td>
<td style="text-align: center;">48.8</td>
<td style="text-align: center;">41.6</td>
<td style="text-align: center;">47.7</td>
<td style="text-align: center;">45.8</td>
</tr>
<tr>
<td style="text-align: center;">QAlign-MetaMathQA<sup></sup></td>
<td style="text-align: center;">38.4</td>
<td style="text-align: center;">62.0</td>
<td style="text-align: center;">69.2</td>
<td style="text-align: center;">67.2</td>
<td style="text-align: center;">62.4</td>
<td style="text-align: center;">52.4</td>
<td style="text-align: center;">64.4</td>
<td style="text-align: center;">46.0</td>
<td style="text-align: center;">49.6</td>
<td style="text-align: center;">59.2</td>
<td style="text-align: center;">44.7</td>
<td style="text-align: center;">62.4</td>
<td style="text-align: center;">57.1</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">Translate-En<sup></sup></td>
<td style="text-align: center;">34.8</td>
<td style="text-align: center;">53.6</td>
<td style="text-align: center;">70.8</td>
<td style="text-align: center;">62.4</td>
<td style="text-align: center;">54.0</td>
<td style="text-align: center;">44.4</td>
<td style="text-align: center;">45.6</td>
<td style="text-align: center;">44.4</td>
<td style="text-align: center;">54.0</td>
<td style="text-align: center;">58.0</td>
<td style="text-align: center;">44.4</td>
<td style="text-align: center;">55.5</td>
<td style="text-align: center;">52.2</td>
</tr>
<tr>
<td style="text-align: center;">LangBridge<sup></sup></td>
<td style="text-align: center;">39.2</td>
<td style="text-align: center;">55.2</td>
<td style="text-align: center;">65.2</td>
<td style="text-align: center;">60.8</td>
<td style="text-align: center;">54.8</td>
<td style="text-align: center;">33.6</td>
<td style="text-align: center;">58.8</td>
<td style="text-align: center;">42.0</td>
<td style="text-align: center;">42.8</td>
<td style="text-align: center;">42.0</td>
<td style="text-align: center;">41.3</td>
<td style="text-align: center;">52.9</td>
<td style="text-align: center;">49.4</td>
</tr>
<tr>
<td style="text-align: center;">MindMerger-Soft<sup></sup></td>
<td style="text-align: center;">55.2</td>
<td style="text-align: center;">65.2</td>
<td style="text-align: center;">68.8</td>
<td style="text-align: center;">69.6</td>
<td style="text-align: center;">63.6</td>
<td style="text-align: center;">60.0</td>
<td style="text-align: center;">68.0</td>
<td style="text-align: center;">56.4</td>
<td style="text-align: center;">59.6</td>
<td style="text-align: center;">60.4</td>
<td style="text-align: center;">57.1</td>
<td style="text-align: center;">65.1</td>
<td style="text-align: center;">62.7</td>
</tr>
<tr>
<td colspan="14" style="text-align: center;"><span class="smallcaps">our methods</span></td>
</tr>
<tr>
<td style="text-align: center;"><strong>LinguaLIFT</strong></td>
<td style="text-align: center;">57.6</td>
<td style="text-align: center;">64.4</td>
<td style="text-align: center;">67.8</td>
<td style="text-align: center;">70.4</td>
<td style="text-align: center;">64.0</td>
<td style="text-align: center;">59.8</td>
<td style="text-align: center;">67.2</td>
<td style="text-align: center;">57.8</td>
<td style="text-align: center;">60.8</td>
<td style="text-align: center;">59.4</td>
<td style="text-align: center;">58.7</td>
<td style="text-align: center;">64.7</td>
<td style="text-align: center;">62.9</td>
</tr>
</tbody>
</table>

**Tabla 20.** Experimental Results on the MGSM Dataset. "LR." "HR." and "Avg." represent the average performance for low-resource languages, high-resource languages, and all languages, respectively. Following prior work (Huang et al., 2024), we classify Bn, Te, Th, and Sw as low-resource languages, while the remaining languages are categorized as high-resource. The asterisk symbol () denotes results taken directly from the published results of Zhu et al. (2024); Huang et al. (2024).

<table id="apdx-mistral-mgsm">
<thead>
<tr>
<th style="text-align: left;"><strong>Mistral-7B as base model</strong></th>
<th style="text-align: center;"><strong>Bn</strong></th>
<th style="text-align: center;"><strong>De</strong></th>
<th style="text-align: center;"><strong>En</strong></th>
<th style="text-align: center;"><strong>Es</strong></th>
<th style="text-align: center;"><strong>Fr</strong></th>
<th style="text-align: center;"><strong>Ja</strong></th>
<th style="text-align: center;"><strong>Ru</strong></th>
<th style="text-align: center;"><strong>Sw</strong></th>
<th style="text-align: center;"><strong>Th</strong></th>
<th style="text-align: center;"><strong>Zh</strong></th>
<th style="text-align: center;"><strong>LR.</strong></th>
<th style="text-align: center;"><strong>HR.</strong></th>
<th style="text-align: center;"><strong>Avg.</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="14" style="text-align: center;"><span class="smallcaps">baseline</span></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">MetaMath<sup></sup></td>
<td style="text-align: center;">38.4</td>
<td style="text-align: center;">70.4</td>
<td style="text-align: center;">78.0</td>
<td style="text-align: center;">71.2</td>
<td style="text-align: center;">70.8</td>
<td style="text-align: center;">50.8</td>
<td style="text-align: center;">67.2</td>
<td style="text-align: center;">16.8</td>
<td style="text-align: center;">34.8</td>
<td style="text-align: center;">57.2</td>
<td style="text-align: center;">30.0</td>
<td style="text-align: center;">66.5</td>
<td style="text-align: center;">55.6</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">MathOctopus-Parallel<sup></sup></td>
<td style="text-align: center;">44.0</td>
<td style="text-align: center;">50.0</td>
<td style="text-align: center;">58.4</td>
<td style="text-align: center;">53.2</td>
<td style="text-align: center;">47.2</td>
<td style="text-align: center;">48.0</td>
<td style="text-align: center;">49.6</td>
<td style="text-align: center;">51.6</td>
<td style="text-align: center;">48.8</td>
<td style="text-align: center;">51.6</td>
<td style="text-align: center;">48.1</td>
<td style="text-align: center;">51.1</td>
<td style="text-align: center;">50.2</td>
</tr>
<tr>
<td style="text-align: center;">QAlign-MetaMathQA<sup></sup></td>
<td style="text-align: center;">45.6</td>
<td style="text-align: center;">59.2</td>
<td style="text-align: center;">65.8</td>
<td style="text-align: center;">63.6</td>
<td style="text-align: center;">59.8</td>
<td style="text-align: center;">49.4</td>
<td style="text-align: center;">60.2</td>
<td style="text-align: center;">55.2</td>
<td style="text-align: center;">51.2</td>
<td style="text-align: center;">57.2</td>
<td style="text-align: center;">50.7</td>
<td style="text-align: center;">59.3</td>
<td style="text-align: center;">56.7</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">Translate-En<sup></sup></td>
<td style="text-align: center;">54.6</td>
<td style="text-align: center;">50.4</td>
<td style="text-align: center;">69.7</td>
<td style="text-align: center;">58.6</td>
<td style="text-align: center;">56.7</td>
<td style="text-align: center;">57.2</td>
<td style="text-align: center;">64.9</td>
<td style="text-align: center;">47.7</td>
<td style="text-align: center;">58.7</td>
<td style="text-align: center;">63.1</td>
<td style="text-align: center;">53.7</td>
<td style="text-align: center;">60.1</td>
<td style="text-align: center;">58.2</td>
</tr>
<tr>
<td style="text-align: center;">LangBridge<sup></sup></td>
<td style="text-align: center;">50.0</td>
<td style="text-align: center;">68.4</td>
<td style="text-align: center;">65.6</td>
<td style="text-align: center;">65.6</td>
<td style="text-align: center;">68.8</td>
<td style="text-align: center;">58.4</td>
<td style="text-align: center;">68.4</td>
<td style="text-align: center;">47.2</td>
<td style="text-align: center;">60.0</td>
<td style="text-align: center;">65.6</td>
<td style="text-align: center;">52.4</td>
<td style="text-align: center;">65.8</td>
<td style="text-align: center;">61.8</td>
</tr>
<tr>
<td style="text-align: center;">MindMerger-Soft<sup></sup></td>
<td style="text-align: center;">57.6</td>
<td style="text-align: center;">69.2</td>
<td style="text-align: center;">79.2</td>
<td style="text-align: center;">71.6</td>
<td style="text-align: center;">69.6</td>
<td style="text-align: center;">57.2</td>
<td style="text-align: center;">68.4</td>
<td style="text-align: center;">53.2</td>
<td style="text-align: center;">59.6</td>
<td style="text-align: center;">68.8</td>
<td style="text-align: center;">56.8</td>
<td style="text-align: center;">69.1</td>
<td style="text-align: center;">65.4</td>
</tr>
<tr>
<td colspan="14" style="text-align: center;"><span class="smallcaps">our methods</span></td>
</tr>
<tr>
<td style="text-align: center;"><strong>LinguaLIFT</strong></td>
<td style="text-align: center;">58.4</td>
<td style="text-align: center;">67.2</td>
<td style="text-align: center;">77.6</td>
<td style="text-align: center;">72.1</td>
<td style="text-align: center;">70.6</td>
<td style="text-align: center;">56.4</td>
<td style="text-align: center;">66.8</td>
<td style="text-align: center;">58.4</td>
<td style="text-align: center;">63.2</td>
<td style="text-align: center;">68.2</td>
<td style="text-align: center;">60.0</td>
<td style="text-align: center;">68.4</td>
<td style="text-align: center;">65.9</td>
</tr>
</tbody>
</table>

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

In some settings, we observed that incorporating multilingual reasoning data during the first stage can further enhance task transfer performance in the second stage. This is evidenced by the improved performance compared to the pure English-only reasoning task training. In the mixed supervision setting, we first fine-tune the language alignment layer during the Language-Align stage using multilingual reasoning instruction data <span class="smallcaps">GSM8KINSTRUCT</span> (Chen et al., 2023) and code-switching translation instruction data, followed by fine-tuning LLM with English-only instruction data, <span class="smallcaps">MetaMathQA</span>. The experimental results on <span class="smallcaps">MGSM</span> are presented in Figure 11. We find that incorporating additional multilingual supervision yields an average performance gain of 1.0% on low-resource reasoning tasks and allows the model to achieve comparable performance on high-resource reasoning tasks relative to the vanilla setting.

# E Supplementary Experiments

## E.1 Quantitative Analysis of the Correlation between Multilingual Alignment and Reasoning Performance

To evaluate the cross-lingual alignment capabilities of LLMs, we employ the multilingual sentence retrieval benchmark **Tatoeba** (Artetxe and Schwenk, 2019), a widely used dataset for assessing ML-LMs. The dataset includes up to 1,000 sentences per language, along with their English translations. In our study, we focus on a subset of the original benchmark that aligns with the language categories used in the reasoning test sets MGSM, ensuring consistency with our experimental design.

We adopt the evaluation procedure outlined in XTREME (Hu et al., 2020) to evaluate cross-lingual alignment. For each language pair, we compute the cosine similarity between sentences in the source language and their closest counterparts in the target language. Following the methods of Yang et al. (2021) and Xie et al. (2022), we use top-1 retrieval accuracy as a metric to quantify alignment between the two languages. A higher retrieval accuracy score indicates a stronger alignment.

**Figura 12.** Performances on MGSM and MSVAMP test set, where languages are sorted by decreasing retrieval accuracy. The trend indicates the mathemtical reasoning performance worsens along with the decreasing cross-lingual alignment degree.

**Tabla 24.** The ablation experiments of the selection of language alignment layer results on the MGSM Dataset. "LR." "HR." and "Avg." represent the average performance for low-resource languages, high-resource languages, and all languages, respectively. Following prior work (Huang et al., 2024), we classify Bn, Te, Th, and Sw as low-resource languages, while the remaining languages are categorized as high-resource.

| **Architecture** | **\# Params** | **Bn** | **De** | **En** | **Es** | **Fr** | **Ja** | **Ru** | **Sw** | **Te** | **Th** | **Zh** | **LR.** | **HR.** | **Avg.** |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Linear | 4 M | 53.1 | 61.6 | 63.9 | 63.8 | 56.6 | 46.0 | 56.6 | 49.0 | 44.4 | 50.6 | 52.2 | 49.3 | 57.2 | 54.4 |
| 2 layers MLP | 10 M | 54.4 | 62.0 | 64.8 | 63.6 | 56.8 | 50.0 | 60.4 | 55.6 | 54.0 | 57.6 | 54.0 | 55.4 | 58.8 | 57.6 |
| 3 layers MLP | 14 M | 53.8 | 62.8 | 66.6 | 56.6 | 58.8 | 50.0 | 59.4 | 53.8 | 53.6 | 58.2 | 53.4 | 54.9 | 58.2 | 57.0 |

**Tabla 25.** Spearman’s rank correlation coefficient ($\rho$) between retrieval accuracy (%) on Tatoeba and reasoning accuracy on MGSM. An asterisk () signifies a statistically significant correlation (p-value \< 0.05).

<table id="tab:rho">
<tbody>
<tr>
<td style="text-align: center;">Test set</td>
<td colspan="2" style="text-align: center;">MGSM</td>
<td colspan="2" style="text-align: center;">MSVAMP</td>
</tr>
<tr>
<td style="text-align: center;">Resource-Level</td>
<td style="text-align: center;">Low-Resource</td>
<td style="text-align: center;">High-Resource</td>
<td style="text-align: center;">Low-Resource</td>
<td style="text-align: center;">High-Resource</td>
</tr>
<tr>
<td style="text-align: center;"><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>ρ</mi><annotation encoding="application/x-tex">\rho</annotation></semantics></math></td>
<td style="text-align: center;">0.733<sup></sup></td>
<td style="text-align: center;">0.778<sup></sup></td>
<td style="text-align: center;">0.763<sup></sup></td>
<td style="text-align: center;">0.783<sup></sup></td>
</tr>
</tbody>
</table>

Figure 12 illustrates that the performance of low-resource languages lags significantly behind that of other languages in both mathematical reasoning tasks and the cross-lingual sentence retrieval task. It also shows that a higher alignment degree tends to correlate with improved mathematical reasoning performance. Additionally, we compute the Spearman’s rank correlation between the retrieval accuracy scores and reasoning performance, as presented in Table 25, which reveals a strong correlation between the two. Both the observed trend and the correlation coefficient confirm that the consistency of multilingual mathematical reasoning is closely tied to the cross-lingual alignment degree.

## E.2 Impact of Different Language Alignment Layers

In the main experiment, we utilize two layers of MLP as the language alignment layers to transfer multilingual alignment from the pre-trained multilingual encoder into the LLMs. We also conducted experiments to evaluate the performance of LinguaLIFT when ablating different mapping layers, as shown in Table 24. In contrast to the findings of Yoon et al. (2024), the two layers of MLP used in our main experiment achieved the best performance. In comparison, using a linear layer resulted in lower performance, likely due to its limited capacity to effectively adapt the pre-trained multilingual encoder to LLM, attributed to the smaller number of parameters.

## E.3 LinguaLIFT CoT Examples

We present three examples of CoT reasoning. These examples demonstrate LinguaLIFT’s ability to effectively understand low-resource languages and generate corresponding mathematical reasoning processes using the chain-of-thought reasoning strategy. Figure 13 illustrates a zero-shot CoT example from the MGSM Bengali dataset. The Vanilla Mono-SFT model produces incorrect reasoning, while the LinguaLIFT model generates a correct reasoning process, ultimately leading to the correct answer. Similar trends are observed in the Thai and Swahili, shown in Figures 14 and 15, where the LinguaLIFT model corrects the erroneous reasoning produced by the Mono-SFT model when presented with the same problems.

**Figura 13.** Example of zero-shot CoT generated by MetaMath-7B when prompted with an example from MGSM Bengali. Translation of the input and response is provided in .

**Figura 14.** Example of zero-shot CoT generated by MetaMath-7B when prompted with an example from MGSM Thai. Translation of the input and response is provided in .

**Figura 15.** Example of zero-shot CoT generated by MetaMath-7B when prompted with an example from MGSM Swahili. Translation of the input and response is provided in .

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
