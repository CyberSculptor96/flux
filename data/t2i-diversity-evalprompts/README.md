---
license: mit
dataset_info:
  features:
  - name: prompt
    dtype: string
  - name: type
    dtype: string
  - name: prompt_id
    dtype: int64
  splits:
  - name: train
    num_bytes: 797913
    num_examples: 4340
  download_size: 391461
  dataset_size: 797913
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# T2I-Diversity Evaluation Prompt Set

## Dataset Summary  
A compact, prompt set for text-to-image containing **4340 English prompts** organised into four increments of descriptive density (minimal → long).  
The 1 085  base prompts were sourced from DrawBench and Parti-Prompts, each **≤ 10 tokens**. For every base prompt, GPT-4o generated *short*, *medium* and *long* rewrites that keep the subject constant while progressively adding detail. This results in 4 variants per concept, enabling stress-tests of text-to-image models across prompt complexity without changing the underlying scene. 


## Intended Uses & Metrics  
* **Primary use** – quantitative or qualitative evaluation of T2I models on:
  * **Text alignment / instruction following** (e.g. PickScore, VQA-Score)  
  * **Aesthetics & diversity** (e.g. LAION-Aesthetics, LPIPS)  
  * **Robustness to prompt length / verbosity**  
* **Secondary use** – prompt-engineering research, ablation studies on caption density trade-offs described in the accompanying paper.

## Dataset Structure  

- prompt — the literal prompt text.
- type — one of the four density levels (original, short_gpt4o, medium_gpt4o, long_gpt4o).
- prompt_id — index of the original minimal prompt; shared by its three rewrites.

## Citation


If you use this dataset, please cite:
```bibtex
@article{brack2025howtotrain,
  title={How to Train your Text-to-Image Model: Evaluating Design Choices for Synthetic Training Captions},
  author={Manuel Brack and Sudeep Katakol and Felix Friedrich and Patrick Schramowski and Hareesh Ravi and Kristian Kersting and Ajinkya Kale},
  journal={arXiv preprint arXiv:2506.16679},
  year={2025}
}
```