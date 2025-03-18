<div align='center'>
<h1>FastCuRL</h1>
<h1>Improving RL Training Efficiency of R1-like Reasoning Models via Curriculum-Guided Iterative Lengthening</h1>

<!-- TODO:  Thread, Paper, Dataset, Weights-->
[![Paper](https://img.shields.io/badge/paper-5f16a8?style=for-the-badge&logo=arxiv&logoColor=white)](FastCuRL.pdf)
[![Blog](https://img.shields.io/badge/Blog-3858bf?style=for-the-badge&logo=homepage&logoColor=white)](https://github.com/nick7nlp/FastCuRL)
<a href="https://huggingface.co/Nickyang/FastCuRL-1.5B-Preview" target="_blank"><img alt="Hugging Face"
    src="https://img.shields.io/badge/HuggingFace-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor"/></a>
</div>

We release **FastCuRL-1.5B-Preview**, a slow-thinking reasoning model that achieves 43.1% accuracy on the AIME 2024 benchmark! We adapt a novel curriculum-guided iterative lengthening reinforcement learning to the distilled 1.5B model and observe continuous performance improvement as training steps increase. To better reproduce our work and advance research progress, we open-source our code, model, and data.


## Key Results

We report Pass@1 accuracy averaged over 16 samples for each problem.

| Model | AIME 2024 | MATH 500 | AMC 2023 | Minerva Math | OlympiadBench | Avg. |
|-------|-----------|-----------|-----------|--------------|---------------|------|
| Qwen2.5-Math-7B-Instruct | 13.3 | 79.8 | 50.6 | 34.6 | 40.7 | 43.8 |
| rStar-Math-7B | 26.7 | 78.4 | 47.5 | - | 47.1 | - |
| Eurus-2-7B-PRIME | 26.7 | 79.2 | 57.8 | 38.6 | 42.1 | 48.9 |
| Qwen2.5-7B-SimpleRL | 26.7 | 82.4 | 62.5 | <strong>39.7</strong> | 43.3 | 50.9 |
| DeepSeek-R1-Distill-Qwen-1.5B | 28.8 | 82.8 | 62.9 | 26.5 | 43.3 | 48.9 |
| Still-1.5B | 32.5 | 84.4 | 66.7 | 29.0 | 45.4 | 51.6 |
| DeepScaleR-1.5B-Preview | 43.1 | 87.8 | 73.6 | 30.2 | 50.0 | 57.0 |
| <strong>FastCuRL-1.5B-Preview</strong> | <strong>43.1</strong> | <strong>88.0</strong> | <strong>74.2</strong> | 31.6 | <strong>50.4</strong> | <strong>57.5</strong> |


## Acknowledgements

- Our training experiments are powered by our heavily modified fork of [verl](https://github.com/volcengine/verl) and [deepscaler](https://github.com/agentica-project/deepscaler).
- Our model is trained on top of [`DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).
