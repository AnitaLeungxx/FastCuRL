<div align='center'>
<h2>FastCuRL: Curriculum Reinforcement Learning with Progressive Context Extension for Efficient Training R1-like Reasoning Models</h2>

<!-- TODO:  Thread, Paper, Dataset, Weights-->
[![Paper](https://img.shields.io/badge/paper-5f16a8?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.17287)
<a href="https://huggingface.co/Nickyang/FastCuRL-1.5B-Preview" target="_blank"><img alt="Hugging Face"
    src="https://img.shields.io/badge/HuggingFace-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor"/></a>
[![Dataset](https://img.shields.io/badge/Datasets-4d8cd8?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/Nickyang/FastCuRL/)
</div>


We release **FastCuRL-1.5B-Preview**, a slow-thinking reasoning model that **outperforms** :chart_with_upwards_trend: the previous SoTA *DeepScaleR-1.5B-Preview* with :rocket: **50% training steps**! We adapt a novel curriculum-guided iterative lengthening reinforcement learning to the *DeepSeek-R1-Distill-Qwen-1.5B* and observe continuous performance improvement as training steps increase. To better reproduce our work and advance research progress, we open-source our code, model, and data.


**The current version of the uploaded paper is unfinished and will be updated. We aim to share some findings in time.**


### Key Results

Training Details. 

| Model | Training Steps  | Training Stages | Number of GPUs Used in Each Stage |
|-------|-----------|-----------|-----------|
| DeepScaleR-1.5B-Preview | ~1,750 | 3 | 8, 16, 32 |
| **FastCuRL-1.5B-Preview** | ~860 | 4 | **8**, **8**, **8**, **8** |

Here, we uniformly set the batch size to 128 for counting training steps, meaning two steps with batch size 64 are counted as one with batch size 128. 

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


### Training Data
Following DeepScaleR, our training dataset consists of 40,315 unique problem-answer pairs compiled from:
- AIME problems (1984-2023)
- AMC problems (before 2023)
- Omni-MATH dataset
- Still dataset

<p>
<img src="img/length_distribution.png" width = "50%" />
</p>

In FastCuRL, we propose a simple condition-sensitive data segmentation approach, which splits the original dataset into three subsets.

[Segmented Datasets](https://huggingface.co/datasets/Nickyang/FastCuRL)

<p>
<img src="img/analysis.png" width = "90%" />
</p>

### Training Strategy
In FastCuRL, we propose a curriculum-guided iterative lengthening approach for improving the RL training efficiency of R1-like reasoning models. Specifically, the four stages are as follows:

- Stage I (8K context,∼160 steps)

<p>
<img src="img/fastcurl_stage1.png" width = "90%" />
</p>

- Stage II (16K context,∼590 steps)

<p>
<img src="img/fastcurl_stage2.png" width = "90%" />
</p>

- Stage III (24K context,∼230 steps)

<p>
<img src="img/fastcurl_stage3.png" width = "90%" />
</p>

- Stage IV (16K context,∼580 steps)

<p>
<img src="img/fastcurl_stage4.png" width = "90%" />
</p>

Overall, we find that during the whole training process, the steps chosen for stage transitions mainly occurred toward the end of each stage, further highlighting the efficiency of the proposed FastCuRL approach.

### Entropy Loss

We observe the changes in entropy loss during the training process and find differences between our training strategy and that of DeepScaleR.

<p>
<img src="img/entropy_loss.png" width = "80%" />
</p>


### Training Scripts

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS

# Run 8K context length training, 160 steps
bash ./scripts/train/run_fastcurl_1.5b_8k_stage1.sh | tee -a fastcurl-1.5b-stage1.log

# Run 16K context length training, 590 steps
bash ./scripts/train/run_fastcurl_1.5b_16k_stage2.sh | tee -a fastcurl-1.5b-stage2.log

# Run 24K context length training, 230 steps
bash ./scripts/train/run_fastcurl_1.5b_24k_stage3.sh | tee -a fastcurl-1.5b-stage3.log

# Run 16K context length training, 580 steps
bash ./scripts/train/run_fastcurl_1.5b_16k_stage4.sh | tee -a fastcurl-1.5b-stage4.log

```

### Evaluate

```bash
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=./fastcurl/data/test/xxx.parquet \
    data.output_path=${OUTPUT_DIR}/xxx.parquet \
    data.n_samples=16 \
    data.batch_size=2048 \
    model.path=${MODEL_PATH} \
    rollout.temperature=0.6 \
    rollout.response_length=32768 \
    rollout.top_k=-1 \
    rollout.top_p=1 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.tensor_model_parallel_size=1
```

### Citation

```bibtex
@misc{fastcurl,
      title={FastCuRL: Curriculum Reinforcement Learning with Progressive Context Extension for Efficient Training R1-like Reasoning Models}, 
      author={Mingyang Song and Mao Zheng and Zheng Li and Wenjie Yang and Xuan Luo and Yue Pan and Feng Zhang},
      year={2025},
      eprint={2503.17287},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.17287}, 
}
```


### Acknowledgements

- Our model is trained on top of [`DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).
- Our training experiments are powered by our heavily modified fork of [verl](https://github.com/volcengine/verl).
- We directly use DeepScaleR's [code](https://github.com/agentica-project/deepscaler) to finish our experiments. However, we have modified parts of the code related to naming conflicts to avoid confusion.
