# NI Sampling: Accelerating Discrete Diffusion Sampling by Token Order Opmization

This is the official implementaion of paper [NI Sampling: Accelerating Discrete Diffusion Sampling by Token Order Opmization](https://openreview.net/forum?id=rrD1U0Izt5&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)) (ICLR26). 

## Overview

**Neural Indicator Sampling (NI Sampling)** is a novel framework designed to accelerate the sampling process of diffusion Large Language Models (LLMs). By training a lightweight neural indicator, we can dynamically predict which tokens should be sampled at each step, significantly reducing redundant computations while maintaining high generation quality. We will release the training code after re-arrangement.

## Prepare

Please install the following dependencies:
```
pip install torch==2.1.2 transformers==4.45.2 accelerate
```

## Download Trained Indicator

We release the checkpoints of our trained indicator at [this link](https://huggingface.co/jsttlgdkycy/NI_Sampling/blob/main/indicator_LLaDA.pth). Please download it first.

## Evaluation

We provide commands to evaluate the indicator on several benchmarks. 

#### Key Hyperparameters
* `prob_threshold`: Confidence threshold for sampling.
* `indicator_threshold`: Threshold for the neural indicator.
* `block_length` / `gen_length & steps`: Adjust these to test under different efficiency settings.

#### Baseline (Confidence Threshold)
```
# GSM8K
accelerate launch --main_process_port 11450 --num_processes 1 eval_llada.py --tasks gsm8k --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=64,prob_threshold=0.9

# MATH
accelerate launch --main_process_port 11450 --num_processes 1 eval_llada.py --tasks minerva_math --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,prob_threshold=0.9

# HumanEval
accelerate launch --main_process_port 11450 --num_processes 1 eval_llada.py --tasks humaneval --model llada_dist --confirm_run_unsafe_code --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,prob_threshold=0.9

# MBPP
accelerate launch --main_process_port 11450 --num_processes 1 eval_llada.py --tasks mbpp --model llada_dist --confirm_run_unsafe_code --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,prob_threshold=0.9
```

#### NI Sampling
```
# GSM8K
accelerate launch --main_process_port 11450 --num_processes 1 eval_llada.py --tasks gsm8k --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=64,prob_threshold=0.95,indicator_path="/PATH/TO/INDICATOR",indicator_threshold=0.89,use_indicator=True

# MATH
accelerate launch --main_process_port 11450 --num_processes 1 eval_llada.py --tasks minerva_math --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,prob_threshold=0.95,indicator_path="/PATH/TO/INDICATOR",indicator_threshold=0.89,use_indicator=True

# HumanEval
accelerate launch --main_process_port 11450 --num_processes 1 eval_llada.py --tasks humaneval --model llada_dist --confirm_run_unsafe_code --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,prob_threshold=0.95,indicator_path="/PATH/TO/INDICATOR",indicator_threshold=0.89,use_indicator=True

# MBPP
accelerate launch --main_process_port 11450 --num_processes 1 eval_llada.py --tasks mbpp --model llada_dist --confirm_run_unsafe_code --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,prob_threshold=0.95,indicator_path="/PATH/TO/INDICATOR",indicator_threshold=0.89,use_indicator=True
```

## Acknowledgement

This codebase is heavily based on [LLaDA](https://github.com/ML-GSAI/LLaDA). We thank the authors for their contribution.

## Citation

If you find our work helpful, please consider citing:
```
@inproceedings{liuni,
  title={NI Sampling: Accelerating Discrete Diffusion Sampling by Token Order Optimization},
  author={Liu, Enshu and Ning, Xuefei and Wang, Yu and Lin, Zinan},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```

