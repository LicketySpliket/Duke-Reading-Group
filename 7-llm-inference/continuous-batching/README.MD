# Continuous Batching
Continuous batching is used in LLM inference servers like llama.cpp and vLLM to handle multiple inference requests with just one model pipeline. Here is a simplified demo in PyTorch of continuous batching.

Reference: [HuggingFace blog](https://huggingface.co/blog/continuous_batching)

## Files
`model.py`: one layer transformer model

`demo.py`: demo with simulated requests with predetermined sequence lengths

`model.ipynb`: contains the same transformer model but with some additional explanation

## Run Demo
```
python demo.py
```