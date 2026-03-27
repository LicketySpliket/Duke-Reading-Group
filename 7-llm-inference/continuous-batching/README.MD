# Continuous Batching
Continuous batching is used in LLM inference servers like llama.cpp and vLLM to handle multiple inference requests with just one model pipeline. Here is a simplified demo in PyTorch of continuous batching.

Reference: [HuggingFace blog](https://huggingface.co/blog/continuous_batching)

## Files
`model.py`: one layer transformer model

`demo.py`: demo with simulated requests with predetermined sequence lengths

`model.ipynb`: contains the same transformer model but with some additional explanation

## Setup
```
$ uv venv
$ source .venv/bin/activate
$ uv pip install torch numpy
```

## Run Demo
```
$ python demo.py
```

## Some Simplifications Compared to Real Inference Servers
- The KV cache is allocated for the max sequence length for all possible slots. In an actual scenario this would be a waste of space.
- Assumes that all prefill can be handled in one forward pass
