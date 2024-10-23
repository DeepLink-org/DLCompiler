# Tuning framework


## Overall Design

dlBLAS is meant to be an operator library for Triton-based operators. As such, kernel developers register their kernels to the library and users ask for a operator by giving operator name and input tensors.

it improves over Triton's autotuner in the following ways:

- **operator selection**: given the same operator, e.g. matmul, there may be different kernel implementations; we want to find the best one based on the input tensors.

- **customized configuration search**: instead of enumerating all possible kernel configurations (BLOCK_SIZE etc.), we want to use advanced algorithm e.g. a bayesian optimizer to search for the best configurations. This needs a flexbile definition of search space and search policy. For DSA hardware, the configuration space is large.

- **caching** the best operator implementation and kernel configurations are cached for the input tensors. It is shape, dtype, device specific.


## Install 

```
cd path-to-Triton_DEEPLINK
```

1. install deps

```
pip install -r python/dlBLAS/requirements.txt
```

2. install packages

```
pip install -e python/dlBLAS/
```
