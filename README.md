# UniqueRandomizer

## Overview

UniqueRandomizer is a data structure for sampling outputs of a randomized
program, such as a neural sequence model, _incrementally_ and _without
replacement_.

*   _Incremental_ sampling: Instead of sampling a large batch of outputs all at
    once, as with beam search, UniqueRandomizer provides samples one at a time.
    This enables flexibility in stopping criteria, such as stopping the sampling
    process as soon as a satisfactory output is found.
*   Sampling _without replacement_: In many applications, a neural model is used
    to produce candidate solutions to some search or optimization problem. In
    such applications it is usually desirable to consider _unique_ candidate
    solutions, since duplicates are typically not useful.

For more details, refer to our paper,
[Incremental Sampling Without Replacement for Sequence Models](https://arxiv.org/abs/2002.09067),
published at ICML 2020.

BibTeX entry:

```
@article{shi2020uniquerandomizer,
  title = {Incremental Sampling Without Replacement for Sequence Models},
  author = {Kensen Shi and David Bieber and Charles Sutton},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  year = {2020}
}
```


## Installation

```
python3 -m pip install --user unique-randomizer
```

This package requires Python 3. The above command automatically installs the
following dependencies as well:

* absl-py >= 0.6.1
* numpy >= 1.15.4
* scipy >= 1.1.0


## Usage

To use UniqueRandomizer, first identify the program or function that you wish to
draw unique samples from, such as the `draw_sample` function in the following
example:

```
def draw_sample(sequence_model, state):
  """Draws a sample (a sequence of token indices) from the sequence model."""
  tokens = []
  token = BOS
  for i in range(MAX_LEN):
    probs, state = sequence_model(token, state)
    token = np.random.choice(np.arange(len(probs)), p=probs)
    if token == EOS:
      break
    tokens.append(token)
  return tokens
```

Note that `draw_sample` can take inputs and can use control flow such as loops,
conditionals, and recursion. There are only two constraints on the `draw_sample`
function:

1.  It must be deterministic given the inputs, except for random choices
    provided by `np.random.choice` (or some other method of selecting a random
    index given a discrete probability distribution).
2.  Two different sequences of random choices must lead to `draw_sample`
    returning different outputs.

Next, add a `UniqueRandomizer` object as an input to `draw_sample`, and use its
`sample_distribution` function to replace `np.random.choice`:

```diff
- def draw_sample(sequence_model, state):
+ def draw_sample(sequence_model, state, randomizer):
    """Draws a sample (a sequence of token indices) from the sequence model."""
    tokens = []
    token = BOS
    for i in range(MAX_LEN):
      probs, state = sequence_model(token, state)
-     token = np.random.choice(np.arange(len(probs)), p=probs)
+     token = randomizer.sample_distribution(probs)
      if token == EOS:
        break
      tokens.append(token)
    return tokens
```

Finally, a simple loop around `draw_sample` can collect unique samples, as
follows:

```
def draw_unique_samples(model, state, num_samples):
  """Draws multiple unique samples from the sequence model."""
  samples = []
  randomizer = unique_randomizer.UniqueRandomizer()
  for _ in range(num_samples):
    samples.append(draw_sample(model, state, randomizer))
    randomizer.mark_sequence_complete()
  return samples
```

## Code Samples

We include a few code samples that demonstrate how to use UniqueRandomizer:

 * `examples/weighted_coin_flips.py`: This provides a very simple example of
   using UniqueRandomizer. The function `flip_two_weighted_coins` simulates
   flipping a pair of weighted coins. The `sample_flips_without_replacement`
   function then uses UniqueRandomizer to efficiently sample outputs of
   `flip_two_weighted_coins` without replacement.

 * `examples/expand_grammar.py`: This defines a Probabilistic Context-Free
   Grammar (PCFG), as well as methods to sample elements of the grammar without
   replacement by using UniqueRandomizer, rejection sampling, and Stochastic
   Beam Search (SBS). The script `examples/expand_grammar_main.py` enables easy
   comparison between the different sampling methods under different scenarios.

 * `examples/sequence_example.py`: This implements sampling without replacement
   from a sequence model, using UniqueRandomizer, Batched UniqueRandomizer,
   rejection sampling, and SBS. The script `examples/sequence_example_main.py`
   enables easy comparison between the different sampling methods under
   different scenarios.

## Disclaimer

This is not an officially supported Google product.
