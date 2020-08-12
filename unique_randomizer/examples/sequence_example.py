# Copyright 2020 The UniqueRandomizer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Demonstrates sampling without replacement for simple sequences."""

import functools
import hashlib
import time
import typing
from typing import List, Optional, Text, Tuple

import numpy as np

from unique_randomizer import stochastic_beam_search as sbs
from unique_randomizer import unique_randomizer as ur


class FakeSequenceModel(object):
  """A sequence "model" that deterministically assigns probabilities to tokens.

  This enables experiments that compare the performance of different
  sampling-without-replacement strategies in different scenarios.

  Vocabulary item 0 is an end-of-sequence marker.

  Attributes:
    sequence_length: The (maximum) number of tokens in a sequence.
    vocabulary_size: The number of distinct sequence elements.
    temperature: The temperature to apply to the probabilities, which are
      sampled from Exponential(1) and normalized. The default temperature of 1.0
      means the probabilities are not changed. A smaller temperature (closer to
      zero) means the probabilities are more skewed, and a larger temperature
      means the probabilities are closer to uniform.
    sleep_millis: The time to sleep, in milliseconds, before returning a
      non-empty probability distribution. This is used to simulate expensive
      operations, such as running a real deep learning model.
    batch_sleep_millis: The time to sleep, in milliseconds, for each batch
      element, when batching probability computations.
    probability_count: The number of calls to next_token_log_probabilities where
      the sequence is not yet complete.
    batch_probability_count: The number of calls to
      next_token_log_probabilities_batched.
  """

  def __init__(self,
               sequence_length: int,
               vocabulary_size: int,
               temperature: float = 1.0,
               sleep_millis: float = 0.0,
               batch_sleep_millis: float = 0.0) -> None:
    """Initializes parameters."""
    self.sequence_length = sequence_length
    self.vocabulary_size = vocabulary_size
    self.temperature = temperature
    self.sleep_millis = sleep_millis
    self.batch_sleep_millis = batch_sleep_millis
    self.probability_count = 0
    self.batch_probability_count = 0

  def sequence_complete(self, prefix: List[int]) -> bool:
    """Returns whether the prefix represents a complete sequence."""
    return (len(prefix) == self.sequence_length or
            # Do not implicitly convert a numpy array to bool.
            (len(prefix) > 0 and prefix[-1] == 0))  # pylint: disable=g-explicit-length-test

  def next_token_log_probabilities(self, prefix: List[int]) -> np.ndarray:
    """Returns log probabilities for the next token as a 1-D numpy array.

    If this sequence is finished, then the returned numpy array will be empty.

    Args:
      prefix: A list of token IDs for the current prefix of the sequence.
    """
    if self.sequence_complete(prefix):
      return np.array([])
    self.probability_count += 1

    original_random_state = np.random.get_state()
    prefix_hash = (int(hashlib.md5(str(prefix).encode()).hexdigest(), 16)
                   % (10 ** 8))
    np.random.seed(prefix_hash)

    unnormalized = np.power(np.random.exponential(size=self.vocabulary_size),
                            1.0 / self.temperature)
    log_probabilities = np.log(unnormalized / np.sum(unnormalized))
    time.sleep(self.sleep_millis / 1000.0)

    np.random.set_state(original_random_state)
    return log_probabilities

  def next_token_log_probabilities_batched(
      self, prefixes: List[List[int]]) -> np.ndarray:
    """Returns log probabilities for the next token in a batched way.

    If a sequence is finished, then the corresponding numpy array will be empty.

    Args:
      prefixes: A list of prefixes, where a prefix is a list of token IDs.
    """
    old_sleep_millis = self.sleep_millis
    self.sleep_millis = 0.0
    results = np.array(
        [self.next_token_log_probabilities(prefix) for prefix in prefixes])
    self.sleep_millis = old_sleep_millis
    self.batch_probability_count += 1
    time.sleep(len(prefixes) * self.batch_sleep_millis / 1000.0)
    return results

  def __repr__(self) -> Text:
    return ('FakeSequenceModel(\n'
            '    sequence_length={},\n'
            '    vocabulary_size={},\n'
            '    temperature={},\n'
            '    sleep_millis={},\n'
            '    batch_sleep_millis={})'.format(
                self.sequence_length, self.vocabulary_size, self.temperature,
                self.sleep_millis, self.batch_sleep_millis))


def sample_with_rejection(model: FakeSequenceModel,
                          num_samples: int) -> List[List[int]]:
  """Samples using naive rejection sampling."""
  # Note: If the number of samples is too large, this may run for a long time
  # (or forever).
  samples = []
  sample_set = set()

  while len(samples) < num_samples:
    # Create a sample.
    prefix = []
    while not model.sequence_complete(prefix):
      distribution = np.exp(model.next_token_log_probabilities(prefix))
      next_token = np.random.choice(np.arange(len(distribution)),
                                    p=distribution)
      prefix.append(next_token)

    # If the sample was already seen, reject it.
    sequence_tuple = tuple(prefix)
    if sequence_tuple in sample_set:
      continue
    sample_set.add(sequence_tuple)
    samples.append(prefix)

  return samples


def sample_with_unique_randomizer(model: FakeSequenceModel,
                                  num_samples: int) -> List[List[int]]:
  """Samples using the UniqueRandomizer."""
  randomizer = ur.UniqueRandomizer()
  samples = []

  while len(samples) < num_samples and not randomizer.exhausted():
    # Create a sample. These are guaranteed to be unique.
    prefix = []
    while not model.sequence_complete(prefix):
      distribution = np.exp(model.next_token_log_probabilities(prefix))
      next_token = randomizer.sample_distribution(distribution)
      prefix.append(next_token)
    randomizer.mark_sequence_complete()
    samples.append(prefix)

  return samples


def sample_with_unique_randomizer_cached(
    model: FakeSequenceModel,
    num_samples: int) -> List[List[int]]:
  """Samples using the UniqueRandomizer with caching."""
  randomizer = ur.UniqueRandomizer()
  samples = []
  while len(samples) < num_samples and not randomizer.exhausted():
    # Create a sample. These are guaranteed to be unique.
    prefix = []
    while not model.sequence_complete(prefix):
      if randomizer.needs_probabilities():
        distribution = np.exp(model.next_token_log_probabilities(prefix))
      else:
        distribution = None
      next_token = randomizer.sample_distribution(distribution)
      prefix.append(next_token)
    randomizer.mark_sequence_complete()
    samples.append(prefix)

  return samples


State = Optional[Tuple[List[int], int]]


def _child_log_probability_fn(states: List[List[int]],
                              model: FakeSequenceModel) -> List[np.ndarray]:
  """For SBS and UR batched."""
  return typing.cast(List[np.ndarray],
                     model.next_token_log_probabilities_batched(states))


def _child_state_fn(
    state_index_pairs: List[Tuple[List[int], int]],
    model: FakeSequenceModel,
) -> List[Tuple[List[int], bool]]:
  """For SBS and UR batched."""
  results = []
  for state, index in state_index_pairs:
    new_state = state + [index]
    results.append((new_state, model.sequence_complete(new_state)))
  return results


def sample_with_stochastic_beam_search(
    model: FakeSequenceModel, num_samples: int) -> List[List[int]]:
  """Samples using Stochastic Beam Search."""
  # A state is a sequence prefix.
  beam_nodes = sbs.stochastic_beam_search(
      child_log_probability_fn=functools.partial(_child_log_probability_fn,
                                                 model=model),
      child_state_fn=functools.partial(_child_state_fn, model=model),
      root_state=[],
      k=num_samples)
  return [node.output for node in beam_nodes]


def sample_with_unique_randomizer_batched(
    model: FakeSequenceModel,
    num_samples: int,
    batch_size: int) -> List[List[int]]:
  """Samples using UniqueRandomizer with SBS for batching."""
  randomizer = ur.UniqueRandomizer()
  outputs = []
  while not randomizer.exhausted() and len(outputs) < num_samples:
    this_batch_size = min(batch_size, num_samples - len(outputs))
    beam_nodes = randomizer.sample_batch(
        child_log_probability_fn=functools.partial(_child_log_probability_fn,
                                                   model=model),
        child_state_fn=functools.partial(_child_state_fn, model=model),
        root_state=[],
        k=this_batch_size)
    outputs.extend(node.output for node in beam_nodes)
  return outputs
