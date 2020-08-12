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
"""This implementation of UniqueRandomizer is simple for educational purposes.

This implementation is designed to match the pseudocode in the UniqueRandomizer
paper. Compared to the better implementation in unique_randomizer.py, the
bare-bones implementation in this file lacks:

  * Efficiency tricks, such as storing unsampled probability masses as a single
    numpy array
  * Some useful functions, such as computing the fraction of the space sampled
  * The Randomizer -> UniqueRandomizer/NormalRandomizer object hierarchy
  * UniqueRandomizer with SBS for batching
"""

from typing import List, Optional

import numpy as np


def _log_subtract(x: float, y: float) -> float:
  """Returns log(exp(x) - exp(y)), or negative infinity if x <= y."""
  # Inspired by https://stackoverflow.com/questions/778047.
  return x + np.log1p(-np.exp(np.minimum(y - x, 0)))


def _sample_log_distribution(log_distribution: List[float]) -> int:
  """Samples from an unnormalized probability distribution in log space."""
  # A slower but more numerically stable solution is discussed at
  # https://stats.stackexchange.com/questions/64081. However, we expect that
  # as the randomizer runs, the probability distribution at each node should
  # not be skewed significantly more than the initial provided distribution,
  # since we will sample more frequently from high-probability choices until
  # the probabilities "even out".
  unnormalized = np.exp(log_distribution - np.max(log_distribution))
  distribution = unnormalized / np.sum(unnormalized)
  return int(np.random.choice(np.arange(len(distribution)), p=distribution))


class _TrieNode(object):
  """A trie node for UniqueRandomizer.

  Attributes:
    parent: The _TrieNode parent of this node, or None if this node is the root.
    children: A list of _TrieNode children. The list will be None if this node
      has never sampled a child yet. The list will be empty if this node is a
      leaf.
    unsampled_log_mass: The current unsampled log probability mass at this node.
  """

  def __init__(self,
               parent: Optional['_TrieNode'],
               unsampled_log_mass: float) -> None:
    """Initializes a _TrieNode."""
    self.parent = parent
    self.children = None
    self.unsampled_log_mass = unsampled_log_mass


def _node_exhausted(node: _TrieNode) -> bool:
  """Returns whether all of the mass at this node has been sampled."""
  if node.children == []:  # Distinguish [] and None. pylint: disable=g-explicit-bool-comparison
    return True  # This node is a leaf.
  if node.children is None:
    return False  # This node is not a leaf but has never been sampled from.
  return all(child.unsampled_log_mass == np.NINF for child in node.children)


class UniqueRandomizer(object):
  """Samples unique sequences of discrete random choices.

  When using a UniqueRandomizer object to provide randomness, the client
  algorithm must be deterministic and behave identically when given a constant
  sequence of choices.

  When a sequence of choices is complete, the client algorithm must call
  `mark_sequence_complete()`. This will update the internal data so that the
  next sampled choices form a new sequence, which is guaranteed to be different
  from previous complete sequences.

  Choices returned by a UniqueRandomizer object respect the initial probability
  distributions provided by the client algorithm, conditioned on the constraint
  that a complete sequence of choices cannot be sampled more than once.

  `sample_distribution()` returns an int in the range [0, num_choices), raising
  a ValueError if all possible sequences of choices have already been sampled.
  """

  def __init__(self) -> None:
    """Initializes a UniqueRandomizer object."""
    self.root_node = _TrieNode(None, 0.0)
    self.current_node = self.root_node

  def sample_distribution(self, distribution: List[float]) -> int:
    """Samples from a given probability distribution."""
    if _node_exhausted(self.current_node):
      raise ValueError('This node has been completely sampled already.')

    current = self.current_node
    if not current.children:
      # This is the first sample from the current node. Set up its children.
      # Note that the current node's unsampled_log_mass is still the initial
      # log probability mass, since the node has never been sampled from before.
      current.children = [
          _TrieNode(parent=current, unsampled_log_mass=float(log_mass))
          for log_mass in np.log(distribution) + current.unsampled_log_mass]

    child_index = _sample_log_distribution([child.unsampled_log_mass
                                            for child in current.children])

    self.current_node = current.children[child_index]
    return child_index

  def mark_sequence_complete(self) -> float:
    """Ends a sequence of choices and returns its log probability."""
    self.current_node.children = []  # This marks the current node as a leaf.

    # Update unsampled log masses for all ancestors.
    sampled_log_mass = self.current_node.unsampled_log_mass
    node = self.current_node
    while node:
      if _node_exhausted(node):
        node.unsampled_log_mass = np.NINF
      else:
        node.unsampled_log_mass = _log_subtract(node.unsampled_log_mass,
                                                sampled_log_mass)
      node = node.parent

    self.current_node = self.root_node
    return sampled_log_mass

  def exhausted(self) -> bool:
    """Returns whether all possible sequences of choices have been sampled."""
    return _node_exhausted(self.root_node)
