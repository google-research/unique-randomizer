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
"""Supports sampling unique sequences of discrete random choices."""

import abc
import math
import typing
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import scipy.special

from unique_randomizer import stochastic_beam_search as sbs


def log_subtract(x: float, y: float) -> float:
  """Returns log(exp(x) - exp(y)), or negative infinity if x <= y."""
  # Inspired by https://stackoverflow.com/questions/778047.
  return x + np.log1p(-np.exp(np.minimum(y - x, 0)))


def sample_log_distribution(log_distribution: np.ndarray) -> np.int64:
  """Samples from an unnormalized probability distribution in log space.

  Args:
    log_distribution: A 1-D numpy array of unnormalized log probabilities.

  Returns:
    An int in the range [0, len(log_distribution)), sampled according to the
    given distribution.
  """
  # A slower but more numerically stable solution is discussed at
  # https://stats.stackexchange.com/questions/64081. However, we expect that
  # as the randomizer runs, the probability distribution at each node should
  # not be skewed significantly more than the initial provided distribution,
  # since we will sample more frequently from high-probability choices until
  # the probabilities "even out".
  unnormalized = np.exp(log_distribution - np.max(log_distribution))
  distribution = unnormalized / np.sum(unnormalized)
  return np.random.choice(np.arange(len(distribution)), p=distribution)


class _TrieNode(object):
  """A trie node for UniqueRandomizer.

  Attributes:
    parent: The _TrieNode parent of this node, or None if this node is the root.
    index_in_parent: The index of this node in the parent, or None if this node
      is the root.
    children: A list of _TrieNode children. A child may be None if it is not
      expanded yet. The entire list will be None if this node has never sampled
      a child yet. The list will be empty if this node is a leaf in the trie.
    unsampled_log_masses: A numpy array containing the current (unsampled) log
      probability mass of each child, or None if this node has never sampled a
      child yet.
    data: A dict capable of storing arbitrary user-provided data for the node.
    sbs_child_state_cache: Used for caching children's states when sampling
      batches.
  """

  def __init__(self, parent: Optional['_TrieNode'],
               index_in_parent: Optional[int]) -> None:
    """Initializes a _TrieNode.

    Args:
      parent: The parent of this node, or None if this node is the root.
      index_in_parent: This node's index in the parent node, or None if this
        node is the root.
    """
    self.parent = parent
    self.index_in_parent = index_in_parent
    self.children = None
    self.unsampled_log_masses = None
    self.data = {}
    self.sbs_child_state_cache = None

  def initial_log_mass_if_not_sampled(self) -> float:
    """Returns this node's initial log probability mass.

    This assumes that no samples have been drawn from this node yet.
    """
    # If no samples have been drawn yet, the unsampled log mass equals the
    # desired initial log mass.
    return (self.parent.unsampled_log_masses[self.index_in_parent]
            # If the node is the root, the initial log mass is 0.0.
            if self.parent else 0.0)

  def sample_child(
      self,
      initial_distribution: Union[np.ndarray, List[float], None]
  ) -> Tuple['_TrieNode', int]:
    """Returns a child _TrieNode according to the given initial distribution.

    This will create the child _TrieNode if it does not already exist.

    Args:
      initial_distribution: A 1-D numpy array containing the initial probability
        distribution that this node should use.

    Returns:
      A tuple of the child _TrieNode and the child's index.
    """
    if not self.children:
      # This is the first sample. Set up children.
      self.children = [None] * len(initial_distribution)
      self.unsampled_log_masses = (np.log(initial_distribution) +
                                   self.initial_log_mass_if_not_sampled())
      # Faster to choose from initial_distribution when it's still accurate
      # (i.e., on the first sample).
      child_index = np.random.choice(np.arange(len(initial_distribution)),
                                     p=initial_distribution)
    else:
      child_index = sample_log_distribution(self.unsampled_log_masses)

    child = self.children[child_index]
    if not child:
      child = self.children[child_index] = _TrieNode(
          parent=self, index_in_parent=child_index)
    return child, int(child_index)

  def mark_leaf(self) -> None:
    """Marks this node as a leaf."""
    self.children = []

  def exhausted(self) -> bool:
    """Returns whether all of the mass at this node has been sampled."""
    # Distinguish [] and None.
    if self.children == []:  # pylint: disable=g-explicit-bool-comparison
      return True
    if self.unsampled_log_masses is None:
      return False  # This node is not a leaf but has never been sampled from.
    return all(np.isneginf(log_mass) for log_mass in self.unsampled_log_masses)

  def mark_mass_sampled(self, log_mass: float) -> None:
    """Recursively subtracts log_mass from this node and its ancestors."""
    if not self.parent:
      return
    if self.exhausted():
      new_log_mass = np.NINF
    else:
      new_log_mass = log_subtract(
          self.parent.unsampled_log_masses[self.index_in_parent], log_mass)
    self.parent.unsampled_log_masses[self.index_in_parent] = new_log_mass
    self.parent.mark_mass_sampled(log_mass)

  def needs_probabilities(self) -> bool:
    """Returns whether this node needs probabilities."""
    return self.children is None


class AllSequencesSampledError(Exception):
  """Raised when all possible sequences have already been sampled."""


class Randomizer(object, metaclass=abc.ABCMeta):
  """Samples sequences of discrete random choices.

  The `sample_*` methods all return an int in the range [0, num_choices).
  """

  def __init__(self) -> None:
    """Initializes this Randomizer object."""
    self._num_sequences_sampled = 0
    self._exhausted = False

  @abc.abstractmethod
  def sample_distribution(
      self,
      probability_distribution: Union[np.ndarray, List[float], None]) -> int:
    """Samples from a given probability distribution (as a list of floats)."""

  def sample_boolean(self, probability_1: float = 0.5) -> int:
    """Samples from a Bernoulli distribution with a given probability of 1."""
    return self.sample_distribution([1 - probability_1, probability_1])

  def sample_uniform(self, num_choices: int) -> int:
    """Samples from a uniform distribution over a given number of choices."""
    return self.sample_distribution(np.ones(num_choices) / num_choices)

  @abc.abstractmethod
  def mark_sequence_complete(self) -> float:
    """Used to mark a complete sequence of choices.

    Returns:
      The log probability of the finished sequence, with respect to the
      initial (given) probability distribution.
    """

  def num_sequences_sampled(self) -> int:
    """Returns the number of complete sequences of choices sampled so far."""
    return self._num_sequences_sampled

  def exhausted(self) -> bool:
    """Returns whether all possible sequences of choices have been sampled."""
    return self._exhausted

  @abc.abstractmethod
  def fraction_sampled(self) -> float:
    """Returns the total probability mass that has been sampled."""

  @abc.abstractmethod
  def needs_probabilities(self) -> bool:
    """Returns whether the current node requires probabilities.

    In UniqueRandomizer, a _TrieNode will need probabilities if it has never
    sampled a child before. Then, it will no longer need probabilities to
    sample a child, since it stores its own updated probabilities. This can
    enable the client to avoid unnecessarily recomputing probabilities.
    """


class UniqueRandomizer(Randomizer):
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

  The `sample_*` methods all return an int in the range [0, num_choices). All of
  these methods raise AllSequencesSampledError if all possible sequences of
  choices have already been sampled.

  Attributes:
    current_node: The current node in the trie.
  """

  def __init__(self) -> None:
    """Initializes a UniqueRandomizer object."""
    super(UniqueRandomizer, self).__init__()
    self._root_node = _TrieNode(None, None)
    self.current_node = self._root_node

  def sample_distribution(
      self,
      probability_distribution: Union[np.ndarray, List[float], None]) -> int:
    """Samples from a given probability distribution (as a list of floats)."""
    if self._exhausted:
      raise AllSequencesSampledError('All sequences of choices have been '
                                     'sampled already.')
    self.current_node, choice_index = self.current_node.sample_child(
        probability_distribution)
    return choice_index

  def mark_sequence_complete(self) -> float:
    """Used to mark a complete sequence of choices.

    Returns:
      The log probability of the finished sequence, with respect to the
      initial (given) probability distribution.
    """
    self._num_sequences_sampled += 1
    self.current_node.mark_leaf()
    log_sampled_mass = self.current_node.initial_log_mass_if_not_sampled()
    self.current_node.mark_mass_sampled(log_sampled_mass)
    self.current_node = self._root_node
    self._exhausted = self._root_node.exhausted()
    return float(log_sampled_mass)

  def fraction_sampled(self) -> float:
    """Returns the total probability mass that has been sampled."""
    if self._exhausted:
      return 1.0
    if not self._root_node.children:
      # The root node has never sampled a child before.
      return 0.0
    return float(1.0 - np.exp(scipy.special.logsumexp(
        self._root_node.unsampled_log_masses)))

  def needs_probabilities(self) -> bool:
    """Returns whether the current node requires probabilities."""
    return self.current_node.needs_probabilities()

  def sample_batch(
      self,
      child_log_probability_fn: Callable[[List[sbs.State]], List[np.ndarray]],
      child_state_fn: Callable[[List[Tuple[sbs.State, int]]],
                               List[Tuple[Union[sbs.State, sbs.Output], bool]]],
      root_state: sbs.State,
      k: int) -> List[sbs.BeamNode]:
    """Samples a batch of outputs using Stochastic Beam Search.

    Nodes in the beam include "states" which can be anything but must contain
    enough information to:
      1. Define a consistent ordering of all children of the node.
      2. Enumerate the probabilities of all children.
      3. Produce the state of the child with a given index.

    Args:
      child_log_probability_fn: A function that takes a list of states and
        returns the log probabilities of the child states of each input state.

      child_state_fn: A function that takes a list of (state, i) pairs and maps
        each to a (ith_child, is_leaf) pair. If ith_child is a leaf state,
        is_leaf should be True, and ith_child will potentially be an actual
        sampled item that should be returned by stochastic_beam_search (it may
        have a different form than other non-leaf states).

      root_state: The state of the root node.
      k: The desired number of samples.

    Returns:
      A list of up to k BeamNode objects, corresponding to the sampled leaves.
    """
    # A state here contains a _TrieNode and the client's state.

    def wrapper_child_log_probability_fn(
        randomizer_states: List[Tuple[_TrieNode, sbs.State]]
    ) -> List[np.ndarray]:
      """Computes child probabilities while updating the trie."""
      results = [None] * len(randomizer_states)
      unexpanded_client_states = []
      unexpanded_indices = []

      for i, (node, client_state) in enumerate(randomizer_states):
        if node.unsampled_log_masses is None:
          # We have never computed this node's child probabilities before.
          unexpanded_client_states.append(client_state)
          unexpanded_indices.append(i)
        else:
          # This node already has unsampled_log_masses set. We just need to
          # normalize them.
          log_unnormalized = node.unsampled_log_masses
          unnormalized = np.exp(log_unnormalized - np.max(log_unnormalized))
          results[i] = np.log(unnormalized / np.sum(unnormalized))

      # Use client's child_log_probability_fn to get probabilities for
      # unexpanded states.
      if unexpanded_client_states:
        client_fn_results = child_log_probability_fn(unexpanded_client_states)
        for i, log_probs in zip(unexpanded_indices, client_fn_results):
          results[i] = log_probs
          node = randomizer_states[i][0]
          node.unsampled_log_masses = (log_probs
                                       + node.initial_log_mass_if_not_sampled())

      return typing.cast(List[np.ndarray], results)

    def wrapper_child_state_fn(
        randomizer_state_index_pairs: List[Tuple[Tuple[_TrieNode, sbs.State],
                                                 int]]
    ) -> List[Tuple[Union[sbs.State, sbs.Output], bool]]:
      """Computes child states while updating the trie."""
      results = [None] * len(randomizer_state_index_pairs)
      unexpanded_client_state_index_pairs = []
      unexpanded_indices = []

      for i, ((node, client_state), child_index) in enumerate(
          randomizer_state_index_pairs):

        # Initialize children structures if needed.
        if node.children is None:
          num_children = len(typing.cast(np.ndarray, node.unsampled_log_masses))
          node.children = [None] * num_children
          node.sbs_child_state_cache = [None] * num_children

        if node.children[child_index] is None:
          # This child has not been created before.
          unexpanded_client_state_index_pairs.append(
              (client_state, child_index))
          unexpanded_indices.append(i)
        else:
          # The child has been created before.
          child_client_state, child_is_leaf = (
              node.sbs_child_state_cache[child_index])
          results[i] = ((node.children[child_index], child_client_state),
                        child_is_leaf)

      # Use client's child_log_probability_fn to get child client states.
      if unexpanded_client_state_index_pairs:
        client_fn_results = child_state_fn(unexpanded_client_state_index_pairs)
        for i, (child_client_state, child_is_leaf) in zip(unexpanded_indices,
                                                          client_fn_results):
          (node, _), child_index = randomizer_state_index_pairs[i]
          child_node = _TrieNode(parent=node, index_in_parent=child_index)
          node.children[child_index] = child_node
          node.sbs_child_state_cache[child_index] = (child_client_state,
                                                     child_is_leaf)
          results[i] = ((child_node, child_client_state), child_is_leaf)

      return typing.cast(List[Tuple[Union[sbs.State, sbs.Output], bool]],
                         results)

    randomizer_beam_nodes = sbs.stochastic_beam_search(
        child_log_probability_fn=wrapper_child_log_probability_fn,
        child_state_fn=wrapper_child_state_fn,
        root_state=(self._root_node, root_state),
        k=k)

    # Update probabilities and remove _TrieNode parts of the states.
    client_beam_nodes = []
    for beam_node in randomizer_beam_nodes:
      leaf_node, client_state = beam_node.output
      log_sampled_mass = leaf_node.initial_log_mass_if_not_sampled()
      leaf_node.mark_mass_sampled(log_sampled_mass)
      client_beam_nodes.append(beam_node._replace(output=client_state))

    self._exhausted = self._root_node.exhausted()
    return client_beam_nodes


class NormalRandomizer(Randomizer):
  """A randomizer where all sequences of choices are independent.

  As opposed to a UniqueRandomizer, a NormalRandomizer can return duplicate
  sequences of choices.

  This does not keep track of the fraction of the search space that was sampled.
  Thus, fraction_sampled() returns a sentinel value of -1.0.
  """

  def __init__(self) -> None:
    """Initializes a NormalRandomizer object."""
    super(NormalRandomizer, self).__init__()
    self._log_probability_sum = 0.0

  def sample_distribution(
      self,
      probability_distribution: Union[np.ndarray, List[float], None]) -> int:
    """Samples from a given probability distribution."""
    index = int(np.random.choice(np.arange(len(probability_distribution)),
                                 p=probability_distribution))
    self._log_probability_sum += math.log(probability_distribution[index])
    return index

  def mark_sequence_complete(self) -> float:
    """Used to mark a complete sequence of choices.

    Returns:
      The log probability of the finished sequence, with respect to the
      initial (given) probability distribution.
    """
    result = self._log_probability_sum
    self._log_probability_sum = 0.0
    return result

  def fraction_sampled(self) -> float:
    """Returns a sentinel value of -1.0. See class docstring."""
    return -1.0

  def needs_probabilities(self) -> bool:
    """Returns whether the current node requires probabilities (always True)."""
    return True
