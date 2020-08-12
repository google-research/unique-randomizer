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
"""Stochastic Beam Search (SBS).

The technique is described in the following paper:

  Wouter Kool, Herke van Hoof, and Max Welling.
  Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling
  Sequences Without Replacement.
  https://arxiv.org/pdf/1903.06059.pdf

The implementation is slightly generalized from the description in the paper,
handling the case where not all leaves are at the same level of the tree.
"""

import typing
from typing import Any, Callable, List, Tuple, Union

import numpy as np

State = Any  # Type alias. pylint: disable=invalid-name
Output = Any  # Type alias. pylint: disable=invalid-name
BeamNode = typing.NamedTuple('BeamNode', [('output', Output),
                                          ('log_probability', float),
                                          ('gumbel', float)])


def sample_gumbels_with_maximum(log_probabilities, target_max):
  """Samples a set of gumbels which are conditioned on having a given maximum.

  Based on https://gist.github.com/wouterkool/a3bb2aae8d6a80f985daae95252a8aa8.

  Args:
    log_probabilities: The log probabilities of the items to sample Gumbels for.
    target_max: The desired maximum sampled Gumbel.

  Returns:
    The sampled Gumbels.
  """
  gumbels = np.random.gumbel(loc=log_probabilities)
  max_gumbel = np.max(gumbels)

  # Use equations (23) and (24) in Appendix B.3 of the SBS paper.

  # Note: Numpy may warn "divide by zero encountered in log1p" on the next code
  # line. This is normal and expected, since one element of
  # `gumbels - max_gumbel` should be zero. The math fixes itself later on, and
  # that element ends up being shifted to target_max.
  v = target_max - gumbels + np.log1p(-np.exp(gumbels - max_gumbel))
  return target_max - np.maximum(v, 0) - np.log1p(np.exp(-np.abs(v)))


def stochastic_beam_search(
    child_log_probability_fn: Callable[[List[State]], List[np.ndarray]],
    child_state_fn: Callable[[List[Tuple[State, int]]],
                             List[Tuple[Union[State, Output], bool]]],
    root_state: State,
    k: int) -> List[BeamNode]:
  """Stochastic Beam Search.

  Nodes in the beam include "states" which can be anything but must contain
  enough information to:
    1. Define a consistent ordering of all children of the node.
    2. Enumerate the probabilities of all children.
    3. Produce the state of the child with a given index.

  Args:
    child_log_probability_fn: A function that takes a list of states and returns
      the log probabilities of the child states of each input state.

    child_state_fn: A function that takes a list of (state, i) pairs and maps
      each to a (ith_child, is_leaf) pair. If ith_child is a leaf state, is_leaf
      should be True, and ith_child will potentially be an actual sampled item
      that should be returned by stochastic_beam_search (it may have a different
      form than other non-leaf states).

    root_state: The state of the root node. This cannot be a leaf node.
    k: The desired number of samples.

  Returns:
    A list of up to k BeamNode objects, corresponding to the sampled leaves.
  """
  if k <= 0:
    return []

  # Data for nodes currently on the beam.
  leaf_log_probs = []
  leaf_gumbels = []
  leaf_outputs = []
  internal_states = [root_state]
  internal_log_probs = [0.0]
  internal_gumbels = [0.0]

  # Expand internal nodes until there are none left to expand.
  while internal_states:
    # Compute child probabilities for all internal nodes.
    child_log_probs_list = child_log_probability_fn(internal_states)

    # Avoid creating tons of BeamNode objects for children of internal nodes
    # (there may be beam_size*node_arity children). Instead pack data into lists
    # for efficiency.
    all_log_probs = []
    all_gumbels = []
    all_states = []
    all_child_indices = []

    # Sample Gumbels for children of internal nodes.
    for node_state, node_log_prob, node_gumbel, child_log_probs in zip(
        internal_states, internal_log_probs, internal_gumbels,
        child_log_probs_list):
      # Note: Numpy may warn "divide by zero encountered in log" on the next
      # code line. This is normal and expected if a child has zero probability.
      # We prevent zero-probability children from being added to the beam.
      log_probabilities = child_log_probs + node_log_prob
      good_indices = np.where(log_probabilities != np.NINF)[0]
      log_probabilities = log_probabilities[good_indices]
      gumbels = sample_gumbels_with_maximum(log_probabilities, node_gumbel)

      all_log_probs.extend(log_probabilities)
      all_gumbels.extend(gumbels)
      all_states.extend([node_state] * len(log_probabilities))
      all_child_indices.extend(good_indices)

    # Select the k best candidates.
    num_internal_candidates = len(all_gumbels)
    num_leaf_candidates = len(leaf_gumbels)
    if k >= num_internal_candidates + num_leaf_candidates:
      # No change to leaf nodes, since all are selected.
      to_expand_states = list(zip(all_states, all_child_indices))
      to_expand_log_probs = all_log_probs
      to_expand_gumbels = all_gumbels

    else:
      # Select the unsorted top k in O(num_candidates) time.
      all_gumbels.extend(leaf_gumbels)
      top_k_indices = np.argpartition(all_gumbels, -k)[-k:]
      to_expand_states = []
      to_expand_log_probs = []
      to_expand_gumbels = []
      leaf_indices = []
      for i in top_k_indices:
        if i >= num_internal_candidates:
          leaf_indices.append(i - num_internal_candidates)
        else:
          to_expand_states.append((all_states[i], all_child_indices[i]))
          to_expand_log_probs.append(all_log_probs[i])
          to_expand_gumbels.append(all_gumbels[i])
      leaf_log_probs = [leaf_log_probs[i] for i in leaf_indices]
      leaf_gumbels = [leaf_gumbels[i] for i in leaf_indices]
      leaf_outputs = [leaf_outputs[i] for i in leaf_indices]

    # Among selected candidates, expand non-leaf nodes.
    internal_log_probs = []
    internal_gumbels = []
    internal_states = []
    child_states = child_state_fn(to_expand_states)
    for log_prob, gumbel, (child_state, is_leaf) in zip(
        to_expand_log_probs, to_expand_gumbels, child_states):
      if is_leaf:
        leaf_log_probs.append(log_prob)
        leaf_gumbels.append(gumbel)
        leaf_outputs.append(child_state)
      else:
        internal_log_probs.append(log_prob)
        internal_gumbels.append(gumbel)
        internal_states.append(child_state)

  # Pack the leaf data into BeamNode objects.
  sampled_nodes = []
  for log_prob, gumbel, output in zip(
      leaf_log_probs, leaf_gumbels, leaf_outputs):
    sampled_nodes.append(BeamNode(output=output, log_probability=log_prob,
                                  gumbel=gumbel))

  # Sort the beam in order of decreasing Gumbels. This corresponds to the order
  # one would get by sampling one-at-a-time without replacement.
  return sorted(sampled_nodes, key=lambda x: x.gumbel, reverse=True)
