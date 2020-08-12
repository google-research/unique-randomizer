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
"""Defines, expands, and enumerates a PCFG using UniqueRandomizer and SBS."""

import math
import re
import time
import typing
from typing import Dict, List, Text, Tuple

import numpy as np

from unique_randomizer import stochastic_beam_search as sbs
from unique_randomizer import unique_randomizer as ur


################################################################################
# Probabilistic Context-Free Grammar (PCFG) definition.

Nonterminal = typing.NamedTuple('Nonterminal', [('expansions', List[Text]),
                                                ('probabilities', List[float])])

GRAMMAR_EXAMPLE = {
    'sentence': Nonterminal(  # (1+1+6)*1953 + 1953^2 + 1 = 3.83M possibilities.
        expansions=['{phrase}.',
                    'miraculously, {phrase}!',
                    '{phrase}, while {phrase}.',
                    'the {animal} exclaimed, "{phrase}!"',
                    'hello world!'],
        probabilities=[0.4, 0.2, 0.2, 0.1, 0.1]),
    'phrase': Nonterminal(  # 3 + 12*5 + 6*5*3 + 5*6*5*12 = 1953 possibilities.
        expansions=['the {fruit} tasted delicious',
                    'the {thing} was {adjective}',
                    'the {animal} ate the {adjective} {fruit}',
                    'the {adjective} {animal} found a {adjective} {thing}'],
        probabilities=[0.4, 0.4, 0.1, 0.1]),
    'thing': Nonterminal(  # 12 possibilities.
        expansions=['{animal}', '{fruit}', 'pebble', 'crystal', 'diamond'],
        probabilities=[0.2, 0.3, 0.4, 0.09, 0.01]),
    'animal': Nonterminal(  # 6 possibilities.
        expansions=['crow', 'goldfish', 'dog', 'monkey', 'rhino', 't-rex'],
        probabilities=[0.3, 0.3, 0.2, 0.15, 0.049, 0.001]),
    'fruit': Nonterminal(  # 3 possibilities.
        expansions=['apple', 'pear', 'dragonfruit'],
        probabilities=[0.5, 0.4, 0.1]),
    'adjective': Nonterminal(  # 5 possibilities.
        expansions=['colorful', 'dull', 'large', 'small', 'rare'],
        probabilities=[0.2, 0.1, 0.35, 0.3, 0.05]),
}  # type: Dict[Text, Nonterminal]


class GrammarError(Exception):
  """Raised when the grammar is malformed."""


def verify_grammar(grammar_dict: Dict[Text, Nonterminal]) -> None:
  """Raises a GrammarError if the grammar is malformed."""
  for nonterminal_name in grammar_dict:
    nonterminal = grammar_dict[nonterminal_name]
    if len(nonterminal.expansions) != len(nonterminal.probabilities):
      raise GrammarError(
          "Nonterminal '{}' has {} expansions and {} probabilities.".format(
              nonterminal_name, len(nonterminal.expansions),
              len(nonterminal.probabilities)))
    if abs(sum(nonterminal.probabilities) - 1.0) > 1e-8:
      raise GrammarError(
          "Nonterminal '{}' has probabilities summing to {}.".format(
              nonterminal_name, sum(nonterminal.probabilities)))
    for expansion in nonterminal.expansions:
      for must_expand in re.findall(r'{.+?}', expansion):
        inner_nonterminal = must_expand[1:-1]
        if inner_nonterminal not in grammar_dict:
          raise GrammarError(
              "Nonterminal '{}' has an expansion '{}' with unknown inner "
              "nonterminal '{}'.".format(
                  nonterminal_name, expansion, inner_nonterminal))


################################################################################
# Exhaustive enumeration of the grammar expansions.


def _enumerate_helper(current_expansion: Text,
                      log_probability_sum: float,
                      result_list: List[Tuple[Text, float]],
                      grammar_dict: Dict[Text, Nonterminal]) -> None:
  """Recursive helper for `enumerate_nonterminal`."""
  match = re.search(r'{.+?}', current_expansion)
  if match:
    to_replace = match.group()
    nonterminal_name = to_replace[1:-1]
    nonterminal = grammar_dict[nonterminal_name]
    for substitution, probability in zip(nonterminal.expansions,
                                         nonterminal.probabilities):
      _enumerate_helper(current_expansion.replace(to_replace, substitution, 1),
                        log_probability_sum + math.log(probability),
                        result_list, grammar_dict)
  else:
    result_list.append((current_expansion, log_probability_sum))


def enumerate_nonterminal(
    nonterminal_name: Text,
    grammar_dict: Dict[Text, Nonterminal]) -> List[Tuple[Text, float]]:
  """Enumerates all nonterminal expansions paired with log-probabilities."""
  result_list = []  # List[Tuple[Text, float]]
  _enumerate_helper('{' + nonterminal_name + '}', 0.0, result_list,
                    grammar_dict)
  return result_list


################################################################################
# Using UniqueRandomizer to sample unique expansions of the grammar.


def _expand_nonterminal(nonterminal_name: Text,
                        grammar_dict: Dict[Text, Nonterminal],
                        randomizer: ur.Randomizer,
                        prob_sleep_millis: float = 0.0) -> Text:
  """Expands a nonterminal using the given randomizer."""
  expansion = '{' + nonterminal_name + '}'
  while True:
    match = re.search(r'{.+?}', expansion)
    if not match:
      break  # Nothing left to expand.
    must_expand = match.group()
    nonterminal = grammar_dict[must_expand[1:-1]]
    if randomizer.needs_probabilities():
      probabilities = nonterminal.probabilities
      # Mimic a scenario where it's expensive to compute the probabilities, such
      # as when using a neural model to predict the nonterminal's replacement.
      time.sleep(prob_sleep_millis / 1000)
    else:
      probabilities = None
    substitution = nonterminal.expansions[
        randomizer.sample_distribution(probabilities)]
    expansion = expansion.replace(must_expand, substitution, 1)
  return expansion


def _sample_expansions(
    num_unique_samples: int,
    nonterminal_name: Text,
    grammar_dict: Dict[Text, Nonterminal],
    randomizer: ur.Randomizer,
    prob_sleep_millis: float = 0.0) -> Tuple[Dict[Text, float], int]:
  """Samples expansions, returning the samples, probabilities, and # tries."""
  expansions_dict = {}  # Dict from the expansion text to its log probability.
  num_tries = 0
  while (len(expansions_dict) < num_unique_samples
         and not randomizer.exhausted()):
    expansion = _expand_nonterminal(nonterminal_name, grammar_dict, randomizer,
                                    prob_sleep_millis)
    log_probability = randomizer.mark_sequence_complete()
    expansions_dict[expansion] = log_probability
    num_tries += 1
  return expansions_dict, num_tries


def sample_with_ur(
    num_unique_samples: int,
    nonterminal_name: Text,
    grammar_dict: Dict[Text, Nonterminal],
    prob_sleep_millis: float = 0.0) -> Tuple[Dict[Text, float], int]:
  """Samples unique expansions of a nonterminal with UniqueRandomizer."""
  return _sample_expansions(num_unique_samples, nonterminal_name, grammar_dict,
                            ur.UniqueRandomizer(), prob_sleep_millis)


def sample_with_rejection(
    num_unique_samples: int,
    nonterminal_name: Text,
    grammar_dict: Dict[Text, Nonterminal],
    prob_sleep_millis: float = 0.0) -> Tuple[Dict[Text, float], int]:
  """Samples unique expansions of a nonterminal using rejection sampling."""
  return _sample_expansions(num_unique_samples, nonterminal_name, grammar_dict,
                            ur.NormalRandomizer(), prob_sleep_millis)


################################################################################
# Using Stochastic Beam Search to sample unique expansions of the grammar.


def sample_with_sbs(
    num_unique_samples: int,
    nonterminal_name: Text,
    grammar_dict: Dict[Text, Nonterminal],
    prob_sleep_millis: float = 0.0) -> Tuple[Dict[Text, float], int]:
  """Samples unique expansions of a nonterminal using Stochastic Beam Search."""

  # A state is a string, e.g., 'the {thing} was {adjective}'.

  def child_log_probability_fn(states: List[Text]) -> List[np.ndarray]:
    """Produces log probabilities of child states."""
    results = []
    for state in states:
      match = re.search(r'{.+?}', state)
      if match:
        to_replace = match.group()
        nonterminal = grammar_dict[to_replace[1:-1]]
        results.append(np.log(nonterminal.probabilities))
        # Mimic a scenario where it's expensive to compute the probabilities,
        # such as when using a neural model to predict the nonterminal's
        # replacement.
        time.sleep(prob_sleep_millis / 1000)
      else:
        raise ValueError('Leaf state encountered but not expected: {}'.format(
            state))
    return results

  def child_state_fn(
      state_index_pairs: List[Tuple[Text, int]]) -> List[Tuple[Text, bool]]:
    """Produces child states."""
    results = []
    for state, child_index in state_index_pairs:
      match = re.search(r'{.+?}', state)
      if match:
        to_replace = match.group()
        nonterminal = grammar_dict[to_replace[1:-1]]
        expansion = nonterminal.expansions[child_index]
        child_state = state.replace(to_replace, expansion, 1)
        child_is_leaf = not bool(re.search(r'{.+?}', child_state))
        results.append((child_state, child_is_leaf))
      else:
        raise ValueError('Leaf state encountered but not expected: {}'.format(
            state))
    return results

  beam_nodes = sbs.stochastic_beam_search(
      child_log_probability_fn=child_log_probability_fn,
      child_state_fn=child_state_fn,
      root_state='{' + nonterminal_name + '}',
      k=num_unique_samples)
  expansions_and_probabilities = [(node.output, node.log_probability)
                                  for node in beam_nodes]
  expansions_dict = dict(expansions_and_probabilities)
  num_tries = len(expansions_dict)
  return expansions_dict, num_tries
