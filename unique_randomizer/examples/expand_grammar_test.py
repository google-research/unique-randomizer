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
"""Tests for expand_grammar.py."""

import math

from absl.testing import absltest
from absl.testing import parameterized
import mock
import scipy.special

from unique_randomizer import unique_randomizer as ur
from unique_randomizer.examples import expand_grammar


class GrammarTest(parameterized.TestCase):
  """Tests for the grammar and grammar-manipulating functions alone."""

  def test_grammar_validates(self):
    expand_grammar.verify_grammar(expand_grammar.GRAMMAR_EXAMPLE)

  @parameterized.named_parameters(
      ('probabilities_and_expansions_mismatch',
       {'my_nonterminal': expand_grammar.Nonterminal(
           expansions=['a', 'b', 'c'],
           probabilities=[0.2, 0.8])},
       "Nonterminal 'my_nonterminal' has 3 expansions and 2 probabilities"),
      ('probability_sum',
       {'my_nonterminal': expand_grammar.Nonterminal(
           expansions=['a', 'b', 'c'],
           probabilities=[0.2, 0.8, 0.1])},
       "Nonterminal 'my_nonterminal' has probabilities summing to 1.1"),
      ('inner_expansion',
       {'my_nonterminal': expand_grammar.Nonterminal(
           expansions=['a', 'b', '{c}'],
           probabilities=[0.2, 0.7, 0.1])},
       "Nonterminal 'my_nonterminal' has an expansion '{c}' with unknown inner "
       "nonterminal 'c'"))
  def test_verify_grammar(self, grammar, error_regex):
    with self.assertRaisesRegex(expand_grammar.GrammarError, error_regex):
      expand_grammar.verify_grammar(grammar)

  @parameterized.named_parameters(
      ('leaf_0', 'animal', 0, 'crow'),
      ('leaf_1', 'animal', 1, 'goldfish'),
      ('non_leaf_0', 'sentence', 0, 'the apple tasted delicious.'),
      ('non_leaf_1', 'sentence', 1, 'miraculously, the pear was dull!'))
  def test_expand_nonterminal(self, nonterminal_name, randomizer_index,
                              expected_expansion):
    mock_randomizer = ur.NormalRandomizer()
    mock_randomizer.sample_distribution = mock.MagicMock(
        return_value=randomizer_index)
    self.assertEqual(
        expand_grammar._expand_nonterminal(
            nonterminal_name, expand_grammar.GRAMMAR_EXAMPLE, mock_randomizer),
        expected_expansion)

  @parameterized.named_parameters(
      ('fruit', 'fruit', 3,
       {'apple': 0.5, 'pear': 0.4, 'dragonfruit': 0.1}),
      ('thing', 'thing', 12,
       {'goldfish': 0.06, 't-rex': 0.0002, 'apple': 0.15, 'pebble': 0.4}),
      ('phrase', 'phrase', 1953,
       {'the pear was dull': 0.4 * 0.3 * 0.4 * 0.1}),
      ('sentence', 'sentence', (1 + 1 + 6)*1953 + 1953**2 + 1,
       {
           'miraculously, the pear was dull!': 0.2 * 0.4 * 0.3 * 0.4 * 0.1,
           'the rhino exclaimed, "the dragonfruit tasted delicious!"':
               0.1 * 0.049 * 0.4 * 0.1,
           'hello world!': 0.1,
       }))
  def test_enumerate_nonterminal(self, nonterminal_name,
                                 expected_num_expansions,
                                 expected_probability_dict):
    actual_results = expand_grammar.enumerate_nonterminal(
        nonterminal_name, expand_grammar.GRAMMAR_EXAMPLE)
    self.assertLen(actual_results, expected_num_expansions)

    actual_results_dict = dict(actual_results)
    # No duplicate expansions.
    self.assertLen(actual_results_dict, expected_num_expansions)

    # Check the given expansions. (expected_probability_dict does not need to
    # be exhaustive.)
    for expansion in expected_probability_dict:
      self.assertAlmostEqual(actual_results_dict[expansion],
                             math.log(expected_probability_dict[expansion]))

    # The probabilities need to sum to 1.
    self.assertAlmostEqual(
        scipy.special.logsumexp([log_probability
                                 for _, log_probability in actual_results]),
        0.0)


class SamplingTest(parameterized.TestCase):
  """Tests sampling of grammar expansions."""

  @parameterized.named_parameters(
      ('fruit', 'fruit'),
      ('animal', 'animal'),
      ('thing', 'thing'),
      ('phrase', 'phrase'))
  def test_unique_randomizer_searches_completely(self, nonterminal_name):
    grammar = expand_grammar.GRAMMAR_EXAMPLE
    enumeration_list = expand_grammar.enumerate_nonterminal(nonterminal_name,
                                                            grammar)
    expected_expansion_probabilities = dict(enumeration_list)

    actual_expansions_dict, num_samples = expand_grammar.sample_with_ur(
        float('inf'), nonterminal_name, grammar)

    # The correct number of expansions, the actual number of expansions, and the
    # actual number of unique expansions should all be equal.
    self.assertTrue(len(enumeration_list) ==  # pylint: disable=g-generic-assert
                    num_samples ==
                    len(actual_expansions_dict))

    # The probabilities should match exactly (no floating point errors), as
    # they're both computed by summing log probabilities.
    self.assertEqual(expected_expansion_probabilities,
                     actual_expansions_dict)

  @parameterized.named_parameters(
      ('fruit', 'fruit'),
      ('animal', 'animal'),
      ('thing', 'thing'),
      ('phrase', 'phrase'))
  def test_normal_randomizer_correct_probabilities(self, nonterminal_name):
    grammar = expand_grammar.GRAMMAR_EXAMPLE
    enumeration_list = expand_grammar.enumerate_nonterminal(nonterminal_name,
                                                            grammar)
    expansion_probabilities = dict(enumeration_list)
    num_samples = min(100, len(expansion_probabilities) - 1)

    expansions_dict, _ = expand_grammar.sample_with_rejection(
        num_samples, nonterminal_name, grammar)

    for expansion, log_probability in expansions_dict.items():
      self.assertIn(expansion, expansion_probabilities)
      self.assertEqual(expansion_probabilities[expansion], log_probability)

  @parameterized.named_parameters(
      ('fruit', 'fruit'),
      ('animal', 'animal'),
      ('thing', 'thing'),
      ('phrase', 'phrase'))
  def test_sbs_searches_completely(self, nonterminal_name):
    grammar = expand_grammar.GRAMMAR_EXAMPLE
    enumeration_list = expand_grammar.enumerate_nonterminal(nonterminal_name,
                                                            grammar)
    expansion_probabilities = dict(enumeration_list)

    actual_expansions_dict, actual_num_samples = expand_grammar.sample_with_sbs(
        len(expansion_probabilities) + 1, nonterminal_name, grammar)

    self.assertEqual(actual_expansions_dict, expansion_probabilities)
    self.assertLen(actual_expansions_dict, actual_num_samples)


if __name__ == '__main__':
  absltest.main()
