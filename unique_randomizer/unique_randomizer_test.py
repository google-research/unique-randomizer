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
"""Tests for unique_randomizer.py."""

import collections
import math

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from unique_randomizer import unique_randomizer as ur


class UniqueRandomizerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('large_small', math.log(12.3), math.log(0.01), math.log(12.29)),
      ('large_medium', math.log(12.3), math.log(4.5), math.log(7.8)),
      ('large_large', math.log(12.3), math.log(12.299), math.log(0.001)),
      ('small_small', math.log(12.3e-5), math.log(0.01e-5), math.log(12.29e-5)),
      ('small_medium', math.log(12.3e-5), math.log(4.5e-5), math.log(7.8e-5)),
      ('small_large', math.log(12.3e-5), math.log(12.299e-5), math.log(1e-8)),
      ('equal', math.log(12.3), math.log(12.3), -float('inf')),
      ('second_greater', math.log(12.3), math.log(12.31), -float('inf')))
  def test_log_subtract(self, x, y, expected):
    self.assertAlmostEqual(ur.log_subtract(x, y), expected)

  def test_sample_log_distribution(self):
    # It's possible but extremely unlikely for this test to fail.
    unnormalized_probabilities = [3e-5, 6e-5, 1e-5]
    log_distribution = np.log(unnormalized_probabilities)
    samples = [ur.sample_log_distribution(log_distribution)
               for _ in range(10000)]

    counter = collections.Counter(samples)
    self.assertEqual(counter[0] + counter[1] + counter[2], 10000)

    self.assertAlmostEqual(counter[0], 0.3 * 10000, delta=250)
    self.assertAlmostEqual(counter[1], 0.6 * 10000, delta=300)
    self.assertAlmostEqual(counter[2], 0.1 * 10000, delta=200)

  def test_root_is_leaf_edge_case(self):
    randomizer = ur.UniqueRandomizer()

    self.assertEqual(randomizer.fraction_sampled(), 0.0)
    self.assertFalse(randomizer.exhausted())
    self.assertEqual(randomizer.num_sequences_sampled(), 0)

    log_probability = randomizer.mark_sequence_complete()

    self.assertEqual(log_probability, math.log(1.0))
    self.assertEqual(randomizer.fraction_sampled(), 1.0)
    self.assertTrue(randomizer.exhausted())
    self.assertEqual(randomizer.num_sequences_sampled(), 1)

    with self.assertRaises(ur.AllSequencesSampledError):
      randomizer.sample_boolean(0.1)

  def test_proportions(self):
    # It's possible but extremely unlikely for this test to fail.
    results = []
    for _ in range(10000):
      randomizer = ur.UniqueRandomizer()
      digits = []
      while not randomizer.exhausted():
        # Choose 2 with probability 0.6, or 0 or 1 with probability 0.2 each.
        if randomizer.sample_boolean(probability_1=0.6):
          digits.append('2')
        else:
          digits.append(str(randomizer.sample_uniform(2)))
        randomizer.mark_sequence_complete()
      results.append(''.join(digits))

    self.assertTrue(all(len(s) == 3 for s in results))
    counter = collections.Counter(results)

    # P('201') = P('210') = 0.6 * 0.5.
    self.assertAlmostEqual(counter['201'], 0.6 * 0.5 * 10000, delta=250)
    self.assertAlmostEqual(counter['210'], 0.6 * 0.5 * 10000, delta=250)

    # P('021') = P('120') = 0.2 * 0.75.
    self.assertAlmostEqual(counter['021'], 0.2 * 0.75 * 10000, delta=200)
    self.assertAlmostEqual(counter['120'], 0.2 * 0.75 * 10000, delta=200)

    # P('012') = P('102') = 0.2 * 0.25.
    self.assertAlmostEqual(counter['012'], 0.2 * 0.25 * 10000, delta=100)
    self.assertAlmostEqual(counter['102'], 0.2 * 0.25 * 10000, delta=100)

  @parameterized.named_parameters(('1', 1), ('2', 2), ('3', 3), ('10', 10))
  def test_ur_sbs_proportions(self, k):
    # This test is analogous to test_proportions.
    # It's possible but extremely unlikely for this test to fail.

    # A state is a pair representing two coin flips. The first flip is biased
    # (60% True). If True, the output is '2'. If False, then the second flip
    # (fair odds) determines whether the output is '0' or '1'.
    # See unique_randomizer_test.py's test_proportions for a procedural
    # representation of this logic.

    def child_log_probability_fn(states):
      results = []
      for state in states:
        first_flip, _ = state
        if first_flip is None:
          results.append(np.log([0.4, 0.6]))
        elif not first_flip:
          results.append(np.log([0.5, 0.5]))
        else:
          raise ValueError('Leaf state encountered unexpectedly.')
      return results

    def child_state_fn(state_index_pairs):
      results = []
      for (first_flip, _), index in state_index_pairs:
        if first_flip is None:
          if index == 0:
            child_state = (False, None)
            results.append((child_state, False))
          elif index == 1:
            output = '2'
            results.append((output, True))
          else:
            raise ValueError('Out of bounds index: {}'.format(index))
        elif not first_flip:
          output = str(index)
          results.append((output, True))
        else:
          raise ValueError('Leaf state encountered unexpectedly.')
      return results

    results = []
    for _ in range(10000):
      randomizer = ur.UniqueRandomizer()
      digit_results = []
      while not randomizer.exhausted():
        beam_nodes = randomizer.sample_batch(
            child_log_probability_fn=child_log_probability_fn,
            child_state_fn=child_state_fn,
            root_state=(None, None),
            k=k)
        digit_results.extend([node.output for node in beam_nodes])
      results.append(''.join(digit_results))

    self.assertTrue(all(len(s) == 3 for s in results))
    counter = collections.Counter(results)

    # P('201') = P('210') = 0.6 * 0.5.
    self.assertAlmostEqual(counter['201'], 0.6 * 0.5 * 10000, delta=250)
    self.assertAlmostEqual(counter['210'], 0.6 * 0.5 * 10000, delta=250)

    # P('021') = P('120') = 0.2 * 0.75.
    self.assertAlmostEqual(counter['021'], 0.2 * 0.75 * 10000, delta=200)
    self.assertAlmostEqual(counter['120'], 0.2 * 0.75 * 10000, delta=200)

    # P('012') = P('102') = 0.2 * 0.25.
    self.assertAlmostEqual(counter['012'], 0.2 * 0.25 * 10000, delta=100)
    self.assertAlmostEqual(counter['102'], 0.2 * 0.25 * 10000, delta=100)

  def test_needs_probabilities(self):
    randomizer = ur.UniqueRandomizer()

    self.assertTrue(randomizer.needs_probabilities())
    first_index = randomizer.sample_distribution([0.9, 0.1])
    self.assertTrue(randomizer.needs_probabilities())
    randomizer.mark_sequence_complete()

    self.assertFalse(randomizer.needs_probabilities())
    second_index = randomizer.sample_distribution(None)
    self.assertTrue(randomizer.needs_probabilities())
    randomizer.sample_boolean(probability_1=0.123)
    self.assertTrue(randomizer.needs_probabilities())
    randomizer.mark_sequence_complete()

    self.assertNotEqual(first_index, second_index)

    self.assertFalse(randomizer.needs_probabilities())
    randomizer.sample_distribution(None)
    self.assertFalse(randomizer.needs_probabilities())
    randomizer.sample_boolean(probability_1=999)
    self.assertTrue(randomizer.needs_probabilities())
    randomizer.mark_sequence_complete()

    self.assertTrue(randomizer.exhausted())

if __name__ == '__main__':
  absltest.main()
