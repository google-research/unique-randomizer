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

from unique_randomizer import unique_randomizer_simple as ur_simple


class UniqueRandomizerSimpleTest(parameterized.TestCase):

  def test_root_is_leaf_edge_case(self):
    randomizer = ur_simple.UniqueRandomizer()
    self.assertFalse(randomizer.exhausted())

    log_mass = randomizer.mark_sequence_complete()

    self.assertEqual(log_mass, math.log(1.0))
    self.assertTrue(randomizer.exhausted())
    with self.assertRaises(ValueError):
      randomizer.sample_distribution([0.9, 0.1])

  def test_sample_proportions(self):
    # It's possible but extremely unlikely for this test to fail.
    results = []
    for _ in range(10000):
      randomizer = ur_simple.UniqueRandomizer()
      digits = []
      while not randomizer.exhausted():
        # Choose 2 with probability 0.6, or 0 or 1 with probability 0.2 each.
        if randomizer.sample_distribution([0.4, 0.6]):
          digits.append('2')
        else:
          digits.append(str(randomizer.sample_distribution([0.5, 0.5])))
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

  def test_correct_probabilities(self):
    result_log_probabilities = {}
    randomizer = ur_simple.UniqueRandomizer()

    while not randomizer.exhausted():
      num_digits = randomizer.sample_distribution([0.1, 0.2, 0.3, 0.4])
      result = ''
      for i in range(num_digits):
        # Digit index 0 is always 0.
        # Digit index 1 is 0 or 1 with equal probability.
        # Digit index 2 is always 1.
        digit = randomizer.sample_distribution([1 - i/2.0, i/2.0])
        result += str(digit)

      log_probability = randomizer.mark_sequence_complete()
      self.assertNotIn(result, result_log_probabilities)
      result_log_probabilities[result] = log_probability

    expected_log_probabilities = {
        '': math.log(0.1),
        '0': math.log(0.2),
        '00': math.log(0.3 * 0.5),
        '01': math.log(0.3 * 0.5),
        '001': math.log(0.4 * 0.5),
        '011': math.log(0.4 * 0.5),
    }

    self.assertEqual(result_log_probabilities, expected_log_probabilities)


if __name__ == '__main__':
  absltest.main()
