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
"""Tests for stochastic_beam_search.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from unique_randomizer import stochastic_beam_search as sbs


class StochasticBeamSearchTest(parameterized.TestCase):

  def test_sample_gumbels_with_maximum(self):
    # It's possible but extremely unlikely for this test to fail.
    probability_distribution = [0.3, 0.6, 0.1]
    log_probabilities = np.log(probability_distribution)
    samples = []
    for _ in range(10000):
      gumbels = sbs.sample_gumbels_with_maximum(log_probabilities,
                                                target_max=-0.123)
      self.assertEqual(max(gumbels), -0.123)
      samples.append(np.argmax(gumbels))

    counter = collections.Counter(samples)
    self.assertEqual(counter[0] + counter[1] + counter[2], 10000)

    self.assertAlmostEqual(counter[0], 0.3 * 10000, delta=250)
    self.assertAlmostEqual(counter[1], 0.6 * 10000, delta=300)
    self.assertAlmostEqual(counter[2], 0.1 * 10000, delta=200)

  @parameterized.named_parameters(('0', 0), ('1', 1), ('2', 2), ('100', 100))
  def test_single_output_edge_case(self, k):

    def child_log_probability_fn(states):
      if states != ['root!']:
        raise ValueError('Unexpected states.')
      return [np.array([0])]

    def child_state_fn(state_index_pairs):
      if state_index_pairs != [('root!', 0)]:
        raise ValueError('Unexpected state_index_pairs.')
      return [('output!', True)]

    beam_nodes = sbs.stochastic_beam_search(
        child_log_probability_fn=child_log_probability_fn,
        child_state_fn=child_state_fn,
        root_state='root!',
        k=k)
    self.assertLen(beam_nodes, min(k, 1))
    if k > 0:
      self.assertEqual(
          beam_nodes[0],
          sbs.BeamNode(output='output!', log_probability=0.0, gumbel=0.0))

  @parameterized.named_parameters(('3', 3), ('10', 10))
  def test_proportions(self, k):
    # This test is analogous to unique_randomizer_test.py's test_proportions.
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
      beam_nodes = sbs.stochastic_beam_search(
          child_log_probability_fn=child_log_probability_fn,
          child_state_fn=child_state_fn,
          root_state=(None, None),
          k=k)
      results.append(''.join([node.output for node in beam_nodes]))

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


if __name__ == '__main__':
  absltest.main()
