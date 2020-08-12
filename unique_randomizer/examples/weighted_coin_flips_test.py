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
"""Tests for weighted_coin_flips.py."""

from absl.testing import absltest
import mock

from unique_randomizer.examples import weighted_coin_flips


class WeightedCoinFlipsTest(absltest.TestCase):

  @mock.patch('builtins.print')
  def test_sample_flips_without_replacement(self, mock_print):
    weighted_coin_flips.sample_flips_without_replacement()
    self.assertEqual(mock_print.call_count, 4)

    printed = [call[0][0] for call in mock_print.call_args_list]
    self.assertIn('100% of the output space has been sampled', printed[-1])

    expected_substrings = [
        "('Heads', 'Heads') with probability 63%",
        "('Heads', 'Tails') with probability 27%",
        "('Tails', 'Heads') with probability  7%",
        "('Tails', 'Tails') with probability  3%",
    ]
    # Each expected substring should appear in exactly 1 printed string.
    for substring in expected_substrings:
      self.assertEqual(sum(substring in p for p in printed), 1)


if __name__ == '__main__':
  absltest.main()
