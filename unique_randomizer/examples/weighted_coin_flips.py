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
"""Using UniqueRandomizer to sample unique results of weighted coin flips."""

import math
from typing import Text, Tuple

from unique_randomizer import unique_randomizer as ur


def flip_two_weighted_coins(randomizer: ur.Randomizer) -> Tuple[Text, Text]:
  """Flips two weighted coins, which land Heads with probability 0.9 and 0.7."""
  possible_flips = ['Tails', 'Heads']
  flip_1 = possible_flips[randomizer.sample_distribution([0.1, 0.9])]
  flip_2 = possible_flips[randomizer.sample_distribution([0.3, 0.7])]
  return flip_1, flip_2


def sample_flips_without_replacement() -> None:
  """Samples the coin flips without replacement, printing out the results."""
  randomizer = ur.UniqueRandomizer()

  # Sample pairs of coin flips until all possible results have been sampled.
  while not randomizer.exhausted():
    sample = flip_two_weighted_coins(randomizer)
    log_probability = randomizer.mark_sequence_complete()

    print('Sample {} is {} with probability {:2.0f}%. '
          'In total, {:3.0f}% of the output space has been sampled.'.format(
              randomizer.num_sequences_sampled(),
              sample,
              math.exp(log_probability) * 100,
              randomizer.fraction_sampled() * 100))


def main():
  sample_flips_without_replacement()

if __name__ == '__main__':
  main()
