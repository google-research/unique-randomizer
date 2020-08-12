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
"""Tests for sequence_example.py."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from unique_randomizer.examples import sequence_example


class FakeSequenceModelTest(parameterized.TestCase):

  def setUp(self):
    super(FakeSequenceModelTest, self).setUp()
    self.model = sequence_example.FakeSequenceModel(
        sequence_length=10,
        vocabulary_size=100,
        temperature=0.5)

  def test_deterministic_distribution(self):
    prefix = [4, 2, 5, 1]
    distribution_1 = self.model.next_token_log_probabilities(prefix)
    np.random.uniform()  # Change numpy's random state.
    distribution_2 = self.model.next_token_log_probabilities(prefix)
    self.assertEqual(distribution_1.tolist(), distribution_2.tolist())

  def test_does_not_interfere_with_numpy_random(self):
    np.random.seed(123)
    uniform_1 = np.random.uniform()

    np.random.seed(123)
    _ = self.model.next_token_log_probabilities([4, 2, 5, 1])
    uniform_2 = np.random.uniform()

    self.assertEqual(uniform_1, uniform_2)

  @parameterized.named_parameters(
      ('empty', [], False),
      ('0', [0], True),
      ('1', [1], False),
      ('10', [1, 0], True),
      ('11', [1, 1], False))
  def test_termination(self, prefix, should_terminate):
    is_complete = self.model.sequence_complete(prefix)
    self.assertEqual(should_terminate, is_complete)
    distribution = self.model.next_token_log_probabilities(prefix)
    # Do not implicitly convert numpy arrays to bool.
    self.assertEqual(should_terminate, len(distribution) == 0)  # pylint: disable=g-explicit-length-test

  def test_uses_sequence_length_and_vocab_size(self):
    prefix = list(range(self.model.sequence_length - 1))
    distribution = self.model.next_token_log_probabilities(prefix)
    self.assertLen(distribution, self.model.vocabulary_size)

    prefix = list(range(self.model.sequence_length))
    distribution = self.model.next_token_log_probabilities(prefix)
    self.assertEmpty(distribution)

  def test_repr(self):
    self.assertIn('FakeSequenceModel', repr(self.model))
    self.assertIn('sequence_length=10', repr(self.model))
    self.assertIn('vocabulary_size=100', repr(self.model))
    self.assertIn('temperature=0.5', repr(self.model))


def _log_samples(samples, method):
  logging.info('First 5 samples for %s:', method)
  for i, sample in enumerate(samples[:5]):
    logging.info('  #%s: %s', i, sample)


class SequenceExampleTest(absltest.TestCase):

  def setUp(self):
    super(SequenceExampleTest, self).setUp()
    # 64 possible sequences.
    self.model = sequence_example.FakeSequenceModel(
        sequence_length=3,
        vocabulary_size=5,
        temperature=0.2)
    self.num_distinct_sequences = 1 + 4 + 16 + 64
    # It so happens that this sequence has probability > 0.5. Further, the 6
    # most likely samples have a total probability > 0.98. Hence it is very
    # likely that this most-likely sequence appears within the first 6 samples.
    self.likely_sequence = [1, 1, 0]
    # Make the behavior of randomizers consistent.
    np.random.seed(123)

  def test_sample_with_rejection(self):
    samples = sequence_example.sample_with_rejection(self.model, 10)
    self.assertLen(samples, 10)
    # All lengths between 1 and 3.
    self.assertTrue(all(1 <= len(sample) <= 3 for sample in samples))
    # All samples are unique.
    self.assertLen(samples, len(set(tuple(sample) for sample in samples)))
    # Likely sequence appears early.
    self.assertIn(self.likely_sequence, samples[:6])
    _log_samples(samples, 'rejection')

  def test_sample_with_unique_randomizer(self):
    samples = sequence_example.sample_with_unique_randomizer(self.model, 100)
    self.assertLen(samples, self.num_distinct_sequences)
    # All lengths between 1 and 3.
    self.assertTrue(all(1 <= len(sample) <= 3 for sample in samples))
    # All samples are unique.
    self.assertLen(samples, len(set(tuple(sample) for sample in samples)))
    # Likely sequence appears early.
    self.assertIn(self.likely_sequence, samples[:6])
    _log_samples(samples, 'UR')

  def test_sample_with_unique_randomizer_cached(self):
    samples = sequence_example.sample_with_unique_randomizer_cached(
        self.model, 100)
    self.assertLen(samples, self.num_distinct_sequences)
    # All lengths between 1 and 3.
    self.assertTrue(all(1 <= len(sample) <= 3 for sample in samples))
    # All samples are unique.
    self.assertLen(samples, len(set(tuple(sample) for sample in samples)))
    # Likely sequence appears early.
    self.assertIn(self.likely_sequence, samples[:6])
    _log_samples(samples, 'UR cached')

  def test_sample_with_stochastic_beam_search(self):
    samples = sequence_example.sample_with_stochastic_beam_search(
        self.model, 100)
    self.assertLen(samples, self.num_distinct_sequences)
    # All lengths between 1 and 3.
    self.assertTrue(all(1 <= len(sample) <= 3 for sample in samples))
    # All samples are unique.
    self.assertLen(samples, len(set(tuple(sample) for sample in samples)))
    # Likely sequence appears early.
    self.assertIn(self.likely_sequence, samples[:6])
    _log_samples(samples, 'SBS')

  def test_sample_with_unique_randomizer_batched(self):
    samples = sequence_example.sample_with_unique_randomizer_batched(
        self.model, num_samples=100, batch_size=5)
    self.assertLen(samples, self.num_distinct_sequences)
    # All lengths between 1 and 3.
    self.assertTrue(all(1 <= len(sample) <= 3 for sample in samples))
    # All samples are unique.
    self.assertLen(samples, len(set(tuple(sample) for sample in samples)))
    # Likely sequence appears early.
    self.assertIn(self.likely_sequence, samples[:6])
    _log_samples(samples, 'UR batched')

if __name__ == '__main__':
  absltest.main()
