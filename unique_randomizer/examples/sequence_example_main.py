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
"""Samples unique sequences using different techniques.

Below we list an example setting where we pretend that batched probability
computations are 10x faster than non-batched ones, followed by results for each
technique on this setting.

Number of samples: 1000
NumPy seed: 123
Model: FakeSequenceModel(
    sequence_length=10,
    vocabulary_size=100,
    temperature=0.1,
    sleep_millis=1.0,
    batch_sleep_millis=0.1)

Sampling technique: ur
Sampling found 1000 samples in 13.363 seconds total.
Number of probability computations: 9936
Number of batches: 0

Sampling technique: ur_cached
Sampling found 1000 samples in 4.106 seconds total.
Number of probability computations: 2657
Number of batches: 0

Sampling technique: ur_batched
Sampling found 1000 samples in 1.888 seconds total.
Number of probability computations: 4247
Number of batches: 42

Sampling technique: ur_batched_max
Sampling found 1000 samples in 2.380 seconds total.
Number of probability computations: 7974
Number of batches: 10

Sampling technique: sbs
Sampling found 1000 samples in 2.072 seconds total.
Number of probability computations: 7974
Number of batches: 10

Sampling technique: rejection
Sampling found 1000 samples in 23.645 seconds total.
Number of probability computations: 18314
Number of batches: 0
"""

import functools
import timeit

from absl import app
from absl import flags
import numpy as np

from unique_randomizer.examples import sequence_example

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_samples', 1000,
                     'The number of unique samples to generate.')
flags.DEFINE_enum('technique', 'ur',
                  ['ur', 'ur_cached', 'ur_batched', 'ur_batched_max',
                   'sbs', 'rejection'],
                  'Which sampling-without-replacement technique to use.')
flags.DEFINE_integer('seed', 123,
                     'Random seed for numpy.')

# These flags define the FakeSequenceModel.
flags.DEFINE_integer('sequence_length', 10,
                     'Number of tokens in the sequence.')
flags.DEFINE_integer('vocabulary_size', 100,
                     'Number of distinct tokens in the vocabulary.')
flags.DEFINE_float('temperature', 0.1,
                   'Temperature to apply to Exponential(1)-distributed '
                   'probabilities.')
flags.DEFINE_float('sleep_millis', 1.0,
                   'Milliseconds to sleep for each non-empty probability '
                   'distribution.')
flags.DEFINE_float('batch_sleep_millis', 0.1,
                   'Milliseconds to sleep for each batch element, when '
                   'batching probability computations.')


def main(unused_argv):
  np.random.seed(FLAGS.seed)
  model = sequence_example.FakeSequenceModel(
      sequence_length=FLAGS.sequence_length,
      vocabulary_size=FLAGS.vocabulary_size,
      temperature=FLAGS.temperature,
      sleep_millis=FLAGS.sleep_millis,
      batch_sleep_millis=FLAGS.batch_sleep_millis)

  if FLAGS.technique == 'rejection':
    sampling_function = sequence_example.sample_with_rejection
  elif FLAGS.technique == 'ur':
    sampling_function = sequence_example.sample_with_unique_randomizer
  elif FLAGS.technique == 'ur_cached':
    sampling_function = (
        sequence_example.sample_with_unique_randomizer_cached)
  elif FLAGS.technique == 'ur_batched':
    sampling_function = functools.partial(
        sequence_example.sample_with_unique_randomizer_batched,
        batch_size=200)
  elif FLAGS.technique == 'ur_batched_max':
    sampling_function = functools.partial(
        sequence_example.sample_with_unique_randomizer_batched,
        batch_size=FLAGS.num_samples)
  elif FLAGS.technique == 'sbs':
    sampling_function = sequence_example.sample_with_stochastic_beam_search
  else:
    raise app.UsageError('Unknown technique: {}'.format(FLAGS.technique))

  print('Number of samples: {}'.format(FLAGS.num_samples))
  print('NumPy seed: {}'.format(FLAGS.seed))
  print('Model: {}'.format(model))

  start_time = timeit.default_timer()
  samples = sampling_function(model=model, num_samples=FLAGS.num_samples)
  elapsed_time = timeit.default_timer() - start_time

  if len(samples) != FLAGS.num_samples:
    raise ValueError('Requested {} samples but only got {}!'
                     .format(FLAGS.num_samples, len(samples)))
  if len(set(tuple(sample) for sample in samples)) != len(samples):
    raise ValueError('Encountered a duplicate sample.')

  print()
  print('Sampling technique: {}'.format(FLAGS.technique))
  print('Sampling found {} samples in {:.3f} seconds total.'
        .format(len(samples), elapsed_time))
  print('Number of probability computations: {}'
        .format(model.probability_count))
  print('Number of batches: {}'
        .format(model.batch_probability_count))


if __name__ == '__main__':
  app.run(main)
