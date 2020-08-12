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
"""Expands a probabilistic context-free grammar using different techniques.

UniqueRandomizer is faster than SBS when probability computations are expensive.
This frequently occurs in machine learning, where a neural model is used to
provide probability distributions. UniqueRandomizer is also incremental, while
SBS is not. Both approaches significantly outperform rejection sampling when
duplicate samples are common.

Example outputs below, using the flags:
  --num_unique=1000 --start_nonterminal=sentence --prob_sleep_millis=2

Sampling technique: ur
Expanding from nonterminal 'sentence'.
Sleeping an extra 2.0 milliseconds per probability distribution computation.
Found 1000 unique expansions in 1000 tries, taking 4.09 seconds total.
On average, each try...
  took 4.092 milliseconds to complete
  was unique with probability 1.0000

Sampling technique: sbs
Expanding from nonterminal 'sentence'.
Sleeping an extra 2.0 milliseconds per probability distribution computation.
Found 1000 unique expansions in 1000 tries, taking 4.71 seconds total.
On average, each try...
  took 4.707 milliseconds to complete
  was unique with probability 1.0000

Sampling technique: rejection
Expanding from nonterminal 'sentence'.
Sleeping an extra 2.0 milliseconds per probability distribution computation.
Found 1000 unique expansions in 3610 tries, taking 36.62 seconds total.
On average, each try...
  took 10.144 milliseconds to complete
  was unique with probability 0.2770
"""

import timeit

from absl import app
from absl import flags

from unique_randomizer.examples import expand_grammar

FLAGS = flags.FLAGS

flags.DEFINE_enum('technique', 'ur', ['ur', 'rejection', 'sbs'],
                  'The sampling technique.')
flags.DEFINE_integer('num_unique', 2000,
                     'The number of unique expansions to generate.')
flags.DEFINE_string('start_nonterminal', 'sentence',
                    'The name of the nonterminal to start expanding from.')
flags.DEFINE_float('prob_sleep_millis', 1.0,
                   'Milliseconds to sleep for each computed probability '
                   'distribution for a nonterminal replacement.')


def main(unused_argv):
  expand_grammar.verify_grammar(expand_grammar.GRAMMAR_EXAMPLE)
  if FLAGS.start_nonterminal not in expand_grammar.GRAMMAR_EXAMPLE:
    raise expand_grammar.GrammarError(
        "The starting nonterminal '{}' is not in the grammar."
        .format(FLAGS.start_nonterminal))

  print('Sampling technique: {}'.format(FLAGS.technique))
  print("Expanding from nonterminal '{}'.".format(FLAGS.start_nonterminal))
  print('Sleeping an extra {} milliseconds per probability distribution '
        'computation.'.format(FLAGS.prob_sleep_millis))

  if FLAGS.technique == 'ur':
    sampling_fn = expand_grammar.sample_with_ur
  elif FLAGS.technique == 'rejection':
    sampling_fn = expand_grammar.sample_with_rejection
  elif FLAGS.technique == 'sbs':
    sampling_fn = expand_grammar.sample_with_sbs
  else:
    raise ValueError('Unhandled technique: {}'.format(FLAGS.technique))

  start_time = timeit.default_timer()
  expansions_dict, num_tries = sampling_fn(
      num_unique_samples=FLAGS.num_unique,
      nonterminal_name=FLAGS.start_nonterminal,
      grammar_dict=expand_grammar.GRAMMAR_EXAMPLE,
      prob_sleep_millis=FLAGS.prob_sleep_millis)
  elapsed_time = timeit.default_timer() - start_time

  print('Found {} unique expansions in {} tries, taking {:.2f} seconds total.'
        .format(len(expansions_dict), num_tries, elapsed_time))
  print('On average, each try...\n'
        '  took {:.3f} milliseconds to complete\n'
        '  was unique with probability {:.4f}'.format(
            elapsed_time * 1000 / num_tries, len(expansions_dict) / num_tries))


if __name__ == '__main__':
  app.run(main)
