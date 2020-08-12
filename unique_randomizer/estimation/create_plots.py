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
"""Binary to create plots for estimation."""

from absl import app
from absl import flags

from unique_randomizer.estimation import hindsight_gumbel_estimation

FLAGS = flags.FLAGS

flags.DEFINE_string('filename',
                    '/tmp/hindsight_gumbel_estimation/plot.png',
                    'Filename for output image.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  hindsight_gumbel_estimation.create_plots(filename=FLAGS.filename)


if __name__ == '__main__':
  app.run(main)
