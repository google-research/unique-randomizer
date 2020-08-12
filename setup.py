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

"""Installs UniqueRandomizer."""

import setuptools

from unique_randomizer import __version__


def get_required_packages():
  """Returns a list of required packages."""
  return [
      'absl-py >= 0.6.1',  # Oct 26, 2018
      'numpy >= 1.15.4',  # Nov 4, 2018
      'scipy >= 1.1.0',  # May 5, 2018
  ]


def run_setup():
  """Installs UniqueRandomizer."""

  with open('README.md', 'r') as fh:
    long_description = fh.read()

  setuptools.setup(
      name='unique-randomizer',
      version=__version__,
      author='Google LLC',
      author_email='no-reply@google.com',
      description='UniqueRandomizer: Incremental Sampling Without Replacement',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/google-research/unique-randomizer',
      packages=setuptools.find_packages(),
      install_requires=get_required_packages(),
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      keywords=('unique randomizer UniqueRandomizer incremental sampling '
                'without replacement sequence models'),
      python_requires='>=3',
  )


if __name__ == '__main__':
  run_setup()
