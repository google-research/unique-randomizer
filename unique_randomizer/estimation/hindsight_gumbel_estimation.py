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
"""Estimating means by sampling Gumbels in hindsight."""

import collections
import functools

from absl import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.special

from unique_randomizer import unique_randomizer as ur


def gumbel_log_survival(x):
  """Returns log P(g > x) for a standard Gumbel g.

  log P(g > x) = log(1 - P(g < x)) = log(1 - exp(-exp(-x))). The implementation
  is more numerically robust than a naive implementation of that formula.

  Args:
    x: The cutoff Gumbel value.
  """
  # Adapted from
  # https://gist.github.com/wouterkool/a3bb2aae8d6a80f985daae95252a8aa8.
  y = np.exp(-x)
  return np.where(x >= 10,
                  -x - y / 2 + y ** 2 / 24 - y ** 4 / 2880,
                  np.log(-np.expm1(-np.exp(-x))))


def truncated_gumbel(log_probability, upper_bound):
  """Samples a Gumbel for a log_probability, given an upper bound."""
  # Adapted from https://cmaddis.github.io/gumbel-machinery.
  if log_probability == -float('inf'):
    return -float('inf')
  gumbel = np.random.gumbel(loc=log_probability)
  return -scipy.special.logsumexp([-gumbel, -upper_bound])


def hindsight_gumbels(log_probabilities):
  """Returns Gumbels that could have produced the samples with probabilities.

  The returned list will have one more element than the input probabilities,
  the last one being the maximum Gumbel for the remaining unsampled items. If
  the samples are exhaustive (probabilities sum to 1), then the last Gumbel is
  -inf.

  Args:
    log_probabilities: The log probabilities of sampled items, in the order that
      they were sampled from a probability proportional to size without
      replacement (PPSWOR) scheme.
  """
  gumbels = []
  unsampled_log_probability = 0.0

  # Sample the maximum Gumbel for all items.
  max_gumbel = np.random.gumbel(loc=unsampled_log_probability)

  for item_log_probability in log_probabilities:
    # The Gumbel for the next sampled item is exactly the maximum Gumbel across
    # all remaining items.
    gumbels.append(max_gumbel)

    # Update the unsampled probability, now that we've sampled the next item.
    unsampled_log_probability = ur.log_subtract(unsampled_log_probability,
                                                item_log_probability)

    # Sample a maximum Gumbel for the remaining unsampled items. This must be at
    # most the previous maximum Gumbel.
    max_gumbel = truncated_gumbel(unsampled_log_probability, max_gumbel)

  # Append the maximum Gumbel for the remaining (truly-unsampled) items.
  gumbels.append(max_gumbel)

  assert len(gumbels) == 1 + len(log_probabilities)
  # Allow a tiny amount of error in case of numerical instability.
  if not all(g1 >= g2 - 1e-5 for g1, g2 in zip(gumbels, gumbels[1:])):
    message = ('Issue in hindsight_gumbels.\n'
               'log_probabilities = {}\n'
               'gumbels = {}').format(
                   log_probabilities, gumbels)
    logging.warn(message)
    print(message)
  return gumbels


def setup_universe(universe_size):
  """Returns the universe of items, probabilities, and values."""
  universe = list(range(universe_size))
  probabilities = np.random.exponential(size=universe_size)
  probabilities = probabilities ** 3  # Skew the probabilities.
  probabilities /= np.sum(probabilities)
  # Skew the values: items with larger probability likely have larger values.
  values = np.random.normal(loc=np.log(probabilities), scale=0.5)
  # Shift values so the minimum is zero.
  values -= np.min(values)
  return universe, probabilities, values


def ppswor_samples(universe, probabilities, num_samples):
  """Samples some items from the universe, using a PPSWOR scheme."""
  results = []
  not_sampled = list(universe)
  for _ in range(num_samples):
    unsampled_probs = probabilities[not_sampled]
    normalized_probs = unsampled_probs / np.sum(unsampled_probs)
    index = np.random.choice(np.arange(len(not_sampled)), p=normalized_probs)
    sample = not_sampled[index]
    results.append((sample, probabilities[sample], normalized_probs[index]))
    not_sampled.remove(sample)

  # This is a list of triples (sample, initial prob., conditional prob.).
  return results


def hindsight_gumbel_estimation(
    universe, probabilities, values, num_samples, normalize, all_samples=None):
  """Hindsight Gumbel Estimation."""
  # Allow repeated_hindsight_gumbel_estimation.
  if all_samples is None:
    results = ppswor_samples(universe, probabilities, num_samples)
    all_samples = [result[0] for result in results]

  # Item probabilities and values, in the order that they were sampled.
  ordered_probabilities = probabilities[all_samples]
  ordered_values = values[all_samples]
  num_samples = len(all_samples)

  estimations = []  # One estimate for every k = 1, ..., num_samples.

  gumbels = hindsight_gumbels(np.log(ordered_probabilities))

  for k in range(1, num_samples + 1):
    # Use the first k samples for estimation. The threshold Gumbel comes from
    # the (k+1)-th sample, or equivalently the "remaining" probability mass
    # (we don't actually need a concrete (k+1)-th sample).
    threshold_gumbel = gumbels[k]
    p = ordered_probabilities[:k]
    if k == len(universe):
      # Otherwise we'll get a warning, if gumbels[k] == -float('inf').
      q = 1
    else:
      q = np.exp(gumbel_log_survival(threshold_gumbel - np.log(p)))
    weight = p / q
    if normalize:
      weight /= np.sum(weight)
    estimate = np.dot(weight, ordered_values[:k])
    estimations.append(estimate)

  return estimations


def repeated_hindsight_gumbel_estimation(
    universe, probabilities, values, num_samples, normalize, repetitions):
  """Uses Hindsight Gumbel Estimation multiple times with different Gumbels."""

  # Use the same samples for each repetition!
  results = ppswor_samples(universe, probabilities, num_samples)
  all_samples = [result[0] for result in results]

  estimations_list = []

  for _ in range(repetitions):
    estimations = hindsight_gumbel_estimation(
        universe, probabilities, values, num_samples, normalize,
        all_samples=all_samples)  # Provide consistent samples.
    estimations_list.append(estimations)

  return np.mean(estimations_list, axis=0)


def ppswor_priority_sampling(
    universe, probabilities, values, num_samples, normalize):
  """Priority Sampling using a PPSWOR sampling scheme."""
  # Adapted from
  # https://github.com/timvieira/blog/blob/master/content/notebook/Priority%20Sampling.ipynb.

  universe_size = len(universe)
  p = probabilities
  f = values

  u = np.random.uniform(0, 1, size=universe_size)
  key = -np.log(u) / p  # ~ Exp(p[i])
  # key = np.random.exponential(scale=1/p)  # Equivalent to the line above.
  order = np.argsort(key)  # Item indices in the order they're chosen.
  ordered_keys = key[order]

  estimations = np.zeros(num_samples)

  for k in range(1, num_samples + 1):
    t = ordered_keys[k] if k < universe_size else np.inf  # Threshold.
    s = order[:k]  # First k sampled items.
    q = 1 - np.exp(-p*t)  # = p(i in s | t).
    weights_s = p[s] / q[s]
    if normalize:
      weights_s /= np.sum(weights_s)
    estimations[k-1] = f[s].dot(weights_s)

  return estimations


def monte_carlo_sampling(universe, probabilities, values, num_samples):
  """Traditional Monte Carlo sampling with replacement."""
  # Adapted from
  # https://github.com/timvieira/blog/blob/master/content/notebook/Priority%20Sampling.ipynb.
  samples = np.random.choice(universe, size=num_samples, p=probabilities,
                             replace=True)
  return np.cumsum(values[samples]) / (1 + np.arange(num_samples))


def create_plots(filename, seed=123):
  """Creates plots for the paper."""
  np.random.seed(seed)

  universe_size = 100
  num_samples = 100
  estimation_repetitions = 2000

  universe, probabilities, original_values = setup_universe(universe_size)

  # Manipulate values here.
  values = original_values

  exact = np.dot(probabilities, values)
  print('Exact value: {}'.format(exact))

  estimation_methods = [
      ('HGE',
       functools.partial(hindsight_gumbel_estimation, normalize=False),
       '#4285F4'),  # Google blue.
      ('HGE, norm.',
       functools.partial(hindsight_gumbel_estimation, normalize=True),
       '#0F9D58'),  # Google green.

      ('Repeated HGE (x10)',
       functools.partial(repeated_hindsight_gumbel_estimation,
                         repetitions=10,
                         normalize=False),
       '#F4B400'),  # Google yellow.
      ('Repeated HGE (x10), norm.',
       functools.partial(repeated_hindsight_gumbel_estimation,
                         repetitions=10,
                         normalize=True),
       '#DB4437'),  # Google red.

      # ('PPSWOR Priority Sampling',
      #  functools.partial(ppswor_priority_sampling, normalize=False),
      #  'red'),
      # ('PPSWOR Priority Sampling, Normalized',
      #  functools.partial(ppswor_priority_sampling, normalize=True),
      #  'darkorange'),

      ('Monte Carlo sampling', monte_carlo_sampling, '#9E9E9E')  # Google gray.
  ]

  estimations_k = list(range(1, num_samples + 1))

  all_estimators_data = collections.defaultdict(list)

  for _ in range(estimation_repetitions):

    for name, method, _ in estimation_methods:
      estimations = method(universe, probabilities, values, num_samples)
      all_estimators_data[name].append(estimations)

  matplotlib.rcParams.update({'font.size': 12})

  plt.figure(facecolor='w', edgecolor='k', figsize=[6.4, 4.8])
  for name, _, color in estimation_methods:
    data = all_estimators_data[name]

    # Cut off first point to reduce noise in the plot.
    cut_data = [x[1:] for x in data]
    cut_estimations_k = estimations_k[1:]

    plt.plot(cut_estimations_k, np.percentile(cut_data, 95, axis=0),
             color=color, linestyle=':', alpha=0.5)
    plt.plot(cut_estimations_k, np.percentile(cut_data, 5, axis=0),
             color=color, linestyle=':', alpha=0.5)

    plt.plot(cut_estimations_k, np.percentile(cut_data, 25, axis=0),
             color=color, linestyle='-', label=name)
    plt.plot(cut_estimations_k, np.percentile(cut_data, 75, axis=0),
             color=color, linestyle='-')

  plt.title('HGE Variations on Synthetic Data')
  plt.axhline(y=exact, color='k', linestyle='--', label='Exact value')
  plt.ylim(exact - 1, exact + 1)
  plt.ylabel('Estimate')
  plt.xlim(0, num_samples)
  plt.xlabel('Number of Samples')
  plt.legend(loc='upper right', fontsize=10)

  print('Saving plot to {}'.format(filename))
  plt.savefig(filename)
