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
"""Sampling approach to the Traveling Salesman Problem."""

import functools
import timeit

from absl import app
from absl import flags
import numpy as np
import scipy.spatial

from unique_randomizer import unique_randomizer as ur

FLAGS = flags.FLAGS

NEAREST_NEIGHBOR = 'nearest_neighbor'
FARTHEST_INSERTION = 'farthest_insertion'
FARTHEST_INSERTION_SAMPLING = 'farthest_insertion_sampling'
FARTHEST_INSERTION_BS = 'farthest_insertion_bs'

flags.DEFINE_integer('dataset_size', 10000, 'The number of TSP instances.')
flags.DEFINE_integer('graph_size', 50, 'The number of nodes in a TSP graph.')
flags.DEFINE_integer('seed', 1234, 'Random seed.')
flags.DEFINE_enum('solver',
                  FARTHEST_INSERTION,
                  [NEAREST_NEIGHBOR, FARTHEST_INSERTION,
                   FARTHEST_INSERTION_SAMPLING, FARTHEST_INSERTION_BS],
                  'The TSP solver to use.')
flags.DEFINE_integer('num_samples', 300, 'The number of samples to use.')
flags.DEFINE_boolean('unique_samples', True, 'Whether to use unique sampling.')
flags.DEFINE_boolean('caching', True,
                     'Whether to cache probabilities during unique sampling.')
flags.DEFINE_float('temperature', 0.2, 'Temperature for sampling.')


def nearest_neighbor(nodes):
  """Nearest neighbor construction of the tour in order."""
  # This code is inspired by
  # https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/tsp/tsp_baseline.py.

  # distances[i][j] is the Euclidean distance from nodes[i] to nodes[j].
  distances = scipy.spatial.distance_matrix(nodes, nodes)
  current_node = 0
  tour = [current_node]
  tour_cost = 0.0
  distance_to_start = distances[current_node].copy()

  for _ in range(len(nodes) - 1):
    # current_node is no longer a valid neighbor (of any other node).
    distances[:, current_node] = np.Inf

    neighbor = distances[current_node].argmin()
    tour_cost += distances[current_node][neighbor]
    tour.append(neighbor)
    current_node = neighbor

  tour_cost += distance_to_start[current_node]
  return tour_cost, tour


def _insertion_cost(distances, previous_node, next_node, inserted_node):
  """Calculates insertion costs of inserting a node into a tour.

  Args:
    distances: A distance matrix.
    previous_node: The node before the inserted node. Can be a vector.
    next_node: The node after the inserted node. Can be a vector.
    inserted_node: The node to insert.

  Returns:
    The extra tour cost(s) when inserting the node at the given location(s).
  """
  return (distances[previous_node, inserted_node]
          + distances[inserted_node, next_node]
          - distances[previous_node, next_node])


def farthest_insertion(nodes, randomizer=None, temperature=1.0, caching=True):
  """Inserts the farthest node from the tour, at the best index."""
  # This code is inspired by
  # https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/tsp/tsp_baseline.py.

  num_nodes = len(nodes)
  distances = scipy.spatial.distance_matrix(nodes, nodes)

  unused_mask = np.ones(num_nodes, dtype=bool)
  tour = []
  for i in range(num_nodes):
    unused_indices = np.flatnonzero(unused_mask)
    if i == 0:
      # Choose the node that is the farthest from any other node.
      next_node = distances.max(1).argmax()
    else:
      # All distances from unused nodes to used nodes.
      candidate_distances = distances[np.ix_(unused_mask, ~unused_mask)]
      # Choose the next node, which is the farthest from the tour.
      next_node = unused_indices[candidate_distances.min(1).argmax()]
    unused_mask[next_node] = False

    if i < 3:
      # The first 3 nodes can be inserted in arbitrary order (symmetry).
      insertion_index = i - 1  # Append node to the end.
    elif randomizer is None:
      # Find the costs for inserting next_node at all possible locations.
      insertion_costs = _insertion_cost(distances, tour, np.roll(tour, -1),
                                        next_node)
      # Find insertion index with lowest insertion cost.
      insertion_index = np.argmin(insertion_costs)
    elif not caching or randomizer.needs_probabilities():
      insertion_costs = _insertion_cost(distances, tour, np.roll(tour, -1),
                                        next_node)
      # Use the insertion costs to define a probability distribution.
      unnormalized = np.power(np.reciprocal(insertion_costs), 1/temperature)
      distribution = unnormalized / np.sum(unnormalized)
      insertion_index = randomizer.sample_distribution(distribution)
    else:
      # Use probabilities in the trie, without computing insertion costs.
      insertion_index = randomizer.sample_distribution(None)
    tour.insert(insertion_index + 1, next_node)

  cost = distances[tour, np.roll(tour, -1)].sum()
  return cost, tour


def farthest_insertion_sampling(nodes, num_samples, unique_samples,
                                temperature, caching=True):
  """Samples using the farthest-insertion heuristic."""
  min_cost, best_tour = farthest_insertion(nodes, randomizer=None)

  randomizer = (ur.UniqueRandomizer() if unique_samples
                else ur.NormalRandomizer())
  for _ in range(1, num_samples):
    cost, tour = farthest_insertion(nodes, randomizer, temperature,
                                    caching=caching)
    randomizer.mark_sequence_complete()
    if cost < min_cost:
      min_cost = cost
      best_tour = tour

  return min_cost, best_tour


def farthest_insertion_bs(nodes, num_samples, temperature):
  """Samples with beam search."""
  num_nodes = len(nodes)
  distances = scipy.spatial.distance_matrix(nodes, nodes)

  unused_mask = np.ones(num_nodes, dtype=bool)

  # Create the starting tour. By symmetry, the first 3 nodes can be inserted in
  # any order.
  root_tour = []
  for i in range(3):
    unused_indices = np.flatnonzero(unused_mask)
    if i == 0:
      # Choose the node that is the farthest from any other node.
      next_node = distances.max(1).argmax()
    else:
      # All distances from unused nodes to used nodes.
      candidate_distances = distances[np.ix_(unused_mask, ~unused_mask)]
      # Choose the next node, which is the farthest from the tour.
      next_node = unused_indices[candidate_distances.min(1).argmax()]
    unused_mask[next_node] = False

    root_tour.insert(i, next_node)

  # Beam nodes include the partial tour, unused mask, and its log probability.
  beam = [(root_tour, unused_mask, 0.0)]
  for _ in range(3, num_nodes):
    candidates = []

    # Expand nodes in the beam.
    for tour, unused_mask, log_prob in beam:
      unused_indices = np.flatnonzero(unused_mask)
      candidate_distances = distances[np.ix_(unused_mask, ~unused_mask)]
      next_node = unused_indices[candidate_distances.min(1).argmax()]
      unused_mask = np.copy(unused_mask)
      unused_mask[next_node] = False

      # Use the insertion costs to define a probability distribution.
      insertion_costs = _insertion_cost(distances, tour, np.roll(tour, -1),
                                        next_node)
      unnormalized = np.power(np.reciprocal(insertion_costs), 1/temperature)
      distribution = unnormalized / np.sum(unnormalized)

      for i, new_log_prob in enumerate(np.log(distribution)):
        new_tour = list(tour)
        new_tour.insert(i + 1, next_node)

        candidates.append(
            (new_tour, unused_mask, log_prob + new_log_prob))

    # Select the best candidates.
    if num_samples >= len(candidates):
      beam = candidates
    else:
      scores = [node[2] for node in candidates]
      top_k_indices = np.argpartition(scores, -num_samples)[-num_samples:]
      beam = [candidates[i] for i in top_k_indices]

  best_tour = None
  best_cost = float('inf')
  for tour, _, _ in beam:
    cost = distances[tour, np.roll(tour, -1)].sum()
    if cost < best_cost:
      best_cost = cost
      best_tour = tour
  assert len(best_tour) == num_nodes
  return best_cost, best_tour


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  np.random.seed(FLAGS.seed)
  data = np.random.uniform(size=(FLAGS.dataset_size, FLAGS.graph_size, 2))
  per_instance_seeds = np.random.randint(1000000, size=(FLAGS.dataset_size))

  if FLAGS.solver == NEAREST_NEIGHBOR:
    solver = nearest_neighbor
  elif FLAGS.solver == FARTHEST_INSERTION:
    solver = farthest_insertion
  elif FLAGS.solver == FARTHEST_INSERTION_SAMPLING:
    solver = functools.partial(farthest_insertion_sampling,
                               num_samples=FLAGS.num_samples,
                               unique_samples=FLAGS.unique_samples,
                               temperature=FLAGS.temperature,
                               caching=FLAGS.caching)
  elif FLAGS.solver == FARTHEST_INSERTION_BS:
    solver = functools.partial(farthest_insertion_bs,
                               num_samples=FLAGS.num_samples,
                               temperature=FLAGS.temperature)
  else:
    raise app.UsageError('Unknown solver: {}'.format(FLAGS.solver))

  start_time = timeit.default_timer()
  solutions = []
  for instance, seed in zip(data, per_instance_seeds):
    np.random.seed(seed)
    solutions.append(solver(instance))
  elapsed_time = timeit.default_timer() - start_time

  dataset_cost = 0
  for cost, tour in solutions:
    if sorted(tour) != list(range(FLAGS.graph_size)):
      raise ValueError('Tour is malformed.')
    dataset_cost += cost

  print('Dataset size: {}'.format(FLAGS.dataset_size))
  print('Graph size: {}'.format(FLAGS.graph_size))
  print('Seed: {}'.format(FLAGS.seed))
  print('Solver: {}'.format(FLAGS.solver))
  if FLAGS.solver == FARTHEST_INSERTION_SAMPLING:
    print()
    print('Sampling-related options:')
    print('  Num samples: {}'.format(FLAGS.num_samples))
    print('  Unique samples: {}'.format(FLAGS.unique_samples))
    print('  Temperature: {}'.format(FLAGS.temperature))
    print('  Caching: {}'.format(FLAGS.caching))
  if FLAGS.solver == FARTHEST_INSERTION_BS:
    print()
    print('Beam search options:')
    print('  Num samples: {}'.format(FLAGS.num_samples))
    print('  Temperature: {}'.format(FLAGS.temperature))
  print()
  print('Time: {:.2f} sec'.format(elapsed_time))
  print('Average cost: {:.5f}'.format(dataset_cost / len(data)))


if __name__ == '__main__':
  app.run(main)
