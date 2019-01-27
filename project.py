# Project by Martina Paganin and Luca Di Liello

import sys
from matplotlib import *
from pylab import *
from numpy.random import RandomState

## Crossovers
from problem import CircuitProblem, arithmetic_crossover, uniform_crossover
## Mutations
from problem import move_mutation, exchange_mutation, inversion_mutation, move_mutation_limited

from utils.geometry import Graph
from multi_objective import run
from utils.rw_graph import readNodesCoordinates, readEdgesList

############################################
############## PARAMETERS ##################
############################################

display = False

args = {}

# Space dimensions of the graph
args['space_dimensions'] = (0, 12) # for both x and y

## Pop and generations
args['pop_size'] = 20
args['max_generations'] = 4000

## Crossover
args['crossover_rate'] = .2
args['crossover_mix'] = .8

## Mutation
args['mutation_rate'] = .8
args['move_radius'] = 4

## Selection
args['tournament_size'] = 5
args['num_selected'] = args['pop_size'] # default equal to pop_size

## Replacement
args['num_elites'] = 2

## Weights of fitnesses
args['fitness_weights'] = [.85,.15]

## Use bounder and constraints
args['use_bounder'] = True
args['constrained'] = False

# Available crossovers: uniform_crossover, circuit_crossover_gene_arithmetic
# Available mutations: inversion_mutation, move_mutation, move_mutation_limited, exchange_mutation
args["variator"] = [move_mutation_limited, uniform_crossover]

############################################
############################################


def choice_without_replacement(rng, n, size):
    result = set()
    while len(result) < size :
        result.add(rng.randint(0, n))
    return result

class NumpyRandomWrapper(RandomState):
    def __init__(self, seed=None):
        super(NumpyRandomWrapper, self).__init__(seed)

    def sample(self, population, k):
        if isinstance(population, int) :
            population = range(population)

        return asarray([population[i] for i in
                        choice_without_replacement(self, len(population), k)])
        #return #self.choice(population, k, replace=False)

    def random(self):
        return self.random_sample()

    def gauss(self, mu, sigma):
        return self.normal(mu, sigma)

#----------------------------------------------------#


if __name__ == '__main__':

    #circuit = Graph(readNodesCoordinates('dataset/nodes.txt'), readEdgesList('dataset/edges.txt'))
    circuit = Graph(nodes=readNodesCoordinates("dataset/simple_circuit_nodi.txt") ,edges=readEdgesList('dataset/simple_circuit_archi.txt'))
    #circuit = Graph(nodes=readNodesCoordinates("dataset/circuit_nodi.txt") ,edges=readEdgesList('dataset/circuit_archi.txt'))

    problem = CircuitProblem(args, circuit)
    # plot the initial circuit
    circuit.plot()

    print 'init intersection_number', circuit.intersection_number()
    print 'init edges_total_length', circuit.edges_total_length()
    #print 'init edges_squareness', circuit.edges_squareness()

    if len(sys.argv) > 1 :
    	rng = NumpyRandomWrapper(int(sys.argv[1]))
    else:
    	rng = NumpyRandomWrapper()

    final_pop, final_pop_fitnesses = run(rng, problem, display=display, **args)

    print 'final fitnesses for best individual', final_pop[0].intersection_number(), final_pop[0].edges_total_length()
    #final_pop[0].plot()
    print 'all fitnesses', [str(x) for x in final_pop_fitnesses]

    # plot 5 best results
    for f in final_pop[:5]: f.plot()
