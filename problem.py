from inspyred import benchmarks
from pylab import *
import copy
import networkx as nx
from inspyred.ec.variators import crossover, mutator
from utils.geometry import Graph

class CombinedObjectives:
    def __init__(self, values, constraints, args):

        if "fitness_weights" in args :
            weights = asarray(args["fitness_weights"])
        else :
            weights = asarray([1 for _ in values])

        # return weighted sum of fitnesses
        self.fitness = sum(asarray(values) * weights)
        if args['constrained']:
            self.fitness -= sum(asarray(constraints))

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __str__(self):
        return str(self.fitness)


class CircuitProblem(benchmarks.Benchmark):
    def __init__(self, constrained=False, initial_graph=None):

        self.initial_graph = initial_graph
        benchmarks.Benchmark.__init__(self, 2, 2)
        # bounder??
        self.maximize = False
        self.constrained=constrained

    def generator(self, random, args):
        individual = copy.deepcopy(self.initial_graph)

        ### INITIAL RANDOMISATION
        ## Generate random individuals
        individual.random_nodes(ranges=args['space_dimensions'])

        ### INITIAL MUTATIONS
        ## Mutate each individual (inversion_mutation)
        #individual = inversion_mutation.single_mutation(random, individual, {'mutation_rate': 1})

        ## Mutate each individual (move_mutation)
        #individual = move_mutation.single_mutation(random, individual, {'mutation_rate': 1})

        ## Mutate each individual (move_mutation_limited)
        #individual = move_mutation_limited.single_mutation(random, individual, {'mutation_rate': 1})

        ## Mutate each individual (exchange_mutation)
        #individual = exchange_mutation.single_mutation(random, individual, {'mutation_rate': 1})

        #individual.plot()
        return individual

    def evaluator(self, candidates, args):
        #print candidates
        fitnesses = []
        for c in candidates:
            f1 = c.intersection_number()
            f2 = c.edges_total_length()
            #f3 = c.edges_squareness()
            #c1 = c.constrains_nodes_edges(min_distance=0.5)

            fitnesses.append(CombinedObjectives([f1, f2], [], args))
        return fitnesses

    def bounder(self, candidate, args):
        ## each candidate's nodes in args['space_dimensions']
        for n in candidate.nodes:
            n.x = args['space_dimensions'][0] if n.x > args['space_dimensions'][0] else (0 if n.x < 0 else n.x)
            n.y = args['space_dimensions'][1] if n.y > args['space_dimensions'][1] else (0 if n.y < 0 else n.y)
        return candidate


########### CROSSOVERS ###########
@crossover
def arithmetic_crossover(random, mom, dad, args):
    ax_alpha = args.setdefault('crossover_mix', 0.5)
    crossover_rate = args.setdefault('crossover_rate', .5)
    bounder = args['_ec'].bounder

    children = []
    if random.random() < crossover_rate:
        bro = copy.deepcopy(dad)
        sis = copy.deepcopy(mom)

        if len(bro.nodes) != len(sis.nodes):
            #print bro, sis
            #print bro.nodes, sis.nodes
            raise Exception("Graphs have different number of nodes ?!?")

        for i in range(len(bro.nodes)):
            bro.nodes[i].x = int(ax_alpha * mom.nodes[i].x + (1 - ax_alpha) * dad.nodes[i].x)
            bro.nodes[i].y = int(ax_alpha * mom.nodes[i].y + (1 - ax_alpha) * dad.nodes[i].y)
            sis.nodes[i].x = int(ax_alpha * dad.nodes[i].x + (1 - ax_alpha) * mom.nodes[i].x)
            sis.nodes[i].y = int(ax_alpha * dad.nodes[i].y + (1 - ax_alpha) * mom.nodes[i].y)

        bro = bounder(bro, args)
        sis = bounder(sis, args)
        children.append(bro)
        children.append(sis)
    else:
        children.append(mom)
        children.append(dad)
    return children

@crossover
def uniform_crossover(random, mom, dad, args):
    ux_bias = args.setdefault('crossover_mix', .5)
    crossover_rate = args.setdefault('crossover_rate', .5)
    children = []
    bro = copy.deepcopy(dad)
    sis = copy.deepcopy(mom)
    if random.random() < crossover_rate:
        for i, (m, d) in enumerate(zip(mom.nodes, dad.nodes)):
            if random.random() < ux_bias:
                bro.nodes[i].__update__(m)
                sis.nodes[i].__update__(d)
    children.append(bro)
    children.append(sis)
    return children


########### MUTATIONS ###########

# select two random nodes p and q, then swap the coordinates of each node in the list between them
# input: [1,2,3,4,5,6,7,8]
# select randomly p = 2, q = 5
# output: [1,2,6,5,4,3,7,8]
@mutator
def inversion_mutation(random, candidate, args):
    rate = args.setdefault('mutation_rate', 0.2)
    mutant = copy.deepcopy(candidate)
    if random.random() < rate:
        size = len(mutant.nodes)
        p = random.randint(0, size-1)
        q = random.randint(0, size-1)
        low, high = min(p, q), max(p, q)
        for a in range((high - low + 1)/ 2):
            mutant.nodes[low + a].__swapvalues__(mutant.nodes[high - a])
    return mutant

# move a node to a random free position
@mutator
def move_mutation(random, candidate, args):
    rate = args.setdefault('mutation_rate', 0.1)
    mutant = copy.deepcopy(candidate)

    if random.random() < rate:
        size = len(mutant.nodes)
        p = random.randint(0, size-1)

        positions = set((node.x, node.y) for node in candidate.nodes)
        coord = (random.randint(*args['space_dimensions']), random.randint(*args['space_dimensions']))
        while (coord in positions):
            coord = (random.randint(*args['space_dimensions']), random.randint(*args['space_dimensions']))

        mutant.nodes[p].x = coord[0]
        mutant.nodes[p].y = coord[1]
    return mutant

# choose a random node and move it into a free node with a maximal distance of "move_radius"
@mutator
def move_mutation_limited(random, candidate, args):
    rate = args.setdefault('mutation_rate', 0.1)
    radius = args.setdefault('move_radius', 5)
    mutant = copy.deepcopy(candidate)
    if random.random() < rate:
        size = len(mutant.nodes)
        p = random.randint(0, size-1)
        x, y = mutant.nodes[p].x, mutant.nodes[p].y

        positions = set((node.x, node.y) for node in candidate.nodes)
        coord = (random.randint(x - radius, x + radius), random.randint(y - radius, y + radius))

        while (coord in positions):
            coord = (random.randint(x - radius, x + radius), random.randint(y - radius, y + radius))

        mutant.nodes[p].x, mutant.nodes[p].y = coord
    return mutant

# choose two random nodes and swap their coordinates
@mutator
def exchange_mutation(random, candidate, args):
    rate = args.setdefault('mutation_rate', 0.1)
    mutant = copy.deepcopy(candidate)
    if random.random() < rate:
        size = len(mutant.nodes)
        p = random.randint(0, size-1)
        neighbor = candidate.near_nodes(mutant.nodes[p])
        n = random.randint(0,len(neighbor)-1)
        mutant.nodes[p].__swapvalues__(neighbor[n])

        print 'candidate', [(node.x, node.y) for node in candidate.nodes]
        print 'mutant', [(node.x, node.y) for node in mutant.nodes]
    return mutant



##### OLD MUTATORS #######

'''
# Not useful anymore given the new dicrete space
@mutator
def coordinates_gaussian_mutation(random, candidate, args):
    mut_rate = args.setdefault('mutation_rate', 0.1)
    mean = args.setdefault('gaussian_mean', 0.0)
    stdev = args.setdefault('gaussian_stdev', 1.0)
    bounder = args['_ec'].bounder

    mutant = copy.deepcopy(candidate)

    for node in mutant.nodes:
        if random.random() < mut_rate:
            node.x += int(random.gauss(mean, stdev))
            node.y += int(random.gauss(mean, stdev))
    mutant = bounder(mutant, args)
    #print 'candidate', [(node.x, node.y) for node in candidate.nodes]
    #print 'mutant', [(node.x, node.y) for node in mutant.nodes]
    return mutant
'''
