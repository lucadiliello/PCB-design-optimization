from pylab import *

from inspyred.ec import terminators, variators, replacers, selectors, evaluators
from inspyred.ec import EvolutionaryComputation

def run(random, problem, display=False, use_bounder=False, variator=None, **kwargs) :

    #create dictionaries to store data about initial population, and lines
    initial_pop_storage = {}

    # A generic genetic algorithm with custom defined mutation and crossover
    algorithm = EvolutionaryComputation(random)
    algorithm.terminator = terminators.generation_termination
    algorithm.variator = variator
    algorithm.selector = selectors.tournament_selection
    algorithm.replacer = replacers.generational_replacement

    if use_bounder:
        kwargs["bounder"] = problem.bounder

    kwargs["problem"] = problem

    # using pp to parallelize evaluations and speed up the process
    final_pop = algorithm.evolve(generator=problem.generator,
                        evaluator=evaluators.parallel_evaluation_pp,
                        pp_evaluator=problem.evaluator,
                        pp_dependencies=(lambda a: a**2,),
                        pp_modules=("import math",
                            "from inspyred import *",
                            "from inspyred import benchmarks",
                            "from problem import CombinedObjectives"),
                        maximize=problem.maximize,
                        initial_pop_storage=initial_pop_storage,
                        **kwargs)

    final_pop_fitnesses = asarray([guy.fitness for guy in final_pop])
    final_pop = asarray([guy.candidate for guy in final_pop])

    return final_pop, final_pop_fitnesses
