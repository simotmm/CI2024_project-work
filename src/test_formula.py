import warnings #per evitare molti warning durante la generazione degli individui
warnings.filterwarnings('ignore', category=RuntimeWarning)

from utils import test_formula, get_problems
import time
from tree_functions import get_full_random_tree, get_random_tree
from fitness import calculate_fitness
import numpy as np
from genetic_operators import crossover
import random
from genetic_operators import mutation
from population import generate_initial_population
from classes import Settings
from genetic_algorithm import genetic_programming_algorithm

PROBLEM = 7
FORMULA = "((((((((np.sinh(x[5]) - x[5]) - x[5]) - x[5]) - x[5]) - x[5]) - x[5]) * 65.000) + ((np.sinh(x[5]) * (88.000 - ((x[5] - x[5]) - 88.000))) + ((((((np.sinh(x[5]) - 87.000) - 88.000) - 87.000) - 87.000) - 87.000) - 87.000)))"



trees = []
depth = 10


PROBLEM = 7


PROBLEMS = get_problems()
p = PROBLEMS[PROBLEM]
x = p.x
y = p.y
x_shape = p.x.shape

depth = 3

"""
for i in range(3):

    if len(trees) < 5:
        tree1 = get_random_tree(depth, x_shape)
        tree2 = get_random_tree(depth, x_shape)
        tree1.fitness = calculate_fitness(tree1, x, y)
        tree2.fitness = calculate_fitness(tree2, x, y)
        trees.append(tree1)
        trees.append(tree2)

    tree1 = random.choice(trees)
    tree2 = random.choice(trees)
    tree3 = crossover(tree1, tree2, depth)
    tree3.fitness = calculate_fitness(tree3, x, y)

    print(f"tree1: \nfitness: {tree1.fitness} \nf(x)={tree1}\n")
    print(f"tree2: \nfitness: {tree2.fitness} \nf(x)={tree2}\n")
    print(f"--> crossover (tree3):    \n    fitness: {tree3.fitness}     \n    f(x)={tree3}\n")

    tree1 = mutate_node(tree1, depth, x_shape)
    tree2 = mutate_node(tree2, depth, x_shape)
    tree3 = mutate_node(tree3, depth, x_shape)
    tree1.fitness = calculate_fitness(tree1, x, y)
    tree2.fitness = calculate_fitness(tree2, x, y)
    tree3.fitness = calculate_fitness(tree3, x, y)

    print(f"--> mutazione:")
    print(f"    tree1: \n    fitness: {tree1.fitness} \n    f(x)={tree1}\n")
    print(f"    tree2: \n    fitness: {tree2.fitness} \n    f(x)={tree2}\n")
    print(f"    tree3: \n    fitness: {tree3.fitness} \n    f(x)={tree3}\n")

    #input()



start = time.time()

population = generate_initial_population(100000, x, y, 2)

end = time.time()-start
print(f"time elapsed: {end}")
"""


PROBLEM = 7






problem = PROBLEMS[PROBLEM]



population_dim = 10000
offspring_size = population_dim // 5
off_mult = offspring_size/population_dim

settings = Settings(
    id=PROBLEM, 
    max_generations=150,
    population_dim=population_dim,
    max_depth=8,
    offspring_mult=off_mult,
    mutation_prob=0.5,
    elitism=0.1
)
problem.settings = settings

sol = genetic_programming_algorithm(problem)

#
# test_formula(FORMULA, PROBLEM)