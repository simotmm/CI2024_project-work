import warnings #per evitare molti warning durante la generazione degli individui
warnings.filterwarnings('ignore', category=RuntimeWarning)

from init import np
from utils import get_problems, get_problem
from genetic_algorithm import genetic_programming_algorithm
from operators import OPERATORS
from tree_functions import node_as_function
from classes import Settings

PROBLEM_ID = 2

PROBLEMS = get_problems()
PROBLEM = get_problem(PROBLEM_ID)

x = PROBLEM.x
y = PROBLEM.y

terminals = PROBLEM.terminals

SETTINGS = Settings(
    id=PROBLEM_ID,
    population_dim=1000,
    max_generations=300,
    max_depth=6,
    mutation_prob=0.5,
    elitism=0.2
)

#PROBLEM 4
SETTINGS = Settings(
    id=PROBLEM_ID,
    population_dim=1000,
    max_generations=600,
    max_depth=6,
    mutation_prob=0.6,
    elitism=0.2
)



SETTINGS = Settings(
    id=PROBLEM_ID,
    population_dim=400,
    max_generations=1400,
    max_depth=6,
    mutation_prob=0.6,
    elitism=0.2
)

PROBLEM.settings = SETTINGS
sol = genetic_programming_algorithm(PROBLEM)


