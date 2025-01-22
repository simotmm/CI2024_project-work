import warnings #per evitare warning durante la generazione degli individui
warnings.filterwarnings('ignore', category=RuntimeWarning)
from init import np, SETTINGS
from utils import get_problems, get_problem, plot_values, set_problems_settings
from genetic_algorithm import genetic_programming_algorithm
from operators import OPERATORS
from tree_functions import node_as_function
from classes import Settings
import sys

PLOT = True
PROBLEM_ID = None       

if len(sys.argv) == 2: # esegui l'algoritmo per un problema specifico con argomento da riga di comando
    id = int(sys.argv[1])
    if id >= 0 and id <= 8: PROBLEM_ID = id
if PROBLEM_ID:
    PROBLEMS = [get_problem(PROBLEM_ID)]
else:
    PROBLEMS = get_problems()

#PROBLEMS = set_problems_settings(PROBLEMS, SETTINGS)



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

def main():
    solutions = []
    all_fitnesses = []
    for problem in PROBLEMS:
        if problem.id == 0: continue #skip the first 
        solution, fitness_values = genetic_programming_algorithm(problem)
        solutions.append(solution)
        all_fitnesses.append((problem.id, fitness_values))
    if PLOT:
        print("\nplot fitness values")
        for solution, (problem_id, fitness_values) in zip(solutions, all_fitnesses):
            print(f"plot: problem {problem_id}\n(close the plot window to continue)\n")
            plot_values(f"problem {problem_id}: fitness", fitness_values)
main()