import warnings #per evitare warning durante la generazione degli individui
warnings.filterwarnings('ignore', category=RuntimeWarning)
from init import SETTINGS_LIST
from utils import get_problems, get_problem, plot_values, set_problems_settings
from genetic_algorithm import genetic_programming_algorithm
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

PROBLEMS = set_problems_settings(PROBLEMS, SETTINGS_LIST)

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
            print(f"plot: problem {problem_id}\n(close the graph window to continue)\n")
            plot_values(f"problem {problem_id}: fitness", fitness_values, tree=solution)
    input("press enter to exit") #per evitare che su windows la finestra si chiuda se il chiamante è all_problems.bat
main()