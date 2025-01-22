import numpy as np
import matplotlib.pyplot as plt
from classes import Problem, Settings

PROBLEM_PATH = "../data/problem_"
PROBLEM_EXTENTION = ".npz"

def sol_to_string(sol):
    return "f(x) = " + str(sol) + "\n"\
            "fitness value: " + str(sol.fitness) + "\n"\

def get_problems():
    problems = []
    for i in range(9):
        problem = np.load(f"{PROBLEM_PATH}{i}{PROBLEM_EXTENTION}")
        problems.append(Problem(i, problem))
    return problems

def get_problem(i: int):
    problems = get_problems()
    return problems[i]

def plot_values(s, values, normalized=False): 
    if len(values) == 0:
        return
    elif len(values) == 1: #per plottare sempre almeno una linea
        values.append(values[0])
    if normalized: 
        # Normalizzazione dei fitness values in modo che siano compresi tra 0 e 1
        max_value = max(values)
        if max_value > 0:  # Evita divisioni per zero
            normalized_values = [fitness / max_value for fitness in values]
        else:
            normalized_values = values  # Se tutti i fitness sono zero, non cambiare i valori
    else:
        normalized_values = values
    # Plottare i valori di fitness (normalizzati o meno)
    plt.plot(normalized_values)
    plt.title(f'{"Normalized " if normalized else ""}{s} values over generations')
    plt.xlabel('Generations')
    plt.ylabel(f'{"Normalized " if normalized else ""}{s}')
    plt.show()


def calculate_variance(x):
    return sum((f-sum(x)/len(x))**2 for f in x)/len(x)

def set_problems_settings(problems: list[Problem], settings: list[Settings]) -> list[Problem]:
    for problem in problems:
        problem.settings = settings[problem.id]
    return problems
