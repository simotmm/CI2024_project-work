from s333017 import f1, f2, f3, f4, f5, f6, f7, f8
from utils import get_problem, get_problems
from fitness import calculate_mse
import sys

f = None
PROBLEM_ID = None

if len(sys.argv) == 2:
    id = int(sys.argv[1])
    if id >= 1 and id <= 8: 
        PROBLEM_ID = id
        if id == 1: f = f1
        elif id == 2: f = f2
        elif id == 3: f = f3
        elif id == 4: f = f4
        elif id == 5: f = f5
        elif id == 6: f = f6
        elif id == 7: f = f7
        elif id == 8: f = f8


def test_formulas():
    if f and PROBLEM_ID:
        problem = get_problem(PROBLEM_ID)
        test_formula(f, problem)
    else:
        f_list = [f1, f2, f3, f4, f5, f6, f7, f6, f8]
        id = 1
        for function in f_list:
            problem = get_problem(id)
            test_formula(function, problem)
            id +=1

def test_formula(f, problem):
    x = problem.x
    y = problem.y
    mse = calculate_mse(f(x), y)
    print(f"problem {problem.id}, mse: {mse}")

test_formulas()