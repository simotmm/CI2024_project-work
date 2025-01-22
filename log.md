# Computational Intelligence - Report Log
Report of activities carried out during the Computational Intelligence ([01URROV](https://didattica.polito.it/pls/portal30/gap.pkg_guide.viewGap?p_cod_ins=01URROV&p_a_acc=2025&p_header=S&p_lang=IT)) 
course at Politecnico di Torino.

Academic Year 2024-2025, first semester.

Simone Tumminelli (s333017)

# Contents
- [Lab 0](#lab-0)
  - [Lab 0 content: Joke about the first lecture](#lab-0-content-joke-about-the-first-lecture)
- [Lab 0 Reviews](#lab-0-reviews)
  - [Lab 0 Review 1](#lab-0-review-1)
  - [Lab 0 Review 2](#lab-0-review-2)
- [Lab 1](#lab-1)
  - [Lab 1 content and code: Set Cover Problem](#lab-1-content-and-code-set-cover-problem)
- [Lab 1 Reviews](#lab-1-reviews)
  - [Lab 1 Review 1](#lab-1-review-1)
  - [Lab 1 Review 2](#lab-1-review-2)
- [Lab 2](#lab-2)
  - [Lab 2 content (md file): Travelling Salesman Problem](#lab-2-content-md-file-travelling-salesman-problem)
  - [Lab 2 code](#lab-2-code)
- [Lab 2 Reviews](#lab-2-reviews)
  - [Lab 2 Review 1](#lab-2-review-1)
  - [Lab 2 Review 2](#lab-2-review-2)
- [Lab 3](#lab-3)
  - [Lab 3 content (md file): n^2-1 Puzzle](#lab-3-content-md-file-n2-1-puzzle)
  - [Lab 3 code](#lab-3-code)
- [Lab 3 Reviews](#lab-3-reviews)
  - [Lab 3 Review 1](#lab-3-review-1)
  - [Lab 3 Review 2](#lab-3-review-2)
- [Final Project Work: Symbolic Regression Genetic Programming](#final-project-work)
  - [Introduction](#introduction)
  - [Requirements](#requirements)
  - [Genetic Algorithm](#genetic-algorithm)
  - [Features](#features)
  - [Steps of the Algorithm](#steps-of-the-algorithm)
    - [Generation of the population](#generation-of-the-population)
    - [Fitness Evaluation](#fitness-evaluation)
    - [Generation of the Offspring and Elite](#generation-of-the-offspring-and-elite)
    - [Genetic Operators](#genetic-operators)
    - [Iteration](#iteration)
    - [End of the process](#end-of-the-process)
  - [Data Structures](#data-structures)
    - [Node](#node)
    - [Problem](#problem)
    - [Settings](#settings)

# Lab 0 
date: Sep 23, 2024

[Lab 0 Link](https://github.com/simotmm/CI2024_lab0)

## Lab 0 content: Joke about the first lecture
"Why did the AI fail the Turing test? Because it kept saying "I'm not a robot", but forgot to check the captcha box."
(joke by chatgpt)


# Lab 0 Reviews
## Lab 0 Review 1
date: Sep 26, 2024

[Lab 0 Review 1 Link](https://github.com/MarcoDelCore/CI2024_lab0/issues/1)
### Lab 0 Review 2 content
Nice joke

## Lab 0 Review 2
date: Sep 26, 2024

[Lab 0 Review 2 Link](https://github.com/carlopantax/CI2024_lab0/issues/1)
### Lab 0 Review 1 content
Good joke, I will bring a chess board too


# Lab 1
Oct 3, 2024 - Oct 9, 2024

[Lab 1 Link](https://github.com/simotmm/CI2024_lab1)

## Lab 1 content and code: Set Cover Problem
Template file from [https://github.com/squillero/computational-intelligence/blob/master/2024-25/set-cover.ipynb](https://github.com/squillero/computational-intelligence/blob/master/2024-25/set-cover.ipynb) 
### Set Cover problem
See: https://en.wikipedia.org/wiki/Set_cover_problem
```
from random import random, seed 
import numpy as np
from icecream import ic
```
#### Reproducible Initialization
If you want to get reproducible results, use rng (and restart the kernel); for non-reproducible ones, use np.random
```
UNIVERSE_SIZE = 100_000
NUM_SETS = 10_000
DENSITY = 0.1
rng = np.random.Generator(np.random.PCG64([UNIVERSE_SIZE, NUM_SETS, int(10_000 * DENSITY)]))
# DON'T EDIT THESE LINES!
SETS = np.random.random((NUM_SETS, UNIVERSE_SIZE)) < DENSITY
for s in range(UNIVERSE_SIZE):
    if not np.any(SETS[:, s]):
        SETS[np.random.randint(NUM_SETS), s] = True
COSTS = np.pow(SETS.sum(axis=1), 1.1)
```
#### Helper Functions
```
def valid(solution):
    """Checks wether solution is valid (ie. covers all universe)"""
    return np.all(np.logical_or.reduce(SETS[solution]))


def cost(solution):
    """Returns the cost of a solution (to be minimized)"""
    return COSTS[solution].sum()
```
#### Have fun!

```
# A dumb solution of "all" sets
solution = np.full(NUM_SETS, True)
ic(valid(solution), cost(solution))
```
ic| valid(solution): np.True_
    cost(solution): np.float64(251175032.4893038)
    (np.True_, np.float64(251175032.4893038))
```
# A random solution with random 50% of the sets
solution = rng.random(NUM_SETS) < .5
ic(valid(solution), cost(solution))
```
ic| valid(solution): np.True_
    cost(solution): np.float64(124526019.03602557)
(np.True_, np.float64(124526019.03602557))
#### Simple RHMC (Random Mutation Hill Climber)
source: lesson held on 03/10/2024
```
def single_mutation(solution):    #tweak - single mutation:
    new_sol = solution.copy()     #change a single value in a random position
    i = rng.integers(0, NUM_SETS) 
    new_sol[i] = not new_sol[i]  
    new_sol
    return new_sol
```
```
def multiple_mutation(solution):              #tweak - multiple mutation: 
    mask = rng.random(NUM_SETS) < .01         #create a boolean mask with 1% probability to be 'true'
    new_sol = np.logical_xor(solution, mask)  #xor: change the 'false' values to 'true' where the mask is 'true'
    return new_sol
```
```
def fitness(solution): #fitness function: tuple (lexicographic ordering)
    return (valid(solution), -cost(solution)) 
solution = rng.random(NUM_SETS) < 0.001      #random starting solution with low probability to have 'true' to start from an invalid solution
solution_fitness = fitness(solution)
first_solution_fitness = solution_fitness
ic(first_solution_fitness)
tweak = multiple_mutation                   #I choose the multiple mutation to tweak the solution

for steps in range(10_000):                      
    new_solution = tweak(solution)               #tweak the current solution
    new_solution_fitness = fitness(new_solution) #evaluate the new solution fitness
    if new_solution_fitness > solution_fitness:  #if the fitness of the new solution is better than the current one
        solution = new_solution                  #the current solution and its fitness are updated
        solution_fitness = fitness(solution)

ic(solution_fitness)
```
ic| first_solution_fitness: (np.False_, np.float64(-352204.3540240815))
ic| solution_fitness: (np.True_, np.float64(-3320313.7959436844))
(np.True_, np.float64(-3320313.7959436844))

##### Results
| Instance | Universe Size | Num Sets | Density | Probability (*) | First Fitness                                | Final Fitness                               | Execution Time |
| -------- | ------------- | -------- | ---     | --------------- | -------------------------------------------- | ------------------------------------------- | -------------- | 
| 1        | 100           | 10       | 0.2     | 0.895           | (np.True_, np.float64(-276.28183998918325))  | (np.True_, np.float64(-276.28183998918325)) | 0.001s         |
| 2        | 1_000         | 100      | 0.2     | 0.3             | (np.False_, np.float64(-10353.644030654768)) | (np.True_, np.float64(-6583.373093326254))  | 0.1s           |
| 3        | 10_000        | 1_000    | 0.2     | 0.05            | (np.False_, np.float64(-162079.19857870534)) | (np.True_, np.float64(-204360.41401215774)) | 0.3s           |
| 4        | 100_000       | 10_000   | 0.1     | 0.001           | (np.False_, np.float64(-352204.3540240815))  | (np.True_, np.float64(-3320313.7959436844)) | 1m 34.9s       |
| 5        | 100_000       | 10_000   | 0.2     | 0.001           | (np.False_, np.float64(-431447.2950779936))  | (np.True_, np.float64(-4793621.294248858))  | 1m 14.0s       |
| 6        | 100_000       | 10_000   | 0.3     | 0.003           | (np.False_, np.float64(-2695762.1963715665)) | (np.True_, np.float64(-11519921.684396852)) | 1m 31.4s       |

((*): " solution = rng.random(NUM_SETS) < 'Probability' ", it is set as low enough to start from an invalid solution)
### Greedy Algorithm
source: I used the algorithm from this website https://www.geeksforgeeks.org/greedy-approximate-algorithm-for-set-cover-problem/

```
def set_cover_greedy(SETS, COSTS):
    universe = set(range(SETS.shape[1])) #all elements
    covered = set()                      #covered elements
    selected = []                        #indexes of selected subsets, we start from an empty solution

    while covered!=universe:
        best_subset = None
        best_ratio = float("inf")

        for i in range(len(SETS)):
            subset = SETS[i]                            
            subset_elements = set(np.where(subset)[0]) #for each subset
            new_elements = subset_elements - covered   #I take the uncovered elements

            if new_elements:                  #if there are uncovered elements  
                current_cost = COSTS[i]       #I compute the ratio cost/coverage
                coverage = len(new_elements)
                ratio = current_cost/coverage

                if ratio < best_ratio:        #if the ratio is the best to this moment
                    best_ratio = ratio        #I choose the current as best subset
                    best_subset = i

        if best_subset is not None:
            selected.append(best_subset)
            covered.update(set(np.where(SETS[best_subset])[0]))
        else:
            print("No valid subset found.")

    return selected

solution = set_cover_greedy(SETS=SETS, COSTS=COSTS)
ic(fitness(solution))
```
ic| fitness(solution): (np.True_, np.float64(-1522897.3303484695))
(np.True_, np.float64(-1522897.3303484695))
This algorithm gives a valid solution but it is very slow: starting from the instance n.4 it takes 9.m 29.0s.



# Lab 1 Reviews
## Lab 1 Review 1
date:  Oct 19, 2024

[Lab 1 Review 1 Link](https://github.com/Blackhand01/CI2024_lab1/issues/2)
### Lab 1 Review 1 content
#### Overview
The proposed solution uses an Hill Climbing algorithm with a single mutation for each call of the `tweak` function. It reaches succesfully the maximum coverage for each instance of the problem proposed.

#### Code and documentation
The code is well organized and the comments are simple and helpful in order to understand the steps of the algorithm.

I appreciated the declaration of the `instances` array in the `Main Execution` part and I also appreciated the results summary in the readme markdown file.

The code ran successfully, there's no bug in it.

#### Possible improvements
- As said, the program reaches the maximum coverage for each instance, but the `valid` function is never called to check if a solution is actually valid. I suggest to add a `fitness` function that combines the `cost` function and the `valid` function.

##### Final comment
Good job! 👍







## Lab 1 Review 2
date:  Oct 19, 2024

[Lab 1 Review 2 Link](https://github.com/SamuelePolito/CI2024_lab1/issues/1)
### Lab 1 Review 1 content
#### Overview
The proposed code presents solution with different approaches, such as Greedy, Hill Climbing (with single and multiple tweaks) and Tweak And Restart. It is shown to reach the maximum coverage for the `(UNIVERSE_SIZE=10000, NUM_SETS=1000, DENSITY=0.2)` instance of the problem.

#### Code and documentation 
The code is well organized and the comments are simple and helpful in order to understand the steps of each algorithm. The code ran successfully with no bugs.

As said, the code reaches the maximum coverage for the `(UNIVERSE_SIZE=10000, NUM_SETS=1000, DENSITY=0.2)` instance of the problem, it does not run the other five instances proposed and the documentation does not show their results.

#### Possible improvements
- To document all the instances of the problem I suggest to add a code section to declare them in an array and run all the algorithms for each element, or, alternatively, manually set the instances in the `Reproducible Initialization` code section and fill a results table in the jupiter notebook file or the readme file.
- The two Hill Climbing algorithms search for the best solution, the fact that a solution is valid or not is shown in the icecream print but it is not checked in the algorithm, I suggest to modify the `fitness` function including the `valid` function in it.
- The code produces a very long print output, in order to track better the various solutions I suggest to save them in an array and then plot a graph.

##### Final comment
Good job👍





# Lab 2
Oct 29, 2024 - Nov 2, 2024
[Lab 2 Link](https://github.com/simotmm/CI2024_lab2)

## Lab 2 content (md file): Travelling Salesman Problem
### Computational Intelligence - Lab 2: TSP
The goal of this lab is to solve the Traveling Salesman Problem ("TSP", https://en.wikipedia.org/wiki/Travelling_salesman_problem) with the given instances (in the "cities" directory) 
using a fast but approximate algorithm and a slower yet more accurate one.

#### Results
##### Greedy Algorithm
| Instance | Steps | Cost (km)  |
| :------- | :---: | ---------: |
| Italy    | 46    |  4,436.03  |
| China    | 726   | 63,962.92  |
| Russia   | 167   | 42,334.16  |
| US       | 326   | 48,050.03  |
| Vanuatu  | 8     |  1,475.53  |

##### Evolutionary Algorithm
| Instance | Steps | Cost (km)  |
| :------- | :---: | ---------: |
| Italy    | 46    |   4,721.66 |
| China    | 726   | 367,689.23 |
| Russia   | 167   |  65,540.95 |
| US       | 326   | 142,000.39 |
| Vanuatu  | 8     |   1,345.54 |

## Lab 2 code 
Copyright **`(c)`** 2024 Giovanni Squillero `<giovanni.squillero@polito.it>`  
[`https://github.com/squillero/computational-intelligence`](https://github.com/squillero/computational-intelligence)  
Free under certain conditions — see the [`license`](https://github.com/squillero/computational-intelligence/blob/master/LICENSE.md) for details.  
```
import logging
from itertools import combinations
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import random
from icecream import ic
logging.basicConfig(level=logging.DEBUG)
INSTANCES = [
    {"cities": pd.read_csv("cities/italy.csv",   header=None, names=["name", "lat", "lon"]), "dist_matrix": None, "name": "Italy"},
    {"cities": pd.read_csv("cities/china.csv",   header=None, names=["name", "lat", "lon"]), "dist_matrix": None, "name": "China"},
    {"cities": pd.read_csv("cities/russia.csv",  header=None, names=["name", "lat", "lon"]), "dist_matrix": None, "name": "Russia"},
    {"cities": pd.read_csv("cities/us.csv",      header=None, names=["name", "lat", "lon"]), "dist_matrix": None, "name": "US"},
    {"cities": pd.read_csv("cities/vanuatu.csv", header=None, names=["name", "lat", "lon"]), "dist_matrix": None, "name": "Vanuatu"}
]
for instance in INSTANCES:
    cities = instance["cities"]
    dist_matrix = np.zeros((len(cities), len(cities)))
    for c1, c2 in combinations(cities.itertuples(), 2):
        dist_matrix[c1.Index, c2.Index] = dist_matrix[c2.Index, c1.Index] = geodesic(
            (c1.lat, c1.lon), (c2.lat, c2.lon)
        ).km
    instance["dist_matrix"] = dist_matrix
    cities.head()
```
#### Lab2 - TSP
https://www.wolframcloud.com/obj/giovanni.squillero/Published/Lab2-tsp.nb
```
def tsp_cost(instance, tsp):
    cities = instance["cities"]
    dist_matrix = instance["dist_matrix"]
    assert tsp[0] == tsp[-1]
    assert set(tsp) == set(range(len(cities)))

    tot_cost = 0
    for c1, c2 in zip(tsp, tsp[1:]):
        tot_cost += dist_matrix[c1, c2]
    return tot_cost
```
#### Greedy Algorithm
```
def greedy(instance, print):
    cities = instance["cities"]
    dist_matrix = instance["dist_matrix"]
    visited = np.full(len(cities), False)
    dist = dist_matrix.copy()
    city = 0
    visited[city] = True
    tsp = list()
    tsp.append(int(city))
    while not np.all(visited):
        dist[:, city] = np.inf
        closest = np.argmin(dist[city])
        #logging.debug(
        #    f"step: {cities.at[city,'name']} -> {cities.at[closest,'name']} ({dist_matrix[city,closest]:.2f}km)"
        #)
        visited[closest] = True
        city = closest
        tsp.append(int(city))
    #logging.debug(
    #    f"step: {cities.at[tsp[-1],'name']} -> {cities.at[tsp[0],'name']} ({dist_matrix[tsp[-1],tsp[0]]:.2f}km)"
    #)
    tsp.append(tsp[0])

    if print:
        logging.info(f" result: Found a path of {len(tsp)-1} steps, total length {tsp_cost(instance, tsp):.2f}km")
    return tsp

i=1
for instance in INSTANCES:
    logging.info(f" instance {i} ({instance["name"]}):")
    greedy(instance, True)
    i += 1 
```
INFO:root: instance 1 (Italy):

INFO:root: result: Found a path of 46 steps, total length 4436.03km

INFO:root: instance 2 (China):

INFO:root: result: Found a path of 726 steps, total length 63962.92km

INFO:root: instance 3 (Russia):

INFO:root: result: Found a path of 167 steps, total length 42334.16km

INFO:root: instance 4 (US):

INFO:root: result: Found a path of 326 steps, total length 48050.03km

INFO:root: instance 5 (Vanuatu):

INFO:root: result: Found a path of 8 steps, total length 1475.53km
#### Evolutionary Algorithm
```
def ea(instance, pop_size=100, generations=500, initial_pop_multiplier=2, mutation_rate=0.1, elitism_rate=0.05):
    # Dati di input: città e matrice delle distanze
    cities = instance["cities"]
    dist_matrix = instance["dist_matrix"]
    num_cities = len(cities)

    # Funzione per calcolare il costo di un percorso
    def path_cost(path):
        return tsp_cost(instance, path + [path[0]])

    # Selezione tramite torneo: seleziona il percorso con il costo minimo tra k individui
    def tournament_selection(population, k=5):
        tournament = random.sample(population, k)
        return min(tournament, key=path_cost)
    
    # Operatore di crossover: combina due genitori per produrre un figlio
    def order_crossover(parent1, parent2):
        start, end = sorted(random.sample(range(num_cities), 2))  # Segmento da copiare dal primo genitore
        child = [None] * num_cities
        child[start:end] = parent1[start:end]  # Copia il segmento nel figlio
        pos = end
        for city in parent2:  # Riempie le restanti città evitando duplicati
            if city not in child:
                if pos >= num_cities:
                    pos = 0
                child[pos] = city
                pos += 1
        return child
    
    # Operatore di mutazione: inverte un sottoinsieme casuale del percorso
    def inversion_mutation(tour):
        i, j = sorted(random.sample(range(num_cities), 2))
        tour[i:j] = reversed(tour[i:j])
        return tour

    # Inizializzazione della popolazione con percorsi casuali
    population = [random.sample(range(num_cities), num_cities) for _ in range(pop_size * initial_pop_multiplier)]
    best_solution = min(population, key=path_cost)  # Trova il percorso migliore iniziale
    best_cost = path_cost(best_solution)

    # Ciclo evolutivo principale
    for generation in range(1, generations + 1):
        new_population = []

        # Elitismo: mantiene una percentuale dei migliori individui
        elite_count = max(1, int(elitism_rate * pop_size))
        elite_individuals = sorted(population, key=path_cost)[:elite_count]
        new_population.extend(elite_individuals)

        # Generazione di nuovi individui attraverso crossover e mutazione
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, k=10)
            parent2 = tournament_selection(population, k=10)

            prob = 0.8 # Applica crossover con probabilità 80%
            if random.random() < prob: 
                child1 = order_crossover(parent1, parent2)
                child2 = order_crossover(parent2, parent1)
            else:  # In caso contrario, copia i genitori direttamente
                child1, child2 = parent1[:], parent2[:]

            # Applica mutazione con probabilità adattiva che decresce nel tempo
            adaptive_mutation_rate = max(0.01, mutation_rate - (mutation_rate * generation / generations))
            if random.random() < adaptive_mutation_rate:
                child1 = inversion_mutation(child1)
            if random.random() < adaptive_mutation_rate:
                child2 = inversion_mutation(child2)

            new_population.extend([child1, child2])

        # Aggiorna la popolazione con i nuovi individui
        population = new_population[:pop_size]

        # Aggiorna la migliore soluzione se è stato trovato un percorso con costo inferiore
        current_best = min(population, key=path_cost)
        current_cost = path_cost(current_best)

        if current_cost < best_cost:
            best_solution = current_best
            best_cost = current_cost
            # logging.info(f"Generation {generation}: New best cost = {best_cost:.2f} km")

    # Completa il percorso tornando alla città di partenza
    best_solution.append(best_solution[0])
    logging.info(f" result: Found a path of {len(best_solution) - 1} steps, total length {best_cost:.2f}km")

    return best_solution, best_cost

i=1
for instance in INSTANCES:
    logging.info(f" instance {i} ({instance["name"]}):")
    ea(instance)
    i += 1 
```
INFO:root: instance 1 (Italy):

INFO:root: result: Found a path of 46 steps, total length 4436.03km

INFO:root: instance 2 (China):

INFO:root: result: Found a path of 726 steps, total length 63962.92km

INFO:root: instance 3 (Russia):

INFO:root: result: Found a path of 167 steps, total length 42334.16km

INFO:root: instance 4 (US):

INFO:root: result: Found a path of 326 steps, total length 48050.03km

INFO:root: instance 5 (Vanuatu):

INFO:root: result: Found a path of 8 steps, total length 1475.53km

INFO:root: instance 1 (Italy):

INFO:root: result: Found a path of 46 steps, total length 4721.66km

INFO:root: instance 2 (China):

INFO:root: result: Found a path of 726 steps, total length 367689.23km

INFO:root: instance 3 (Russia):

INFO:root: result: Found a path of 167 steps, total length 65540.95km

INFO:root: instance 4 (US):

INFO:root: result: Found a path of 326 steps, total length 142000.39km

INFO:root: instance 5 (Vanuatu):

INFO:root: result: Found a path of 8 steps, total length 1345.54km


# Lab 2 Reviews
## Lab 2 Review 1
date: Nov 17, 2024

[Lab 2 Review 1 Link](https://github.com/AliEdrisabadi/CI2024_lab2/issues/5)
### Lab 1 Review 1 Content
#### Overview
The proposed code presents solution with different approaches: the greedy one that is more faster but less accurate and an avolutionary one that is slower; both approaches have the "italy" instance of the problem as input.

#### Code and documentation 
The code is well organized, there are some comments and I think they are sufficient to understand the two algorithm used. The code ran successfully with no bugs. The output is clear but in the greedy one it is shown a not valid total distance ("inf"). The readme file is empty.

As said, the code reaches a solution for the "italy" instance of the problem, the other four instances proposed are not included in the code and their results are not shown in the documentation.

#### Possible improvements
- To document all the instances of the problem I suggest to add a code section to declare them in an array and run the algorithms for each element, or, alternatively, manually set the instances in the data loading section and fill a results table in the jupiter notebook file or the readme file.
- I suggest to use a "requirements.txt" file to install all the dependencies at once using "pip install -r requirements.txt".
- In the greedy algorithm the total distance shown is "infinite", because the `DIST_MATRIX` is modified during the execution, I suggest to use a list for all the cities not already visited, in order to use the `DIST_MATRIX` to calculate correctly the final cost.
- In the evolutionary algorithm it would be good to start with individuals generated by the greedy algorithm.

##### Final comment
Good job!👍



## Lab 2 Review 2
date Nov 17, 2024

[Lab 2 Review 2 Link](https://github.com/LorenzoFormentin/CI2024_lab2/issues/3)
### Lab 1 Review 2 Content
#### Overview
The proposed code presents solution with different approaches: one that is more faster but less accurate and another that is slower and more accurate, using a bigger number of generations, adaptive mutation rate and elitism.

#### Code and documentation 
The code is well organized, there are very few comments but I don't think it's a problem, since the readme file explains the two algorithms well and clearly. The code ran successfully with no bugs. I really appreciated the clarity of the output and the progress bar, nice touch!

The code reaches a solution for the "vanuatu" instance of the problem, the other four instances proposed are not included in the code and their results are not shown in the documentation.

#### Possible improvements
- To document all the instances of the problem I suggest to add a code section to declare them in an array and run the algorithms for each element, or, alternatively, manually set the instances in the `Data Loading` code section and fill a results table in the jupiter notebook file or the readme file.
- I suggest to use a "requirements.txt" file to install all the dependencies at once using "pip install -r requirements.txt".
- For the current best individual is it possible to store the the fitness score, in order to non calculate it again.

##### Final comment
Very good job!👍


# Lab 3
Nov 18, 2024 - Nov 19, 2024

[Lab 3 Link](https://github.com/simotmm/CI2024_lab3)

## Lab 3 content (md file): n^2-1 Puzzle
### Computational Intelligence - Lab 3: n^2-1 Puzzle

The goal of this lab is to solve efficently a generic n^2-1 puzzle 
(also known as Gem Puzzle, Boss Puzzle, Mystic Square, etc. (https://en.wikipedia.org/wiki/Magic_square)) using path-search algorithms.
The problem consists of rearranging a board with numbered tiles into a specific goal configuration by moving tiles into an empty space.

#### Strategies Implemented
(Source: lecture held on date 14/11/2024)

##### A*
The A* algorithm is an informed search method that combines the cost of reaching a state `(g)` with a heuristic estimate of the cost to reach the goal `(h)`. The priority function for this algorithm is defined as `priority(s) = g(s) + h(s)`.
- `g(s)`: The cumulative cost to reach state `s` (calculated from the initial state).
- `h(s)`: A heuristic estimate of the cost to reach the goal from state `s`. The heuristic used in this implementation is the misplaced tiles heuristic, which counts the number of tiles not in their goal positions, ignoring empty tiles.
This algorithm is optimal when the heuristic is admissible (never overestimates the true cost).

##### Breadth-First Search
BFS is an uninformed search method that explores the state space level by level. It does not use a heuristic and treats all actions as having equal cost.
The priority is determined by the order of insertion, ensuring a FIFO behavior, this guarantees that all states at the current depth are explored before moving deeper.
It lways finds the shortest solution in terms of the number of actions, but it's computationally expensive for large puzzles.

## Lab 3 code
Copyright **`(c)`** 2024 Giovanni Squillero `<giovanni.squillero@polito.it>`  
[`https://github.com/squillero/computational-intelligence`](https://github.com/squillero/computational-intelligence)  
Free under certain conditions — see the [`license`](https://github.com/squillero/computational-intelligence/blob/master/LICENSE.md) for details.

### n-puzzle problem
```
RANDOM_SEED = 1242
SIZE = 3
```
```
import logging
from random import seed, choice
from typing import Callable
import numpy as np
from queue import PriorityQueue
logging.basicConfig(format="%(message)s", level=logging.INFO)
```
```
class State:
    def __init__(self, data: np.ndarray):
        self._data = data.copy()
        self._data.flags.writeable = False

    def __hash__(self):
        return hash(bytes(self._data))

    def __eq__(self, other):
        return bytes(self._data) == bytes(other._data)

    def __lt__(self, other):
        return bytes(self._data) < bytes(other._data)

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return repr(self._data)

    @property
    def data(self):
        return self._data

    def copy_data(self):
        return self._data.copy()
```

#### Search Algorithm (global search)
```
def search(
    initial_state: State,
    goal_test: Callable,
    parent_state: dict,
    state_cost: dict,
    priority_function: Callable,
    unit_cost: Callable,
):
    frontier = PriorityQueue() 
    parent_state.clear()      
    state_cost.clear()          

    state = initial_state       # start from the initial state
    parent_state[state] = None  # the initial state has no parent
    state_cost[state] = 0       # cost to reach the initial state is zero

    while state is not None and not goal_test(state):  # process states until goal is reached or no states remain
        for a in possible_actions(state):              # iterate over all possible actions from the current state
            new_state = result(state, a)               # determine the resulting state after applying the action
            cost = unit_cost(a)                        # compute the cost of this action
            if new_state not in state_cost and new_state not in frontier.queue:  
                parent_state[new_state] = state  # if the state is new: set the current state as its parent, calculate the cumulative cost and add it to the frontier with its priority                                
                state_cost[new_state] = state_cost[state] + cost 
                frontier.put(new_state, priority_function(new_state))  
                logging.debug(f"Added new node to frontier (cost={state_cost[new_state]})")
            elif new_state in frontier.queue and state_cost[new_state] > state_cost[state] + cost:  
                old_cost = state_cost[new_state] # if the state is already in the frontier with a higher cost: store the old cost for logging, update its parent to the current state and update its cumulative cost
                parent_state[new_state] = state 
                state_cost[new_state] = state_cost[state] + cost  
                logging.debug(f"Updated node cost in frontier: {old_cost} -> {state_cost[new_state]}")
        if frontier:  # if there are more states to process in the frontier get the state with the highest priority
            state = frontier.get()  
        else:
            state = None

    path = list()  # initialize the solution path
    s = state
    while s:                        # reconstruct the path from the goal state to the initial state
        path.append(s.copy_data())  # copy state data to avoid modifying the original state
        s = parent_state[s]         # move to the parent state

    logging.info(f"Found a solution in {len(path):,} steps; visited {len(state_cost):,} states") 
    return list(reversed(path))  # return the path in the correct order, from initial to goal state

```

#### Graph search for the the n-puzzle
```
seed(RANDOM_SEED)

GOAL = State(np.array(list(range(1, SIZE**2)) + [0]).reshape((SIZE, SIZE)))
logging.info(f"Goal:\n{GOAL}")

def goal_test(state):
    return state == GOAL

# (R, C) -> UP / RIGHT / DOWN / LEFT
MOVES = [np.array(_) for _ in [(-1, 0), (0, +1), (+1, 0), (0, -1)]]

def find_empty_space(board: np.ndarray):
    t = np.where(board == 0)
    return np.array([t[0][0], t[1][0]])

def is_valid(board: np.ndarray, action):
    return all(0 <= (find_empty_space(board) + action)[i] < board.shape[i] for i in [0, 1])

def possible_actions(state: State):
    return (m for m in MOVES if is_valid(state._data, m))

def result(state, action):
    board = state.copy_data()
    space = find_empty_space(board)
    pos = space + action
    board[space[0], space[1]] = board[pos[0], pos[1]]
    board[pos[0], pos[1]] = 0
    return State(board)

INITIAL_STATE = GOAL
for r in range(5_000):
    INITIAL_STATE = result(INITIAL_STATE, choice(list(possible_actions(INITIAL_STATE))))

print("\nInitial State:")
print(INITIAL_STATE)
```
Goal:
[[1 2 3]
 [4 5 6]
 [7 8 0]]

Initial State:
[[1 8 4]
 [6 5 2]
 [7 3 0]]

 #### A*
```
parent_state = dict()  # dictionary to store the parent of each state
state_cost = dict()    # dictionary to store the cumulative cost for each state

def h(state): # heuristic function: number of misplaced tiles (ignoring empty tiles)
    return np.sum((state._data != GOAL._data) & (state._data > 0))

final = search(
    INITIAL_STATE,                                     # starting state of the search
    goal_test=goal_test,                               # function to test if the goal state is reached
    parent_state=parent_state,                         # pass the parent_state dictionary for tracking
    state_cost=state_cost,                             # pass the state_cost dictionary for tracking
    priority_function=lambda s: state_cost[s] + h(s),  # combine the cost-so-far and the heuristic value
    unit_cost=lambda a: 1,                             # define a uniform unit cost for all actions
)
```
Found a solution in 143 steps; visited 6,654 states

#### Breadth-First
```
parent_state = dict()  # dictionary to store the parent of each state
state_cost = dict()    # dictionary to store the cumulative cost for each state

final = search(
    INITIAL_STATE,                                # starting state of the search
    goal_test=goal_test,                          # function to test if the goal state is reached
    parent_state=parent_state,                    # pass the parent_state dictionary for tracking
    state_cost=state_cost,                        # pass the state_cost dictionary for tracking
    priority_function=lambda s: len(state_cost),  # use a constant priority for breadth-first (queue behavior)
    unit_cost=lambda a: 1,                        # define a uniform unit cost for all actions
)
```
Found a solution in 143 steps; visited 6,654 states


# Lab 3 Reviews 
## Lab 3 Review 1
date: Dec 1, 2024

[Lab 3 Review 1 Link](https://github.com/rosfi12/CI2024_lab3/issues/2)
### Lab 3 Review 1 content
#### Overview
The proposed code presents a good solution with different approaches: BFS, A* and Greedy algorithms.

#### Code and documentation 
The code is well organized and the comments are clear and helpful in order to understand the algorithms. The code ran successfully with no bugs and the results output is clear.

The `readme` file is empty but I don't think this is a problem because the in the code the comments and the output are clear.

#### Possible improvements
- I suggest to use a "requirements.txt" file to install all the dependencies at once using "pip install -r requirements.txt".
- The use of the `is_solvable` function can be removed because the initial state of the puzzle is generated from the solution.
- In the `readme` file it would be good to have a brief description for the problem and the used approaches, as well as a table with the results.

##### Final comment
Very good job!👍



## Lab 3 Review 2
date: Dec 1, 2024

[Lab 3 Review 2 Link](https://github.com/UtkuKepir/CI2024_lab3/issues/2)
### Lab 3 Review 2 content
#### Overview
The proposed code presents a very good solution with a lot of different approaches such as A*, DFS and BFS, each one with different strategies as well.

#### Code and documentation 
The code is well organized, the comment are not a lot but I think they are sufficient to understand the goal of each section. The code ran successfully with no bugs and the results output is very clear, well done.

The `readme` file is simple and well structured, thank you for compiling all the results in the table, it helps a lot to make comparisons between the different strategies.

#### Possible improvements
- On the side of the algorithms I have nothing to suggest, I think you explored the strategies to solve the problem in a more than excellent way, undoubtedly more than I did.
- I suggest to use a "requirements.txt" file to install all the dependencies at once using "pip install -r requirements.txt".
- In the `readme` file there are three image links that don't work, maybe you forgot to upload the png files in the repo.

##### Final comment
Very good job! Kudos👍


# Final Project Work: Symbolic Regression Genetic Programming
4 Jan, 2025 - 23 Jan 2025

[Final Project Work Link](https://github.com/simotmm/CI2024_project-work)

## Introduction
The goal of this project is to find the best mathematical function that approximates the values of a gicen dataset.

The approach used is Symbolic Regression with Genetic Programming Algorithm, the function is encoded using sintax tree.



## Requirements
- Python 3.12.8 +
- matplotlib
- numpy
- tqdm



## Genetic Algorithm
In the genetic algorithm every syntax tree is treated as an individual in a population. The population evolves by generation in generation and the best individuals are preserved. At the end of the process the result is the best individual.


## Features
- **Supported operations**
  - Arithmetic operations: `+`, `-`, `*`, `/`.
  - Trigonometric operations: `sin`, `cos`, `tan`
  - Others: `sqrt`, `log`, `abs`
  - *Note*: `sqrt`, `log` and `div` are implemented in a safe way to avoid runtime errors. The operand of `sqrt` and `log` will be always in absolute value, and a very small constant (`1e-40`) is added to thsecond operand of `div`, as well as to the operand `sqrt` and `log`. This operations are preserved in the final result tree function.
  All the 
  - *Note*: all the operators are expressed in `numpy` (e.g. `a+b` will be `np.add(a,b)`).
- **Visualization**
  - During the evolution process: progress bar.
  - At the end of the evolution process: is shown a graph that illustrates the trend of evolution over the generations.


## Steps of the Algorithm
### Generation of the population
Given a maximum depth a list of individuals is generated with ramped method: a small part is a set of full trees (depth is maximum) and the other part is a set of not complete trees with depth between `0` and `max_depth` (`0` depth: a node with no children).

### Fitness evaluation
All the individuals in the population are evaluated and sorted by the fitness metric. The best individual is eventually updated.
Fitness Metric: based on mean square error. (`fitness`=`-mse`, higher values mean better results).

### Generation of the Offspring and Elite
The `offspring` is the result of **Genetic Operators**, it will be used, as next population, combined with the `elite`, the individuals with the best fitness values, they will be preserved in the next generation.

#### Genetic Operators
##### Mutation or Crossover
According to the mutation probability in the `Settings` is chosen Mutation or Crossover as genetic operator.
###### Parent Selection and Crossover
- Parent Selection: two individuals are randomly chosen from the population.
- *Note*: the first idea for the selection was a double tournament selection with fitness hole (choose n=2 individuals, and select the one with the best fitness with a hight probability) but the trend of the results suggested to remove that computational cost and choose randomly, to increase diversity and add a litte speed up to the process.
- Crossover: features of the two individuals are combined to create a new one.
###### Mutation
An individual is randomly selected from the population and their features are changed. Types of mutation used are: subtree mutation, hoist_mutation, collapse mutation, expansion mutation and point mutation.

### Iteration
The process from Fitness Evaluation to Mutation is repeated `max_generation` times.

### End of the process
At the end of the process the best individual and the history of best fitness values is returned.


## Data Structures
### Node
The `Node` object represents an individual of the popolation. It is, in fact, a tree with an associated value and eventually one or two child nodes, depending on the type of the value.
#### Node types
- Constant: an `integer` or a `floating point` value. It has no child nodes.
- Variable: a `string` in the form of `x[i]`, with `i` depending on the first dimension of the input dataset. It has no child nodes.
- Operator:
  - Single Operator: a custom `function` among `sin`, `cos`, `tan`, `log` and `sqrt`. It has one child node.
  - Double Operator: a custom `function` among `add`, `subtract`, `multiply` and `divide`. I has two child nodes.
### Problem
The `Problem` object represents the problem to solve, it contains the data from the corresponding `.npz` file and an object `Settings`.
### Settings
The `Settings` object is a set of parameters such as population dimension, number of generation, maximum tree depth, probabiliy of mutation and percentage of elitism.


## Repository Structure
- [s333017.py](./s333017.py): module that contains the best functions fuond by the algorithm for each problem proposed.
- [/data](./data/): it contains the dataset for the problems proposed, the form of each dataset is: input values `x` and output values `y` (`x` can be multi-dimensional).
- [log.md](./log.md): report that contains the log of activities during the course.
- [/src](./src/): root folder of genetic algorithm program. it contain a [runnable example](./src/example.ipynb), the main that runs the algorithm for all the problems and all the modules.
  - `main.py`: runs the algoritm for all the problems. It accept a command line parameter to run the algorithm for a specific problm. (example: to run the algoritm for the `problem_4.npz`: type in terminal "python main.py 4").
  - modules: 
    - `classes.py`: definition of classes for `Node`, `Problem` and `Settings` and global constants.
    - `operators.py`: definition of operators dictionaries and functions.
    - `init.py`: initialization of randomness and settings space for all the problems.
    - `fitness.py`: fitness evaluation functions.
    - `genetic_algoritm.py`: main algorithm and offspring functions.
    - `genetic_operators.py`: crossover and mutation functions.
    - `population.py`: population generation and parent selection functions.
    - `tree_functions.py`: functions generate full trees and nodes.
    - `utils.py`: utility functions to get data from dataset,to plot fitness graph, and others.
  - [/scripts](./src/scripts/): simple .bat scripts to run all the probems in parallel or run a single problem.


## Table of Result
| Problem | Mean Square Error      |
|:-------:|:----------------------:|
| 1       | 7.125940794232773e-34  |
| 2       | 13987717013674.275     |
| 3       | 5.306486117932578e-29  |
| 4       | 2.21376097012696e-05   |
| 5       | 7.804262963600968e-19  |
| 6       | 1.4783925196927175e-05 |
| 7       | 32.653988150939874     |
| 8       | 29932.624601487092     |


