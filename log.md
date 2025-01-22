# Computational Intelligence - Report Log

Simone Tumminelli (s333017)

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

## Lab 1 content: Set Cover Problem
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

### Results
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

## Lab 2 content: Travelling Salesman Problem
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

### Code
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
