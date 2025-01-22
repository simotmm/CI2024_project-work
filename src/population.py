import numpy as np
from init import random
from classes import Node
from fitness import calculate_fitness
from tree_functions import create_full_random_tree, create_random_tree

PRINT_AVERAGE_DEPTH = True

def generate_initial_population(population_dim: int, terminals: list[str], tree_depth: int, full_tree_prob: int = 0.001) -> list[Node]:
    population = []
    tree_depth = 2 #impostare a 2 se il fitness cresce lentamente
    
    while len(population) < population_dim:
        if random.random() <= full_tree_prob:
            population.append(create_full_random_tree(tree_depth, terminals))
        else:
            population.append(create_random_tree(tree_depth, terminals))

    if PRINT_AVERAGE_DEPTH:
        print(f"population generater, average depth in {len(list(population))} trees: {average_tree_depth(list(population))}")
    return population


#selezione candidati parent
def parent_selection(population: list[Node], x: np.ndarray, y: np.ndarray) -> Node:
    
    def tournament_1v1(population, hole_prob=0.15):
        i1 = population[random.randint(len(population))]
        i2 = population[random.randint(len(population))]
        if calculate_fitness(i1, x, y) > calculate_fitness(i2, x, y): 
            best = i1
            other = i2
        else: 
            best = i2
            other = i1
        if random.random() <= hole_prob: #fitness hole to reduce bloat
            return other
        return best

    def tournament_selection(population, tournament_size, hole_prob=0.1):
        tournament = random.sample(list(population), tournament_size)
        if random.random() <= hole_prob: #fitness hole to reduce bloat
            return min(tournament, key=lambda ind: ind.fitness)
        return max(tournament, key=lambda ind: ind.fitness)

    #valid_population = population#[i for i in population if i.fitness != -np.inf]
    #if not valid_population:
    #    raise ValueError("La popolazione non contiene individui validi.")
    #return tournament_selection(population, tournament_size=2)
    return tournament_1v1(population)


def average_tree_depth(population):
    if not population: return 0
    return sum(tree.depth() for tree in population) / len(population)
