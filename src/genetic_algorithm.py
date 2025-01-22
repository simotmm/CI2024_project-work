import time
from tqdm import tqdm
from init import np, random
from fitness import calculate_fitness
from classes import Problem, Settings, Node
from utils import plot_values, seconds_to_str
from tree_functions import print_tree_structure
from genetic_operators import crossover, mutation
from population import generate_initial_population, parent_selection


PLOT = False
PRINT_STRUCTURE = False
RESET_POPULATION = False 
TOURNAMENT_SELECTION = False


def genetic_programming_algorithm(problem: Problem) -> Node:
    #dati problema e setting
    settings: Settings = problem.settings
    x = problem.x
    y = problem.y
    terminals = problem.terminals
    population_size = settings.population_dim
    num_generations = settings.max_generations
    max_depth = settings.max_depth
    mutation_prob = settings.mutation_prob
    elitism = settings.elitism
    if RESET_POPULATION:
        max_no_improving_gens = int(num_generations/10)
        no_improving_gens = 0

    # Inizio
    start_time = time.time()
    elitism_size = int(population_size * elitism)
    offspring_size = population_size - elitism_size
    best_fitness = -np.inf
    best_individual = None
    fitness_values = []

    print(problem)
    print(settings)
    #popolazione iniziale
    print(f"creation of the inititial population...")
    population = generate_initial_population(population_size, terminals, max_depth)

    #barra di avanzamento con tqdm
    with tqdm(total=num_generations, desc="genetic_algorithm", unit="gen") as pbar:
        for generation in range(num_generations):

            #calcolo fitness 
            this_gen_fitness_values = get_fitness_values(population, x, y)
            i = np.argmax(this_gen_fitness_values)
            current_best = population[i]
            current_best_fitness = this_gen_fitness_values[i]
            fitness_values.append(current_best_fitness)

            #update values tqdm bar
            pbar.set_postfix({
                "best Fitness": f"{current_best_fitness}",
                "depth": current_best.depth()
            })

            #aggiornamento del migliore
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best
            elif RESET_POPULATION:
                no_improving_gens += 0

            #elitismo: rimangono i migliori elitism_size individui
            sorted_population = sort_population_by_fitness(population, this_gen_fitness_values)
            elite = sorted_population[:elitism_size]

            if RESET_POPULATION and no_improving_gens > max_no_improving_gens: #modalità che dovrebbe introdurre più diversità
                population = generate_initial_population(population_size, terminals, max_depth)
                no_improving_gens = 0

            #offspring
            #offspring = generate_offspring(population, offspring_size, mutation_prob, max_depth, terminals, x, y)
            offspring = generate_offspring(population, offspring_size, mutation_prob, max_depth, terminals)
            population = elite + offspring  #popolazione per la prossima gen

            pbar.update(1) #bar update

    if PLOT: plot_values(f"problem {problem.id}: fitness", fitness_values, tree=best_individual)
    end_time = time.time() - start_time
    print(f"process complete, elapsed time: {seconds_to_str(end_time)}")
    print(f"best fitness: {best_fitness}, tree depth: {best_individual.depth()}")
    print(f"f{problem.id}(x) = {best_individual}")
    if PRINT_STRUCTURE: print_tree_structure(best_individual)
    print()
    return best_individual, fitness_values


def generate_offspring(population, offspring_size, mutation_prob, max_depth, terminals, x=None, y=None):
    offspring = []
    while len(offspring) < offspring_size:
        if random.random() <= mutation_prob:
            if TOURNAMENT_SELECTION and x and y:
                p1 = parent_selection(population, x, y)
                p2 = parent_selection(population, x, y)
                parents = [p1, p2]
            else:
                parents = random.sample(population, 2)
            child = crossover(parents[0], parents[1], max_depth)
        else:
            parent = random.choice(population)
            child = mutation(parent, max_depth, terminals)
        offspring.append(child)
    
    return offspring


def get_fitness_values(population: list[Node], x: np.ndarray, y: np.ndarray) -> list[float]:
    values = []
    for individual in population:
        values.append(calculate_fitness(individual, x, y))
    return values


def sort_population_by_fitness(population: list[Node], fitness_values: list[float]):
    paired = zip(fitness_values, population)
    sorted_paired = sorted(paired, key=lambda x: x[0], reverse=True)
    return [tree for _, tree in sorted_paired]
