import time
import random
import numpy as np
from classes import Settings
random.seed(int(time.time()))
np.random.seed(int(time.time()))


SETTINGS_LIST = [
    #PROBLEM 0 SETTINGS: not used, default values (DO NOT REMOVE: necessary for )
    Settings(id=0),

    #PROBLEM 1 SETTINGS: default values
    Settings(id=1),

    #PROBLEM 2 SETTINGS
    Settings(
        id=2,
        population_dim=1000,
        max_generations=600,
        max_depth=6,
        mutation_prob=0.6,
        elitism=0.2
    ),

    #PROBLEM 3 SETTINGS
    Settings(
        id=3,
        population_dim=100_000,
        max_generations=10000,
        max_depth=6,
        mutation_prob=0.6,
        elitism=0.2
    ),

    #PROBLEM 4 SETTINGS
    Settings(
        id=4,
        population_dim=1000,
        max_generations=600,
        max_depth=6,
        mutation_prob=0.6,
        elitism=0.2
    ),

    #PROBLEM 5 SETTINGS
    Settings(
        id=5,
        population_dim=1000,
        max_generations=600,
        max_depth=6,
        mutation_prob=0.6,
        elitism=0.2
    ),

    #PROBLEM 6 SETTINGS
    Settings(
        id=6,
        population_dim=1000,
        max_generations=600,
        max_depth=6,
        mutation_prob=0.6,
        elitism=0.2
    ),

    #PROBLEM 7 SETTINGS
    Settings(
        id=7,
        population_dim=1000,
        max_generations=600,
        max_depth=6,
        mutation_prob=0.6,
        elitism=0.2
    ),

    #PROBLEM 8 SETTINGS
    Settings(
        id=8,
        population_dim=1000,
        max_generations=600,
        max_depth=6,
        mutation_prob=0.6,
        elitism=0.2
    ),

]


SETTINGS = [
    


#PROBLEM 4





]