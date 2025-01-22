from classes import Node
from init import np
from tree_functions import get_random_node

# crossover:
#  prende due individui e ne genera uno nuovo
def crossover(n1: Node, n2: Node, depth: int, prob=0.5) -> Node:
    if n1.is_leaf() or n2.is_leaf():
        return get_random_node(np.random.choice([n1, n2]))
    mixed = []
    for n in n1.children:
        mixed.append(crossover(n, n2, depth))
    the_creature = Node(value=n1.value, type=n1.type, children=mixed)
    if the_creature.depth() > depth:
        return np.random.choice([n1, n2])
    return the_creature
