from init import random, np
from tree_functions import create_random_tree, generate_random_terminal_node, get_random_node
from classes import Node
from operators import OPERATORS, DOUBLE_OPERATORS, SINGLE_OPERATORS


def crossover(n1: Node, n2: Node, depth: int, prob=0.5) -> Node:
    if n1.is_leaf() or n2.is_leaf():
        return get_random_node(n2)
    if random.random() <= prob: 
        return get_random_node(n2)
    mixed = []
    for c in n1.children:
        mixed.append(crossover(c, n2, depth))
    the_creature = Node(n1.value, mixed)
    if the_creature.depth() > depth:
        return n1 
    return the_creature


def mutation(tree: Node, max_depth: int, terminals: list[str]):

    MUTATION_TYPES = {
        "subtree_mutation": 0.575,  
        "point_mutation": 0.175,
        "expansion_mutation": 0.05,  
        "hoist_mutation": 0.1,    
        "collapse_mutation": 0.1
    }

    def expansion(tree: Node, max_depth: int, terminals: list[str]) -> Node:
        if tree is None: return None
        node = get_random_node(tree) #modifica per riferimento
        node = create_random_tree(max_depth, terminals)
        return tree

    def subtree(tree: Node, max_depth: int, terminals: list[str], prob_to_stop: int = 0.1) -> Node:
        if tree.is_leaf() or random.random() <= prob_to_stop or max_depth <= 0:  # Replace subtree
            return create_random_tree(2, terminals)
        new_tree = Node(tree.value, [subtree(child, max_depth - 1, terminals) for child in tree.children])
        if(new_tree.depth() > max_depth):
            return tree
        return new_tree

    def hoist(tree: Node, depth: int) -> Node:
        if tree is None or depth <= 0 or tree.is_leaf(): return tree
        new_c = []
        for c in tree.children:
            c = hoist(c, depth - 1)
            new_c.append(c)
        return get_random_node(tree)

    def collapse(tree: Node, terminals: list[str], prob_to_stop: int = 0.25) -> Node:
        if tree.is_leaf(): return tree
        if random.random() < prob_to_stop:  # Collapse this subtree into a terminal
            return generate_random_terminal_node(terminals)
        new_c = []
        for c in tree.children:
            c = collapse(c, terminals)
            new_c.append(c)
        return Node(tree.value, new_c)

    def point(tree: Node, terminals: list[str], prob_to_stop: int = 0.2) -> Node:
        if tree.is_leaf() or random.random() <= prob_to_stop:
            if tree.is_leaf():
                return generate_random_terminal_node(terminals)
            else:
                op, _ = random.choice([op for op in OPERATORS.values() if len(tree.children) == op[1]])
                return Node(op, tree.children)
        return Node(tree.value, [point(child, terminals) for child in tree.children])

    chosen_mutation = random.choices(
        list(MUTATION_TYPES.keys()),
        weights = list(MUTATION_TYPES.values()),
        k = 1 
    )[0]

    if chosen_mutation == "subtree_mutation":
        return subtree(tree, max_depth, terminals)
    elif chosen_mutation == "collapse_mutation":
        return collapse(tree, terminals)
    elif chosen_mutation == "point_mutation":
        return point(tree, terminals)
    elif chosen_mutation == "hoist_mutation":
        return hoist(tree, max_depth)
    elif chosen_mutation == "expansion_mutation":
        return expansion(tree, max_depth, terminals)