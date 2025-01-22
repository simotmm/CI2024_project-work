from init import random, np
from classes import Node
from operators import OPERATORS

def create_random_tree(depth: int, terminals: int, prob_to_stop: int = 0.5) -> Node:
    if depth <= 0 or random.random() <= prob_to_stop: 
        return generate_random_terminal_node(terminals)
    func, arity = random.choice(list(OPERATORS.values()))
    children = [create_random_tree(depth - 1, terminals) for _ in range(arity)]
    return Node(func, children)


def create_full_random_tree(depth: int, terminals: int) -> Node:
    if depth <= 0: 
        return generate_random_terminal_node(terminals)
    func, arity = random.choice(list(OPERATORS.values()))
    children = [create_random_tree(depth - 1, terminals) for _ in range(arity)]
    return Node(func, children)


def generate_random_terminal_node(terminals) -> Node:
    terminal = random.choice(terminals)
    if terminal.startswith("x["): return Node(terminal)
    return Node(generate_random_constant())


def get_random_node(tree: Node) -> Node:
    if not tree.children or random.random() <= 0.5:
        return tree
    return get_random_node(random.choice(tree.children))


def generate_random_constant() -> int | float:
    const = random.randint(-100, 100) / 10
    if random.random() <= 0.5: return int(const)
    return const


def node_to_function(node: Node):
    if node is None: return None
    if node.is_x():  # Estrazione dell'indice, ritorna tutti i valori di x corrispondenti
        # Estrai il numero tra parentesi quadre come indice
        i = int(node.value[node.value.index("[") + 1: node.value.index(']')])
        return lambda x, y=None: x[i, :]  # Aggiungi un argomento facoltativo per y
    
    if node.is_constant():  # Ritorna sempre il valore costante per tutte le colonne di x
        return lambda x: np.full(x.shape[1], node.value)
    
    f = node.value  # Singolo o doppio operatore
    inputs = []
    for child in node.children:
        inputs.append(node_to_function(child))
    return lambda x: f(*[input(x) for input in inputs])


def node_as_function(node: Node, x):
    if node is None: return None
    if node.is_x():
        i = int(node.value[node.value.index("[") + 1: node.value.index(']')])
        return x[i, :]
    elif node.is_constant():
        return np.full(x.shape[1], node.value)
    #else: è single o double operator
    f = node.value
    inputs = []
    for child in node.children:
        inputs.append(node_as_function(child, x))
    return f(*inputs)


def print_tree_structure(node: Node, depth: int = 0):
    if node is None: return
    spazi = "  " * depth
    if depth == 0: print(f"tree structure: ")
    if callable(node.value): s = node.value.__name__
    else: s = node.value
    print(f" {spazi}(depth {depth}: {s}) ")
    for c in node.children:
        print_tree_structure(c, depth+1)