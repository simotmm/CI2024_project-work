NP_PREFIX = "np."
from operators import OPERATORS, SINGLE_OPERATORS, DOUBLE_OPERATORS

#classi per impostazioni, dati del problema, individio (nodo)

class Problem:
    def __init__(self,
                  id: int, 
                  data, 
                  settings: "Settings" = None):
        self.id = id
        self.data = data
        self.x = data["x"]
        self.y = data["y"]
        self.settings = settings or Settings(id)
        self.terminals = [f"x[{i}]" for i in range(self.x.shape[0])]
        self.terminals.append("const")

    def __str__(self):
        return f"problem {self.id}:\n" + \
               f"x shape: {self.x.shape}"


class Settings:
    def __init__(self, 
                 id: int, #chiave primaria con Problem
                 population_dim: int = 1000, 
                 max_generations: int = 20, 
                 max_depth: int = 8, 
                 mutation_prob: float = 0.5, 
                 elitism: float = 0.2,
                 offspring_mult: float = 0.0,
                 ):
        self.id = id
        self.population_dim = population_dim
        self.max_generations = max_generations
        self.max_depth = max_depth
        self.mutation_prob = mutation_prob
        self.elitism = elitism
        self.offspring_mult = offspring_mult

    def __str__(self):
        return f"population size: {self.population_dim}, " + \
               f"max generations: {self.max_generations}, " + \
               f"max depth {self.max_depth}, " + \
               f"mutation probability {self.mutation_prob*100}%, " + \
               f"elitism: {self.elitism*100}%" #\ + 
               #f"offspring mult: {self.offspring_mult}, " #non più usato, offspring_mult ora è (1-self.elitism)


class Node:
    def __init__(self, 
                 value, #int | float | str | callable, 
                 children: list["Node"] = None):
        self.value = value
        self.children = children or []
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_constant(self):
        return isinstance(self.value, (int, float))
    
    def is_x(self):
        return isinstance(self.value, str) and self.value.startswith("x[")
    
    def is_operator(self):
        return callable(self.value)
    
    def is_single_operator(self):
        return self.value in SINGLE_OPERATORS.values()
    
    def is_double_operator(self):
        return self.value in DOUBLE_OPERATORS.values()
    
    def tot_children(self):
        if self.is_leaf(): return 0
        return len(self.children)

    def __str__(self):
        if self.is_operator():
            f_np_name = f"{NP_PREFIX}{self.value.__name__}"
            inner = ",".join(str(child) for child in self.children)
            return f"{f_np_name}({inner})"
        return str(self.value)
    
    def depth(self):
        if self.is_leaf(): return 0
        l_d = 0
        r_d = 0
        if len(self.children) == 2:
            l_d = self.children[0].depth()
            r_d = self.children[1].depth()
        elif len(self.children) == 1:
            l_d = self.children[0].depth()
        return max(l_d, r_d) + 1 