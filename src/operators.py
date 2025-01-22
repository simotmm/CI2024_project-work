from init import np

TERMINALS = ["x", "const"]
SAFE_CONSTANT = 1e-40

def add(x,y):      return np.add(x,y)
def subtract(x,y): return np.subtract(x,y)
def multiply(x,y): return np.multiply(x,y)
def divide(x,y):   return np.divide(x,y+SAFE_CONSTANT) # trattete opportunamente in fase di conversione a stringa
def divmod(x,y):   return np.divmod(x,y+SAFE_CONSTANT)
def floor_divide(x,y): return np.floor_divide(x,y+SAFE_CONSTANT)
def pow(x,y):      return np.pow(x+SAFE_CONSTANT,y) 
def log(x):        return np.log(np.abs(x+SAFE_CONSTANT))
def sqrt(x):       return np.sqrt(np.abs(x+SAFE_CONSTANT)) 
def sin(x):        return np.sin(x)
def cos(x):        return np.cos(x)
def tan(x):        return np.tan(x)
def abs(x):        return np.abs(x)
def arcsin(x):     return np.arcsin(x)
def arccos(x):     return np.arccos(x)
def arctan(x):     return np.arctan(x)
def sinh(x):       return np.sinh(x)
def cosh(x):       return np.cosh(x)
def tanh(x):       return np.tanh(x)

OPERATORS = {
    "add":      (add, 2),
    "subtract": (subtract, 2),
    "multiply": (multiply, 2),
    "divide":   (divide, 2), 
    "abs":      (abs, 1), 
    "log":      (log, 1),    
    "sqrt":     (sqrt, 1), 
    "sin":      (sin, 1),
    "cos":      (cos, 1),      
    "tan":      (tan, 1),
    #"pow":      (pow, 2),  #rimossi, generano troppi nan
    #"arcsin":   (arcsin, 1), 
    #"arccos":   (arccos, 1),
    #"arctan":   (arctan, 1),
    #"sinh":     (sinh, 1),
    #"cosh":     (cosh, 1),
    #"tanh":     (tanh, 1),
    #"divmod":   (divmod, 2),
    #"floor_divide": (floor_divide, 2)
}

SINGLE_OPERATORS = {    
    "abs":      (abs, 1),
    "log":      (log, 1),    
    "sqrt":     (sqrt, 1), 
    "sin":      (sin, 1),
    "cos":      (cos, 1),      
    "tan":      (tan, 1),    
    #"arcsin":   (arcsin, 1),
    #"arccos":   (arccos, 1),
    #"arctan":   (arctan, 1),
    #"sinh":     (sinh, 1),
    #"cosh":     (cosh, 1),
    #"tanh":     (tanh, 1)
}

DOUBLE_OPERATORS = {
    "add":      (add, 2),
    "subtract": (subtract, 2),
    "multiply": (multiply, 2),
    "div":      (divide, 2),
    #"pow":      (pow, 2),
    #"divmod":   (divmod, 2),
    #"floor_divide": (floor_divide, 2)
}


