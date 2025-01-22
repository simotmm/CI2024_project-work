from init import np
from tree_functions import node_as_function, node_to_function

def calculate_fitness(tree, x, y):
    predictions = node_as_function(tree, x) 
    mse = calculate_mse(predictions, y)
    #try: predictions = evaluate_tree(tree, x)
    #except: return -np.inf
    #if np.any(np.isnan(predictions)): return -np.inf
    mse = calculate_mse(predictions, y)
    if(np.iscomplex(mse)): return -np.inf
    #r2 = r_2(y, predictions)
    #fitness = r_2 - mse
    return -mse

def calculate_mse(y_pred: np.ndarray, y: np.ndarray) -> float:
    return np.mean((y_pred - y) ** 2)

def r_2(y: np.array, y_pred: np.array) -> float: #coefficiente di determinazione R^2 
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - y_pred)**2)
    return 1 - (ss_res / ss_tot)