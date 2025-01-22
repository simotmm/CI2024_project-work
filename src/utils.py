from init import np
import matplotlib.pyplot as plt
from classes import Problem


PROBLEM_PATH = "../data/problem_"
PROBLEM_EXTENTION = ".npz"


def get_problems():
    problems = []
    for i in range(9):
        problem = np.load(f"{PROBLEM_PATH}{i}{PROBLEM_EXTENTION}")
        problems.append(Problem(i, problem))
    return problems


def get_problem(i: int):
    problems = get_problems()
    return problems[i]


def plot_values(s, values, normalized=False, tree=None):
    if len(values) == 0:
        return
    elif len(values) == 1:  # Per plottare sempre almeno una linea
        values.append(values[0])

    if normalized: 
        # Normalizzazione dei fitness values in modo che siano compresi tra 0 e 1
        max_value = max(values)
        if max_value > 0:  # Evita divisioni per zero
            normalized_values = [fitness / max_value for fitness in values]
        else:
            normalized_values = values  # Se tutti i fitness sono zero, non cambiare i valori
    else:
        normalized_values = values

    # Plottare i valori di fitness (normalizzati o meno)
    plt.plot(normalized_values)
    plt.title(f'{"normalized " if normalized else ""}{s} values over generations')
    plt.xlabel('generations')
    plt.ylabel(f'{"normalized " if normalized else ""}{s}')

    # Aggiungere informazioni extra sotto il riquadro del grafico
    text_info = f"\nbest fitness value: {values[-1]}"
    if tree is not None:
        text_info += f"\nf(x) = {tree}"

    # Posizionare il testo sotto il riquadro del grafico
    plt.gca().text(
        0.5, -0.2,  # Posizione relativa nel grafico (x: centro, y: sotto il grafico)
        text_info,
        ha='center',  # Allineamento orizzontale
        va='center',  # Allineamento verticale
        transform=plt.gca().transAxes,  # Coordinate relative al sistema di assi
        fontsize=10
    )

    plt.tight_layout()  # Adatta i margini per evitare sovrapposizioni
    plt.show()


def set_problems_settings(problems, setting_list):
    settings_dict = {setting.id: setting for setting in setting_list}
    for problem in problems:
        if problem.id in settings_dict:
            problem.settings = settings_dict[problem.id]
    return problems


def calculate_variance(x):
    return sum((f-sum(x)/len(x))**2 for f in x)/len(x)

def seconds_to_str(time):
    days = int(time // 86400)
    time %= 86400
    hours = int(time // 3600)
    time %= 3600
    minutes = int(time // 60)
    seconds = time % 60
    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds > 0 or not parts:
        parts.append(f"{seconds:.2f} second{'s' if seconds != 1 else ''}")
    return ", ".join(parts)