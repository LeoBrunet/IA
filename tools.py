from math import ceil, floor
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
import numpy as np

'''
    Quelques fonctions utiles pour le TDm1.
'''

def plot(data, x_axis=None, y_axis=None, title=None):
    '''
        Affichage de points (x = linéaire).
    '''
    fig = plt.figure()
    ls = fig.add_subplot(111)

    min_x = 0
    max_x = len(data)
    min_y = min(data)
    max_y = max(data)

    ls.set_xlim([min_x, max_x])
    ls.set_ylim([min_y, max_y])

    X = np.linspace(0, len(data), len(data))
    ls.plot(X, data, color='red', linewidth=1)

    if x_axis:
        ls.set_xlabel(x_axis)
    if y_axis:
        ls.set_ylabel(y_axis)
    if title:
        ls.set_title(title)

    plt.show()

def plot_data(data, a=None, b=None, x_axis=None, y_axis=None, title=None):
    '''
        Affichage de points
                    - data : data : liste de couples de points (x, y)
        Optionnel :
                    - a et b : coefficients d'une droite
                    - x_axis, y_axis : légendes des axes
                    - title : titre du graphique

    '''

    fig = plt.figure()
    ls = fig.add_subplot(111)

    estimation = mpatches.Patch(color='green', label='Estimation')
    fig.legend(handles=[estimation])

    min_x = min([x[0] for x in data])
    max_x = max([x[0] for x in data])
    min_y = min([x[1] for x in data])
    max_y = max([x[1] for x in data])

    ls.set_xlim([0, max_x + (max_x - min_x) / 4])
    ls.set_ylim([0, max_y + (max_y - min_y) / 4])

    ls.scatter([x[0] for x in data], [y[1] for y in data], color='green')

    if a is not None and b is not None:
        pred = [[0, 0], [max_x, a * max_x + b]]
        ls.plot([x[0] for x in pred], [y[1] for y in pred], color='red', linewidth=1)

    if x_axis:
        ls.set_xlabel(x_axis)
    if y_axis:
        ls.set_ylabel(y_axis)
    if title:
        ls.set_title(title)

    plt.show()

def compute_poly(X, coefs, reverse_coefs=True, target=None):
    '''
        Permet de calculer les valeurs d'un polynome pour des valeurs X en entrée
                    - X : suite de valeurs
                    - coefs : coefficients du polynome
        Optionnel :
                    - reverse_coefs : permet de retourner les coefficients dans
                                      le vecteur
                    - target : valeurs attendues (vecteur de même dimension que
                               X), pour calculer l'erreur
    '''
    coefs_to_use = coefs.copy()
    if reverse_coefs:
        coefs_to_use.reverse()

    Y = []

    for x in X:
        val = 0
        for i, coef in enumerate(coefs_to_use):
            val += coef * x**i
        Y.append(val)

    if target:
        error = MSE(zip(Y, target), coefs_to_use)
        # error = sum([abs(Y[i] - target[i]) for i in range(len(target))])
        return Y, error
    return Y

def plot_data_poly(data, coefs, title=None):
    '''
        Affichage de données et d'un polynome
                    - data : les données à afficher (couples de points (x, y))
                    - coefs : les coefficients du polynome à afficher
        Optionnel :
                    - title : titre du graphique
    '''
    max_x = max(x[0] for x in data)
    nb_pts = 100
    X = [(max_x * x / nb_pts) for x in range(nb_pts + 1)]
    Y = compute_poly(X, coefs, reverse_coefs=True)

    fig = plt.figure()
    ls = fig.add_subplot(111)

    min_x = min([x[0] for x in data])
    max_x = max([x[0] for x in data])
    min_y = min([x[1] for x in data])
    max_y = max([x[1] for x in data])

    ls.set_xlim([0, max_x + (max_x - min_x) / 4])
    ls.set_ylim([0, max_y + (max_y - min_y) / 4])

    ls.plot([x for x in X], [y for y in Y], color='blue', linewidth=1)
    ls.scatter([x[0] for x in data], [y[1] for y in data], color='green')

    if title:
        fig.set_title(title)

    plt.show()

def plot_multi_poly(data, all_coefs, title=None, fig_title=None):
    '''
        Affichage de données et d'un polynome
                    - data : les données à afficher (couples de points (x, y))
                    - coefs : les coefficients des polynomes à afficher (liste
                              de coefficients pour chaque polynome)
        Optionnel :
                    - title : titre de chaque graphique
                    - fig_title : titre de la figure entière
    '''
    size = len(all_coefs)

    final_x = 0

    for i in range(10):
        if pow(i, 2) < size:
            final_x += 1

    final_y = ceil(size / final_x)

    fig, axs = plt.subplots(final_y, final_x, sharex=False, sharey=False)

    if fig_title:
        fig.suptitle(fig_title)

    max_x = max(x[0] for x in data)
    nb_pts = 100
    X = [(max_x * x / nb_pts) for x in range(nb_pts + 1)]

    min_x = min([x[0] for x in data])
    max_x = max([x[0] for x in data])
    min_y = min([x[1] for x in data])
    max_y = max([x[1] for x in data])

    for i, coefs in enumerate(all_coefs):
        Y = compute_poly(X, coefs, reverse_coefs=True)

        current_axis = axs[floor(i / final_x)][i % final_x]

        current_axis.set_xlim([0, max_x + (max_x - min_x) / 4])
        current_axis.set_ylim([0, max_y + (max_y - min_y) / 4])

        current_axis.plot([x for x in X], [y for y in Y], color='blue', linewidth=1)
        current_axis.scatter([x[0] for x in data], [y[1] for y in data], color='green')

        if title:
            if len(title) == 1:
                current_axis.set_title(title)
            else:
                current_axis.set_title(title[i])

    plt.show()



def poly_numpy(data, degree):
    '''
        Permet de calculer automatiquement les coefficient d'un polynome le
        plus adapté aux données fournies en entrée.
                    - data : données en entrée
                    - degree : le degré voulu du polynome
    '''
    x_data = [x[0] for x in data]
    y_data = [y[1] for y in data]
    coefs = list(np.polyfit(x_data, y_data, degree))
    return coefs

def LSE(data, coefs, reverse_coefs=True):
    '''
        Calcule le LSE entre les données et le polynome.
    '''
    dmm_sum = 0
    coefs_to_use = coefs.copy()

    if reverse_coefs:
        coefs_to_use.reverse()

    for data_pt in data:
        val = 0
        for i, coef in enumerate(coefs_to_use):
            val += coef * data_pt[0]**i

        dmm_sum += pow(data_pt[1] - val, 2)

    return dmm_sum

def MSE(data, coefs, reverse_coefs=True):
    '''
        Calcule le MSE entre les données et le polynome.
    '''
    dmm_sum = LSE(data, coefs, reverse_coefs=reverse_coefs)
    return (1 / len(data)) * dmm_sum

def reg_lin(data):
    reg = LinearRegression(fit_intercept=False, normalize=False)
    reg.fit([[x[0]] for x in data], [[y[1]] for y in data])
    # pred = reg.predict([[x[0]] for x in data])
    return reg.coef_[0][0]

def print_2d_list(lst):
    for i, row in enumerate(lst):
        for j, value in enumerate(row):
            print(value, end=' ')
        print('\n')

def get_min_2d_list(lst):
    min_val = sys.maxsize
    idx = [-1, -1]

    for i in range(len(lst)):
        for j in range(len(lst[i])):
            if lst[i][j] < min_val:
                min_val = lst[i][j]
                idx = [i, j]

    return min_val, idx

def generate_points(nb=20, mean=None, cov=None, display=False):
    if mean is None:
        mean = [100, 5]

    if cov is None:
        cov = [[1, 0.5], [2500, 100]]

    x, y = np.random.multivariate_normal(mean, cov, nb).T

    if display:
        plt.scatter(x, y)
        plt.show()

    return x, y


if __name__ == '__main__':
    generate_points(20, True)

### UNUSED ###
# def find_best_coef_poly(data, to_print=False):
#     X = [x[0] for x in data]
#     Y_target = [x[1] for x in data]
#     best_error = [99999, None]

#     for i in range(-100, 100):
#         print(i)
#         for j in range(-100, 100):
#             for k in range(-100, 100):
#                 for l in range(-100, 100):
#                     _, error = compute_poly(X, [i, j, k, l], target=Y_target)
#                     if error < best_error[0]:
#                         best_error[0] = error
#                         best_error[1] = [i, j, k, l]

#     if to_print:
#         print(f'Best coefs: {best_error[1]}')
#         print(f'Y estimated: {compute_poly(X, best_error[1])}')

#     return best_error[1]