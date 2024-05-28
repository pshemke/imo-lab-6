import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import time
import multiprocessing as mp
from copy import deepcopy

def score(cities, paths):
    cycle_1 = paths[0] + [paths[0][0]]
    cycle_2 = paths[1] + [paths[1][0]]
    score_1=sum(cities[cycle_1[i], cycle_1[i+1]] for i in range(len(cycle_1) - 1))
    score_2=sum(cities[cycle_2[i], cycle_2[i+1]] for i in range(len(cycle_2) - 1))

    return score_1+score_2

def delta_insert(cities, path, i, city):
    a, b = path[i - 1], path[i]
    return cities[a, city] + cities[city, b] - cities[a, b]

def delta_replace_vertex(cities, path, i, city):
    path_len = len(path)
    a, b, c = path[(i - 1)%path_len], path[i], path[(i+1)%path_len]
    return cities[a, city] + cities[city, c] - cities[a, b] - cities[b, c]

def delta_replace_vertices_outside(cities, paths, i, j):
    return delta_replace_vertex(cities, paths[0], i, paths[1][j]) + delta_replace_vertex(cities, paths[1], j, paths[0][i])

def delta_replace_vertices_inside(cities, path, i, j):
    path_len = len(path)
    a, b, c = path[(i - 1)%path_len], path[i], path[(i+1)%path_len]
    d, e, f = path[(j-1)%path_len], path[j], path[(j+1)%path_len]
    if j-i == 1:
        return cities[a,e]+cities[b,f]-cities[a,b]-cities[e,f]
    elif (i, j) == (0, len(path)-1):
        return cities[e, c] + cities[d, b] - cities[b, c] - cities[d, e]
    else:
        return cities[a,e] + cities[e,c] + cities[d,b] + cities[b,f] -cities[a,b]-cities[b,c]-cities[d,e] - cities[e,f] 

def delta_replace_edges_inside(cities, path, i, j):
    path_len = len(path)
    if (i, j) == (0, len(path)-1):
        a, b, c, d = path[i], path[(i+1)%path_len], path[(j-1)%path_len], path[j]
    else:
        a, b, c, d = path[(i - 1)%path_len], path[i], path[j], path[(j+1)%path_len]
    return cities[a, c] + cities[b, d] - cities[a, b] - cities[c, d]

def outside_candidates(paths):
    indices = list(range(len(paths[0]))), list(range(len(paths[1])))
    indices_pairwise = list(itertools.product(*indices))
    return indices_pairwise

def inside_candidates(path):
    combinations = []
    for i in range(len(path)):
        for j in range(i+1, len(path)):
            combinations.append([i, j])
    return combinations

def replace_vertices_outside(paths, i, j):
    temp = paths[0][i]
    paths[0][i] = paths[1][j]
    paths[1][j] = temp

def replace_vertices_inside(path, i, j):
    temp = path[i]
    path[i] = path[j]
    path[j] = temp
    
def replace_edges_inside(path, i, j):
    if (i, j) == (0, len(path)-1):
        temp = path[i]
        path[i] = path[j]
        path[j] = temp     
    path[i:j+1] = reversed(path[i:j+1])
    
def regret(args):
    cities, start_idx = args
    n = cities.shape[0]
    unvisited = list(range(n))
    
    tour1 = [unvisited.pop(start_idx)]
    nearest_to_first_1 = [cities[tour1[0]][j] for j in unvisited]
    tour1.append(unvisited.pop(np.argmin(nearest_to_first_1)))

    start_city_2_idx = np.argmax([cities[tour1[0]][i] for i in unvisited])
    tour2 = [unvisited.pop(start_city_2_idx)]

    nearest_to_first_2 = [cities[tour2[0]][j] for j in unvisited]
    tour2.append(unvisited.pop(np.argmin(nearest_to_first_2)))

    nearest_to_tour_1 = [cities[tour1[0]][j] + cities[tour1[1]][j] for j in unvisited]
    tour1.append(unvisited.pop(np.argmin(nearest_to_tour_1)))

    nearest_to_tour_2 = [cities[tour2[0]][j] + cities[tour2[1]][j] for j in unvisited]
    tour2.append(unvisited.pop(np.argmin(nearest_to_tour_2)))

    while len(unvisited) > 0:
        for tour in [tour1, tour2]:
            regrets = []
            for city in unvisited:
                distances = [cities[tour[i]][city] + cities[city][tour[i+1]] - cities[tour[i]][tour[i+1]] for i in range(len(tour)-1)]
                distances.append(cities[tour[0]][city] + cities[city][tour[-1]] - cities[tour[-1]][tour[0]])
                distances.sort()
                regret = distances[1] - distances[0]
                regret -= 0.37 * distances[0]
                regrets.append((regret, city))
            regrets.sort(reverse=True)
            best_city = regrets[0][1]
            tour_distances = [cities[tour[i]][tour[i+1]] for i in range(len(tour)-1)]
            best_increase = float('inf')
            best_index = -1
            for i in range(len(tour_distances)):
                increase = cities[best_city][tour[i]] + cities[best_city][tour[i+1]] - tour_distances[i]
                if increase < best_increase:
                    best_increase = increase
                    best_index = i + 1
            tour.insert(best_index, best_city)
            unvisited.remove(best_city)
    return [tour1,tour2]

def regret_destroy_fix(tours,remove_count):
    removed_1 = random.sample(tours[0], k = remove_count//2)
    removed_2 = random.sample(tours[1], k = remove_count//2)
    for i in range(len(removed_1)):
        tours[0].remove(removed_1[i])
        tours[1].remove(removed_2[i])

    unvisited = removed_1 + removed_2

    while len(unvisited) > 0:
        for tour in [tours[0], tours[1]]:
            regrets = []
            for city in unvisited:
                distances = [cities[tour[i]][city] + cities[city][tour[i+1]] - cities[tour[i]][tour[i+1]] for i in range(len(tour)-1)]
                distances.append(cities[tour[0]][city] + cities[city][tour[-1]] - cities[tour[-1]][tour[0]])
                distances.sort()
                regret = distances[1] - distances[0]
                regret -= 0.37 * distances[0]
                regrets.append((regret, city))
            regrets.sort(reverse=True)
            best_city = regrets[0][1]
            tour_distances = [cities[tour[i]][tour[i+1]] for i in range(len(tour)-1)]
            best_increase = float('inf')
            best_index = -1
            for i in range(len(tour_distances)):
                increase = cities[best_city][tour[i]] + cities[best_city][tour[i+1]] - tour_distances[i]
                if increase < best_increase:
                    best_increase = increase
                    best_index = i + 1
            tour.insert(best_index, best_city)
            unvisited.remove(best_city)
    return [tours[0],tours[1]]

class Steepest(object):
    def __init__(self, cities):
        self.cities = cities
        self.delta = delta_replace_edges_inside
        self.replace = replace_edges_inside
        self.moves = [self.outside_vertices_trade_best, self.inside_trade_best]
    
    def outside_vertices_trade_best(self, cities, paths):
        candidates = outside_candidates(paths)
        scores = np.array([delta_replace_vertices_outside(cities, paths, i, j) for i, j in candidates])
        best_result_idx = np.argmin(scores)
        if scores[best_result_idx] < 0:
            return replace_vertices_outside, (paths, *candidates[best_result_idx]), scores[best_result_idx]
        return None, None, scores[best_result_idx]
            
    def inside_trade_best(self, cities, paths):
        combinations = inside_candidates(paths[0]), inside_candidates(paths[1])
        scores = np.array([[self.delta(cities, paths[idx], i, j) for i, j in combinations[idx]] for idx in range(len(paths))])
        best_path_idx, best_combination = np.unravel_index(np.argmin(scores), scores.shape)
        best_score = scores[best_path_idx, best_combination]
        if best_score < 0:
            return self.replace, (paths[best_path_idx], *combinations[best_path_idx][best_combination]), best_score
        return None, None, best_score 
    
    def __call__(self, paths):
        paths = deepcopy(paths)
        start = time.time()
        while True:
            replace_funs, args, scores = list(zip(*[move(self.cities, paths) for move in self.moves]))
            best_score_idx = np.argmin(scores)
            if scores[best_score_idx] < 0:
                replace_funs[best_score_idx](*args[best_score_idx])
            else:
                break
        return time.time()-start, paths
    
def pairwise_distances(points):
    num_points = len(points)
    dist_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])

    return dist_matrix

def plot_optimized_tours(positions, cycle1, cycle2, method):
    cycle1.append(cycle1[0])
    cycle2.append(cycle2[0])

    plt.figure()
    plt.plot(positions[cycle1, 0], positions[cycle1, 1], linestyle='-', marker='o', color='r', label='Cycle 1')
    plt.plot(positions[cycle2, 0], positions[cycle2, 1], linestyle='-', marker='o', color='b', label='Cycle 2')

    plt.legend()
    plt.title(method)
    plt.savefig(method)

def index(xs, e):
    try:
        return xs.index(e)
    except:
        return None
        
def find_node(paths, a):
    i = index(paths[0], a)
    if i is not None: return 0, i
    i = index(paths[1], a)
    if i is not None: return 1, i
    print(paths)
    assert False, f'City {a} must be in either cycle'
    
def remove_at(xs, sorted_indices):
    for i in reversed(sorted_indices):
        del(xs[i])

def reverse(xs, i, j):
    n = len(xs)
    d = (j - i) % n
    for k in range(abs(d)//2+1):
        a, b = (i+k)%n, (i+d-k)%n
        xs[a], xs[b] = xs[b], xs[a]

def insert_move(moves, move):
    delta_x = move[0]
    for i, x in enumerate(moves):
        delta_y = x[0]
        if delta_x < delta_y:
            moves.insert(i, move)
            return
        elif delta_x == delta_y:
            return
    moves.append(move)

def has_edge(cycle, a, b):
    for i in range(len(cycle) - 1):
        x, y = cycle[i], cycle[i+1]
        if (a, b) == (x, y): return +1
        if (a, b) == (y, x): return -1
        
    x, y = cycle[-1], cycle[0]
    if (a, b) == (x, y): return +1
    if (a, b) == (y, x): return -1
    return 0

def any_has_edge(paths, a, b):
    for i in range(2):
        status = has_edge(paths[i], a, b)
        if status != 0: return i, status
    return None, 0

def delta_swap_node(D, x1, y1, z1, x2, y2, z2):
    return D[x1,y2] + D[z1,y2] - D[x1,y1] - D[z1,y1] + D[x2,y1] + D[z2,y1] - D[x2,y2] - D[z2,y2]

def make_swap_node(cities, paths, cycycle1, i, cycycle2, j):
    cycle1, cycle2 = paths[cycycle1], paths[cycycle2]
    D = cities
    n, m = len(cycle1), len(cycle2)
    x1, y1, z1 = cycle1[(i-1)%n], cycle1[i], cycle1[(i+1)%n]
    x2, y2, z2 = cycle2[(j-1)%m], cycle2[j], cycle2[(j+1)%m]
    delta = delta_swap_node(cities, x1, y1, z1, x2, y2, z2)
    move = delta, NODE, cycycle1, cycycle2, x1, y1, z1, x2, y2, z2
    return delta, move

def delta_swap_edge(cities, a, b, c, d):
    if a == d or a == b or a == c or b == c or b == d or c == d: return 1e8
    return cities[a, c] + cities[b, d] - cities[a, b] - cities[c, d]

def gen_swap_edge_2(cities, cycle, i, j):
    n = len(cycle)
    nodes = cycle[i], cycle[(i+1)%n], cycle[j], cycle[(j+1)%n]
    return (delta_swap_edge(cities, *nodes), *nodes)

def delta_swap_edge_2(cities, cycle, i, j):
    return gen_swap_edge_2(cities, cycle, i, j)[0]

def gen_swap_edge(n):
    return [(i, (i+d)%n) for i in range(n) for d in range(2, n-1)]

def gen_swap_node(n, m):
    return [(i, j) for i in range(n) for j in range(m)]

def init_moves(cities, paths):
    moves = []
    for k in range(2):
        cycle = paths[k]
        n = len(cycle)
        for i, j in gen_swap_edge(n):
            delta, a, b, c, d = gen_swap_edge_2(cities, cycle, i, j)
            if delta < 0: moves.append((delta, EDGE, a, b, c, d))
    for i, j in gen_swap_node(len(paths[0]), len(paths[1])):
        delta, move = make_swap_node(cities, paths, 0, i, 1, j)
        if delta < 0: moves.append(move)
    return moves

def random_cycle(args):
    cities, _ = args
    n = cities.shape[0]
    remaining = list(range(n))
    random.shuffle(remaining)
    paths = [remaining[:n//2], remaining[n//2:]]
    return paths

EDGE, NODE = range(2)
def apply_move(paths, move):
    kind = move[1]
    if kind == EDGE:
        _, _, a, _, c, _ = move
        (cycle1, i), (cycle2, j) = find_node(paths, a), find_node(paths, c)
        cycle = paths[cycle1]
        n = len(cycle)
        reverse(cycle, (i+1)%n, j)
    elif kind == NODE:
        _, _, cycle1, cycle2, _, a, _, _, b, _ = move
        i, j = paths[cycle1].index(a), paths[cycle2].index(b)
        paths[cycle1][i], paths[cycle2][j] = paths[cycle2][j], paths[cycle1][i]
    else:
        assert False, 'Invalid move type'


class MSLS:
    def __init__(self,cities):
        self.cities = cities

    def __call__(self, paths):
        start = time.time()
        #paths jest nie używane, wewnętrznie generujemy 100 rozwiązań losowych
        solutions = list(map(random_cycle, [(cities, i) for i in range(100)]))
        _, new_solutions = zip(*list(map(Steepest(self.cities), solutions)))
        new_scores = [score(self.cities, x) for x in new_solutions]
        best_idx = np.argmin(new_scores)
        best = new_solutions[best_idx]
        return time.time() - start, best


class ILS1:
    def __init__(self,cities,file):
        self.cities = cities
        self.local_search = Steepest(self.cities)
        self.perturbation_count = 0
        self.time_limit = 0
        self.swap_count = 10 #wymieniamy 10 wierzcholkow miedzy cyklami
        if file == 'kroa':
            self.time_limit = 357.370771
        elif file == 'krob':
            self.time_limit = 358.975393

    def __call__(self,paths):
        cycles = deepcopy(paths)
        start = time.time()
        _, current_solution = self.local_search(cycles)
        while time.time() - start < self.time_limit:
            experimental_solution = deepcopy(current_solution)
            #perturbacja
            current_swaps = 0
            while current_swaps < self.swap_count:
                #wybierz losowy wierzcholek z cykli i je zamien
                node_1 = random.choice(experimental_solution[0])
                node_2 = random.choice(experimental_solution[1])
                i, j = experimental_solution[0].index(node_1), experimental_solution[1].index(node_2)
                experimental_solution[0][i], experimental_solution[1][j] = experimental_solution[1][j], experimental_solution[0][i]
                current_swaps += 1
            #wersja z local_searchem
            _, experimental_solution = self.local_search(experimental_solution)
            if(score(self.cities,experimental_solution) < score(self.cities,current_solution)):
                current_solution = experimental_solution
            self.perturbation_count += 1

        return time.time() - start, current_solution, self.perturbation_count
         

class ILS2_local_search:
    def __init__(self,cities,file):
        self.cities = cities
        self.local_search = Steepest(self.cities)
        self.perturbation_count = 0
        self.time_limit = 0
        self.remove_count = 60 # pozbywamy sie przy kazdej perturbacji 60 wierzcholkow
        if file == 'kroa':
            self.time_limit = 357.370771
        elif file == 'krob':
            self.time_limit = 358.975393

    def __call__(self,paths):
        cycles = deepcopy(paths)
        start = time.time()
        _, current_solution = self.local_search(cycles)
        while time.time() - start < self.time_limit:
            experimental_solution = deepcopy(current_solution)
            #perturbacja
            regret_destroy_fix(experimental_solution,self.remove_count)
            #wersja z local_searchem
            _, experimental_solution = self.local_search(experimental_solution)
            if(score(self.cities,experimental_solution) < score(self.cities,current_solution)):
                current_solution = experimental_solution
            self.perturbation_count += 1

        return time.time() - start, current_solution, self.perturbation_count
    
class ILS2:
    def __init__(self,cities,file):
        self.cities = cities
        self.local_search = Steepest(self.cities)
        self.perturbation_count = 0
        self.time_limit = 0
        self.remove_count = 60 # pozbywamy sie przy kazdej perturbacji 60 wierzcholkow
        if file == 'kroa':
            self.time_limit = 357.370771
        elif file == 'krob':
            self.time_limit = 358.975393

    def __call__(self,paths):
        cycles = deepcopy(paths)
        start = time.time()
        _, current_solution = self.local_search(cycles)
        while time.time() - start < self.time_limit:
            experimental_solution = deepcopy(current_solution)
            #perturbacja
            regret_destroy_fix(experimental_solution,self.remove_count)
            #wersja z local_searchem
            #_, experimental_solution = self.local_search(experimental_solution)
            if(score(self.cities,experimental_solution) < score(self.cities,current_solution)):
                current_solution = experimental_solution
            self.perturbation_count += 1

        return time.time() - start, current_solution, self.perturbation_count

class Natura_local_search:
    def __init__(self,cities,file):
        self.cities = cities
        self.local_search = Steepest(self.cities)
        self.perturbation_count = 0
        self.time_limit = 0
        if file == 'kroa':
            self.time_limit = 357.370771
        elif file == 'krob':
            self.time_limit = 358.975393

    def __call__(self):
        start = time.time()
        #wygeneruj poczatkowa populacje - DONE
        solutions = list(map(random_cycle, [(cities, i) for i in range(20)]))
        _, population = zip(*list(map(Steepest(self.cities), solutions)))
        while time.time() - start < self.time_limit:
            #Wylosuj dwa różne rozwiązania (rodziców) stosując rozkład równomierny - DONE
            parents = random.sample(population, k = 2)
            #Skonstruuj rozwiązanie potomne y poprzez rekombinację rodziców - WIP
            child = self.recombine(parents) #niezrobiona
            #y := Lokalne przeszukiwanie (y) (opcjonalnie) - DONE
            _, child = self.local_search(child)
            #jeżeli y jest lepsze od najgorszego rozwiązania w populacji i (wystarczająco) różne od wszystkich rozwiązań w populacji - WIP
            new_scores = [score(self.cities, x) for x in population]
            worst_idx = np.argmax(new_scores)
            if score(self.cities, child) < new_scores[worst_idx]: #WIP
                #Dodaj y do populacji i usuń najgorsze rozwiązanie - DONE
                population.pop(worst_idx)
                population.append(child)

            self.perturbation_count += 1

        new_scores = [score(self.cities, x) for x in new_solutions]
        best_idx = np.argmin(new_scores)
        best = new_solutions[best_idx]
        
        return time.time() - start, best, self.perturbation_count
    
    def recombine(self,parents):
        pass

score_results = []
time_results = []
perturbation_results = []

#wersja dla msls
"""
for file in ['kroa','krob']:
    coords = pd.read_csv(file, sep=' ')
    positions=np.array([coords['x'], coords['y']]).T
    cities = np.round(pairwise_distances(np.array(positions)))
    variants = [MSLS(cities)]
    #variants = [MSLS(cities),ILS1(cities),ILS2(cities)]
    for solve in [random_cycle]:
        solutions = list(map(solve, [(cities, i) for i in range(10)]))
        scores = [score(cities, x) for x in solutions]
        score_results.append(dict(file=file, function=solve.__name__, search="none", min=int(min(scores)), mean=int(np.mean(scores)), max=int(max(scores))))
        best_idx = np.argmin(scores)
        best = solutions[best_idx]
        plot_optimized_tours(positions, *best, f'dataset - {file}, cycle - {solve.__name__}')
        for variant in variants:
            times, new_solutions = zip(*list(map(variant, solutions)))
            new_scores = [score(cities, x) for x in new_solutions]
            best = new_solutions[best_idx]
            plot_optimized_tours(positions, *best, f'dataset - {file}, cycle - {solve.__name__}, method - {(type(variant).__name__).lower()}')
            score_results.append(dict(file=file, function=solve.__name__, search=type(variant).__name__, min=int(min(new_scores)), mean=int(np.mean(new_scores)), max=int(max(new_scores))))
            time_results.append(dict(file=file, function=solve.__name__, search=type(variant).__name__, min=float(min(times)), mean=float(np.mean(times)), max=float(max(times))))
scores = pd.DataFrame(score_results)
times = pd.DataFrame(time_results)
print(scores)
print(times)
scores.to_csv('csv_scores.csv', index=False)  
times.to_csv('csv_times.csv', index=False) 
 """
#wersja dla ILS1
for file in ['kroa','krob']:
    coords = pd.read_csv(file, sep=' ')
    positions=np.array([coords['x'], coords['y']]).T
    cities = np.round(pairwise_distances(np.array(positions)))
    variants = [ILS1(cities,file),ILS2_local_search(cities,file),ILS2(cities,file)]
    #variants = [MSLS(cities),ILS1(cities),ILS2(cities)]
    for solve in [random_cycle]:
        solutions = list(map(solve, [(cities, i) for i in range(10)]))
        scores = [score(cities, x) for x in solutions]
        score_results.append(dict(file=file, function=solve.__name__, search="none", min=int(min(scores)), mean=int(np.mean(scores)), max=int(max(scores))))
        best_idx = np.argmin(scores)
        best = solutions[best_idx]
        plot_optimized_tours(positions, *best, f'dataset - {file}, cycle - {solve.__name__}')
        for variant in variants:
            times, new_solutions, perturbation = zip(*list(map(variant, solutions)))
            new_scores = [score(cities, x) for x in new_solutions]
            best = new_solutions[best_idx]
            plot_optimized_tours(positions, *best, f'dataset - {file}, cycle - {solve.__name__}, method - {(type(variant).__name__).lower()}')
            score_results.append(dict(file=file, function=solve.__name__, search=type(variant).__name__, min=int(min(new_scores)), mean=int(np.mean(new_scores)), max=int(max(new_scores))))
            time_results.append(dict(file=file, function=solve.__name__, search=type(variant).__name__, min=float(min(times)), mean=float(np.mean(times)), max=float(max(times))))
            perturbation_results.append(dict(file=file, function=solve.__name__, search=type(variant).__name__, min=float(min(perturbation)), mean=float(np.mean(perturbation)), max=float(max(perturbation))))
scores = pd.DataFrame(score_results)
perturbations = pd.DataFrame(perturbation_results)
times = pd.DataFrame(time_results)
print(scores)
print(times)
print(perturbations)
scores.to_csv('csv_scores.csv', index=False)  
times.to_csv('csv_times.csv', index=False) 
perturbations.to_csv('csv_perturbations.csv', index=False) 