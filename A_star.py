# This file has been written such that it serves as a API call for performing A* search
from Predictor import Predictor as pred
import concurrent.futures as cf

def cluster_task(x, y, cell=None):
            if not cell:
                cell = pred()
            cell.recluster()
            return cell, x, y

class A_star:
    def __init__(self):
        self.pred_mat = [[None] * 29 for _ in range(29)]

        with cf.ProcessPoolExecutor() as executor:
            fs = []
            for i in range(len(self.pred_mat)):
                for j in range(len(self.pred_mat[i])):
                    fs.append(executor.submit(cluster_task, i, j)) if i != j else None

            prog = 0
            print('Initiation progress: 0%', end='\r')
            for f in cf.as_completed(fs):
                res, i, j = f.result()
                self.pred_mat[i][j] = res
                prog += 100
                if j == 0:
                    print(f'Initiation progress: {prog // len(fs)}%', end='\r')
            print('Initiation progress: 100%')

    def a_star(self, graph, origin, destination, graph_neighbor_coordinates_keys_to_index, TIME):
        open_list = [origin]

        g_score = {node: float('inf') for node in graph}
        g_score[origin] = 0
        parent = {origin: None}
        f_score = {node: float('inf') for node in graph}
        f_score[origin] = self.pred_mat[graph_neighbor_coordinates_keys_to_index[origin]][graph_neighbor_coordinates_keys_to_index[destination]].heuristic(TIME)

        while open_list:
            current = min(open_list, key=lambda node: f_score[node])
            open_list.remove(current)

            if current == destination:
                path = []
                while current:
                    path.append(current)
                    current = parent[current]
                return path[::-1]   # Returns a list of the names of nodes
            
            for neighbor, cost in graph[current].items():
                tentative_g = g_score[current] + cost

                if tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.pred_mat[graph_neighbor_coordinates_keys_to_index[neighbor]][graph_neighbor_coordinates_keys_to_index[destination]].heuristic(TIME) if neighbor != destination else 0
                    parent[neighbor] = current

                    if neighbor not in open_list:
                        open_list.append(neighbor)

        return 'Path NotFoundError\n'
    
    # Use to recluster data
    def update_pred(self):
        with cf.ProcessPoolExecutor() as executor:
            fs = []
            for i in len(range(self.pred_mat)):
                for j in len(range(self.pred_mat[i])):
                    fs.append(executor.submit(cluster_task, i, j, self.pred_mat[i][j]))

            prog = 0
            print('Updation progress: 0%', end='\r')
            for f in cf.as_completed(fs):
                res, i, j = f.result()
                self.pred_mat[i][j] = res
                prog += 100
                if j == 0:
                    print(f'Updation progress: {prog // len(fs)}%', end='\r')
                
    # Get path costs at a given time
    def cost_at_time(self, u, v, TIME):
        if u == v:
            return 0
        else:
            return int(self.pred_mat[u][v].heuristic(TIME)) if self.pred_mat[u][v] else 9999