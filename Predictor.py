from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from matplotlib import pyplot as plt

n_pts = 10
noise = 0.5
len_x = 1440
min_y = 5
max_y = 20

class Predictor:
    def __init__(self, data: np.ndarray = None, max_k: int = 15, k_iter: int = 20, bias: float = 0.25):
        self.data: np.ndarray = data if data else self.generate_data()
        self.max_k = max_k + 1
        self.k_iter = k_iter
        self.bias = bias

        self.clusters = {}
        self.pred = []

    def generate_data(self):
        X = []
        freq = 1 + np.random.rand() * 3
        shift = np.random.randint(0, n_pts) 
        for i in range(n_pts):
            v = (shift + i) * 2 * np.pi * freq / n_pts
            X.append((
                ((i / 100) % (2 * np.pi)) * len_x,
                (np.sin(v) + 1 + np.random.rand() * noise) * (max_y - min_y) + min_y
            ))

        X.sort(key=lambda x: x[0])
        return np.array(X)

    # Clustering functions
    def __distance(self, p1, p2, bias: float = 1):
        t = p1.copy() - p2.copy()

        if bias != 1:
            t *= np.array((bias, len_x))
        else:
            if abs(t[0]) > len_x * 0.5:
                t[0] = len_x - abs(t[0])
            t *= np.array((1, 0.5 * len_x / abs(max_y - min_y)))

        return np.sqrt(np.sum((t)**2))

    def __gen_clusters(self, k: int):
        self.clusters = {}

        for i in range(k):
            centre = self.data[i * self.data.shape[0] // k]
            cluster = {
            'centre' : centre,
            'points' : []
            }
            self.clusters[i] = cluster

    def __assign_clusters(self):
        k = len(self.clusters)

        for x in self.data:
            dist = []

            for j in range(k):
                dis = self.__distance(x, self.clusters[j]['centre'])
                dist.append(dis)
            
            self.clusters[np.argmin(dist)]['points'].append(x)

    def __update_clusters(self):
        k = len(self.clusters)

        for i in range(k):
            points = np.array(self.clusters[i]['points'])
            if points.shape[0] > 0:
                new_centre = points.mean(axis=0)
                self.clusters[i]['centre'] = new_centre
                self.clusters[i]['points'] = []

    def __pred_cluster(self):
        
        k = len(self.clusters)
        pred = []


        for x in self.data:
            dist = []
            for j in range(k):
                dist.append(self.__distance(x, self.clusters[j]['centre']))
            self.clusters[np.argmin(dist)]['points'].append(x)
            pred.append(np.argmin(dist))

        return pred

    def __silhouette(self):
        scores = []
        for i in self.clusters:
            agg_scr = 0
            neighbours = []

            # Skip empty clusters
            if len(self.clusters[i]['points']) == 0:
                continue

            for x in self.clusters[i]['points']:
                in_dist = 0
                out_dist = 0
                for j in self.clusters:
                    if len(self.clusters[j]['points']) == 0:
                        continue

                    dist = self.__distance(x, self.clusters[j]['centre'])
                    if i != j:
                        neighbours.append(dist)
                    else:
                        in_dist = dist

            out_dist = np.min(neighbours)
            scr = (out_dist - in_dist) / np.max((in_dist, out_dist))
            agg_scr += scr

            scores.append(agg_scr / len(self.clusters[i]['points']))
        return scores

    def recluster(self, show: bool=False):
        agg_scr = []
        for k in range(4, self.max_k):
            self.__gen_clusters(k)
            for i in range(self.k_iter):
                self.__assign_clusters()
                self.__update_clusters()
            self.pred = self.__pred_cluster()
            scores = self.__silhouette()
            agg_scr.append((k, np.mean(scores)))

        agg_scr.sort(key=lambda x: x[1])
        self.__gen_clusters(agg_scr.pop()[0])
        # for i in range(len(self.clusters)):
            # print(self.clusters[i]['centre'])
        for _ in range(self.k_iter):
            self.__assign_clusters()
            self.__update_clusters()
            # print('\n\n')
            # for i in range(len(self.clusters)):
                # print(self.clusters[i]['centre'])
        self.pred = self.__pred_cluster()

        if show:
            self.show_clusters()

    def show_clusters(self):
        plt.scatter(self.data[:, 0], self.data[:, 1], c = self.pred)
        for i in self.clusters:
            centre = self.clusters[i]['centre']
            plt.scatter(centre[0], centre[1], marker = '^', c = 'red')
        plt.xlabel('Minutes from midnight')
        plt.ylabel('Travel time (in mins)')
        plt.show()


    # KNN-means and prediction functions
    def __gaussian_average(self, time: float):
        prod = 0
        total = 0
        for x in self.data:
            if time - 150 < x[0] < time + 150:
                wei = np.e ** (-0.5 * ((x[0] - time) * 125 / (len_x / 100)) ** 2)
                prod += wei * x[1]
                total += wei
        return prod / total if total > 0 else prod


    def __knn_m(self, time: float, new_y: float, show: bool):
        knn = KNeighborsClassifier(n_neighbors=self.data.shape[0] // 10)
        knn.fit(self.data, self.pred)
        new_point = np.array([(time, new_y)])

        prediction = knn.predict(new_point)
        fin_X = np.concatenate((self.data, new_point))
        self.pred.append(prediction[0])

        tot_dist = 0
        clust = self.clusters[prediction[0]]['points'].copy()
        for pt in clust:
            tot_dist += self.__distance(pt, new_point, self.bias)

        final_y = 0
        for pt in clust:
            final_y += pt[1] * self.__distance(pt, new_point, self.bias) / tot_dist

        if show:
            plt.scatter(fin_X[:, 0], fin_X[:, 1], c = self.pred)
            plt.scatter(time, new_y, c = 'red')
            plt.scatter(time, final_y, marker = '*', s = 250, c = 'white', edgecolors = 'black', linewidths=0.75)
            for i in self.clusters:
                centre = self.clusters[i]['centre']
                print(centre)
                plt.scatter(centre[0], centre[1], marker = '^', c = 'red')
            plt.xlabel('Minutes from midnight')
            plt.ylabel('Travel time (in mins)')
            plt.show()

        self.data = np.concatenate((self.data, np.array([(time, final_y)])))
        return final_y

    def heuristic(self, time: float, show: bool=False):
        trav_t = self.__gaussian_average(time)
        return self.__knn_m(time, trav_t, show)