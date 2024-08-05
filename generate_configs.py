import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    N = 20
    J = 1

    # cluster update steps
    T_therm = 1000

    def initialize():
        return 2 * np.random.randint(2, size=((N, N))) - np.ones((N, N))

    # Wolff cluster algorithm
    def cluster_update(configuration, T):
        magnetization = np.sum(configuration)
        size = 0
        visited = np.zeros((N, N))
        cluster = []
        i, j = np.random.randint(N, size=2)
        cluster.append((i, j))
        visited[i, j] = 1
        while len(cluster) > 0:
            i, j = cluster.pop()
            i_left = (i + 1) % N
            i_right = (i + N - 1) % N
            j_up = (j + 1) % N
            j_down = (N + j - 1) % N
            neighbors = [(i_left, j), (i_right, j), (i, j_up), (i, j_down)]
            for neighbor in neighbors:
                if visited[neighbor] == 0 and configuration[neighbor] == configuration[i,
                                                                                       j] and np.random.random() < (1 - np.exp(-2 * J / T)):
                    cluster.append(neighbor)
                    visited[neighbor] = 1
                    size += 1
            configuration[i, j] *= -1
        return size

    def cluster_update_switch(configuration, T):
        magnetization = np.sum(configuration)
        size = 0
        visited = np.zeros((N, N))
        cluster = []
        i, j = np.random.randint(N, size=2)
        cluster.append((i, j))
        visited[i, j] = 1
        while len(cluster) > 0:
            i, j = cluster.pop()
            i_left = (i + 1) % N
            i_right = (i + N - 1) % N
            j_up = (j + 1) % N
            j_down = (N + j - 1) % N
            neighbors = [(i_left, j), (i_right, j), (i, j_up), (i, j_down)]
            for neighbor in neighbors:
                if visited[neighbor] == 0 and configuration[neighbor] == configuration[i,
                                                                                       j] and np.random.random() < (1 - np.exp(-2 * J / T)):
                    cluster.append(neighbor)
                    visited[neighbor] = 1
                    size += 1
                    configuration[i, j], configuration[neighbor] = configuration[neighbor], configuration[i, j]
        return size

    def single_update(configuration, T):
        i, j = np.random.randint(N, size=2)
        i_left = (i + 1) % N
        i_right = (i + N - 1) % N
        j_up = (j + 1) % N
        j_down = (N + j - 1) % N
        neighbors = [(i_left, j), (i_right, j), (i, j_up), (i, j_down)]
        nb = np.random.randint(4, size=1)[0]

        ii, jj = neighbors[nb]
        if configuration[ii, jj] == configuration[i,
                                                  j] and np.random.random() < (1 - np.exp(-2 * J / T)):
            configuration[i, j] *= -1

    train_configs = []
    train_labels = []

    num_T = 100
    min_T = 0.05
    max_T = 5

    # how many configurations per temperature
    num_conf = 500

    T_c = 2.27
    Temps = np.linspace(min_T, max_T, num_T)

    for i, T in tqdm(enumerate(Temps)):
        configuration = initialize()
        csize = []
        for _ in range(T_therm):
            csize.append(cluster_update(configuration, T))
        T_A = int(N**2 / (2 * np.mean(csize))) * 2 + 1
        for i in range(num_conf * T_A):
            cluster_update(configuration, T)
            if i % T_A == 0:
                train_configs.append(np.reshape(configuration.copy(), N**2))
                train_labels.append(T)

    np.savetxt("labels_%ix%i.txt" % (N, N), train_labels, fmt='%.2e')
    np.savetxt("configs_%ix%i.txt" % (N, N), train_configs, fmt='%.2e')
