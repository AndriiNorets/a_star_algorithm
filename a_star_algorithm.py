import matplotlib.pyplot as plt
import numpy as np
import heapq

# Funkcja dla heurystyki (odległość Manhattan)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Algorytm A* z wizualizacją
def astar(graph, start, goal, visualize=False):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    if visualize:
        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(-0.5, len(graph[0]), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(graph), 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)

        # Rysujemy siatkę
        for row in range(len(graph)):
            for col in range(len(graph[0])):
                if graph[row][col] == 1:  # Zajęte punkty (czarne)
                    ax.plot(col, row, 's', color='black', ms=20)
                else:  # Wolne punkty (niebieskie)
                    ax.plot(col, row, 's', color='lightblue', ms=20)

        plt.ion()

    while open_list:
        current = heapq.heappop(open_list)[1]

        if visualize:
            # Aktualny punkt - rysujemy go jako zielony (obecnie badany)
            ax.plot(current[1], current[0], 'o', color='green', ms=15)  
            plt.draw()
            plt.pause(0.5)  # Opóźnienie dla wizualizacji

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            if visualize:
                # Rysujemy ścieżkę (czerwona)
                for node in path:
                    ax.plot(node[1], node[0], 'o', color='red',ms=10)  
                    plt.draw()
                    plt.pause(0.1)
                plt.ioff()
                plt.show()

            return path

        for neighbor in get_neighbors(graph, current):
            tentative_g_score = g_score[current] + 1

            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                
                if neighbor not in [i[1] for i in open_list]:
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    if visualize:
        plt.ioff()
        plt.show()
    return None  # Ścieżka nie została znaleziona

# Funkcja, która zwraca sąsiadów danej komórki
def get_neighbors(graph, node):
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Góra, dół, lewo, prawo
    for direction in directions:
        neighbor = (node[0] + direction[0], node[1] + direction[1])
        if 0 <= neighbor[0] < len(graph) and 0 <= neighbor[1] < len(graph[0]) and graph[neighbor[0]][neighbor[1]] != 1:
            neighbors.append(neighbor)
    return neighbors

if __name__ == "__main__":
    # Definicja siatki (0 - wolne, 1 - zajęte)
    graph = [
        [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0]
    ]


    start_x = int(input("Wpisz X dla 'start' point: "))
    start_y = int(input("Wpisz Y dla 'start' point: "))
    start = (start_x, start_y)

    end_x = int(input("Wpisz X dla 'end' point: "))
    end_y = int(input("Wpisz Y dla 'end' point: "))
    end = (end_x, end_y)

    path = astar(graph, start, end, visualize=True)

    if path:
        print("Znaleziono ścieżkę:", path)
    else:
        print("Ścieżka nie została znaleziona")
