import matplotlib.pyplot as plt
import numpy as np
import heapq
import math

def wczytaj_siatke_z_pliku(nazwa_pliku):
    with open(nazwa_pliku, 'r') as plik:
        siatka = []
        for linia in plik:
            siatka.append([int(x) for x in linia.split()])
    return siatka

def heurystyka(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def znajdz_sasiadow(siatka, punkt):
    sasiedzi = []
    kierunki = [
        ("dół", (1, 0)),
        ("lewo", (0, -1)),
        ("góra", (-1, 0)),
        ("prawo", (0, 1))
    ]

    # Dostępne sąsiedzi
    for kierunek, przesuniecie in kierunki:
        sasiad = (punkt[0] + przesuniecie[0], punkt[1] + przesuniecie[1])
        if 0 <= sasiad[0] < len(siatka) and 0 <= sasiad[1] < len(siatka[0]) and siatka[sasiad[0]][sasiad[1]] != 5:
            sasiedzi.append((kierunek, sasiad))
    return sasiedzi

# A*
def astar(siatka, start, cel, wizualizacja=False):
    if siatka[start[0]][start[1]] == 5:
        print(f"Błąd: Punkt startowy {start} jest komórką zajętą.")
        return None
    if siatka[cel[0]][cel[1]] == 5:
        print(f"Błąd: Punkt końcowy {cel} jest komórką zajętą.")
        return None
    
    # Ustalamy listę punktów do odwiedzenia
    otwarta_lista = []
    heapq.heappush(otwarta_lista, (0, start))
    poprzednicy = {}
    g_score = {start: 0}
    f_score = {start: heurystyka(start, cel)}

    #Wizualizacja siatki
    if wizualizacja:
        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(-0.5, len(siatka[0]), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(siatka), 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)

        for rzad in range(len(siatka)):
            for kolumna in range(len(siatka[0])):
                if siatka[rzad][kolumna] == 5:
                    ax.plot(kolumna, rzad, 's', color='black', ms=20)
                else:
                    ax.plot(kolumna, rzad, 's', color='lightblue', ms=20)
        plt.ion()

    while otwarta_lista:
        # Bierzemy punkt o najmniejszym f_score
        obecny = heapq.heappop(otwarta_lista)[1]

        if wizualizacja:
            ax.plot(obecny[1], obecny[0], 'o', color='green', ms=15)
            plt.draw()
            plt.pause(0.1)

        if obecny == cel:
            sciezka = []
            while obecny in poprzednicy:
                sciezka.append(obecny)
                obecny = poprzednicy[obecny]
            sciezka.append(start)
            sciezka.reverse()

            if wizualizacja:
                for punkt in sciezka:
                    ax.plot(punkt[1], punkt[0], 'o', color='red', ms=10)
                    plt.draw()
                    plt.pause(0.001)
                plt.ioff()
                plt.show()

            return sciezka
        
        #Opracowanie sąsiadów obecnego punktu
        for kierunek, sasiad in znajdz_sasiadow(siatka, obecny):
            # Aktualny kierunek
            print(f"Sprawdzam kierunek: {kierunek}")  
            koszt_tymczasowy = g_score[obecny] + 1

            if koszt_tymczasowy < g_score.get(sasiad, float('inf')):
                poprzednicy[sasiad] = obecny
                g_score[sasiad] = koszt_tymczasowy
                f_score[sasiad] = koszt_tymczasowy + heurystyka(sasiad, cel)
                if sasiad not in [i[1] for i in otwarta_lista]:
                    heapq.heappush(otwarta_lista, (f_score[sasiad], sasiad))

    if wizualizacja:
        plt.ioff()
        plt.show()
    return None

if __name__ == "__main__":
    siatka = wczytaj_siatke_z_pliku("grid.txt")

    start = (0, 0)
    cel = (19, 19)

    sciezka = astar(siatka, start, cel, wizualizacja=True)

    if sciezka:
        print("Znaleziono ścieżkę:")
        print(sciezka)
    else:
        print("Nie znaleziono ścieżki.")

