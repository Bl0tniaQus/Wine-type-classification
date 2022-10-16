#importowanie modułów potrzebnych do niektórych
#operacji matematycznych i wykresów
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random

#funkcja inicjalizująca wagi LW między warstwą ukrytą a wyjściową
#stosuje uproszczony, niemacierzowy model
#każdemu neuronowi z warstwy ukrytej przyporządkowuje liczbę odpowiadającą konkretnej klasie
#odwzorowuje proporcje między wystąpieniami poszczególnych klas

def LW_init(T, n): # T - zbiór etykiet, n - liczba neuronów warstwy srodkowej
    m = len(T)  #liczba rekordów danych
    klasy = set(T) #zbiór klas
    wystapienia = [] #lista zawierająca ilość wystąpień danej klasy
    res = [] #lista wynikowa
    wystapieniaNorm = []#lista zawierająca zadaną ilość wystąpień danej klasy w wektorze wagowym
    for i in klasy:
        wystapienia.append([T.count(i), i]) #zliczanie wystąpień klas
    wystapienia = sorted(wystapienia)
    for j in range(len(wystapienia)):
        wystapieniaNorm.append([round((1 / m) * wystapienia[j][0] * n), wystapienia[j][1]]) #"normalizowanie" wystąpień
    for k in range(len(wystapieniaNorm)):
        for i in range(wystapieniaNorm[k][0]):
            res.append(wystapieniaNorm[k][1]) #tworzenie wektora wagowego
    #zagwarantowanie wypełnienia całej długości wektora
    if len(res) < n:
        for i in range(n - len(res)):
            res.append(wystapieniaNorm[len(wystapieniaNorm) - 1][1])
    if len(res) > n:
        res = res[0:n]
    return res
#funkcja losująca 80% unikatowych rekordów danych
def wybierzUczace(dane):
    return random.sample(range(len(dane)),int(0.8*len(dane)))
#funkcja wybierająca rekordy do testowania, wszystkie, które nie zostały wybrane jako uczące
def wybierzTestujace(dane_ind, dane):
    test_ind = []
    for i in range(len(dane)):
        if i not in dane_ind:
            test_ind.append(i)
    return test_ind

#klasa obsługująca pobranie danych z pliku i utworzenie z nich odpowiednich zbiorów
class Dane:
    def __init__(self, nazwa):
        self.plik = open(nazwa, "r")
        self.lines = [i.replace(',', ' ').replace('\n', '') for i in self.plik.readlines()]
        self.values = [i.split(' ') for i in self.lines]
        for i in range(0, len(self.values)):
            for j in range(0, len(self.values[1])):
                #if self.values[i][j].isnumeric():
                    self.values[i][j] = float(self.values[i][j])  
        self.T = [i[0] for i in self.values]
        self.Ts = sorted(self.T)
        self.P = [i[1:14] for i in self.values]
        self.Pt = np.transpose(self.P)
        self.Ps = sorted(self.P)
        self.Pn = []
        for i in self.Pt:
            vec = []
            minp = min(i)
            maxp = max(i)
            for x in i:
                xnorm = (1 - (-1)) * (x - minp) / (maxp - minp) + (-1) #normalizacja do przedziału <-1,1>
                vec.append(xnorm)
            self.Pn.append(vec)
        self.Pn = np.transpose(self.Pn)

#utworzenie obiektu z danymi
dane = Dane("wine.txt")

S1 = [7,10,15,20,30,40,50,60,70,80,90,100,110,120,130,150,200]  #liczba neuronów warstwy ukrytej
LR = [0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]#współczynnik uczenia
C = len(set(dane.T))  # liczba neuronów wyjściowych / liczba klas
n_cech = len(dane.Pt) #liczba cech
licznik = 0 #licznik wykonanych pętli
epoki = 15 #liczba epok
Q = [] #macierz zawierający PK poszczególnych sieci

for ind_S1 in range(len(S1)): #pętla iterująca po różnych ilościach neuronów
    Qv = [] #wiersz macierzy PK
    for ind_lr in range(len(LR)): #pętla iterująca po różnych współczynnikach uczenia
        uczace = wybierzUczace(dane.Pn) #wybór danych uczących
        testujace = wybierzTestujace(uczace,dane.Pn) #wybór daynch testujących
        #IW = [np.random.randint(-100, 100, n_cech) / 10000 for i in range(S1[ind_S1])]
        IW = [np.zeros(n_cech) / 1000 for i in range(S1[ind_S1])] #inicjacja wag IW
        LW = LW_init(dane.T, S1[ind_S1]) #inicjacja wag LW
        lr = LR[ind_lr] #wybór współczynnika uczenia
        poprawne = 0 #ilość poprawnych klasyfikacji
        for t in range(epoki): #pętla epok
            for rekord in uczace: #pętla wybierająca kolejne rekordy uczące
                ED_vec = [] #wektor odległości wektora od poszczególnych neuronów
                ED = 0
                T = dane.T[rekord] #poprawna etykieta wektora
                for c in range(S1[ind_S1]): #pętla obliczająca odległośi wektora od neuronów
                    ED = 0
                    for cecha in range(n_cech):
                        ED = ED + (dane.Pn[rekord][cecha] - IW[c][cecha]) ** 2
                    ED = math.sqrt(ED)
                    ED_vec.append(ED)
                idmin = ED_vec.index(min(ED_vec)) #wybór minimalnej odległości
                a = LW[idmin] #wyznaczona etykieta dla rekordu danych
                if a != T:
                    LW[idmin] = T #"przepięcie" neuronu na inną etykietę
                znak = 1
                if a == T:
                    znak = -1

                for i in range(len(IW[idmin])): #pętla aktualizacji wag IW dla zwycięskiego neuronu
                    IW[idmin][i] = IW[idmin][i] - (lr/((t+1)**2) * (dane.Pn[rekord][i] - IW[idmin][i])) * znak
                    #IW[idmin][i] = IW[idmin][i] - (lr * (dane.Pn[rekord][i] - IW[idmin][i])) * znak

        for d in testujace: #pętla testująca PK
            T = dane.T[d]
            ED_vec = []
            for c in range(S1[ind_S1]):
                ED = 0
                for cecha in range(n_cech):
                    ED = ED + (dane.Pn[d][cecha] - IW[c][cecha]) ** 2
                ED = math.sqrt(ED)
                ED_vec.append(ED)
            idmin = ED_vec.index(min(ED_vec))
            a = LW[idmin]
            if a == T: #sprawdzenie słusznosci wyboru
                poprawne = poprawne + 1
        licznik = licznik + 1
        q = (poprawne / len(testujace)) * 100 #wyznaczenie PK dla sieci
        Qv.append(q)
        postep = round(((licznik / (len(S1) * len(LR))) * 100), 2)
        print(str(postep) +"% "+str(q))
    Q.append(Qv)

#część wykresowa programu, znacznie modyfikowana między eksperymentami
X, Y = np.meshgrid(S1,LR)
Q = np.array(Q)

ax = plt.axes(projection="3d")

ax.plot_surface(X, Y, np.transpose(Q), cmap="plasma")
ax.set_xlabel('S1', fontsize=15)
ax.set_ylabel('LR', fontsize=15)
ax.set_zlabel('PK [%]', fontsize=15)
plt.title("PK(S1,LR), dla " + str(epoki) + " epok")
plt.show()