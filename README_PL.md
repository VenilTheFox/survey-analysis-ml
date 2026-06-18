[Wersja angielska (English version)](./README.md)
# survey-analysis-ml
Analiza i modelowanie predykcyjne zdrowia psychicznego w branży technologicznej z wykorzystaniem regresji logistycznej, SVM oraz sieci neuronowej MLP.

Projekt analizuje dane ankietowe od pracowników branży technologicznej dotyczące środowiska pracy, wsparcia w zakresie zdrowia psychicznego, benefitów oraz historii leczenia. Celem jest predykcja, czy dana osoba skorzystała z leczenia zdrowia psychicznego na podstawie zgłoszonych czynników.

---

## Zbiór danych
W projekcie wykorzystano zbiór danych Mental Health in Tech Survey (2014) dostępny na Kaggle:
https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey

---

## Struktura projektu i workflow

### 1. Czyszczenie danych i inżynieria cech
* Uzupełniono brakujące wartości na podstawie założeń wynikających z kontekstu danych (np. `self_employed`, `work_interfere`).
* Ujednolicono i uproszczono niespójne wartości tekstowe w kolumnie `Gender`, redukując je do trzech klas: `male`, `female`, `other`.
* Usunięto błędne i skrajne wartości w kolumnie `Age`, aby zapewnić poprawność danych.
* Zakodowano zmienne kategoryczne:
  * Kodowanie porządkowe (ordinal encoding) dla zmiennych o naturalnym porządku (np. `work_interfere`, `leave`, `no_employees`)
  * Kodowanie one-hot dla zmiennych nominalnych z użyciem `pd.get_dummies`
* Przeskalowano zmienne numeryczne przy użyciu `MinMaxScaler` dopasowanego wyłącznie do danych treningowych, aby uniknąć wycieku danych.

### 2. Analiza danych (EDA)
* Przeanalizowano zależności między czynnikami związanymi z pracą a korzystaniem z leczenia zdrowia psychicznego przy użyciu wykresów liczności (count plots).
* Zbadano wpływ czynników organizacyjnych (np. anonimowość, wsparcie przełożonych, benefity) na decyzję o podjęciu leczenia.

### 3. Selekcja cech (RFECV)
* Zastosowano metodę Recursive Feature Elimination with Cross-Validation (RFECV) z regresją logistyczną jako modelem bazowym.
* Optymalizacja zestawu cech została przeprowadzona na podstawie F1-score w 5-krotnej walidacji krzyżowej.
* Usunięto cechy mało istotne lub redundantne, redukując wymiarowość danych.


### 4. Budowa modeli i walidacja krzyżowa
Trzy modele zostały wytrenowane i porównane:

* Regresja logistyczna (RFECV): Model bazowy o wysokiej interpretowalności, trenowany na wyselekcjonowanych cechach.
* Support Vector Machine (SVM, kernel RBF): Model nieliniowy wykorzystujący kernel RBF do odwzorowania złożonych zależności w danych.
* Multilayer Perceptron (MLP): Sieć neuronowa z dwiema warstwami ukrytymi (10, 5), funkcją aktywacji `tanh` oraz optymalizatorem `adam`, modelująca nieliniowe zależności.

Wszystkie modele oceniano przy użyciu stratifikowanej 5-krotnej walidacji krzyżowej na zbiorze treningowym.


### 5. Ewaluacja modeli
* Ostateczna ocena została przeprowadzona na niezależnym zbiorze testowym.
* Zastosowane metryki: Accuracy, Precision, Recall, F1-score.
* Wykorzystano macierze pomyłek (confusion matrix) do analizy błędów klasyfikacji.

---

## Podsumowanie wyników

| Model | Accuracy | Precision | Recall | F1-score |
|------|----------|-----------|--------|----------|
| **Regresja logistyczna (RFECV)** | 0.797 | 0.797 | 0.803 | 0.800 |
| **SVM (RBF kernel)** | 0.825 | 0.817 | 0.843 | 0.829 |
| **MLP (Multilayer Perceptron)** | 0.845 | 0.828 | 0.874 | 0.851 |

---

## Kluczowe wnioski
1. Wszystkie modele osiągnęły stosunkowo dobre wyniki na danych ankietowych.
2. Model MLP uzyskał najlepsze wyniki, jednak różnice między modelami nie były duże.
3. SVM zapewnił dobry kompromis między skutecznością a generalizacją.
4. Regresja logistyczna oferuje najlepszą interpretowalność przy konkurencyjnych wynikach.

---

## Technologie
* **Język:** Python 3.x
* **Analiza danych:** `pandas`, `numpy`
* **Wizualizacja:** `matplotlib`, `seaborn`
* **Uczenie maszynowe:** `scikit-learn`
