# Od zera do inzyniera - materialy do nauki AI Engineering

To repozytorium to **materialy towarzyszace** do dokumentu / ebooka  
**„Od zera do inzyniera”**.

Repo NIE jest pojedynczym projektem.
To jest **zestaw taskow, przykladow i pelnych rozwiazan**, ktore prowadza krok po kroku od totalnych podstaw do poziomu **mid / mid+ AI Engineer**.

Glowne zalozenie:
- najpierw **rozumiesz**
- potem **robisz**
- na koncu **widzisz jak to wyglada w wersji produkcyjnej**

---

## Jak to sie laczy z Docsem / ebookiem

Glowne tlumaczenia, narracja i teoria sa w dokumencie:

https://docs.google.com/document/d/1FMbR2LuNaM9ti3eofmtqUBzzqtZumdM8v_RZSuERW_I

Ten dokument:
- tlumaczy pojecia od zera
- prowadzi przez pipeline (dane -> model -> inference -> monitoring)
- mowi **dlaczego cos robimy**, a nie tylko „jak”

To repo:
- daje **kod**
- daje **taski**
- daje **pelne odpowiedzi i referencje**

Czytaj to razem. Jedno bez drugiego traci sens.

---

## Struktura repo
```od-zera-do-inzyniera/
├── computer-vision/
│ ├── README.md
│ ├── src/
│ ├── scripts/
│ └── data/
│
├── odpowiedzi/
│ └── cv-yolo-prod/
│ ├── README.md
│ ├── src/
│ ├── scripts/
│ └── out/
│
├── .gitignore
└── README.md <-- (ten plik)
```

### computer-vision/
To sa **taski edukacyjne**.

Cel:
- zrozumiec jak CV dziala w praktyce
- zobaczyc pipeline end-to-end
- dotknac problemow typu threshold, false positives, domain shift

Tu **nie wszystko jest „produkcyjne”**.
To sa zadania, ktore masz:
- uruchomic
- zmodyfikowac
- zepsuc i naprawic

---

### odpowiedzi/
Tu sa **pelne odpowiedzi referencyjne**.

Cel:
- pokazac jak ten sam problem wyglada na poziomie **mid**
- pokazac strukture kodu, ktora da sie obronic na rozmowie
- pokazac monitoring mindset, a nie tylko „model dziala”

Jesli:
- utkniesz
- chcesz porownac swoje rozwiazanie
- chcesz zobaczyc jak to powinno wygladac „na serio”

→ zagladasz tutaj.

---

## Jak z tego korzystac (bardzo wazne)

Rekomendowany flow:

1. Czytasz rozdzial w Docsie
2. Wchodzisz do odpowiedniego folderu w repo
3. Robisz task **samodzielnie**
4. Dopiero potem zagladasz do `odpowiedzi/`
5. Porownujesz:
   - strukture
   - decyzje
   - trade-offy

Jesli tylko kopiujesz kod:
- oszukujesz sama siebie
- i nic z tego nie zostanie

---

## Poziom i target

Ten material NIE jest:
- szybkim kursem
- „learn AI in 7 days”
- akademickim wywodem

To jest:
- material pod realna prace
- dla osob, ktore chca rozumiec co sie dzieje w produkcji
- i nie chca robic czegos pokroju „modele odpalam z tutoriala”

Docelowy poziom:
- **mid / mid+ AI Engineer**
- z podstawami DevOps i MLOps

---

## Co dalej

To repo bedzie roslo razem z dokumentem:
- NLP / LLM
- RAG
- serving (FastAPI)
- monitoring
- deployment (Docker, cloud)

Jesli cos jest niejasne - to znaczy, ze **jeszcze tam dojdziemy**.

GL;HF

yours truly
n


