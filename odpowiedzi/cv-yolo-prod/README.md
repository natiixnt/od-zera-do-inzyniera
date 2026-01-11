# cv-yolo-prod (odpowiedzi) - mid level, dziala i da sie debugowac

Masz tu dzialajacy pipeline na prawdziwym modelu (YOLO z ultralytics), plus outputy tak zrobione, zeby:
- dalo sie to zautomatyzowac (batch po folderze)
- dalo sie to logowac/monitorowac (JSON + CSV z metrykami)
- dalo sie debugowac (PNG z boxami)

Jesli to brzmi jak "za duzo", to spokojnie - ten projekt to miniatura tego, co potem robisz w pracy XD

---

## Co ten program robi (high level)

Mamy dwa entrypointy:
- scripts/run_on_image.py - inferencja na jednym obrazie
- scripts/run_on_folder.py - inferencja na calym folderze obrazow + CSV z metrykami

Kazdy run robi:
1) wczytuje obraz (cv2.imread)
2) odpala YOLO (ultralytics) i dostaje detekcje (boxy + confidence + class)
3) rysuje boxy na obrazie (debug wizualny)
4) zapisuje wynik jako PNG
5) zapisuje detale jako JSON (zeby nie zgadywac co wykryl)
6) mierzy latency (ile trwa inference)

W wersji folderowej dodatkowo:
- liczy proste metryki per obraz (detections_count, avg_conf, latency_ms)
- zapisuje metrics.csv, zeby miec szybki przeglad

---

## Struktura folderow

- src/cv_yolo/config.py
  Settings z envow (model, prog confidence, out_dir)

- src/cv_yolo/predict.py
  Wrapper na YOLO. Zwraca liste Detection:
  - label (nazwa klasy)
  - confidence (pewnosc)
  - box_xyxy (x1,y1,x2,y2)

- src/cv_yolo/visualize.py
  Rysowanie boxow i opisow na obrazie

- scripts/run_on_image.py
  CLI: jeden obraz -> out png + out json + latency

- scripts/run_on_folder.py
  CLI: folder obrazow -> out/batch/ png + json + metrics.csv

---

## Jak uruchamiac (Windows PowerShell)

W tym projekcie importy ida z src/, wiec najprosciej ustawic PYTHONPATH.

W folderze odpowiedzi/cv-yolo-prod:

1) aktywuj venv (jak juz masz, pomin)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

2) instalacja
python -m pip install --upgrade pip setuptools wheel
pip install --only-binary=:all: -r requirements.txt

3) ustaw PYTHONPATH (zeby python widzial src/)
C:\Users\User\Desktop\praca\od-zera-do-inzyniera\odpowiedzi\cv-yolo-prod\src = (Resolve-Path ".\src").Path

4) run na jednym obrazie
python .\scripts\run_on_image.py --image ..\..\computer-vision\data\images\sample_01.jpg --out .\out\result.png

5) batch na folderze
python .\scripts\run_on_folder.py --dir ..\..\computer-vision\data\images --conf 0.5

---

## Parametry ktore maja znaczenie

--conf
To jest prog confidence. To nie jest "prawda", to jest pokretlo.

- nizszy prog -> wiecej wykryc, wiecej smieci (false positives)
- wyzszy prog -> mniej wykryc, wiecej pominiec (false negatives)

Dobierasz to pod koszt bledu:
- alarmy / auto akcje -> prog w gore
- "lepiej wykryc za duzo niz za malo" -> prog w dol i potem walidacja

--model
Domyslnie yolov8n.pt (maly, szybki).
Mozesz podmienic na np. yolov8s.pt (wiekszy, wolniejszy, czesto lepszy).

---

## Co jest w JSON (i po co to w ogole)

PNG z boxami jest fajny do oka, ale:
- nie da sie tego latwo analizowac programowo
- nie da sie porownac runow po zmianie modelu
- nie da sie zrobic monitoringu

Dlatego zapisujemy JSON:
- detections: lista obiektow (label, confidence, box)
- stats: detections_count, latency_ms (i w batch avg_conf)

Przyklad:
- jak nagle detections_count spada do 0 na obrazach gdzie zawsze bylo 5, to jest sygnal problemu
- jak avg_conf spada o 30 procent, to moze byc gorsze swiatlo, brudna kamera, inny kadr (drift)

---

## Co jest w metrics.csv (mid-level monitoring)

W batch runie tworzysz metrics.csv z kolumnami:
- image
- detections_count
- avg_conf
- latency_ms

To jest prosty monitoring "bez etykiet".
Na prodzie czesto nie masz ground truth, wiec nie policzysz accuracy.
Patrzysz na symptomy:
- trend liczby detekcji
- trend confidence
- latency (czy serwer zaczyna mulic)
- rozklad klas (tu jeszcze tego nie liczymy, ale mozna)

---

## Debug checklist (jak cos nie gra)

1) detekcje = 0 i myslisz ze powinno cos byc
- sprawdz czy obraz sie wczytal (sciezka)
- zmniejsz prog conf: --conf 0.25
- sprawdz na innym obrazie (np. z czlowiekiem / autem)
- zobacz JSON, czy sa detekcje ale wyciete progiem

2) dziwne boxy / smieci w tle
- to normalne, model jest statystyczny
- podnies prog conf
- w prawdziwym systemie dorzucasz reguly (np. ignoruj bardzo male boxy)

3) latency nagle rosnie
- CPU vs GPU
- obciazenie systemu
- zbyt duzy model

4) import error: No module named cv_yolo
- nie ustawilas PYTHONPATH na src/
  C:\Users\User\Desktop\praca\od-zera-do-inzyniera\odpowiedzi\cv-yolo-prod\src = (Resolve-Path ".\src").Path


