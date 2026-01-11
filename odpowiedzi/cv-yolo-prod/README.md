# CV YOLO prod (odpowiedzi)

To jest projekt pomocniczy do rozdzialu Computer Vision. Cel jest prosty: miec dzialajacy end to end pipeline na prawdziwym modelu, zeby nie gadac o CV w abstrakcji.

Co tu jest:
- YOLO (ultralytics) jako prawdziwy detektor
- skrypt inference na obrazie
- rysowanie boxow
- konfiguracja przez .env / zmienne srodowiskowe

## Struktura

- src/cv_yolo/predict.py - wrapper na YOLO, zwraca liste detekcji
- src/cv_yolo/visualize.py - rysuje boxy
- scripts/run_on_image.py - uruchomienie na jednym obrazie

## Setup

Najstabilniej na Windows dziala Python 3.12 (w CV/ML ekosystem to jest standard).

W folderze odpowiedzi/cv-yolo-prod:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

Jesli pip zacznie kombinowac z kompilacja numpy, wymus whelle:
pip install --only-binary=:all: -r requirements.txt

## Run

Najprosciej:

python scripts/run_on_image.py --image path\to\image.jpg

Z wlasnym progiem confidence:
python scripts/run_on_image.py --image path\to\image.jpg --conf 0.35

Z innym modelem:
python scripts/run_on_image.py --image path\to\image.jpg --model yolov8s.pt

Output laduje w out/.

## Jak myslec o conf_threshold (praktycznie)

To jest pokretlo, nie prawda objawiona.

- nizszy prog -> wiecej wykryc, wiecej smieci (false positives)
- wyzszy prog -> mniej wykryc, wiecej pomijania (false negatives)

Prog dobierasz pod koszt bledu:
- alarmy i automatyczne akcje -> zwykle prog w gore
- bezpieczenstwo i nie-mozemy-niczego-pominac -> zwykle prog w dol, potem druga walidacja

## Co bys monitorowala na prodzie (bez etykiet)

Bez etykiet nie policzysz accuracy, wiec patrzysz na symptomy:
- ile detekcji na klatke (trend w czasie)
- rozklad confidence (czy robi sie nienaturalnie niski)
- klasy (czy nagle znika jakas klasa)
- latency (czy inference nie zaczyna mulic)

To sa sygnaly driftu albo problemu z kamera (swiatlo, brud, ostrosc).

