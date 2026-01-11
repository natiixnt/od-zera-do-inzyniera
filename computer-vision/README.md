# computer-vision

Mini repo do ogarniecia CV pipeline end to end:
- load image
- preprocess
- (dummy) detector
- threshold + NMS
- draw boxes
- domain shift demo

## install
Na szybko (venv):
python -m venv .venv
.venv\Scripts\activate
pip install opencv-python numpy

## run
python scripts/run_on_image.py --image data/images/sample_01.jpg --out out/sample_01.png --threshold 0.5

python scripts/sweep_threshold.py --image data/images/sample_01.jpg --min 0.1 --max 0.9 --step 0.1

python scripts/domain_shift_demo.py --image data/images/sample_01.jpg --out_dir out/domain_shift --threshold 0.5
