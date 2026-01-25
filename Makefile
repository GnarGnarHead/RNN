PY ?= python3
INPUT ?= input.txt
STEPS ?= 8000
K ?= 2

.PHONY: download run sweep

download:
	$(PY) scripts/download_tiny_shakespeare.py --out $(INPUT)

run:
	$(PY) settle_rnn_charlm.py --text-path $(INPUT) --k-settle $(K) --steps $(STEPS)

sweep:
	$(PY) settle_rnn_charlm.py --text-path $(INPUT) --steps $(STEPS) --sweep-k 1 2 4 8 --out-dir runs --run-name sweep
