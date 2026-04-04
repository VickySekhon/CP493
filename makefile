PYTHON = python3
PIP = pip

TOP_K = 10
MAX_PAGES = 100
OUTPUT = csvs

.PHONY: install run-all figs clean

install:
	$(PIP) install -r requirements.txt

run-all:
	$(PYTHON) src/predict_quality.py --top-k $(TOP_K) --max-pages $(MAX_PAGES) --output $(OUTPUT)

figs:
	$(PYTHON) src/figures/entropy_vs_cer.py --top-k $(TOP_K)
	$(PYTHON) src/figures/roc_thresholds.py --top-k $(TOP_K)
	$(PYTHON) src/figures/stratified_analysis.py --top-k $(TOP_K)

clean:
	rm -rf figures/*.png csvs/*.csv