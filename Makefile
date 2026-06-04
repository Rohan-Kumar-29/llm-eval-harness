.PHONY: setup run report all test lint dashboard clean

PYTHON = .venv/Scripts/python
PIP    = .venv/Scripts/pip

setup:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete. Activate with: .venv\\Scripts\\Activate.ps1"

run:
	$(PYTHON) -m evalharness run

report:
	$(PYTHON) -m evalharness report

all:
	$(PYTHON) -m evalharness all

smoke:
	$(PYTHON) -m evalharness run --smoke

test:
	.venv/Scripts/pytest tests/ -v

lint:
	.venv/Scripts/ruff check src/ tests/
	.venv/Scripts/ruff format --check src/ tests/

dashboard:
	.venv/Scripts/streamlit run dashboard/app.py

clean:
	rm -rf .cache/ results/*.parquet results/*.png
	@echo "Cleaned run artifacts."
