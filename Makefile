.PHONY: smoke notebook clean

smoke:
	python scripts/smoke_check.py

notebook:
	python scripts/run_notebook.py

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
