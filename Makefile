.PHONY: install test ingest keys run chat eval up down
export PYTHONPATH := src

install:            ## editable install with all extras
	pip install -e ".[all]"

test:               ## unit + end-to-end tests (offline, no cloud needed)
	python -m unittest discover -s tests -t . -v

ingest:             ## index the sample dataset with the configured providers
	python -m support_assistant.cli.ingest --source data/support_tickets_1000.csv

keys:               ## demo API keys for tenant CMP-101 (printed once)
	python -m support_assistant.cli.keys --tenant CMP-101 --role customer
	python -m support_assistant.cli.keys --tenant CMP-101 --role agent

run:                ## start the API on :8080
	uvicorn support_assistant.api.app:app_factory --factory --host 0.0.0.0 --port 8080 --reload

chat:               ## terminal chat as a CMP-101 customer
	python -m support_assistant.cli.chat --tenant CMP-101 --role customer

eval:               ## retrieval / isolation / leakage / threshold-sweep report -> reports/
	python -m support_assistant.cli.evaluate --source data/support_tickets_1000.csv --samples 300

up:                 ## full local stack: api + qdrant + mysql
	docker compose up --build -d

down:
	docker compose down
