# Simple workflow helpers
.PHONY: install run synth clean

install:
	python -m pip install -r requirements.txt

run:
	python run.py

synth:
	python run.py --generate-synth

clean:
	rm -rf outputs/*
