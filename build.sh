#!/bin/bash

rm -rf build dist *.egg-info
python -m build
python -m twine check dist/*
python -m twine upload dist/*
rm -rf ./dist
rm -rf ./src/tuningtron.egg-info
rm -rf ./src/tuningtron/__pycache__

