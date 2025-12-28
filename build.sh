#!/bin/bash

rm -rf build dist *.egg-info
rm -rf ./dist
rm -rf ./src/tuningtron.egg-info
rm -rf ./src/tuningtron/__pycache__

python -m build
python -m twine check dist/*
python -m twine upload dist/*

