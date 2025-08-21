#!/bin/bash
clear
source ./myenv/bin/activate
pytest -v --maxfail=1 --disable-warnings