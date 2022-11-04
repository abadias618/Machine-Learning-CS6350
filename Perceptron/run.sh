#!/bin/bash
python3 driver.py bank-note/train.csv bank-note/test.csv standard &&
python3 driver.py bank-note/train.csv bank-note/test.csv voted &&
python3 driver.py bank-note/train.csv bank-note/test.csv average
