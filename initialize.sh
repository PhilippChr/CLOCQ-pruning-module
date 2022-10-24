#!/usr/bin/bash 

# create directories
mkdir -p data
mkdir -p results
mkdir -p out

# download 
wget https://qa.mpi-inf.mpg.de/clocq/smart/SMART2022-EL-wikidata-train.json -P data/
wget https://qa.mpi-inf.mpg.de/clocq/smart/SMART2022-EL-wikidata-test.json -P data/
wget https://qa.mpi-inf.mpg.de/clocq/smart/model.bin -P data/