Post-hoc pruning module for adapting CLOCQ to entity linking
============

# Description
This repository contains the code for our submission to the [SMART 2022 Task](https://smart-task.github.io/2022/),
and builds upon the [CLOCQ repository](https://github.com/PhilippChr/CLOCQ).


In case of any questions, please [let us know](mailto:pchristm@mpi-inf.mpg.de).

# Code Usage
Please first clone and install the CLOCQ code [CLOCQ repository](https://github.com/PhilippChr/CLOCQ) for accessing the [public CLOCQ API](https://clocq.mpi-inf.mpg.de).
Downloading any data is not required.

Clone the repo via:
```bash
    git clone https://github.com/PhilippChr/CLOCQ-pruning-module.git
    cd CLOCQ-pruning-module/
    conda create --name clocq python=3.8
    conda activate clocq

    # install dependencies
	git clone https://github.com/PhilippChr/CLOCQ.git
    cd CLOCQ/
    pip install -e .
    cd ..
    pip install transformers
```

Pytorch is required as well:
```bash
	# install PyTorch without CUDA
    conda install pytorch torchvision torchaudio -c pytorch

    # install PyTorch for CUDA 10.2 (using GPU)
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

    # install PyTorch for CUDA 11.3 (using GPU)
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

# SMART results 
First download the [SMART data](https://github.com/smart-task/smart-2022-datasets/tree/main/EL_entity_linking/wikidata) and put it in `./data`.
You can use the following script, that also downloads the trained model and initializes required folders.
```bash
	bash initialize.sh
```

Next, split the original train set in a train and dev split:
```bash
    python prepare_smart_data.py
```

Then, run original CLOCQ code on the data to identify potential linkings.
The parameters can be adjusted in the [config](config.yml).
```bash
    python run_clocq.py <PATH_TO_INPUT> <PATH_TO_OUTPUT> [<PATH_TO_CONFIG>]
```
For example:
```bash
    python run_clocq.py data/SMART2022-EL-wikidata-train-split.json results/SMART2022-EL-wikidata-train-split-clocq.json
    python run_clocq.py data/SMART2022-EL-wikidata-dev-split.json results/SMART2022-EL-wikidata-dev-split-clocq.json
    python run_clocq.py data/SMART2022-EL-wikidata-test.json results/SMART2022-EL-wikidata-test-split-clocq.json
```

Then, train the pruning module:
```bash
    python pruning_module.py --train <PATH_TO_TRAIN> <PATH_TO_DEV> [<PATH_TO_CONFIG>]
```
For example:
```bash
    python pruning_module.py --train results/SMART2022-EL-wikidata-train-split-clocq.json results/SMART2022-EL-wikidata-dev-split-clocq.json
```

Finally, run the pruning module via:
```bash
    python pruning_module.py --inference <PATH_TO_INPUT> <PATH_TO_OUTPUT> [<PATH_TO_CONFIG>]
```
For example:
```bash
    python pruning_module.py --inference results/SMART2022-EL-wikidata-test-split-clocq.json results/SMART2022-EL-wikidata-test-final-results.json
```

You can find the results in the specified output file (`results/SMART2022-EL-wikidata-test-final-results.json` in the example).