# Usage: python prepare_smart_data.py
import json
import random
random.seed(7)

NUM_DEV_SAMPLE = 10000 # dev: 10k, train: ~40k
input_path = "data/SMART2022-EL-wikidata-train.json"

# open data
with open(input_path, "r") as fp:
	data = json.load(fp)

# randomly shuffle + create new dev/train splits
random.shuffle(data)
dev_data = data[:NUM_DEV_SAMPLE]
train_data = data[NUM_DEV_SAMPLE:]

# store new splits
input_path = "data/SMART2022-EL-wikidata-train-split.json"
with open(output_path, "w") as fp:
	fp.write(json.dumps(train_data, indent=4))
input_path = "data/SMART2022-EL-wikidata-dev-split.json"
with open(output_path, "w") as fp:
	fp.write(json.dumps(dev_data, indent=4))