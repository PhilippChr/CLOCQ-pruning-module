# Usage:    python pruning_module.py --train <PATH_TO_TRAIN> <PATH_TO_DEV> [<PATH_TO_CONFIG>]
# Or:       python pruning_module.py --inference <PATH_TO_INPUT> <PATH_TO_OUTPUT> [<PATH_TO_CONFIG>]
import os
import sys
import json
import yaml
import logging
import random

from tqdm import tqdm
from pathlib import Path

from pruning_model import PruningModel

def get_config(path):
    """Load the config dict from the given .yml file."""
    with open(path, "r") as fp:
        config = yaml.safe_load(fp)
    return config

def train(config, train_path, dev_path):
    """Train the model."""
    # train model
    model = PruningModel(config)
    model.train(train_path, dev_path)

def inference(config, input_path, output_path):
    """Run inference."""
    # load model
    model = PruningModel(config)
    _load(model)

    # load data
    with open(input_path, "r") as fp:
        data = json.load(fp)

    # run inference
    for instance in tqdm(data):
        mentions = model.inference(instance["question"])
        instance["predicted_mentions"] = mentions
        native_clocq_linkings = instance["linkings"]
        instance["native_clocq_linkings"] = native_clocq_linkings
        instance["entities"] = _prune_linkings(config, native_clocq_linkings, mentions)

    # store results
    with open(output_path, "w") as fp:
        fp.write(json.dumps(data, indent=4))

def _prune_linkings(config, linkings, mentions):
    """Prune the linkings provided by the original CLOCQ method using the predicted mentions."""
    mentions = set([mention.lower() for mention in mentions])
    output_linkings = set()
    for linking in linkings:
        # apply k (don't consider rank-5 results if k=1)
        if isinstance(config["clocq_k"], int) and linking["rank"] >= config["clocq_k"]:
            continue
        else:
            # check for exact match
            if linking["question_word"].lower() in mentions:
                output_linkings.add(linking["item"]["id"])
            else:
                # relaxed match: linking mention appears in predicted mention
                for mention in mentions:
                    if linking["question_word"].lower() in mention:
                        output_linkings.add(linking["item"]["id"])
                
                # relaxed match: predicted mention appears in linking mention
                for mention in mentions:
                    if mention in linking["question_word"].lower():
                        output_linkings.add(linking["item"]["id"])
    return list(output_linkings)

def _load(model):
    """Load the model."""
    model.load()
    model.set_eval_mode()
    model = True

#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception(
            "Usage: python pruning_module.py --train <PATH_TO_TRAIN> <PATH_TO_DEV> [<PATH_TO_CONFIG>]\nOR python pruning_module.py --inference <PATH_TO_INPUT> <PATH_TO_OUTPUT> [<PATH_TO_CONFIG>]"
        )

    # load params
    function = sys.argv[1]
    config_path = sys.argv[4] if len(sys.argv) > 4 else "config.yml"
    config = get_config(config_path)

    # train: train model
    if function == "--train":
        train_path = sys.argv[2]
        dev_path = sys.argv[3]
        train(config, train_path, dev_path)

    # inference: add predictions to data
    elif function == "--inference":
        input_path = sys.argv[2]
        output_path = sys.argv[3]
        inference(config, input_path, output_path)