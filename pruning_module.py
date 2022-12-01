import os
import sys
import json
import yaml
import logging
import random

from tqdm import tqdm
from pathlib import Path

from pruning_model import PruningModel

class PruningModule:
    def __init__(self, config_path="config.yml"):
        self.config = self.get_config(config_path)
        self.model = PruningModel(self.config)
        self.model_loaded = False

    def get_config(self, path):
        """Load the config dict from the given .yml file."""
        with open(path, "r") as fp:
            config = yaml.safe_load(fp)
        return config

    def train(self, train_path, dev_path):
        """Train the model."""
        # train model
        self.model.train(train_path, dev_path)

    def inference(self, input_path, output_path):
        """Run inference."""
        # load model
        self._load()

        # load data
        with open(input_path, "r") as fp:
            data = json.load(fp)

        # run inference
        for instance in tqdm(data):
            clocq_linkings = instance["kb_item_tuple"]
            instance["clocq_linkings"] = clocq_linkings

            entity_linkings, predicted_mentions = self.get_entity_linkings(instance["question"], clocq_linkings)
            instance["predicted_mentions"] = predicted_mentions
            instance["entities"] = [linking["item"]["id"] for linking in entity_linkings]

        # store results
        with open(output_path, "w") as fp:
            fp.write(json.dumps(data, indent=4))

    def inference_on_question(self, question):
        # load model (if not done already)
        self._load()
        
        predicted_mentions = self.model.inference(question)
        return predicted_mentions

    def get_entity_linkings(self, question, linkings):
        """Prune the entity linkings provided by the original CLOCQ method using the predicted mentions."""
        # load model
        self._load()

        # predict the relevant mentions (via seq2seq model)
        predicted_mentions = self.model.inference(question)
        predicted_mentions = set([mention.lower() for mention in predicted_mentions])

        # prune predicate linkings
        entity_linkings = [linking for linking in linkings if linking["item"]["id"][0] == "Q"]
        
        # prune irrelevant entity linkings
        output_linkings = list()
        for linking in entity_linkings:
            # apply k (don't consider rank-5 results if k=1)
            if isinstance(self.config["clocq_k"], int) and linking["rank"] >= self.config["clocq_k"]:
                continue
            else:
                # check for exact match
                if linking["question_word"].lower() in predicted_mentions:
                    output_linkings.append({
                        "item": linking["item"],
                        "mention": linking["question_word"]
                    })
                else:
                    # relaxed match: linking mention appears in predicted mention
                    for mention in predicted_mentions:
                        if linking["question_word"].lower() in mention:
                            output_linkings.append({
                                "item": linking["item"],
                                "mention": linking["question_word"]
                            })
                    
                    # relaxed match: predicted mention appears in linking mention
                    for mention in predicted_mentions:
                        if mention in linking["question_word"].lower():
                            output_linkings.append({
                                "item": linking["item"],
                                "mention": linking["question_word"]
                            })
        return output_linkings, list(predicted_mentions)

    def get_relation_linkings(self, question, linkings, top_ranked=True):
        """Prune the relation linkings provided by the original CLOCQ method."""
        # prune entity linkings
        relation_linkings = [
            (kb_item["item"], kb_item["question_word"])
            for kb_item in linkings
            if kb_item["item"]["id"] and kb_item["item"]["id"][0] == "P"
        ]

        # prune lower ranked relations
        mentions = set()
        output_linkings = list() # output
        linkings_dict = dict() # dictionary to keep track of linkings per mention
        for relation, mention in relation_linkings:
            # skip 2nd ranked linking for same mention
            if top_ranked and mention in linkings_dict:
                continue
            mentions.add(mention)
            linkings_dict[mention] = True
            output_linkings.append({
                "item": relation,
                "mention": mention
            })
        return output_linkings, list(mentions)

    def _load(self):
        """Load the model."""
        if not self.model_loaded:
            self.model.load()
            self.model.set_eval_mode()
            self.model_loaded = True

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
    pruning_module = PruningModule(config_path)

    # train: train model
    if function == "--train":
        train_path = sys.argv[2]
        dev_path = sys.argv[3]
        pruning_module.train(train_path, dev_path)

    # inference: add predictions to data
    elif function == "--inference":
        input_path = sys.argv[2]
        output_path = sys.argv[3]
        pruning_module.inference(input_path, output_path)
