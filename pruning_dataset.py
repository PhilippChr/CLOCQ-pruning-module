import json
import torch

def output_to_text(mentions):
    return "|".join(mentions)

class PruningDataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, path):
        self.tokenizer = tokenizer

        self.k = config["pruning_module_clocq_k"]
        self.config = config

        input_encodings, output_encodings, dataset_length = self._load_data(path)
        self.input_encodings = input_encodings
        self.output_encodings = output_encodings
        self.dataset_length = dataset_length

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.input_encodings.items()}
        labels = self.output_encodings["input_ids"][idx]
        item = {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": labels,
        }
        return item

    def __len__(self):
        return self.dataset_length

    def _load_data(self, path):
        """
        Opens the file, and loads the data into
        a format that can be put into the model.

        The input dataset should be annotated using
        the silver_annotation.py class.

        The whole history is given as input.
        """
        # open data
        with open(path, "r") as fp:
            data = json.load(fp)

        inputs = list()
        outputs = list()

        for instance in data:
            # identify "gold" mentions (=mentions that should be linked)
            mentions = list()
            gold_entities = set(instance["gold_entities"])
           
            disambiguations = instance["linkings"]
            for disambiguation in disambiguations:
                if isinstance(self.k, int) and disambiguation["rank"] >= self.k:
                    continue
                if disambiguation["item"]["id"] in gold_entities:
                    mentions.append(disambiguation["question_word"])

            inputs.append(instance["question"])
            outputs.append(output_to_text(mentions))

        # encode
        input_encodings = self.tokenizer(
            inputs, padding=True, truncation=True, max_length=self.config["max_input_length"]
        )
        output_encodings = self.tokenizer(
            outputs, padding=True, truncation=True, max_length=self.config["max_input_length"]
        )
        dataset_length = len(inputs)

        return input_encodings, output_encodings, dataset_length
