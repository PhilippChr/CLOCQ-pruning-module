import os
import torch
import transformers

from pathlib import Path

import pruning_dataset as dataset


class PruningModel(torch.nn.Module):
    def __init__(self, config):
        super(PruningModel, self).__init__()
        self.config = config
        
        # select model architecture
        if config["architecture"] == "BART":
            self.model = transformers.BartForConditionalGeneration.from_pretrained(
                "facebook/bart-base"
            )
            self.tokenizer = transformers.BartTokenizerFast.from_pretrained("facebook/bart-base")
        elif config["architecture"] == "T5":
            self.model = transformers.T5ForConditionalGeneration.from_pretrained("t5-base")
            self.tokenizer = transformers.T5TokenizerFast.from_pretrained("t5-base")
        else:
            raise Exception(
                "Unknown architecture for module specified in config: currently, only T5-base (=T5) and BART-base (=BART) are supported."
            )

    def set_eval_mode(self):
        """Set model to eval mode."""
        self.model.eval()

    def save(self):
        """Save model."""
        model_path = self.config["model_path"]
        # create dir if not exists
        model_dir = os.path.dirname(model_path)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def load(self):
        """Load model."""
        if torch.cuda.is_available():
            state_dict = torch.load(self.config["model_path"])
        else:
            state_dict = torch.load(self.config["model_path"], torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def train(self, train_path, dev_path):
        """Train model."""
        # load datasets
        train_dataset = dataset.PruningDataset(self.config, self.tokenizer, train_path)
        dev_dataset = dataset.PruningDataset(self.config, self.tokenizer, dev_path)
        # arguments for training
        training_args = transformers.Seq2SeqTrainingArguments(
            output_dir="results",  # output directory
            num_train_epochs=self.config["num_train_epochs"],  # total number of training epochs
            per_device_train_batch_size=self.config[
                "per_device_train_batch_size"
            ],  # batch size per device during training
            per_device_eval_batch_size=self.config[
                "per_device_eval_batch_size"
            ],  # batch size for evaluation
            warmup_steps=self.config[
                "warmup_steps"
            ],  # number of warmup steps for learning rate scheduler
            weight_decay=self.config["weight_decay"],  # strength of weight decay
            logging_dir="logs",  # directory for storing logs
            logging_steps=1000,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end="True"
            # predict_with_generate=True
        )
        # create the object for training
        trainer = transformers.Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
        )
        # training progress
        trainer.train()
        # store model
        self.save()

    def inference(self, input):
        """Run the model on the given input."""
        # encode
        input_encodings = self.tokenizer(
            input,
            padding=True,
            truncation=True,
            max_length=self.config["max_input_length"],
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            input_encodings = input_encodings.to(torch.device("cuda"))
        # generate
        output = self.model.generate(
            input_ids=input_encodings["input_ids"],
            attention_mask=input_encodings["attention_mask"],
            no_repeat_ngram_size=self.config["no_repeat_ngram_size"],
            num_beams=self.config["num_beams"],
            early_stopping=self.config["early_stopping"],
        )
        # decoding
        mentions_text = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        # obtain question words
        mentions = self.get_mentions(mentions_text)
        return mentions

    def get_mentions(self, mentions_text):
        mentions = mentions_text.split("|")
        mentions = [m.strip() for m in mentions]
        return mentions

