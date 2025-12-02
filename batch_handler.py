import sys
sys.path.insert(1, '../atp/')

class BatchHandler:
    def __init__(self, config, data_handler, start=None, stop=None):
        self.config = config
        self.data_handler = data_handler
        self.batch_size = config.args.batch_size

        if not start or not stop:
            start = 0
            stop = self.batch_size

        self.start = start
        self.stop = stop
        self.base_toks = {
            key: { "input_ids": self.data_handler.base_toks[key]["input_ids"][self.start:self.stop], "attention_mask": self.data_handler.base_toks[key]["attention_mask"][self.start:self.stop]} for key in self.data_handler.base_toks
        }

        self.base_qs_toks = {
            key: { "input_ids": self.data_handler.base_qs_toks[key]["input_ids"][self.start:self.stop], "attention_mask": self.data_handler.base_qs_toks[key]["attention_mask"][self.start:self.stop]} for key in self.data_handler.base_qs_toks
        }

        self.source_qs_toks = {
            key: { "input_ids": self.data_handler.source_qs_toks[key]["input_ids"][self.start:self.stop], "attention_mask": self.data_handler.source_qs_toks[key]["attention_mask"][self.start:self.stop]} for key in self.data_handler.source_qs_toks
        }

        self.response_start_positions = {
            "base": {
                key: self.data_handler.response_start_positions["base"][key][self.start:self.stop] for key in self.data_handler.response_start_positions["base"]
            },
            "pyreft": self.data_handler.response_start_positions["pyreft"][self.start:self.stop]
        }

        self.prompting_toks = {
            "input_ids": self.data_handler.prompting_toks["input_ids"][self.start:self.stop],
            "attention_mask": self.data_handler.prompting_toks["attention_mask"][self.start:self.stop]
        }

        self.pyreft_toks = {
            "input_ids": self.data_handler.pyreft_toks["input_ids"][self.start:self.stop],
            "attention_mask": self.data_handler.pyreft_toks["attention_mask"][self.start:self.stop]
        }

        if self.config.args.eval_extant:
            self.mmlu_toks = {
                "input_ids": self.data_handler.mmlu_prompts["input_ids"][self.start:self.stop],
                "attention_mask": self.data_handler.mmlu_prompts["attention_mask"][self.start:self.stop]
            }

            self.mmlu_answers = self.data_handler.mmlu_answers[self.start:self.stop]

        if self.config.args.eval_test_dataset:
            self.eval_test_dataset = {
                "queries": {
                    'input_ids': self.data_handler.eval_test_dataset["queries"]['input_ids'][self.start:self.stop],
                    'attention_mask': self.data_handler.eval_test_dataset["queries"]['attention_mask'][self.start:self.stop]
                },
                "answers": self.data_handler.eval_test_dataset["answers"][self.start:self.stop] if self.data_handler.eval_test_dataset["answers"] is not None else None
            }

    def update(self, start=None, stop=None):
        if start is None or stop is None:
            self.start = self.start + self.batch_size
            self.stop = self.start + self.batch_size
        else:
            self.start = start
            self.stop = stop
        print(f"Updating batch handler: start={self.start}, stop={self.stop}")
        self.base_toks = {
            key: { "input_ids": self.data_handler.base_toks[key]["input_ids"][self.start:self.stop], "attention_mask": self.data_handler.base_toks[key]["attention_mask"][self.start:self.stop]} for key in self.base_toks
        }
        
        self.base_qs_toks = {
            key: { "input_ids": self.data_handler.base_qs_toks[key]["input_ids"][self.start:self.stop], "attention_mask": self.data_handler.base_qs_toks[key]["attention_mask"][self.start:self.stop]} for key in self.base_qs_toks
        }

        self.source_qs_toks = {
            key: { "input_ids": self.data_handler.source_qs_toks[key]["input_ids"][self.start:self.stop], "attention_mask": self.data_handler.source_qs_toks[key]["attention_mask"][self.start:self.stop]} for key in self.source_qs_toks
        }
        self.response_start_positions = {
            "base": {
                key: self.data_handler.response_start_positions["base"][key][self.start:self.stop] for key in self.response_start_positions["base"]
            },
            "pyreft": self.data_handler.response_start_positions["pyreft"][self.start:self.stop]
        }

        self.prompting_toks = {
            "input_ids": self.data_handler.prompting_toks["input_ids"][self.start:self.stop],
            "attention_mask": self.data_handler.prompting_toks["attention_mask"][self.start:self.stop]
        }

        self.pyreft_toks = {
            "input_ids": self.data_handler.pyreft_toks["input_ids"][self.start:self.stop],
            "attention_mask": self.data_handler.pyreft_toks["attention_mask"][self.start:self.stop]
        }

        if self.config.args.eval_extant:
            self.mmlu_toks = {
                "input_ids": self.data_handler.mmlu_prompts["input_ids"][self.start:self.stop],
                "attention_mask": self.data_handler.mmlu_prompts["attention_mask"][self.start:self.stop]
            }
            self.mmlu_answers = self.data_handler.mmlu_answers[self.start:self.stop]
        if self.config.args.eval_test_dataset:
            self.eval_test_dataset = {
                "queries": {
                    'input_ids': self.data_handler.eval_test_dataset["queries"]['input_ids'][self.start:self.stop],
                    'attention_mask': self.data_handler.eval_test_dataset["queries"]['attention_mask'][self.start:self.stop]
                },
                "answers": self.data_handler.eval_test_dataset["answers"][self.start:self.stop] if self.data_handler.eval_test_dataset["answers"] is not None else None
            }
        