from transformers import Trainer, TrainingArguments
from models.utils.taxonomy_graph import TaxonomyGraph
import networkx as nx
from models.utils.subgraph import SimpleTaxonomyModel, SubgraphManager
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Dict
import numpy as np

class TaxonomyExpander:
    def __init__(self, subgraph_manager: SubgraphManager, subgraph_predictor: SimpleTaxonomyModel, llm="google/flan-t5-base", use_peft=True):
        self.tokenizer = AutoTokenizer.from_pretrained(llm)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(llm)
        self.subgraph_predictor = subgraph_predictor
        self.subgraph_manager = subgraph_manager
        if use_peft:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q", "v"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            self.model = get_peft_model(base_model, lora_config)
        else:
            self.model = base_model

    def extract_subgraph_prompt(self, concepts, query: str, description: str, graph: TaxonomyGraph) -> str:
        prob = self.subgraph_predictor(query)
        max_index = np.argmax(prob)
        subgraph = self.subgraph_manager.create_positive_subgraph([concepts[max_index]], [concepts[max_index]], graph)


        prompt_lines = ["You are an assistant to hypernym prediction and sorting", "Given a term, its context and a subgraph containing the hypernym of this term, You need to rank these candidate terms in the subgraph which are most possible to be the hypernym or parent term to the given term and return the list"]
        prompt_lines.append(f"Query: {query}")
        prompt_lines.append(f"Description: {description}")
        prompt_lines.append(f"Subgraph for '{query}':")
        for u, v in subgraph.edges:
            prompt_lines.append(f"- {u} --> {v}")            
        return "\n".join(prompt_lines)

    def predict_parent(self, prompt: str, max_tokens: int = 20) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        output = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def prepare_finetune_data(self, concepts, graph: TaxonomyGraph, labeled_data: List[Dict]):
        inputs, targets = [], []
        for item in labeled_data:
            candidate = item["node"]
            description = item["desciption"]
            parent = item["parent"]
            prompt = self.extract_subgraph_prompt(concepts, candidate, description, graph)
            inputs.append(prompt)
            targets.append(parent)
        return inputs, targets

    def finetune(self, concepts, train_graph: TaxonomyGraph, train_data: List[Dict]):
        inputs, targets = self.prepare_finetune_data(concepts, train_graph, train_data)
        class DatasetWrapper:
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                return {
                    "input_ids": self.encodings["input_ids"][idx],
                    "attention_mask": self.encodings["attention_mask"][idx],
                    "labels": self.labels["input_ids"][idx]
                }

            def __len__(self):
                return len(self.encodings["input_ids"])

        encodings = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        labels = self.tokenizer(targets, padding=True, truncation=True, return_tensors="pt")
        dataset = DatasetWrapper(encodings, labels)

        training_args = TrainingArguments(
            output_dir="./taxonomy-expansion",
            per_device_train_batch_size=8,
            num_train_epochs=1,
            learning_rate=1e-4,
            logging_dir="./logs",
            save_strategy="no",
            evaluation_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )
        
        trainer.train()