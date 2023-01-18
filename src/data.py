
import json
import pandas as pd

from datasets import Dataset
from torch.utils.data import DataLoader


def Table2TextDataLoader(fname, tokenizer, batch_size, mode="train"):
    """
    Build Data Loader

    """
    def preprocess_function(examples):
        tokenizer_input = tokenizer([tokenizer.bos_token+s+tokenizer.eos_token for s in examples["table"]],
                                    padding="max_length", max_length=512, truncation=True, return_tensors="pt", return_token_type_ids=False)
        encoder_input_ids = tokenizer_input["input_ids"]
        encoder_attention_mask = tokenizer_input["attention_mask"]

        if mode=="train":
            tokenizer_output = tokenizer([tokenizer.bos_token+s+tokenizer.eos_token for s in examples["text"]],
                                        padding="max_length", max_length=512, truncation=True, return_tensors="pt", return_token_type_ids=False)
            decoder_input_ids = tokenizer_output["input_ids"]
            decoder_attention_mask = tokenizer_output["attention_mask"]

            return {
                "input_ids": encoder_input_ids,
                "attention_mask": encoder_attention_mask,
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
            }

        return {
            "input_ids": encoder_input_ids,
            "attention_mask": encoder_attention_mask,
        }

    dataset = load_dataset(fname, mode)
    dataset = dataset.map(
        preprocess_function, batched=True, num_proc=8, remove_columns=dataset.column_names
    ).with_format("torch")

    dataloader = DataLoader(dataset, shuffle=(True if mode=="train" else False), batch_size=batch_size)

    return dataloader


def jsonlload(fname):
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        j_list = [json.loads(line) for line in lines]

    return j_list


def jsonldump(j_list, fname):
    with open(fname, "w", encoding='utf-8') as f:
        for json_data in j_list:
            f.write(json.dumps(json_data, ensure_ascii=False)+'\n')


def jsonl2df(j_list, mode):
    data_dict = {"table": []}
    if mode == "train":
        data_dict["text"] = []

    for j in j_list:
        td = []
        for row in j['input']['table']:
            td.append("[TAB]".join([d['value'] for d in row]))
        table = "[NL]".join(td)
        
        if mode == "train":
            for text in j['output']:
                data_dict["table"].append(table)
                data_dict["text"].append(text)
        else:
            data_dict["table"].append(table)
    
    df = pd.DataFrame(data_dict)
    return df


def load_dataset(fname, mode):
    j_list = jsonlload(fname)
    df = jsonl2df(j_list, mode)
    dataset = Dataset.from_pandas(df)

    return dataset