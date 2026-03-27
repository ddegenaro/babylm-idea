from datasets import load_dataset

class TextDataCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        texts = [e["text"] for e in examples]
        batch = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding='longest',
            return_tensors='pt',
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

def get_data(train_rows: int, eval_rows: int, seed: int):
    
    ds = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small")

    if train_rows > -1:
        train_dataset = ds["train"].shuffle(seed=seed).select(range(train_rows))
    else:
        train_dataset = ds["train"].shuffle(seed=seed)
    
    if "validation" in ds:
        if eval_rows > -1:
            eval_dataset = ds["validation"].shuffle(seed=seed).select(range(eval_rows))
        else:
            eval_dataset = ds["validation"].shuffle(seed=seed)
    else:
        eval_dataset = None
    
    return train_dataset, eval_dataset
