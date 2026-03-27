from datasets import load_dataset

ds = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small")

def get_data(train_rows: int, eval_rows: int, seed: int):

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