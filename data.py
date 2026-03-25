from datasets import load_from_disk

DATA_DIR = 'data'

tokenized = load_from_disk(DATA_DIR)

def get_data(train_rows: int, eval_rows: int, seed: int):

    if train_rows > -1:
        train_dataset = tokenized["train"].shuffle(seed=seed).select(range(train_rows))
    else:
        train_dataset = tokenized["train"].shuffle(seed=seed)
        
    if eval_rows > -1:
        eval_dataset = tokenized["validation"].shuffle(seed=seed).select(range(eval_rows))
    else:
        eval_dataset = tokenized["validation"].shuffle(seed=seed)
    
    return train_dataset, eval_dataset