import os
import torch
import random
import json
from collections import OrderedDict
from itertools import product

import numpy as np
from transformers import (
    GPT2TokenizerFast,
    GPT2Config, GPT2LMHeadModel,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)

from data import get_data
from embed_pos_gpt import EmbedPOSGPTLMHead

embed = True

seed = 444
train_rows = -1
eval_rows = 10_000
 
MAX_LENGTH = 64
n_embd = 384
n_layer = 4
n_head = 4

num_train_epochs = 5
lr = 5e-4
wd = 1e-2
warmup_steps = 300

grid = OrderedDict({
    'nums_pos_tags': [[8], [16], [32], [64]],
    'insert_after': [[1], [2], [3], [1, 2]],
    'expand_and_contract': [True, False]
})

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

os.makedirs('experiments', exist_ok=True)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=MAX_LENGTH,
    n_embd=n_embd,
    n_layer=n_layer,
    n_head=n_head,
    pad_token_id = tokenizer.pad_token_id,
    bos_token_id = tokenizer.bos_token_id,
    eos_token_id = tokenizer.eos_token_id
)

train_dataset, eval_dataset = get_data(
    train_rows = train_rows,
    eval_rows = eval_rows,
    seed = seed
)

for experiment_setup in product(*grid.values()):
    
    nums_pos_tags, insert_after, expand_and_contract = experiment_setup
    
    try:
        experiment_num = str(max([int(i) for i in os.listdir('experiments')]) + 1)
    except:
        experiment_num = '1'
    
    CHECKPOINT_DIR = f'experiments/{experiment_num}'
    
    os.makedirs(CHECKPOINT_DIR)
    
    json.dump(
        {
            'nums_pos_tags': nums_pos_tags,
            'insert_after': insert_after,
            'expand_and_contract': expand_and_contract,
            'train_rows': train_rows,
            'eval_rows': eval_rows,
            'seed': seed,
            'max_length': MAX_LENGTH,
            'n_embd': n_embd,
            'n_layer': n_layer,
            'n_head': n_head,
            'num_train_epochs': num_train_epochs,
            'lr': lr,
            'wd': wd,
            'warmup_steps': warmup_steps,
            'embed': embed
        },
        open(os.path.join(CHECKPOINT_DIR, 'hparams.json'), 'w+'),
        indent=4
    )

    if embed:
        model = EmbedPOSGPTLMHead(config, nums_pos_tags, insert_after, expand_and_contract)
    else:
        model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    
    # breakpoint()

    # The training args
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        save_steps=2000,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=1000,
        logging_steps=500,
        learning_rate=lr,
        weight_decay=wd,
        warmup_steps=warmup_steps,
        fp16=False,
        report_to=["tensorboard"],
        logging_dir=f"{CHECKPOINT_DIR}/logs",
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = data_collator,
    )

    trainer.train()
