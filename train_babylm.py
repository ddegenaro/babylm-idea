import os
import random
import json
import glob
from collections import OrderedDict
from itertools import product

import numpy as np
import torch
from torch import nn
from transformers import (
    GPT2Config, GPT2LMHeadModel,
    Trainer, TrainingArguments,
    GPT2Tokenizer
)

from data import get_data, TextDataCollator
from embed_pos_gpt import EmbedPOSGPT2LMHeadModel

seed = 444
train_rows = 10_000 # -1 means all rows
eval_rows = 10_000
 
MAX_LENGTH = 1024
n_embd = 128
n_layer = 12
n_head = 4

num_train_epochs = 5
lr = 5e-4
wd = 1e-2
warmup_steps = 300

embed = True # use experimental technique

if embed:
    grid = OrderedDict({
        'nums_pos_tags': [[2], [4], [8], [16], [32], [64]],
        'insert_after': [[1]],
        'expand_and_contract': [True],
        'pos_activation': [nn.ReLU()]
    })
else:
    grid = OrderedDict({
        'nums_pos_tags': [None],
        'insert_after': [None],
        'expand_and_contract': [None],
        'pos_activation': [None]
    })

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.mps.is_available():
    DEVICE = 'mps'
print(f'Using device: {DEVICE}')

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

os.makedirs('experiments', exist_ok=True)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
data_collator = TextDataCollator(tokenizer, max_length=MAX_LENGTH)

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

print('Loading data...')
train_dataset, eval_dataset = get_data(
    train_rows = train_rows,
    eval_rows = eval_rows,
    seed = seed
)
print('Done.')

print('Searching finished experiments...')
existing_hparams = [
    json.load(open(f)) for f in glob.glob('experiments/*/hparams.json')
]
for existing_hparam in existing_hparams:
    del existing_hparam['param_count']
print('Done.')

for experiment_setup in product(*grid.values()):
    
    nums_pos_tags, insert_after, expand_and_contract, pos_activation = experiment_setup
    
    hparams = {
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
        'embed': embed,
        'pos_activation': repr(pos_activation) if pos_activation is not None else None
    }
    
    if not embed:
        hparams['nums_pos_tags'] = None
        hparams['insert_after'] = None
        hparams['expand_and_contract'] = None
    
    if hparams in existing_hparams:
        print(f'Skipping: {experiment_setup}, already found.')
        continue
    
    try:
        exps = os.listdir('experiments')
        try:
            exps.remove('.DS_Store')
        except:
            pass
        experiment_num = str(max([int(i) for i in exps]) + 1)
    except:
        experiment_num = '1'
    CHECKPOINT_DIR = f'experiments/{experiment_num}'
    os.makedirs(CHECKPOINT_DIR)

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        save_steps=2000,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=1000,
        logging_steps=500,
        learning_rate=lr,
        weight_decay=wd,
        warmup_steps=warmup_steps,
        fp16=False,
        report_to=["tensorboard"],
        logging_dir=f"{CHECKPOINT_DIR}/logs",
        remove_unused_columns=False,
    )
    
    if embed:
        model = EmbedPOSGPT2LMHeadModel(
            config,
            nums_pos_tags,
            insert_after,
            expand_and_contract,
            pos_activation
        )
    else:
        model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    
    hparams['param_count'] = sum(p.numel() for p in model.parameters())
    json.dump(
        hparams,
        open(os.path.join(CHECKPOINT_DIR, 'hparams.json'), 'w+'),
        indent=4
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = data_collator,
    )
    
    print(f'Training {type(model)} on device {DEVICE}:')
    for key, val in hparams.items():
        if type(val) == int:
            print(f'\t{key}: {val:,}')
        else:
            print(f'\t{key}: {val}')

    trainer.train()
