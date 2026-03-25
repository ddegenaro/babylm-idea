from torch import nn

from transformers import GPT2Config, AutoTokenizer, GPT2Model

from embed_pos_gpt import EmbedPOSGPT

num_pos_tags = 16 # list of pos tag set sizes for each layer
insert_after = -1 # list of zero-indexed layers, a single layer index, or -1 for all layers
expand_and_contract = False # whether to make each layer a 2-layer FFNN versus a single WX+B
pos_activation = nn.ReLU() # activation to put between layers, ignored if expand_and_contract==False

config = GPT2Config(
    n_positions=128,
    n_embd=256,
    n_layer=6,
    n_head=4
)

prompt = ['hello world, how are you?']

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2Model(config)
custom_model = EmbedPOSGPT(
    config,
    nums_pos_tags=num_pos_tags,
    insert_after=insert_after,
    expand_and_contract=expand_and_contract,
    pos_activation=pos_activation
)

def param_count(m: nn.Module):
    return sum(p.numel() for p in m.parameters())

inputs = tokenizer(
    prompt,
    return_tensors='pt',
    truncation=True,
    padding='longest'
)

print(f'custom model params: {param_count(custom_model):,}')
print(f'  base model params: {param_count(model):,}')
print('\n')

print('inputs:')
print(inputs)
print('\n')

print('outputs:')
print(custom_model(**inputs))
print('\n')

print('args:')
print('nums_pos_tags:', custom_model.nums_pos_tags)
print('insert_after:', custom_model.insert_after)
print('\n')

print('params:')
if expand_and_contract:
    print('pos_selectors_bot:', custom_model.pos_selectors_bot)
    print('pos_selectors_top:', custom_model.pos_selectors_top)
else:
    print('pos_selectors:', custom_model.pos_selectors)
print('wpose:', custom_model.wpose)