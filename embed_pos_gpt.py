from typing import Union

import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.masking_utils import create_bidirectional_mask, create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import auto_docstring, logging
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2LMHeadModel

logger = logging.get_logger(__name__)

@auto_docstring
class EmbedPOSGPT2Model(GPT2Model):
    
    def __init__(
        self,
        config,
        nums_pos_tags: Union[int, list[int]] = 16,
        insert_after: Union[int, list[int]] = 2,
        expand_and_contract: bool = False,
        pos_activation: nn.Module = nn.ReLU()
    ):
        r"""
        nums_pos_tags (`Union[int, list[int]]`):
            How many POS tags to learn at each corresponding layer mentioned in insert_after.
        insert_after (`Union[int, list[int]]`):
            Which layers after which to insert a learnable POS tag module.
        expand_and_contract (`bool`):
            Whether to use a small MLP as the classifier instead of just a linear layer.
        pos_activation (`nn.Module`):
            Activation function to use inside the MLP if using it.
        """
        
        super().__init__(config)
        
        self.expand_and_contract = expand_and_contract
        self.pos_activation = pos_activation
        
        if type(nums_pos_tags) == int:
            nums_pos_tags = [nums_pos_tags]
        self.nums_pos_tags = nums_pos_tags
        
        if type(insert_after) == int:
            insert_after = [insert_after]
        self.insert_after = insert_after
        
        if self.insert_after == [-1]:
            self.insert_after = list(range(config.n_layer))
        
        if len(self.nums_pos_tags) == 1 and len(self.insert_after) > 1:
            self.nums_pos_tags = [self.nums_pos_tags[0] for _ in range(len(self.insert_after))]
        
        assert len(self.insert_after) == len(self.nums_pos_tags), f"Must provide a num pos tags for each layer in insert_after, or else a single number to use for all layers."
        
        for i in self.insert_after:
            assert i < config.num_hidden_layers, f"Each layer specified in insert_after must exist. Layer {i} does not exist with {config.num_hidden_layers} hidden layers."
        
        if expand_and_contract:
            
            self.pos_selectors_bot = nn.ModuleDict({
                str(self.insert_after[i]): nn.Linear(config.hidden_size, 4 * config.hidden_size)
                for i in range(len(self.insert_after))
            })
            
            self.pos_selectors_top = nn.ModuleDict({
                str(self.insert_after[i]): nn.Linear(4 * config.hidden_size, self.nums_pos_tags[i])
                for i in range(len(self.insert_after))
            })
        else:
            self.pos_selectors = nn.ModuleDict({
                str(self.insert_after[i]): nn.Linear(config.hidden_size, self.nums_pos_tags[i])
                for i in range(len(self.insert_after))
            })
        
        self.wpose = nn.ModuleDict({
            str(self.insert_after[i]): nn.Embedding(self.nums_pos_tags[i], self.embed_dim)
            for i in range(len(self.insert_after))
        })
        
        
  
    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """
        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        # based on pattern from src/transformers/models/whisper/modeling_whisper.py::WhisperDecoder
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)

            if self.config.add_cross_attention and not isinstance(past_key_values, EncoderDecoderCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache(config=self.config))

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        # Attention mask.
        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        encoder_attention_mask = None
        if encoder_hidden_states is not None:
            encoder_attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        for i, block in enumerate(self.h):
            hidden_states = block(
                hidden_states,
                past_key_values if not (self.gradient_checkpointing and self.training) else None,
                causal_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                position_ids=position_ids,
                **kwargs,
            )
            
            # breakpoint()
            
            # ADDITIONAL
            if i in self.insert_after:
                key = str(i)
                if self.expand_and_contract:
                    selections = torch.argmax(
                        self.pos_selectors_top[key](
                            self.pos_activation(self.pos_selectors_bot[key](hidden_states))
                        ),
                        dim=-1
                    )
                else:
                    selections = torch.argmax(
                        self.pos_selectors[key](hidden_states),
                        dim=-1
                    )
                embeds = self.wpose[key](selections)
                hidden_states += embeds

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)

        past_key_values = past_key_values if use_cache else None
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
        
@auto_docstring(
    custom_intro="""
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """
)
class EmbedPOSGPT2LMHeadModel(GPT2LMHeadModel):
    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    def __init__(
        self,
        config,
        nums_pos_tags,
        insert_after,
        expand_and_contract,
        pos_activation
    ):
        r"""
        nums_pos_tags (`Union[int, list[int]]`):
            How many POS tags to learn at each corresponding layer mentioned in insert_after.
        insert_after (`Union[int, list[int]]`):
            Which layers after which to insert a learnable POS tag module.
        expand_and_contract (`bool`):
            Whether to use a small MLP as the classifier instead of just a linear layer.
        pos_activation (`nn.Module`):
            Activation function to use inside the MLP if using it.
        """
        
        super().__init__(config)
        self.transformer = EmbedPOSGPT2Model(
            config,
            nums_pos_tags=nums_pos_tags,
            insert_after=insert_after,
            expand_and_contract=expand_and_contract,
            pos_activation=pos_activation
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()