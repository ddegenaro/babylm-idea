from typing import Optional, Union

import torch
from torch import nn

from transformers import GPT2LMHeadModel, GPT2Model
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.masking_utils import create_causal_mask
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.utils import logging

logger = logging.get_logger(__name__)

class EmbedPOSGPT(GPT2Model):
    
    def __init__(
        self,
        config,
        nums_pos_tags: Union[int, list[int]] = 16,
        insert_after: Union[int, list[int]] = 2,
        expand_and_contract: bool = False,
        pos_activation: nn.Module = nn.ReLU()
    ):
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
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPastAndCrossAttentions]:
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # based on pattern from src/transformers/models/whisper/modeling_whisper.py::WhisperDecoder
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            elif isinstance(past_key_values, tuple):
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.53.0. "
                    "You should pass an instance of `Cache` instead, e.g. "
                    "`past_key_values=DynamicCache.from_legacy_cache(past_key_values)`."
                )
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

            if self.config.add_cross_attention and not isinstance(past_key_values, EncoderDecoderCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache(config=self.config))

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        # Attention mask.
        # ._update_causal_mask() and ._prepare_4d_causal_attention_mask_with_cache_position() copied from LlamaModel
        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif self._attn_implementation != "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.h):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                past_key_values if not (self.gradient_checkpointing and self.training) else None,
                cache_position,
                causal_mask,
                head_mask[i],
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

            hidden_states = outputs[0]
            
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

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        past_key_values = past_key_values if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, past_key_values, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
        
class EmbedPOSGPTLMHead(GPT2LMHeadModel):

    def __init__(
        self,
        config,
        nums_pos_tags: Union[int, list[int]] = 16,
        insert_after: Union[int, list[int]] = 2,
        expand_and_contract: bool = False,
        pos_activation: nn.Module = nn.ReLU()
    ):
        super().__init__(config)
        self.transformer = EmbedPOSGPT(
            config,
            nums_pos_tags,
            insert_after,
            expand_and_contract,
            pos_activation
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()