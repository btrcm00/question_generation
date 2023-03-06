import torch
from typing import Union, Optional, Tuple
from torch import nn
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.utils.tensorboard import SummaryWriter
from transformers import MBartForConditionalGeneration, MBartConfig
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.mbart.modeling_mbart import shift_tokens_right
from transformers.utils import logging

from common.constants import OUTPUT_PATH

logger = logging.get_logger(__name__)


class BartPhoPointer(MBartForConditionalGeneration):
    def __init__(self, config: MBartConfig, model_config):
        super(BartPhoPointer, self).__init__(config)

        if model_config is not None:
            self.pgen_board = SummaryWriter(log_dir=model_config.get("logging_dir", ""))
            self.pgen_counter = 0
        else:
            self.pgen_board = SummaryWriter(log_dir=OUTPUT_PATH + "/logging/logging31_10/")
            self.pgen_counter = 0

        self.p_gen_sigmoid = nn.Sigmoid()

        self.context_linear = nn.Linear(config.d_model, 1)
        self.init_weights_(self.context_linear)
        self.s_t_linear = nn.Linear(config.d_model, 1)
        self.init_weights_(self.s_t_linear)
        self.x_t_linear = nn.Linear(config.d_model, 1)
        self.init_weights_(self.x_t_linear)

        self.attn_linear = nn.Linear(config.d_model, 1)
        self.init_weights_(self.attn_linear)

        self.pgen_bias = nn.Parameter(torch.zeros(1))
        self.init_weights_(self.pgen_bias)
        self.attn_bias = nn.Parameter(torch.zeros(1))
        self.init_weights_(self.attn_bias)

    @staticmethod
    def init_weights_(module: Union[nn.Linear, nn.Parameter]):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Parameter):
            module.data.zero_()

    def calculate_final_distribution_decoder(self, input_ids, lm_logits, attn_scores, encoder_last_hidden,
                                             decoder_inputs, decoder_outputs):
        context_vec = torch.bmm(attn_scores, encoder_last_hidden)  # h* = Dot{a, h}
        # print(context_vec.size())

        # p_gen = sigmoid(Wh.h* + Wx.x_t + Ws.s_t)
        p_gen = self.context_linear(context_vec) + self.x_t_linear(decoder_inputs) + self.s_t_linear(
            decoder_outputs) + self.pgen_bias

        # p_gen = self.p_gen_layer_norm(p_gen)
        p_gen = torch.sigmoid(p_gen)

        if self.training:
            if self.pgen_counter % 20 == 0:
                print("max p_gen in batch:", torch.max(p_gen))
                print("min p_gen in batch:", torch.min(p_gen))

                self.pgen_board.add_scalars("P_gen", {"min": torch.min(p_gen), "max": torch.max(p_gen),
                                                      "mean": torch.mean(p_gen)},
                                            self.pgen_counter)

            self.pgen_counter += 1

        # Apply attention score with vocab size
        # Create zeros matrix with dim [Bsz, vocab_size, dst_len]
        attn_score_vocab = torch.zeros(lm_logits.size(), device=torch.device(lm_logits.device)).transpose(1, 2)

        # create index with index of each word in vocab to pass to scatter_ function
        input_ids = input_ids.unsqueeze(dim=-1)
        index = input_ids.repeat(1, 1, decoder_outputs.size()[1])  # [Bsz, src_len, dst_len]

        # Copy attn score of each word in src to attn_score_vocab corresponding to index
        attn_distribution = attn_score_vocab.scatter_add_(dim=1, index=index,
                                                          src=attn_scores.transpose(1, 2)).transpose(1, 2)

        # p_gen*P_vocab + (1-p_gen)*attn_distribution
        # if not self.training:
        #     p_gen[p_gen>0.15] = 1
        vocab_diss = nn.functional.softmax(lm_logits, dim=-1)
        final_distribution = p_gen * vocab_diss + (1 - p_gen) * attn_distribution

        return final_distribution

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            entity_weight: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        attn_scores = outputs.cross_attentions[-1].mean(dim=1)
        # attn_scores = outputs.cross_attentions[-1][:, 0, :, :]
        lm_logits = self.calculate_final_distribution_decoder(input_ids=input_ids, lm_logits=lm_logits,
                                                              attn_scores=attn_scores,
                                                              encoder_last_hidden=outputs.encoder_last_hidden_state,
                                                              decoder_inputs=outputs.decoder_hidden_states[0],
                                                              decoder_outputs=outputs.decoder_hidden_states[-1]
                                                              )

        masked_lm_loss = None
        if labels is not None:
            # if self.use_pointer:
            loss_fct = NLLLoss(reduction='none')
            logits = torch.log(lm_logits)
            masked_lm_loss = loss_fct(logits.reshape(-1, self.config.vocab_size), labels.view(-1))
            entity_weight_reshaped = entity_weight.view(-1)
            # print("WEIGHTTTTTT",entity_weight_reshaped)
            masked_lm_loss = torch.sum(masked_lm_loss * entity_weight_reshaped) / torch.sum(entity_weight_reshaped)

            # # loss_fct = CrossEntropyLoss()
            # loss_fct = CrossEntropyLoss(reduction='none')
            # masked_lm_loss = loss_fct(lm_logits.reshape(-1, self.config.vocab_size), labels.view(-1))
            # entity_weight = entity_weight.view(-1)
            # masked_lm_loss = torch.sum(masked_lm_loss * entity_weight) / torch.sum(entity_weight)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids: torch.LongTensor,
            past: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]] = None,
            **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": kwargs["inputs_for_pointer"],
            "entity_weight": kwargs["entity_weight"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ):
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor

        # entity_inputs = encoder_kwargs.pop("entity_inputs")
        entity_weight = encoder_kwargs.pop("entity_weight")
        model_kwargs["encoder_outputs"] = encoder(**encoder_kwargs)
        model_kwargs["inputs_for_pointer"] = inputs_tensor
        # model_kwargs["entity_inputs"] = entity_inputs
        model_kwargs["entity_weight"] = entity_weight
        return model_kwargs
