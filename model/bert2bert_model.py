import warnings
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, NLLLoss

from abc import ABC
from typing import Optional, Tuple, Union
from transformers import (
    EncoderDecoderModel,
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    EncoderDecoderConfig,
)
from transformers.data.metrics import DEPRECATION_WARNING
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right
from torch.utils.tensorboard import SummaryWriter
from transformers.utils import logging
from common.constants import OUTPUT_PATH

logger = logging.get_logger(__name__)


class QG_EncoderDecoderModel(EncoderDecoderModel, ABC):
    def __init__(
            self,
            config: Optional[PretrainedConfig] = None,
            encoder: Optional[PreTrainedModel] = None,
            decoder: Optional[PreTrainedModel] = None,
            training_config=None
    ):
        super().__init__(config, encoder, decoder)
        if training_config is not None:
            self.pgen_board = SummaryWriter(log_dir=training_config.get("logging_dir", ""))
            self.pgen_counter = 0
            self.use_pointer = training_config.get("use_pointer", False)
        else:
            self.use_pointer = True
            self.pgen_board = SummaryWriter(log_dir=OUTPUT_PATH + "/logging/logging17_8/")
            self.pgen_counter = 0

        if self.use_pointer:
            self.p_gen_sigmoid = nn.Sigmoid()

            self.context_linear = nn.Linear(self.decoder.config.hidden_size, 1)
            self.init_weights_(self.context_linear)
            self.s_t_linear = nn.Linear(self.decoder.config.hidden_size, 1)
            self.init_weights_(self.s_t_linear)
            self.x_t_linear = nn.Linear(self.decoder.config.hidden_size, 1)
            self.init_weights_(self.x_t_linear)

            self.attn_linear = nn.Linear(self.decoder.config.hidden_size, 1)
            self.init_weights_(self.attn_linear)

            self.pgen_bias = nn.Parameter(torch.zeros(1))
            self.init_weights_(self.pgen_bias)
            self.attn_bias = nn.Parameter(torch.zeros(1))
            self.init_weights_(self.attn_bias)

            # self.vocab_dist_linear = nn.Linear()

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
        index = torch.tile(input_ids, [1, 1, decoder_outputs.size()[1]])  # [Bsz, src_len, dst_len]

        # Copy attn score of each word in src to attn_score_vocab corresponding to index
        attn_distribution = attn_score_vocab.scatter_add_(dim=1, index=index,
                                                          src=attn_scores.transpose(1, 2)).transpose(1, 2)

        # p_gen*P_vocab + (1-p_gen)*attn_distribution
        vocab_diss = nn.functional.softmax(lm_logits, dim=-1)
        final_distribution = p_gen * vocab_diss + (1 - p_gen) * attn_distribution

        return final_distribution, p_gen

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            entity_weight: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
            past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
                self.encoder.config.hidden_size != self.decoder.config.hidden_size
                and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
        if self.use_pointer:
            attn_scores = decoder_outputs.cross_attentions[-1].mean(dim=1)
            # attn_scores = decoder_outputs.cross_attentions[-1][:, 0, :, :]
            logits, p_gen = self.calculate_final_distribution_decoder(input_ids=input_ids, lm_logits=logits,
                                                                      attn_scores=attn_scores,
                                                                      encoder_last_hidden=encoder_outputs.last_hidden_state,
                                                                      decoder_inputs=decoder_outputs.hidden_states[0],
                                                                      decoder_outputs=decoder_outputs.hidden_states[-1]
                                                                      )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            warnings.warn(DEPRECATION_WARNING, FutureWarning)
            if self.use_pointer:
                loss_fct = NLLLoss(reduction='none')
                logits = torch.log(logits)
                loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))
                entity_weight_reshaped = entity_weight.view(-1)
                # print("WEIGHTTTTTT",entity_weight_reshaped)
                loss = torch.sum(loss * entity_weight_reshaped) / torch.sum(entity_weight_reshaped)

                # loss += torch.sum(torch.abs(p_gen.squeeze(-1) - p_gen_weight))
                # print(loss)


            else:
                # loss_fct = CrossEntropyLoss()
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))
                entity_weight = entity_weight.view(-1)
                loss = torch.sum(loss * entity_weight) / torch.sum(entity_weight)

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs
        # if not self.training:
        #     return Seq2SeqLMOutput(
        #         loss=loss,
        #         logits=logits,
        #         past_key_values=decoder_outputs.past_key_values,
        #         decoder_hidden_states=decoder_outputs.hidden_states,
        #         decoder_attentions=decoder_outputs.attentions,
        #         cross_attentions=decoder_outputs.cross_attentions,
        #         encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #         encoder_hidden_states=encoder_outputs.hidden_states,
        #         encoder_attentions=encoder_outputs.attentions,
        #     ), p_gen

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

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
        entity_weight = encoder_kwargs.pop("entity_weight")

        model_kwargs["encoder_outputs"] = encoder(**encoder_kwargs)
        model_kwargs["inputs_for_pointer"] = inputs_tensor
        model_kwargs["entity_weight"] = entity_weight
        return model_kwargs

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

    @classmethod
    def from_encoder_decoder_pretrained(
            cls,
            encoder_pretrained_model_name_or_path: str = None,
            decoder_pretrained_model_name_or_path: str = None,
            *model_args,
            **kwargs
    ):
        # Add training_config to model
        training_config = kwargs.get("training_config", None)

        kwargs_encoder = {
            argument[len("encoder_"):]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
        return cls(encoder=encoder, decoder=decoder, config=config, training_config=training_config)
