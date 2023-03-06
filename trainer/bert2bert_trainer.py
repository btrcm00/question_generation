import torch
import os
import json
import argparse
import numpy as np
from datasets import load_metric

from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.trainer_utils import IntervalStrategy
from transformers import SchedulerType, AutoTokenizer
from transformers.training_args import OptimizerNames

from common.common_keys import MODEL_QUESTION_TYPE_INPUT, SPECIAL_TOKENS
from common.constants import DATA_PATH, OUTPUT_PATH, SPECIAL_TOKENS_PATH
from common.config import QuestionType
from trainer.seq2seq_trainer import QGTrainer
from model.bert2bert_model import QG_EncoderDecoderModel
from dataset_constructor.dataloader import FQG_dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

metric = load_metric("sacrebleu")
tokenizer = None
sent_limit = 0


def model_config():
    training_parser = argparse.ArgumentParser()
    training_parser.add_argument('--output_dir', default=OUTPUT_PATH + "/checkpoint/checkpoint20_9/", type=str,
                                 help='output directory')
    training_parser.add_argument("--logging_dir", default=OUTPUT_PATH + "/logging/logging20_9/", type=str,
                                 help="Tensorboard Logging Folder")
    training_parser.add_argument('-b', '--batch_size', default=4, type=int, help='mini-batch size (default: 32)')
    training_parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight Decay")
    training_parser.add_argument("--save_total_limit", default=5, type=int, help="Total Limit")
    training_parser.add_argument("--lr", default=2e-5, type=float, help="Learning rate")
    training_parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient Accumulation "
                                                                                            "Steps")
    training_parser.add_argument("--eval_steps", default=2000, type=int, help="Num steps per evaluation ")

    training_parser.add_argument("--logging_steps", default=2000, type=int, help="Num Steps per logging ")
    training_parser.add_argument("--save_steps", default=2000, type=int, help="Num Steps per saving ")
    training_parser.add_argument("--device", default="cuda", type=str, help="Visible Cuda")
    training_parser.add_argument("--num_train_epochs", default=10, type=int, help="Num Epochs")
    training_parser.add_argument("--restore_checkpoint", default=True, type=bool)
    training_parser.add_argument("--restore_folder",
                                 default=OUTPUT_PATH + "/checkpoint/checkpoint17_9/checkpoint-54000",
                                 type=str)

    training_parser.add_argument('--dataset_folder', default=DATA_PATH + "/FQG_data/bert2bert/new_dataset_30_8/",
                                 type=str,
                                 help='directory '
                                      'of dataset')
    training_parser.add_argument("--encoder_pretrained_path", default="vinai/phobert-base", type=str,
                                 help="Tokenizer name or path")
    training_parser.add_argument("--decoder_pretrained_path", default="vinai/phobert-base", type=str,
                                 help="Tokenizer name or path")
    training_parser.add_argument("--generation_num_beams", default=1, type=int)
    training_parser.add_argument("--metric_for_best_model", default="bleu", type=str)
    training_parser.add_argument("--warm_up_ratio", default=0.1, type=float, help="Warm-up ratio for scheduler")
    training_parser.add_argument('--generation_max_length', default=128, type=int,
                                 help='maximum question token number')
    training_parser.add_argument('--input_max_length', default=256, type=int,
                                 help='maximum context token number')

    training_parser.add_argument('--use_pointer', default=True, type=bool,
                                 help='whether or not using pointer generator in training model')

    return training_parser.parse_args()


def collate_(data):
    global sent_limit
    batch = {}
    for feature_key in data[0].keys():
        # prepare for tag feature ids
        if isinstance(data[0][feature_key], dict):
            batch[feature_key] = {
                k: torch.tensor(np.array([ele[feature_key][k].numpy() for ele in data])) if k not in ["ques_type_id",
                                                                                                      MODEL_QUESTION_TYPE_INPUT] else [
                    ele[feature_key][k] for ele in data]
                for k in data[0][feature_key].keys()
            }
        elif feature_key in ["entity_weight", "p_gen_weight"]:
            batch[feature_key] = torch.vstack([ele[feature_key] for ele in data])
        # input ids vs attention mask
        else:
            batch[feature_key] = torch.tensor([ele[feature_key][:sent_limit] for ele in data])
    return batch


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # decoded_preds = [ele.split("?")[0] + "?" for ele in decoder]

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    print("_______________________________________")
    print("_______________________________________")
    print("PRED : ", decoded_preds)
    print("LABEL: ", decoded_labels)
    print("_______________________________________")
    print("_______________________________________")

    result = metric.compute(predictions=decoded_preds, references=[[ele] for ele in decoded_labels])
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def main():
    global tokenizer
    global sent_limit
    training_config = model_config()
    sent_limit = training_config.input_max_length

    if not training_config.restore_checkpoint:
        # add new special tokens
        tokenizer = AutoTokenizer.from_pretrained(training_config.encoder_pretrained_path)
        new_special_tokens = json.load(open(SPECIAL_TOKENS_PATH))[
                                 SPECIAL_TOKENS] + [e.name for e in QuestionType]
        special_tokens_dict = {
            "additional_special_tokens": [f"<{id_.upper()}>" for id_ in list(set(new_special_tokens))]}
        tokenizer.add_special_tokens(special_tokens_dict)
        # ==========================================

        bert2bert = QG_EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=training_config.encoder_pretrained_path,
            decoder_pretrained_model_name_or_path=training_config.decoder_pretrained_path,
            training_config=training_config.__dict__
        )

        # resize embeddings of encoder and decoder
        bert2bert.encoder.resize_token_embeddings(len(tokenizer))
        bert2bert.decoder.resize_token_embeddings(len(tokenizer))
        # ==========================================

        # define parameters for beam search decoding
        bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
        bert2bert.config.eos_token_id = tokenizer.sep_token_id
        bert2bert.config.pad_token_id = tokenizer.pad_token_id
        bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size
        bert2bert.config.output_attentions = True
        bert2bert.config.output_hidden_states = True
        # bert2bert.load_state_dict(torch.load("/TMTAI/KBQA/minhbtc/ACS-QG/QASystem/TextualQA/QuestionAnswering_Generation/FQG_output/checkpoint11_8/checkpoint-184000/pytorch_model.bin", map_location=torch.device(training_config.device)))
        # ==========================================
        print(22222222222222222222, bert2bert.config.vocab_size)
    else:
        tokenizer = AutoTokenizer.from_pretrained(training_config.restore_folder)
        bert2bert = QG_EncoderDecoderModel.from_pretrained(training_config.restore_folder,
                                                           training_config=training_config.__dict__)
    bert2bert.to(training_config.device)
    train_dataset, valid_dataset, test_dataset = FQG_dataset.get_dataset(config=training_config,
                                                                         dataset_folder=training_config.dataset_folder,
                                                                         tokenizer=tokenizer,
                                                                         added_new_special_tokens=True
                                                                         )

    training_args = Seq2SeqTrainingArguments(
        output_dir=training_config.output_dir,
        evaluation_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        learning_rate=training_config.lr,
        weight_decay=training_config.weight_decay,
        save_total_limit=training_config.save_total_limit,
        num_train_epochs=training_config.num_train_epochs,
        predict_with_generate=True,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        eval_steps=training_config.eval_steps,
        fp16=True,
        push_to_hub=False,
        # dataloader_pin_memory=True,
        dataloader_num_workers=2,
        logging_dir=training_config.logging_dir,
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        logging_first_step=False,
        optim=OptimizerNames.ADAMW_TORCH,
        generation_max_length=training_config.generation_max_length,
        generation_num_beams=training_config.generation_num_beams,
        load_best_model_at_end=True,
        metric_for_best_model=training_config.metric_for_best_model,
        warmup_ratio=training_config.warm_up_ratio,
        lr_scheduler_type=SchedulerType.COSINE_WITH_RESTARTS
    )

    trainer = QGTrainer(
        model=bert2bert,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_,
        compute_metrics=compute_metrics,
        tokenizer=train_dataset.tokenizer
    )

    # trainer.train(resume_from_checkpoint=False)
    print(trainer.evaluate(valid_dataset))


if __name__ == "__main__":
    main()
