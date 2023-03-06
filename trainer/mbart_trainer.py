import argparse
import json
import numpy as np
import torch

from datasets import load_metric
from transformers import SchedulerType, AutoTokenizer
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import OptimizerNames
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from common.common_keys import *
from common.config import QuestionType, PipelineConfig
from common.constants import *
from dataset_constructor.dataloader import FQG_dataset
from model.bartpho import BartPhoPointer
from trainer.seq2seq_trainer import QGTrainer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def model_config():
    training_parser = argparse.ArgumentParser()
    training_parser.add_argument('--output_dir', default=OUTPUT_PATH + "/checkpoint/checkpoint_bart_24_2/", type=str,
                                 help='output directory')
    training_parser.add_argument("--logging_dir", default=OUTPUT_PATH + "/logging/logging_bart_24_2/", type=str,
                                 help="Tensorboard Logging Folder")
    training_parser.add_argument('-b', '--batch_size', default=2, type=int, help='mini-batch size (default: 32)')
    training_parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight Decay")
    training_parser.add_argument("--save_total_limit", default=5, type=int, help="Total Limit")
    training_parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate")
    training_parser.add_argument("--gradient_accumulation_steps", default=2, type=int, help="Gradient Accumulation "
                                                                                            "Steps")
    training_parser.add_argument("--eval_steps", default=5, type=int, help="Num steps per evaluation ")

    training_parser.add_argument("--logging_steps", default=5, type=int, help="Num Steps per logging ")
    training_parser.add_argument("--save_steps", default=5, type=int, help="Num Steps per saving ")
    training_parser.add_argument("--device", default="cpu", type=str, help="Visible Cuda")
    training_parser.add_argument("--num_train_epochs", default=1, type=int, help="Num Epochs")
    training_parser.add_argument("--restore_checkpoint", default=True, type=bool)
    training_parser.add_argument("--restore_folder",
                                 default=INFERENCE_PATH + "/checkpoint/bartpho_pointer_22_9/",
                                 type=str)

    training_parser.add_argument('--dataset_folder', default=DATA_PATH + "/FQG_data/bartpho/bart_dataset_19_9/",
                                 type=str,
                                 help='directory '
                                      'of dataset')
    training_parser.add_argument("--pretrained_path", default="vinai/bartpho-word", type=str,
                                 help="Tokenizer name or path")
    training_parser.add_argument("--generation_num_beams", default=5, type=int)
    training_parser.add_argument("--metric_for_best_model", default="bleu", type=str)
    training_parser.add_argument("--warm_up_ratio", default=0.1, type=float, help="Warm-up ratio for scheduler")
    training_parser.add_argument('--generation_max_length', default=200, type=int,
                                 help='maximum question token number')
    training_parser.add_argument('--input_max_length', default=512, type=int,
                                 help='maximum context token number')

    training_parser.add_argument('--use_pointer', default=True, type=bool,
                                 help='whether or not using pointer generator in training model')

    return training_parser.parse_args()


class ModelTrainer:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metric = load_metric("sacrebleu")
        self.sent_limit = config.pipeline_input_max_length
        self.tokenizer: AutoTokenizer
        if not config.training_restore_checkpoint:
            # add new special tokens
            self.tokenizer = AutoTokenizer.from_pretrained(config.pipeline_pretrained_path)
            new_special_tokens = json.load(open(SPECIAL_TOKENS_PATH))[
                                     SPECIAL_TOKENS] + [e.name for e in QuestionType]
            special_tokens_dict = {
                "additional_special_tokens": [f"<{id_.upper()}>" for id_ in list(set(new_special_tokens))]}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            # ==========================================

            self.model = BartPhoPointer.from_pretrained(
                config.pipeline_pretrained_path,
                model_config=config
            )
            # resize embeddings of encoder and decoder
            self.model.resize_token_embeddings(len(self.tokenizer.get_vocab()))
            # ==========================================
            # model = BartPhoPointer.from_pretrained(config.training_restore_folder,
            #                                        model_config=training_config.__dict__)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.training_restore_folder)
            self.model = BartPhoPointer.from_pretrained(config.training_restore_folder,
                                                        model_config=config.__dict__)
        self.model.to(config.pipeline_device)
        self.train_dataset, self.valid_dataset, self.test_dataset = FQG_dataset.get_dataset(
            config=PipelineConfig(**vars(config)),
            tokenizer=self.tokenizer,
            added_new_special_tokens=True
        )

    def collate_(self, data):
        batch = {}
        for feature_key in data[0].keys():
            # prepare for tag feature ids
            if isinstance(data[0][feature_key], dict):
                batch[feature_key] = {
                    k: torch.tensor(np.array([ele[feature_key][k].numpy() for ele in data])) if k not in [
                        "ques_type_id",
                        MODEL_QUESTION_TYPE_INPUT] else [
                        ele[feature_key][k] for ele in data]
                    for k in data[0][feature_key].keys()
                }
            elif feature_key in ["entity_weight", "p_gen_weight"]:
                batch[feature_key] = torch.vstack([ele[feature_key] for ele in data])
            # input ids vs attention mask
            else:
                batch[feature_key] = torch.tensor([ele[feature_key][:self.sent_limit] for ele in data])
        return batch

    def compute_metrics(self, eval_predictions):
        predictions, labels = eval_predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # decoded_predictions = [ele.split("?")[0] + "?" for ele in decoder]

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        print("_______________________________________")
        print("_______________________________________")
        print("PRED : ", decoded_predictions)
        print("LABEL: ", decoded_labels)
        print("_______________________________________")
        print("_______________________________________")

        result = self.metric.compute(predictions=decoded_predictions, references=[[ele] for ele in decoded_labels])
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def main(self):
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.training_output_dir,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_strategy=IntervalStrategy.STEPS,
            per_device_train_batch_size=self.config.training_batch_size,
            per_device_eval_batch_size=self.config.training_batch_size,
            learning_rate=self.config.training_learning_rate,
            weight_decay=self.config.training_weight_decay,
            save_total_limit=self.config.training_save_total_limit,
            num_train_epochs=self.config.training_num_epochs,
            predict_with_generate=True,
            gradient_accumulation_steps=self.config.training_gradient_accumulation_steps,
            eval_steps=self.config.training_eval_steps,
            fp16=True,
            push_to_hub=False,
            # dataloader_pin_memory=True,
            dataloader_num_workers=2,
            logging_dir=self.config.training_logging_dir,
            logging_strategy=IntervalStrategy.STEPS,
            logging_steps=self.config.training_logging_steps,
            save_steps=self.config.training_save_steps,
            logging_first_step=False,
            optim=OptimizerNames.ADAMW_TORCH,
            generation_max_length=self.config.pipeline_output_max_length,
            generation_num_beams=self.config.training_generation_num_beams,
            load_best_model_at_end=True,
            metric_for_best_model=self.config.training_metrics,
            warmup_ratio=self.config.training_warm_up_ratio,
            lr_scheduler_type=SchedulerType.COSINE_WITH_RESTARTS
        )

        _trainer = QGTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            data_collator=self.collate_,
            compute_metrics=self.compute_metrics,
            tokenizer=self.train_dataset.tokenizer
        )

        _trainer.train(resume_from_checkpoint=self.config.training_restore_checkpoint)
        # print(trainer.evaluate(valid_dataset))
        # with mlflow.start_run():


if __name__ == "__main__":
    training_config = model_config()

    config = PipelineConfig(
        training_output_dir=training_config.output_dir,
        training_use_pointer=training_config.use_pointer,
        pipeline_input_max_length=training_config.input_max_length,
        pipeline_output_max_length=training_config.generation_max_length,
        training_warm_up_ratio=training_config.warm_up_ratio,
        training_metrics=training_config.metric_for_best_model,
        training_generation_num_beams=training_config.generation_num_beams,
        pipeline_pretrained_path=training_config.pretrained_path,
        pipeline_dataset_folder=training_config.dataset_folder,
        training_restore_folder=training_config.restore_folder,
        training_restore_checkpoint=training_config.restore_checkpoint,
        training_num_epochs=training_config.num_train_epochs,
        pipeline_device=training_config.device,
        training_save_steps=training_config.save_steps,
        training_logging_steps=training_config.logging_steps,
        training_eval_steps=training_config.eval_steps,
        training_gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        training_learning_rate=training_config.lr,
        training_save_total_limit=training_config.save_total_limit,
        training_weight_decay=training_config.weight_decay,
        training_batch_size=training_config.batch_size,
        training_logging_dir=training_config.logging_dir,
    )
    trainer = ModelTrainer(config=config)
    trainer.main()
