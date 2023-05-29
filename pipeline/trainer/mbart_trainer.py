import argparse
import json
import os
import numpy as np
import torch
import logging

from datasets import load_metric
from transformers import SchedulerType, AutoTokenizer, AutoConfig
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import OptimizerNames
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from common.common_keys import *
from common.config import QuestionType, PipelineConfig
from common.constants import *
from pipeline.dataset_constructor.dataloader import FQG_dataset
from pipeline.trainer.model.bartpho import BartPhoPointer
from pipeline.trainer.seq2seq_trainer import QGTrainer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class ModelTrainer:
    def __init__(self, config: PipelineConfig = None):
        self.config = config if config is not None else PipelineConfig()
        self.metric = load_metric("sacrebleu")
        if self.config.training_restore_checkpoint:
            if self.config.pipeline_onnx:
                pass
            self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(self.config.training_restore_folder)
            self.model = BartPhoPointer.from_pretrained(self.config.training_restore_folder, model_config=self.config)
            self.model_config = AutoConfig.from_pretrained(self.config.training_restore_folder)
        else:
            self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(self.config.pipeline_pretrained_path)
            self.add_new_special_tokens()
            # ==========================================

            self.model = BartPhoPointer.from_pretrained(self.config.pipeline_pretrained_path, model_config=self.config)
            # resize embeddings of encoder and decoder
            self.model.resize_token_embeddings(len(self.tokenizer.get_vocab()))
            self.model_config = AutoConfig.from_pretrained(self.config.pipeline_pretrained_path)

        self.model.to(self.config.pipeline_device)
        self.train_dataset, self.valid_dataset, self.test_dataset = None, None, None
        # FQG_dataset.get_dataset(
        #     config=self.config,
        #     tokenizer=self.tokenizer
        # )
        self.logger = logging.getLogger(__name__)
        self.init_trainer()

    def init_trainer(self):
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
            fp16=self.config.pipeline_device == "cuda",
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

        self.trainer = QGTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            data_collator=self.collate_,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer
        )

    def add_new_special_tokens(self):
        new_special_tokens = json.load(open(SPECIAL_TOKENS_PATH))[SPECIAL_TOKENS] + [e.name for e in QuestionType]
        special_tokens_dict = {
            "additional_special_tokens": [f"<{id_.upper()}>" for id_ in list(set(new_special_tokens))]}
        self.tokenizer.add_special_tokens(special_tokens_dict)

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
                batch[feature_key] = torch.tensor(
                    [ele[feature_key][:self.config.pipeline_input_max_length] for ele in data])
        return batch

    def compute_metrics(self, eval_predictions):
        predictions, labels = eval_predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        self.logger.info("_______________________________________")
        self.logger.info("_______________________________________")
        self.logger.info("PRED : ", decoded_predictions)
        self.logger.info("LABEL: ", decoded_labels)
        self.logger.info("_______________________________________")
        self.logger.info("_______________________________________")

        result = self.metric.compute(predictions=decoded_predictions, references=[[ele] for ele in decoded_labels])
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def run(self, mode: str = "train"):
        if hasattr(self, mode):
            func = getattr(self, mode)
            func()

    def temp(self):
        onnx_config = BartPhoPointerOnnxConfig(self.model_config)
        onnx_path = f"{self.config.training_output_dir}/onnx/model.onnx"

        from transformers.onnx import export
        from pathlib import Path
        onnx_inputs, _ = export(self.tokenizer, self.model, onnx_config, onnx_config.default_onnx_opset,
                                Path(onnx_path))
        print(onnx_inputs)

    def export_onnx(self):
        example = self.tokenizer(
            "<WHERE> Khi áp_dụng cho các loài chim , thuật_ngữ \" đặc_hữu \" đề_cập đến bất_kỳ loài nào chỉ có ở một khu_vực địa_lý cụ_thể . Không có giới_hạn cho một khu_vực , sẽ không sai nếu nói rằng tất_cả các loài chim là đặc_hữu của Trái_Đất . Nhưng trong thực_tế , khu_vực lớn nhất mà thuật_ngữ này được sử_dụng phổ_biến là <ANS> một quốc_gia ( ví_dụ như đặc_hữu của Việt_Nam ) hoặc . một vùng hay tiểu_vùng địa_lý_học động_vật ( ví_dụ như đặc_hữu của Đông_Nam_Á </ANS> ) .",
            return_tensors="pt", padding="max_length", max_length=self.config.pipeline_input_max_length,
            truncation=True)

        example = example.to(self.config.pipeline_device)
        example[ENTITY_WEIGHT] = torch.ones((1, self.config.pipeline_output_max_length)).to(self.config.pipeline_device)
        example[ATTENTION_MASK] = example.pop(ATTENTION_MASK).to(dtype=example[ENTITY_WEIGHT].dtype)
        example.pop("token_type_ids")

        onnx_path = f"{self.config.training_output_dir}/onnx"
        if not os.path.isdir(onnx_path):
            os.makedirs(onnx_path)
        # example = self.collate_([example])
        self.model.eval()
        torch.onnx.export(
            self.model,
            # args=(*list(example.values()),),
            args=(example[INPUT_IDS], example[ENTITY_WEIGHT], example[ATTENTION_MASK]),
            f=f"{onnx_path}/model.onnx",
            input_names=[INPUT_IDS, ENTITY_WEIGHT, ATTENTION_MASK],
            output_names=["logits"],
            opset_version=11,
            dynamic_axes={
                INPUT_IDS: {0: "batch", 1: "input_length"},
                ENTITY_WEIGHT: {0: "batch", 1: "output_length"},
                ATTENTION_MASK: {0: "batch", 1: "input_length"}
            }
        )

    def test_onnx(self):
        example = self.tokenizer(
            "<WHERE> Khi áp_dụng cho các loài chim , thuật_ngữ \" đặc_hữu \" đề_cập đến bất_kỳ loài nào chỉ có ở một khu_vực địa_lý cụ_thể . Không có giới_hạn cho một khu_vực , sẽ không sai nếu nói rằng tất_cả các loài chim là đặc_hữu của Trái_Đất . Nhưng trong thực_tế , khu_vực lớn nhất mà thuật_ngữ này được sử_dụng phổ_biến là <ANS> một quốc_gia ( ví_dụ như đặc_hữu của Việt_Nam ) hoặc . một vùng hay tiểu_vùng địa_lý_học động_vật ( ví_dụ như đặc_hữu của Đông_Nam_Á </ANS> ) .",
            padding="max_length", max_length=self.config.pipeline_input_max_length, truncation=True)
        import numpy as np
        for e in [INPUT_IDS, ATTENTION_MASK]:
            example[e] = [example[e]]
        example[ENTITY_WEIGHT] = [[1] * self.config.pipeline_input_max_length]
        example.pop("token_type_ids")
        # example.pop(ATTENTION_MASK)
        example[ATTENTION_MASK] = example.pop(ATTENTION_MASK)

        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if self.config.pipeline_device != "cpu":
            providers.append("CUDAExecutionProvider")
        session = ort.InferenceSession(f"{self.config.training_output_dir}/onnx/model.onnx", opts, providers=providers)
        output = session.run(output_names=["logits"], input_feed=dict(example))[0]
        abcd = np.argmax(output, axis=-1)
        print(abcd)
        k = self.tokenizer.batch_decode(abcd, skip_special_tokens=True)
        print(k)

    def train(self):
        self.trainer.train(resume_from_checkpoint=self.config.training_restore_checkpoint)
        # if self.config.pipeline_onnx:
        #     print("Exporting ...")
        #     self.export_onnx()

    def eval(self):
        self.logger.info(self.trainer.evaluate(eval_dataset=self.test_dataset))


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run(mode="train")
