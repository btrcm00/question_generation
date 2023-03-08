import argparse

from common.common_keys import *
from common.constants import *
from common.config import PipelineConfig
from dataset_constructor.prepare_dataset import DatasetConstructor
from trainer.mbart_trainer import ModelTrainer


class QGPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.dataset_constructor: DatasetConstructor
        self.trainer: ModelTrainer

    def run(self):
        self.dataset_constructor = DatasetConstructor(constructor_config=self.config)
        self.dataset_constructor.run()

        self.trainer = ModelTrainer(config=self.config)
        self.trainer.run()


def pipeline_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_output_dir', default=f"{OUTPUT_PATH}/checkpoint/checkpoint_bart_7_3/", type=str,
                        help='output directory')
    parser.add_argument("--training_logging_dir", default=f"{OUTPUT_PATH}/logging/logging_bart_7_3/", type=str,
                        help="Tensorboard Logging Folder")
    parser.add_argument('-b', '--training_batch_size', default=4, type=int, help='mini-batch size (default: 32)')
    parser.add_argument("--training_weight_decay", default=0.01, type=float, help="Weight Decay")
    parser.add_argument("--training_save_total_limit", default=5, type=int, help="Total Limit")
    parser.add_argument("--training_learning_rate", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--training_gradient_accumulation_steps", default=2, type=int, help="Gradient Accumulation "
                                                                                            "Steps")
    parser.add_argument("--training_eval_steps", default=2000, type=int, help="Num steps per evaluation ")

    parser.add_argument("--training_logging_steps", default=2000, type=int, help="Num Steps per logging ")
    parser.add_argument("--training_save_steps", default=2000, type=int, help="Num Steps per saving ")
    parser.add_argument("--pipeline_device", default="cuda", type=str, help="Visible Cuda")
    parser.add_argument("--training_num_epochs", default=10, type=int, help="Num Epochs")
    parser.add_argument("--training_restore_checkpoint", action='store_true')
    parser.add_argument("--training_restore_folder", default=f"{INFERENCE_PATH}/checkpoint/bartpho_pointer_22_9/",
                        type=str)
    parser.add_argument("--training_generation_num_beams", default=5, type=int)
    parser.add_argument("--training_metrics", default="bleu", type=str)
    parser.add_argument("--training_warm_up_ratio", default=0.1, type=float, help="Warm-up ratio for scheduler")

    parser.add_argument('--training_use_pointer', action='store_false',
                        help='whether or not using pointer generator in training model')

    parser.add_argument("--constructor_num_of_threads", default=1, type=int)
    parser.add_argument("--pipeline_pretrained_path", default="vinai/bartpho-word", type=str,
                        help="Tokenizer name or path")
    parser.add_argument("--pipeline_special_tokens_path", default=SPECIAL_TOKENS_PATH, type=str)
    parser.add_argument("--pipeline_input_max_length", default=512, type=str)
    parser.add_argument("--pipeline_output_max_length", default=256, type=str)
    parser.add_argument("--pipeline_dataset_folder", default=f"{TRAING_DATASET_FOLDER}/new_dataset",
                        help='directory of dataset')
    parser.add_argument("--constructor_ratio", default=[0.9, 0.05, 0.05], type=list)

    return parser.parse_args()


if __name__ == "__main__":
    config = pipeline_config()
    config = PipelineConfig(**vars(config))
    pipeline = QGPipeline(config=config)
    pipeline.run()
