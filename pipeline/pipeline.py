from common.config import PipelineConfig
from dataset_constructor.prepare_dataset import DatasetConstructor
from trainer.mbart_trainer import ModelTrainer


class QGPipeline:
    def __init__(self, config: PipelineConfig):
        self.dataset_constructor = DatasetConstructor(constructor_config=config)
        self.trainer = ModelTrainer(config=config)

    def pipeline(self):
        self.dataset_constructor.main()
        self.trainer.main()


if __name__ == "__main__":
    pass
