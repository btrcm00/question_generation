import argparse
import datetime
import pytz
from flask import Flask, request, jsonify

from common.common_keys import *
from common.config import PipelineConfig, Config
from pipeline.inference.base_pipeline import QuestionSampler
# from inference.refactor_pipeline import QuestionSampler
from common.constants import *
from pipeline.trainer.model.bartpho import BartPhoPointer

app = Flask(__name__)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def api_config():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--folder_checkpoint", default=OUTPUT_PATH + "/checkpoint/checkpoint_bart_16_3/", type=str)
    parser.add_argument("--folder_checkpoint",
                        default=f"{INFERENCE_PATH}/checkpoint/bartpho_all_chatgptdata_all_steps_partial_no_questype_27_3",
                        type=str)
    parser.add_argument('--input_max_length', default=512, type=int,
                        help='maximum context token number')
    parser.add_argument('--output_max_length', default=256, type=int,
                        help='maximum context token number')
    parser.add_argument('--model_device', default="cpu", type=str)
    parser.add_argument('--parallel_input_processing', action='store_true')
    parser.add_argument('--sampling_return_entity', action='store_true')
    parser.add_argument('--inference_batch_size', default=4, type=int)
    parser.add_argument("--training_logging_dir", default=f"{OUTPUT_PATH}/logging/base", type=str,
                        help="Tensorboard Logging Folder")
    parser.add_argument("--api_port", default=35234, type=int)

    return parser.parse_known_args()


config, _ = api_config()
deploy_time = datetime.datetime.now(pytz.timezone("Asia/SaiGon")).strftime('%H:%M:%S_%d/%m/%Y')
pipeline_config = PipelineConfig(
    training_output_dir=config.folder_checkpoint,
    pipeline_input_max_length=config.input_max_length,
    pipeline_output_max_length=config.output_max_length,
    pipeline_device=config.model_device,
    sampling_parallel_input_processing=config.parallel_input_processing,
    sampling_inference_batch_size=config.inference_batch_size,
    training_logging_dir=config.training_logging_dir,
    sampling_return_entity=config.sampling_return_entity
)
sampler = QuestionSampler(pipeline_config)


@app.route('/question_gen/bart_predict', methods=['POST'])
def bart_predict():
    data = request.json
    for ele in [MODEL_INPUT, MODEL_QUESTION_TYPE_INPUT]:
        if ele not in data.keys():
            return {'suggest_reply': 'ERROR NOT ENOUGH PARAM', 'id_job': '', 'check_end': True}

    num_beams = data.get("num_beams", 1)
    num_return_sequences = data.get("num_return_sequences", 1)
    output = sampler.predict(data, num_beams=num_beams, num_return_sequences=num_return_sequences)

    return jsonify({
        "checkpoint": sampler.best_checkpoint_path.split("/")[-1],
        "latest-deploy-time": deploy_time,
        "predict": output.dict()
    })


@app.route('/question_gen/bart_samplings', methods=['POST'])
def bart_samplings():
    data = request.json
    for ele in [PASSAGE]:
        if ele not in data.keys():
            return {'suggest_reply': 'ERROR NOT ENOUGH PARAM', 'id_job': '', 'check_end': True}

    output, processed_passage = sampler.sampling(**data)
    return jsonify({
        "checkpoint": sampler.best_checkpoint_path.split("/")[-1],
        "latest-deploy-time": deploy_time,
        "processed_passage": processed_passage,
        **output.dict()
    })
    # return jsonify({**_output, **output.dict()})


def create_app():
    return app


if __name__ == '__main__':
    app.run(host=Config.service_host, port=config.api_port)
