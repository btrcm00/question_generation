import argparse
from flask import Flask, request, jsonify

from common.common_keys import *
from common.config import PipelineConfig, Config
from inference.sampling_pipeline import QuestionSampler
# from inference.refactor_pipeline import QuestionSampler
from common.constants import *
from model.bartpho import BartPhoPointer

app = Flask(__name__)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def mbart_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_checkpoint", default=INFERENCE_PATH + "/checkpoint/bartpho_pointer_22_9/", type=str)
    parser.add_argument('--input_max_length', default=512, type=int,
                        help='maximum context token number')
    parser.add_argument('--output_max_length', default=256, type=int,
                        help='maximum context token number')
    parser.add_argument('--model_device', default="cpu", type=str)
    parser.add_argument('--parallel_input_processing', action='store_true')
    parser.add_argument('--inference_batch_size', default=4, type=int)
    return parser.parse_args()


bart_config = mbart_config()
config = PipelineConfig(
    training_output_dir=bart_config.folder_checkpoint,
    pipeline_input_max_length=bart_config.input_max_length,
    pipeline_output_max_length=bart_config.output_max_length,
    pipeline_device=bart_config.model_device,
    sampling_parallel_input_processing=bart_config.parallel_input_processing,
    sampling_inference_batch_size=bart_config.inference_batch_size
)
sampler = QuestionSampler(config)

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
        "bart_generated_question": output
    })


@app.route('/question_gen/bart_samplings', methods=['POST'])
def bart_samplings():
    data = request.json
    for ele in [PASSAGE]:
        if ele not in data.keys():
            return {'suggest_reply': 'ERROR NOT ENOUGH PARAM', 'id_job': '', 'check_end': True}

    output = sampler.sampling(**data)
    return jsonify({
        "bart_samplings": output
    })


def create_app():
    return app


if __name__ == '__main__':
    app.run(host=Config.service_host, port=35234)
