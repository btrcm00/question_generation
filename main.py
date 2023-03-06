# import json
# import os
# from tqdm import tqdm
# import regex as re
#
# mapp = ""
# folder = "/TMTAI/KBQA/minhbtc/ACS-QG/QASystem/TextualQA/QuestionAnswering_Generation/dataset/sampling_dataset/wiki_sampling/original"
# all_files = os.listdir(folder)
# print(len(all_files))
#
# with open("/TMTAI/KBQA/minhbtc/ACS-QG/QASystem/TextualQA/QuestionAnswering_Generation/dataset/sampling_dataset/wiki_sampling/id_link.txt", "r", encoding="utf8") as f:
#     mapp = f.readlines()
#
# print(len(mapp))
# mapp = [ele[:-1] for ele in mapp if "[]" not in ele]
# mapping = {}
# output = {}
#
# for line in tqdm(mapp):
#     qid, title = line.split("|||")
#     title = title[2:-2]
#     data = None
#     if not os.path.isfile(f"{folder}/{qid}.txt"):
#         continue
#     with open(f"{folder}/{qid}.txt", "r", encoding="utf8") as f:
#         data = f.read()
#     data = re.split(r"\n+", data)
#     data = [ele.strip() for ele in data if len(ele) > 100 and not len(ele)==len(ele.encode())]
#     if data:
#         output[title] = data
#
# print(len(list(output.keys())))
# json.dump(output, open("es_search.json", "w", encoding="utf8"), indent=4, ensure_ascii=False)
#
# import os
# from random import random, randint
# from mlflow import log_metric, log_param, log_artifacts
#
# if __name__ == "__main__":
#     # Log a parameter (key-value pair)
#     log_param("param1", randint(0, 100))
#
#     # Log a metric; metrics can be updated throughout the run
#     log_metric("foo", random())
#     log_metric("foo", random() + 1)
#     log_metric("foo", random() + 2)
#
#     # Log an artifact (output file)
#     if not os.path.exists("outputs"):
#         os.makedirs("outputs")
#     with open("outputs/test.txt", "w") as f:
#         f.write("hello world!")
#     log_artifacts("outputs")
