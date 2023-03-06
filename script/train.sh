#! /bin/sh
CUDA_VISIBLE_DEVICES=-1 python trainer/marian_trainer.py \
--output_dir FQG_output/ \
--batch_size 26 \
--device cuda \
--logging_dir logging/logging15_6_add_bias/ \
--lr 2e-5 \
--weight_decay 0.01 \
--save_total_limit 4 \
--gradient_accumulation_steps 1 \
--num_train_epochs 30 \
--restore_checkpoint False \
--restore_folder FQG_output \
--dataset_folder dataset/FQG_data/6_6/ \
--marian_pretrained_path Helsinki-NLP/opus-mt-en-vi \
--generation_num_beams 5 \
--metric_for_best_model bleu \
--warm_up_ratio 0.1 \
--generation_max_length 200 \
--input_max_length 512 \
--eval_steps 500 \
--logging_steps 500 \
--save_steps 500 \

