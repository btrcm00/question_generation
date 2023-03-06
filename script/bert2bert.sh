#! /bin/sh
CUDA_VISIBLE_DEVICES=-1 python trainer/bert2bert_trainer.py \
--output_dir FQG_output/checkpoint24_6_newdata \
--batch_size 2 \
--device cuda \
--logging_dir logging/logging24_6_newdata/ \
--lr 2e-5 \
--weight_decay 0.01 \
--save_total_limit 4 \
--gradient_accumulation_steps 1 \
--num_train_epochs 30 \
--restore_checkpoint False \
--restore_folder FQG_output \
--dataset_folder dataset/FQG_data/29_6_splitdata/ \
--encoder_pretrained_path vinai/phobert-base \
--decoder_pretrained_path vinai/phobert-base \
--generation_num_beams 5 \
--metric_for_best_model bleu \
--warm_up_ratio 0.1 \
--generation_max_length 200 \
--input_max_length 256 \
--eval_steps 5000 \
--logging_steps 5000 \
--save_steps 5000 \

