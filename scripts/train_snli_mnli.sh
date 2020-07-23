

## bert trained on snli only 

python -m bert_scripts.sen_pair_classification \
       --task_name polarity \
       --bert_model bert-base-uncased \
       --output_dir bert_scripts/runs/challenge_train_hard_dev_snli/ \
       --data_dir generate_challenge/data/snli_train/ \
       --do_train --do_eval --do_lower_case \
       --max_seq_length 128 \
       --learning_rate 2e-5

# bert trained on smnli+mnli

python -m bert_scripts.sen_pair_classification \
       --task_name polarity \
       --bert_model bert-base-uncased \
       --output_dir bert_scripts/runs/challenge_train_hard_dev_snli_mnli/ \
       --data_dir generate_challenge/data/snli_mnli_train/ \
       --do_train --do_eval \
       --do_lower_case \
       --max_seq_length 128 \
       --learning_rate 2e-5
