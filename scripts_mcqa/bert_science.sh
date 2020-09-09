SCRIPT_LOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LIB_LOC="$( dirname $SCRIPT_LOC )"
echo "working from ${LIB_LOC}"

#####################################
# TRAIN SCIENCE MODEL FROM SCRATCH  #
#####################################


## LOCATION OF SCIENCE DATA
SCIENCE_TRAIN=${SRC}/scripts_mcqa/etc/science_data

MODEL=${BERT_SCIENCE}/bert_science

## train a bert model on science

## UNCOMMENT TO RUN 
python -m mcqa_code.arc_mc \
       --data_dir ${SCIENCE_TRAIN} \
       --model_type bert --model_name_or_path bert-large-uncased-whole-word-masking \
       --do_train --do_eval --do_lower_case \
       --max_seq_length 110 --learning_rate 2e-5 \
       --num_train_epochs 3 --save_steps -1 --seed 42 \
       --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 8 \
       --num_choices 5 --no_save_checkpoints \
       --override \
       --output_dir _experiments/bert_large_collected_science

# ## evaluate provided science model on probe datasets


##########################################
# EVALUATE ON EXISTING PROBES ZERO-SHOT  #
##########################################

DEFS=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/definitions/
HYPERNYMS=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/hypernymy/
HYPONYMS=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/hyponymy/
SYNONYMS=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/synonymy/
DICTIONARY=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/dictionary_qa/

## create directory first 
## mkdir _experiments/bert

## datasets
EVAL="dev" # change to `test` if desired

## UNCOMMENT TO RUN 
for dataset in ${DEFS} ${HYPERNYMS} ${HYPONYMS} ${DICTIONARY} ${SYNONYMS}; do
    ###
    dname=$(basename ${dataset})
    ##
    echo "runing ${run_output}"
    python -m mcqa_code.arc_mc \
        --data_dir ${dataset}/ \
        --model_type bert \
        --model_name_or_path bert-large-uncased-whole-word-masking \
        --do_eval \
        --do_lower_case --max_seq_length 100  \
        --save_steps -1 --seed 42 \
        --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 32 \
        --num_choices 5 \
        --no_save_checkpoints \
        --run_existing ${MODEL} \
        --dev_name ${dataset}/${EVAL}.jsonl \
        --output_dir _experiments/bert/${dname}_${EVAL} \
        --override
done


######################################
# FINAL EVALUATION WITH INOCULATION  #
######################################

### NOTE: these are the final inoculation experiments (after inoculation fine-tuning)
## with the best found parameters (i.e., standard parameters such as lr, as well as amount
## of probe data to inoculate on, whether to also tune jontly on additional science data)

## make sure output directory exists
## mkdir ${LIB_LOC}/_experiments/bert/

for cond in "hyponymy 3000 0.00001 extra_1" "synonymy 3000 0.00001 extra_1" "definitions 3000 0.00001 extra_2" "dictionary_qa 3000 0.00001 NONE" "hypernymy 1000 0.00001 extra_2"; do
    set -- ${cond}
    dname=${1}
    dosage=${2}
    lr=${3}
    extra=${4}

    dataset=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data
    run_output=_experiments/bert/science_${dname}_test
    data_dir=${dataset}/inoculate_${dosage}_${extra}
    if [ ${extra} == "NONE" ]
    then
        data_dir=${dataset}/inoculate_${dosage}
    fi

    log_out=/tmp/experiment_run.log

    python -m mcqa_code.arc_mc \
        --data_dir ${data_dir}/ \
        --model_type bert \
        --model_name_or_path bert-large-uncased-whole-word-masking \
        --do_train --do_eval \
        --do_lower_case --max_seq_length 100 --learning_rate ${lr}  \
        --save_steps -1 --seed 42 \
        --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 32 \
        --num_choices 5 \
        --no_save_checkpoints \
        --inoculate ${MODEL} \
        --dev_name ${dataset}/dev.jsonl \
        --output_dir ${run_output} \
        --remove_model \
        --override \
        --dev_name2 ${dataset}/test.jsonl &>${log_out}
    ####
    mv ${log_out} ${run_output}/bert_log.log
done



#############################
# ## sketch of inoculation  #
#############################

### tuning on done on a smaller dev set called `small_dev.jsonl`


## SCIENCE_DEV=${SCIENCE_TRAIN}/large_dev.jsonl

# for dataset in ${HYPERNYMS} ${HYPONYMS} ${DEFS} ${SYNONYMY} ${DICT}; do
#     for dosage in 3000 2000 500 250 100; do
#         for lr in 2e-5 5e-5 0.00001 0.00005; do
#             for extra in 1 2 3; do
#                 ###
#                 dname=$(basename ${dataset})
#                 run_output=experiments/bert/science_${dname}_inoculate_${dosage}_${lr}_extra_${extra}

#                 ##
#                 echo "runing ${run_output}"
#                 python -m mcqa_code.arc_mc \
#                     --data_dir ${dataset}/inoculate_${dosage}_extra_${extra}/ \
#                     --model_type bert \
#                     --model_name_or_path bert-large-uncased-whole-word-masking \
#                     --do_train --do_eval \
#                     --do_lower_case --max_seq_length 100 --learning_rate ${lr}  \
#                     --save_steps -1 --seed 42 \
#                     --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 32 \
#                     --num_choices 5 \
#                     --no_save_checkpoints \
#                     --inoculate ${SCIENCE_MODEL} \
#                     --dev_name ${dataset}/inoculate_${dosage}_extra_${extra}/small_dev.jsonl \
#                     --output_dir ${run_output} \
#                     --remove_model \
#                     --dev_name2 ${SCIENCE_DEV}
#             done
#         done
#     done
# done


##########################
# ANSWER ONLY BASELINES  #
##########################

DEFS=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/definitions/
HYPERNYMS=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/hypernymy/
HYPONYMS=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/hyponymy/
SYNONYMS=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/synonymy/
DICTIONARY=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/dictionary_qa/

output=${LIB_LOC}/experiments/bert
for dataset in ${DEF} ${HYPERNYMS} ${HYPONYMS} ${SYNONYMS} ${DICTIONARY}; do
    dname=$(basename ${dataset})
    tmp_out=/tmp/_experiment_out.log
    
    python -m mcqa_code.arc_mc \
        --data_dir ${dataset}/question_only \
        --model_type bert \
        --model_name_or_path bert-large-uncased-whole-word-masking \
        --do_train --do_eval \
        --do_lower_case --max_seq_length 100 --learning_rate 2e-5  \
        --save_steps -1 --seed 42 \
        --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 32 \
        --num_choices 5 \
        --no_save_checkpoints \
        --dev_name ${dataset}/qo_test.jsonl \
        --dev_name2 ${dataset}/qo_test.jsonl \
        --output_dir ${output}/bert_${dname}_question_only_with_test \
        --override \
        --remove_model &>${tmp_out}
    ## move over log
    mv ${tmp_out} ${output}/bert_${dname}_question_only_with_test
