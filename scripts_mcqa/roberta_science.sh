SCRIPT_LOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LIB_LOC="$( dirname $SCRIPT_LOC )"
echo "working from ${LIB_LOC}"

## make a roberta output 
##mkdir _experiments/roberta
ROBERTA_OUT="_experiments/roberta_mcqa"
## script for generating the config for each run
JSON_BUILD=${LIB_LOC}/mcqa_code/_make_inoc_json.py


### SWITCH TO ALLENNLP BRANCH
echo "SWitching to AllenNLP branch"
cd ${allennlp}

## switch to conda environment
## conda activate roberta

########################################
# ## TRAIN SCIENCE MODEL FROM SCRATCH  #
########################################

## check that file pointers are correct

python -m allennlp.run train ${LIB_LOC}/scripts_mcqa/etc/models/roberta_science_config.json \
        -s ${LIB_LOC}/_experiments/roberta_large_collected_science  \
        --file-friendly-logging &

## NOTE: in the json config, the dats size is slightly off (and was based on an earlier iteration
# of experiments); I re-ran it again with the correct number and the numbers went down slightly, so
## I stuck with this.

###########################################################
# TEST ROBERTA ON ZERO-SHOT ON PROBES WITH SCIENCE MODEL  #
###########################################################

DEFS=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/definitions/
HYPERNYMS=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/hypernymy/
HYPONYMS=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/hyponymy/
SYNONYMS=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/synonymy/
DICTIONARY=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/dictionary_qa/

for dataset in ${DEFS} ${HYPERNYMS} ${HYPONYMS} ${DICTIONARY} ${SYNONYMS}; do
    dname=$(basename ${dataset})
    out_dir=${ROBERTA_OUT}/roberta_${dname}_dev
    echo "now running ${dataset}"

    #echo "trying to make ${out_dir}"
    if [ ! -d ${out_dir} ]; then
        mkdir -p ${out_dir};
    fi
    tmp_log=/tmp/tmp_log.log
    
    python -m allennlp.run evaluate_custom \
       --evaluation-data-file ${dataset}/dev.jsonl \
       --metadata-fields "id,question_text,choice_text_list,correct_answer_index,answer_index,label_logits,label_probs,choice_context_list" \
       --output-file ${out_dir}/eval-output.jsonl \
       --cuda-device 0 \
       --overrides '{"iterator":{"batch_size":32}}' \
       ${ROBERTA_SCIENCE} &> ${tmp_log}

    mv ${tmp_log} ${out_dir}/

    ## test
    out_dir=${ROBERTA_OUT}/roberta_${dname}_test
    tmp_log=/tmp/tmp_log.log

    echo "trying to make ${out_dir}"
    if [ ! -d ${out_dir} ]; then
        mkdir -p ${out_dir};
    fi

    env CUDA_VISIBLE_DEVICES=0 python -m allennlp.run evaluate_custom \
       --evaluation-data-file ${dataset}/test.jsonl \
       --metadata-fields "id,question_text,choice_text_list,correct_answer_index,answer_index,label_logits,label_probs,choice_context_list" \
       --output-file ${out_dir}/eval-output.jsonl \
       --cuda-device 0 \
       --overrides '{"iterator":{"batch_size":32}}' \
       ${ROBERTA_SCIENCE} &> ${tmp_log}

    cp ${tmp_log} ${out_dir}/
    rm -rf ${tmp_log}
done


##########################################################
# FINAL TEST/DEV INOCULATION RUNS FOR SCORING ROBERTA ON #
# PROBES                                                 #
##########################################################

## eval data 
SCIENCE_EVAL=${LIB_LOC}/scripts_mcqa/etc/science_data/larger_dev.jsonl


## best hyper parameters
EVAL="dev"

## loop through all datasets with optimal settings 
for cond in "synonymy 3000 0.00001 NONE" "definitions 3000 0.00001 extra_1" "hyponymy 3000 0.00001 extra_2" "dictionary_qa 3000 0.00001 extra_2" "hypernymy 1000 0.00001 extra_2"; do
    set -- ${cond}
    dname=${1}
    dosage=${2}
    lr=${3}
    learning_rate=${lr}
    extra=${4}

    ## dataset and output 
    dataset=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/${dname}
    out_dir=${ROBERTA_OUT}/science_${dname}_${EVAL}

    
    dataset_dir=${dataset}/inoculate_${dosage}_${extra}
    if [ ${extra} == "NONE" ]
    then
        dataset_dir=${dataset}/inoculate_${dosage}
    fi


    ## compute the size of the training data
    train_details=`wc -l ${dataset_dir}/train.jsonl`
    set -- ${train_details}
    ## the actual train amount 
    amount=${1}

    ## build the config 
    python ${JSON_BUILD} ${amount} ${learning_rate} ${dataset_dir} ${SCIENCE_EVAL} /tmp/ 2
    tmp_json=/tmp/_tmp_json_2.json
    out_log=/tmp/out_log_1.log

    ## fine-tune
    python -m allennlp.run fine-tune \
                        -m ${ROBERTA_SCIENCE} \
                        -c ${tmp_json} \
                        -s ${out_dir} &> ${out_log}

    echo "moving the temporary json"
    mv ${tmp_json} ${out_dir}/config.json
    mv ${out_log} ${out_dir}/out_log.log

    echo "cleaning a bit now"
    rm -rf ${out_dir}/metrics*
    rm -rf ${out_dir}/best*
    rm -rf ${out_dir}/vocabulary/*
    rm -rf ${out_dir}/model_state*
    rm -rf ${out_dir}/training_state*

    tmp_log=/tmp/tmp_log_1.log
    echo "run evaluating ${dosage}"
    python -m allennlp.run evaluate_custom \
	--evaluation-data-file ${dataset}/${EVAL}.jsonl \
	--metadata-fields "id,question_text,choice_text_list,correct_answer_index,answer_index,label_logits,label_probs,choice_context_list" \
	--output-file ${out_dir}/eval-output-${EVAL}.jsonl \
	--cuda-device 0 \
	--overrides '{"iterator":{"batch_size":32}}' \
	${out_dir}/model.tar.gz &> ${tmp_log}

    echo "removing model"
    rm -rf ${out_dir}/model.tar.gz
    mv ${tmp_log} ${out_dir}/dev_log.log     
done


############################
# ## sketch of inoculation #
############################

# NOTE: below is a sketch of inoculation. The goal is to find
# the dosage (amount or probe data to train on), learning rate, etc.. amount of
# extra science data to add that give highest aggregrate score
# on SCIENCE_EVAL and probe data. Given the large size of the science devs
# we due tuning on a smaller subset called `small_dev.jsonl`

# DEFS=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/definitions/
# HYPERNYMS=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/hypernymy/
# HYPONYMS=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/hyponymy/
# SYNONYMS=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/synonymy/
# DICTIONARY=${LIB_LOC}/scripts_mcqa/etc/wordnet_exp_data/dictionary_qa/

# for dataset_stuff in "${HYPERNYMS} hypernymy" "${HYPONYMS} hyponyms" "${DEFS} definitions" "${SYNONYMY} synonymy" "${DICT} dictionary_qa"; do
#     set -- ${dataset_stuff}
#     dataset=${1}
#     name=${2}
#     for dosage in  "100" "250" "500" "1000" "2000" "3000"; do    ## amount of inoculation data
#         for add_amount in 1 2 3 ; do  ## added science amount. Remove this loop to 
#             dataset_dir=${dataset}/inoculate_${dosage}_extra_${add_amount}/
#             for learning_rate in "2e-5" "5e-5" 0.00001 0.00005; do ## learning rate
#                 ## generate the json config
#                 python ${JSON_BUILD} ${dosage} ${learning_rate} ${dataset_dir} ${SCIENCE_EVAL} /tmp/ 2

#                 # ## temporary json 
#                 tmp_json=/tmp/_tmp_json_2.json
#                 cat ${tmp_json}

#                 # ## experiment output
#                 out_dir=${ROBERTA_OUT}/science_${name}_inoculate_${dosage}_${learning_rate}_extra_${add_amount} 
                
#                 # ## fine-tune
#                 env CUDA_VISIBLE_DEVICES=1 python -m allennlp.run fine-tune \
#                         -m ${SCIENCE} \
#                         -c ${tmp_json} \
#                         -s ${out_dir}
            
#                 # ## removing json
#                 echo "moving the temporary json"
#                 ##
#                 mv ${tmp_json} ${out_dir}/config.json

#                 # ## remov auxiliary model files
#                 rm -rf ${out_dir}/metrics*
#                 rm -rf ${out_dir}/best*
#                 rm -rf ${out_dir}/vocabulary/*
#                 rm -rf ${out_dir}/model_state*
#                 rm -rf ${out_dir}/training_state*
#                 rm -rf ${out_dir}/model.tar.gz
#             done
#         done
#     done
# done
