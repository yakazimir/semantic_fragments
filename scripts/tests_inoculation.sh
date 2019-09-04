## switch up to
SCRIPT_LOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LIB_LOC="$( dirname $SCRIPT_LOC )"
cd ${LIB_LOC}

final_path='generate_challenge/final_tests'
mnli='generate_challenge/final_tests/decomp/mnli_dev.json'

## models from https://github.com/nelson-liu/inoculation-by-finetuning
DECOMP_MODEL=https://s3-us-west-2.amazonaws.com/ai2-nelsonl/adversarial/models/decomposable_attention_original_mismatched/model.tar.gz
DECOMP_CONFIG=scripts/configs/decomp_config.json
ESIM_MODEL=https://s3-us-west-2.amazonaws.com/ai2-nelsonl/adversarial/models/esim_original_mismatched/model.tar.gz
ESIM_CONFIG=scripts/configs/esim_config.json
OUT='_experiments'

#!!!!!!!!!!!!!!
## change environment 
#conda deactivate
#conda activate allennlp


#######################
# ## DECOMP test loop #
#######################

CONFIG=${DECOMP_CONFIG}
MODEL=${DECOMP_MODEL}
name=decomp

for cond in "0.0004 quantifier" "0.0004 conditional" "0.001 simple_fragment" "0.0001 simple_negation_only" "0.0004 counting" "0.0001 comparative" "0.0001 hard_fragment" "0.0001 boolean_coordination_only";
do
    set -- ${cond}
    LR=${1}
    ecase=${2}

    ## data
    train=${final_path}/decomp/${ecase}/train.json
    dev=${final_path}/decomp/${ecase}/dev.json
    test=${final_path}/decomp/${ecase}/test.json

    ## call allennlp
    allennlp fine-tune -m ${MODEL} \
         -o '{"trainer": {"optimizer": {"lr": "'${LR}'"}, "num_serialized_models_to_keep": 0}, "train_data_path": "'${train}'","validation_data_path" : "'${dev}'", "test_data_path": "'${test}'"}' \
         -c ${CONFIG} \
         -s ${OUT}/${ecase}_${name}_final_test/

    ## remove associated files (takes up a lot of space otherwise)
    rm -rf ${OUT}/${ecase}_${name}_final_test/metrics*
    rm -rf ${OUT}/${ecase}_${name}_final_test//model.tar.gz
    rm -rf ${OUT}/${ecase}_${name}_final_test/best*
    rm -rf ${OUT}/${ecase}_${name}_final_test/training_state*
    rm -rf ${OUT}/${ecase}_${name}_final_test/model_state*
done


## INOCULATION
# for cond in generate_challenge/simple_fragment generate_challenge/hard_fragment  generate_challenge/brazil_fragments/boolean_coordination_only generate_challenge/brazil_fragments/quantifier generate_challenge/brazil_fragments/counting generate_challenge/brazil_fragments/simple_negation_only generate_challenge/brazil_fragments/conditional generate_challenge/brazil_fragments/comparative
# do
#     for num_examples in 50 100 250 500 1000 1500 2000 3000; do
#         for learning_rate in 0.000001 0.00001 0.0001 0.0004 0.001 0.01; do
#             ## inoculate
# 	    name=`echo $(basename $cond)`
# 	    train=${cond}/inoculate_${num_examples}/train.json
# 	    dev=${cond}/inoculate_${num_examples}/dev.json
# 	    mnli_mm_test=${cond}/inoculate_${num_examples}_retest_mnli/dev.json
# 	    #cat ${train}
#             allennlp fine-tune -m https://s3-us-west-2.amazonaws.com/ai2-nelsonl/adversarial/models/decomposable_attention_original_mismatched/model.tar.gz -o '{"trainer": {"optimizer": {"lr": '${learning_rate}'}, "num_serialized_models_to_keep": 0}, "train_data_path": "'${train}'","validation_data_path" : "'${dev}'", "test_data_path": "'${mnli_mm_test}'"}' -c bin/decomp_config.json -s task_spec_fine_tune/${name}_decomp_mnli_${num_examples}_challenge_${learning_rate}/
# 	    echo "DELETING STUFF"
# 	    rm -rf task_spec_fine_tune/${name}_decomp_mnli_${num_examples}_challenge_${learning_rate}/metrics*
# 	    rm -rf task_spec_fine_tune/${name}_decomp_mnli_${num_examples}_challenge_${learning_rate}/model.tar.gz
# 	    rm -rf task_spec_fine_tune/${name}_decomp_mnli_${num_examples}_challenge_${learning_rate}/model_state*
# 	    rm -rf task_spec_fine_tune/${name}_decomp_mnli_${num_examples}_challenge_${learning_rate}/best*
# 	    rm -rf -rf task_spec_fine_tune/${name}_decomp_mnli_${num_examples}_challenge_${learning_rate}/training_state*
# 	    rm -rf -rf task_spec_fine_tune/${name}_decomp_mnli_${num_examples}_challenge_${learning_rate}/log/*
# 	    rm -rf -rf task_spec_fine_tune/${name}_decomp_mnli_${num_examples}_challenge_${learning_rate}/vocabulary/*
#         done
#     done
# done


###########
# ## ESIM #
###########

CONFIG=${ESIM_CONFIG}
MODEL=${ESIM_MODEL}
name=esim

for cond in "0.0001 quantifier" "0.0001 conditional" "0.0001 simple_fragment" "0.0004 simple_negation_only" "0.0004 counting" "0.0004 comparative" "0.0001 hard_fragment" "0.0004 boolean_coordination_only";
do
    set -- ${cond}
    LR=${1}
    ecase=${2}

    ## data
    train=${final_path}/esim/${ecase}/train.json
    dev=${final_path}/esim/${ecase}/dev.json
    test=${final_path}/esim/${ecase}/test.json

    ## call allennlp
    allennlp fine-tune -m ${MODEL} \
         -o '{"trainer": {"optimizer": {"lr": "'${LR}'"}, "num_serialized_models_to_keep": 0}, "train_data_path": "'${train}'","validation_data_path" : "'${dev}'", "test_data_path": "'${test}'"}' \
         -c ${CONFIG} \
         -s ${OUT}/${ecase}_${name}_final_test/

    ## remove associated files (takes up a lot of space otherwise)
    rm -rf ${OUT}/${ecase}_${name}_final_test/metrics*
    rm -rf ${OUT}/${ecase}_${name}_final_test//model.tar.gz
    rm -rf ${OUT}/${ecase}_${name}_final_test/best*
    rm -rf ${OUT}/${ecase}_${name}_final_test/training_state*
    rm -rf ${OUT}/${ecase}_${name}_final_test/model_state*
done


## INOCULATION SKETCH
# for cond in generate_challenge/simple_fragment generate_challenge/hard_fragment generate_challenge/brazil_fragments/boolean_coordination_only generate_challenge/brazil_fragments/quantifier generate_challenge/brazil_fragments/counting generate_challenge/brazil_fragments/simple_negation_only generate_challenge/brazil_fragments/conditional generate_challenge/brazil_fragments/comparative
# do
#     for num_examples in 50 100 250 500 1000 1500 2000 3000; do ## dataset sizes
#         for learning_rate in 0.000001 0.00001 0.0001 0.0004 0.001 0.01; do ## learning rates (allennlp does early stopping, so we are finding optimal # iterations)
#             ## inoculate
# 	    name=`echo $(basename $cond)`
# 	    train=${cond}/inoculate_${num_examples}/train.json
# 	    dev=${cond}/inoculate_${num_examples}/dev.json
# 	    mnli_mm_test=${cond}/inoculate_${num_examples}_retest_mnli/dev.json
# 	    #cat ${train}
#             allennlp fine-tune \
#                      -m https://s3-us-west-2.amazonaws.com/ai2-nelsonl/adversarial/models/esim_original_mismatched/model.tar.gz \
#                      -o '{"trainer": {"optimizer": {"lr": '${learning_rate}'}, "num_serialized_models_to_keep": 0}, "train_data_path": "'${train}'","validation_data_path" : "'${dev}'", "test_data_path": "'${mnli_mm_test}'"}' \
#                      -c bin/esim_config.json \
#                      -s task_spec_fine_tune/${name}_esim_mnli_${num_examples}_challenge_${learning_rate}/

# 	    echo "DELETING STUFF"
# 	    rm -rf task_spec_fine_tune/${name}_esim_mnli_${num_examples}_challenge_${learning_rate}/metrics*
# 	    rm -rf task_spec_fine_tune/${name}_esim_mnli_${num_examples}_challenge_${learning_rate}/model.tar.gz
# 	    rm -rf task_spec_fine_tune/${name}_esim_mnli_${num_examples}_challenge_${learning_rate}/model_state*
# 	    rm -rf task_spec_fine_tune/${name}_esim_mnli_${num_examples}_challenge_${learning_rate}/best*
# 	    rm -rf -rf task_spec_fine_tune/${name}_esim_mnli_${num_examples}_challenge_${learning_rate}/training_state*
# 	    rm -rf -rf task_spec_fine_tune/${name}_esim_mnli_${num_examples}_challenge_${learning_rate}/log/*
# 	    rm -rf -rf task_spec_fine_tune/${name}_esim_mnli_${num_examples}_challenge_${learning_rate}/vocabulary/*
#         done
#     done
# done


#########
# BERT  #
#########

##!!!!!!!!!!!!!!!!!!1
### switch environment
#conda deactivate
#conda activate nli

## seed=42 (default)

name=bert
bert_model='bert_scripts/bert_models/mnli_snli/pytorch_model.bin'
bert_config='bert_scripts/bert_models/mnli_snli/bert_config.json'

for cond in "0.000001 42 quantifier" "0.000001 42 conditional" "2e-5 42 simple_fragment" "0.000001 42 simple_negation_only" "2e-5 27 counting" "0.000001 42 comparative" "0.000001 42 hard_fragment" "0.000001 42 boolean_coordination_only";
do
    set -- ${cond}
    LR=${1}
    seed=${2} ## random seed, which sometimes makes a difference 
    ecase=${3}

    ## data
    train=${final_path}/bert/${ecase}/

    python -m bert_scripts.sen_pair_classification \
           --task_name polarity \
           --bert_model bert-base-uncased \
           --output_dir ${OUT}/${ecase}_${name}_final_test/ \
           --data_dir ${train} \
           --do_eval --do_lower_case \
           --max_seq_length 128 \
           --learning_rate ${LR} \
           --seed ${seed} \
           --inoculate ${bert_model} \
           --bert_config ${bert_config} \
           --do_train \
           --remove_model
done


##################
# # INOCULATION  #
##################

## this is just a sketch of what was done, in this case for MNLI mistmached. 

# for cond in generate_challenge/simple_fragment generate_challenge/hard_fragment generate_challenge/brazil_fragments/boolean_coordination_only generate_challenge/brazil_fragments/quantifier generate_challenge/brazil_fragments/counting generate_challenge/brazil_fragments/simple_negation_only generate_challenge/brazil_fragments/conditional generate_challenge/brazil_fragments/comparative; do
#     DATADIR=${cond}
#     NAME=`basename ${cond}`
#     for i in 50 100 250 500 1000 1500 2000 3000 # the different dosages 
#     do
#         ## run on challenge sets and get scores 
#         python -m bert_scripts.sen_pair_classification \
#                --task_name polarity \
#                --bert_model bert-base-uncased \
#                --output_dir bert_scripts/runs/${NAME}_inoculate_snli_mnli_${i}_challenge \
#                --data_dir ${DATADIR}/inoculate_${i} \ ## replace ${DATADIR}/inoculate_${i}_retest_mnli to retest on original MNLI mistmatched. 
#                --do_eval --do_lower_case --max_seq_length 128 \
#                --learning_rate 0.000001 \ ## we had an additional looked that tested 2e-5, 0.000001, 0.0000001; 0.000001 is almost always the best 
#                --inoculate ${bert_model} \
#                --bert_config ${bert_config} \
#                --do_train \
#                --seed \ # we experimented with 42 27 919; we found this to be fairly stable 
#                --remove_model
#         ## trains the same model again then restests on orignal task
#     done
# done

