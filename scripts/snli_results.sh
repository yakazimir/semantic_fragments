### reproducing the SNLI test results

SCRIPT_LOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LIB_LOC="$( dirname $SCRIPT_LOC )"
echo ${LIB_LOC}
echo "switching to top directory "
cd ${LIB_LOC}

## the experimet output directory 
EXP=_experiments


## dependencies
# breaking_json=breaking_nli_datasets/dataset.jsonl
# bert_models=bert_scripts/bert_models
## models 

################################
# SNLI TEST  AND BREAKING NLI  #
################################

## switch environment (if needed)

#!!!!!!!!!!!!
#conda deactivate
#conda activate nli

#########
# BERT  #
#########

## BERT + snli => snli test  
python -m bert_scripts.sen_pair_classification --task_name polarity \
       --bert_model bert-base-uncased  \
       --output_dir ${EXP}/bert_snli_snli_test \
       --data_dir /home/kyler/projects/NLI/generate_challenge/orig_test/snli \
       --do_eval \
       --do_lower_case \
       --max_seq_length 128 \
       --run_existing bert_scripts/bert_models/snli/pytorch_model.bin \
       --bert_config bert_scripts/bert_models/snli/bert_config.json

## BERT + snli => breaking nli
python -m bert_scripts.sen_pair_classification --task_name polarity \
       --bert_model bert-base-uncased  \
       --output_dir ${EXP}/bert_snli_breaking_nli_test \
       --data_dir breaking_nli_dataset/ \
       --do_eval \
       --do_lower_case \
       --max_seq_length 128 \
       --run_existing bert_scripts/bert_models/snli/pytorch_model.bin \
       --bert_config bert_scripts/bert_models/snli/bert_config.json

## out-of-the-box performance on challenge sets

## monotonicity fragments : should get 54.8% 58.8%, respectively 
for cond in generate_challenge/simple_fragment generate_challenge/hard_fragment; do
    python -m bert_scripts.sen_pair_classification --task_name polarity \
       --bert_model bert-base-uncased  \
       --output_dir ${EXP}/bert_snli_${cond} \
       --data_dir ${cond}/test/ \
       --do_eval \
       --do_lower_case \
       --max_seq_length 128 \
       --run_existing bert_scripts/bert_models/snli/pytorch_model.bin \
       --bert_config bert_scripts/bert_models/snli/bert_config.json
done

## logic fragments
for cond in generate_challenge/brazil_fragments/boolean_coordination_only generate_challenge/brazil_fragments/quantifier generate_challenge/brazil_fragments/counting generate_challenge/brazil_fragments/simple_negation_only generate_challenge/brazil_fragments/conditional generate_challenge/brazil_fragments/comparative; do
    python -m bert_scripts.sen_pair_classification --task_name polarity \
       --bert_model bert-base-uncased  \
       --output_dir ${EXP}/bert_snli_${cond} \
       --data_dir ${cond}/test/ \
       --do_eval \
       --do_lower_case \
       --max_seq_length 128 \
       --run_existing bert_scripts/bert_models/snli/pytorch_model.bin \
       --bert_config bert_scripts/bert_models/snli/bert_config.json
done
## expected results on last run (Sept.3 2019):
#boolean       => 61.3
#quantifier    => 35.1
#counting      => 57.0
#negation      => 49.1
#conditional   => 36.3
#comparative   => 38.3
## avg: 46.1 


## BERT + snli + mnli => snli test
python -m bert_scripts.sen_pair_classification --task_name polarity \
       --bert_model bert-base-uncased  \
       --output_dir ${EXP}/bert_snli_mnli_snli_test  \
       --data_dir /home/kyler/projects/NLI/generate_challenge/orig_test/snli \
       --do_eval \
       --do_lower_case \
       --max_seq_length 128 \
       --run_existing bert_scripts/bert_models/mnli_snli/pytorch_model.bin \
       --bert_config bert_scripts/bert_models/mnli_snli/bert_config.json

## BERT + snli + mnli => breaking NLI 
python -m bert_scripts.sen_pair_classification --task_name polarity \
       --bert_model bert-base-uncased \
       --output_dir ${EXP}/bert_snli_mnli_breaking_nli_test/ \
       --data_dir breaking_nli_dataset/ \
       --do_eval \
       --do_lower_case \
       --max_seq_length 128 \
       --learning_rate 2e-5 \
       --run_existing bert_scripts/bert_models/mnli_snli/pytorch_model.bin \
       --bert_config bert_scripts/bert_models/mnli_snli/bert_config.json

## enumerate through fragments
## monotonicity fragments
for cond in generate_challenge/simple_fragment generate_challenge/hard_fragment; do
    python -m bert_scripts.sen_pair_classification --task_name polarity \
       --bert_model bert-base-uncased  \
       --output_dir ${EXP}/bert_snli_mnli_${cond} \
       --data_dir ${cond}/test/ \
       --do_eval \
       --do_lower_case \
       --max_seq_length 128 \
       --run_existing bert_scripts/bert_models/mnli_snli/pytorch_model.bin \
       --bert_config bert_scripts/bert_models/mnli_snli/bert_config.json
done
## simple => 61.7
## hard => 64.


## logic fragments
for cond in generate_challenge/brazil_fragments/boolean_coordination_only generate_challenge/brazil_fragments/quantifier generate_challenge/brazil_fragments/counting generate_challenge/brazil_fragments/simple_negation_only generate_challenge/brazil_fragments/conditional generate_challenge/brazil_fragments/comparative; do
    python -m bert_scripts.sen_pair_classification --task_name polarity \
       --bert_model bert-base-uncased  \
       --output_dir ${EXP}/bert_snli_mnli_${cond} \
       --data_dir ${cond}/test/ \
       --do_eval \
       --do_lower_case \
       --max_seq_length 128 \
       --run_existing bert_scripts/bert_models/mnli_snli/pytorch_model.bin \
       --bert_config bert_scripts/bert_models/mnli_snli/bert_config.json
done
#boolean       => 43.2
#quantifier    => 33.8
#counting      => 55.1
#negation      => 39.2
#conditional   => 66.3
#comparative   => 46.5
## avg: 47.3



# on negation challenge

#######################################
# ## ESIM and decomposable attention  #
#######################################

## switch environment (if needed) 

#!!!!!!!!!!!!
#conda deactivate
#conda activate allennlp

## ESIM + snli (+ elmo) => snli test  
allennlp evaluate  https://s3-us-west-2.amazonaws.com/allennlp/models/esim-elmo-2018.05.17.tar.gz \
         https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_test.jsonl
## ESIM + snli (+ elmo) => breaking nli test
allennlp evaluate https://s3-us-west-2.amazonaws.com/allennlp/models/esim-elmo-2018.05.17.tar.gz  \
         breaking_nli_dataset/dataset.jsonl

## monotonicity fragments
for cond in generate_challenge/simple_fragment generate_challenge/hard_fragment; do
    allennlp evaluate https://s3-us-west-2.amazonaws.com/allennlp/models/esim-elmo-2018.05.17.tar.gz \
             ${cond}/test.json
done
## simple : 65.6
## hard : 60.1 
## avg. 62.8

## logic fragments
for cond in generate_challenge/brazil_fragments/boolean_coordination_only generate_challenge/brazil_fragments/quantifier generate_challenge/brazil_fragments/counting generate_challenge/brazil_fragments/simple_negation_only generate_challenge/brazil_fragments/conditional generate_challenge/brazil_fragments/comparative; do
    allennlp evaluate https://s3-us-west-2.amazonaws.com/allennlp/models/esim-elmo-2018.05.17.tar.gz \
             ${cond}/test.json
done
#boolean       => 40.5
#quantifier    => 58. 
#counting      => 49.9
#negation      => 33.
#conditional   => 49.5
#comparative   => 35.2
## avg:  44.3


## decomposable attention + snli (+ elmo) => snli test
allennlp evaluate  https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz \
         https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_test.jsonl
# decomposable attention + snli (+ elmo) => breaking NLI 
allennlp evaluate  https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz \
         breaking_nli_dataset/dataset.jsonl

## monotonicity fragments
for cond in generate_challenge/simple_fragment generate_challenge/hard_fragment; do
    allennlp evaluate https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz ${cond}/test.json
done
## simple : 51.
## hard : 45.8
## avg. 48.4

## logic fragments
for cond in generate_challenge/brazil_fragments/boolean_coordination_only generate_challenge/brazil_fragments/quantifier generate_challenge/brazil_fragments/counting generate_challenge/brazil_fragments/simple_negation_only generate_challenge/brazil_fragments/conditional generate_challenge/brazil_fragments/comparative; do
    allennlp evaluate https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz ${cond}/test.json
done
#boolean       => 37.6
#quantifier    => 45.3
#counting      => 49.3
#negation      => 32.8
#conditional   => 52.6
#comparative   => 35.2
## avg: 42.1

