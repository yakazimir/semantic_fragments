## set data directory here

## set your version here 
HYP_ONLY="/home/kyler/third_party/hypothesis-only-NLI"

SCRIPT_LOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LIB_LOC="$( dirname $SCRIPT_LOC )"
DATA=${LIB_LOC}
echo "getting into the hypothesis only library..."
cd ${HYP_ONLY}

####################
# HYPOTHESIS ONLY  #
####################

## logic
for cond in  generate_challenge/hard_fragment/ generate_challenge/simple_fragment/
do
    echo "RUNNING "+${cond}
    python src/train.py --embdfile data/embds/glove.840B.300d.txt \
           --train_lbls_file ${DATA}/${cond}/hypothesis_only/cl_challenge_train_lbl_file \
           --val_lbls_file ${DATA}/${cond}/hypothesis_only/cl_challenge_val_lbl_file \
           --train_src_file ${DATA}/${cond}/hypothesis_only/cl_challenge_train_source_file \
           --val_src_file ${DATA}/${cond}/hypothesis_only/cl_challenge_val_source_file \
           --test_lbls_file ${DATA}/${cond}/hypothesis_only/cl_challenge_test_lbl_file \
           --test_src_file ${DATA}/${cond}/hypothesis_only/cl_challenge_test_source_file \
           --outputdir ${DATA}/${cond}/hypothesis_only/results \
           --n_epochs 15 
done


for cond in  generate_challenge/brazil_fragments/boolean_coordination_only generate_challenge/brazil_fragments/quantifier generate_challenge/brazil_fragments/counting generate_challenge/brazil_fragments/simple_negation_only generate_challenge/brazil_fragments/comparative generate_challenge/brazil_fragments/conditional
do
    echo "RUNNING "+${cond}
    python src/train.py --embdfile data/embds/glove.840B.300d.txt \
           --train_lbls_file ${DATA}/${cond}/hypothesis_only/cl_challenge_train_lbl_file \
           --val_lbls_file ${DATA}/${cond}/hypothesis_only/cl_challenge_val_lbl_file \
           --train_src_file ${DATA}/${cond}/hypothesis_only/cl_challenge_train_source_file \
           --val_src_file ${DATA}/${cond}/hypothesis_only/cl_challenge_val_source_file \
           --test_lbls_file ${DATA}/${cond}/hypothesis_only/cl_challenge_test_lbl_file \
           --test_src_file ${DATA}/${cond}/hypothesis_only/cl_challenge_test_source_file \
           --outputdir ${DATA}/${cond}/hypothesis_only/results \
           --n_epochs 15 
done

# echo "now running premise-only baselines"
for cond in  generate_challenge/hard_fragment/ generate_challenge/simple_fragment/
do
    echo "RUNNING "+${cond}
    python src/train.py --embdfile data/embds/glove.840B.300d.txt \
           --train_lbls_file ${DATA}/${cond}/premise_only/cl_challenge_train_lbl_file \
           --val_lbls_file ${DATA}/${cond}/premise_only/cl_challenge_val_lbl_file \
           --train_src_file ${DATA}/${cond}/premise_only/cl_challenge_train_source_file \
           --val_src_file ${DATA}/${cond}/premise_only/cl_challenge_val_source_file \
           --test_lbls_file ${DATA}/${cond}/premise_only/cl_challenge_test_lbl_file \
           --test_src_file ${DATA}/${cond}/premise_only/cl_challenge_test_source_file \
           --outputdir ${DATA}/${cond}/premise_only/results --n_epochs 15 
done
# hard fragment   => 59.8
# simple fragment => 55.0 



for cond in generate_challenge/brazil_fragments/boolean_coordination_only generate_challenge/brazil_fragments/quantifier generate_challenge/brazil_fragments/counting generate_challenge/brazil_fragments/simple_negation_only generate_challenge/brazil_fragments/comparative generate_challenge/brazil_fragments/conditional
do
    echo "RUNNING "+${cond}
    python src/train.py --embdfile data/embds/glove.840B.300d.txt \
           --train_lbls_file ${DATA}/${cond}/premise_only/cl_challenge_train_lbl_file \
           --val_lbls_file ${DATA}/${cond}/premise_only/cl_challenge_val_lbl_file \
           --train_src_file ${DATA}/${cond}/premise_only/cl_challenge_train_source_file \
           --val_src_file ${DATA}/${cond}/premise_only/cl_challenge_val_source_file \
           --test_lbls_file ${DATA}/${cond}/premise_only/cl_challenge_test_lbl_file \
           --test_src_file ${DATA}/${cond}/premise_only/cl_challenge_test_source_file \
           --outputdir ${DATA}/${cond}/premise_only/results --n_epochs 15 
done

echo "now running premise+hypothesis baselines"

for cond in generate_challenge/brazil_fragments/boolean_coordination_only generate_challenge/brazil_fragments/quantifier generate_challenge/brazil_fragments/counting generate_challenge/brazil_fragments/simple_negation_only generate_challenge/brazil_fragments/comparative generate_challenge/brazil_fragments/conditional
do
    echo "RUNNING "+${cond}
    python src/train.py --embdfile data/embds/glove.840B.300d.txt \
           --train_lbls_file ${DATA}/${cond}/hypothesis_premise/cl_challenge_train_lbl_file \
           --val_lbls_file ${DATA}/${cond}/hypothesis_premise/cl_challenge_val_lbl_file \
           --train_src_file ${DATA}/${cond}/hypothesis_premise/cl_challenge_train_source_file \
           --val_src_file ${DATA}/${cond}/hypothesis_premise/cl_challenge_val_source_file \
           --test_lbls_file ${DATA}/${cond}/hypothesis_premise/cl_challenge_test_lbl_file \
           --test_src_file ${DATA}/${cond}/hypothesis_premise/cl_challenge_test_source_file \
           --outputdir ${DATA}/${cond}/hypothesis_premise/results --n_epochs 15 
done
# hard fragment   => 66.2
# simple fragment => 52.1 

for cond in  generate_challenge/hard_fragment/ generate_challenge/simple_fragment/
do
    echo "RUNNING "+${cond}
    python src/train.py --embdfile data/embds/glove.840B.300d.txt \
           --train_lbls_file ${DATA}/${cond}/hypothesis_premise/cl_challenge_train_lbl_file \
           --val_lbls_file ${DATA}/${cond}/hypothesis_premise/cl_challenge_val_lbl_file \
           --train_src_file ${DATA}/${cond}/hypothesis_premise/cl_challenge_train_source_file \
           --val_src_file ${DATA}/${cond}/hypothesis_premise/cl_challenge_val_source_file \
           --test_lbls_file ${DATA}/${cond}/hypothesis_premise/cl_challenge_test_lbl_file \
           --test_src_file ${DATA}/${cond}/hypothesis_premise/cl_challenge_test_source_file \
           --outputdir ${DATA}/${cond}/hypothesis_premise/results --n_epochs 15 
done

## my numbers

#logic fragments

# hypothesis-only/premise-only/hypothesis+premise
# bool        => 40.9 47.5 46.8
# quantifier  => 70.0 52.0 87.2
# counting    => 61.8 65.0 77.0
# negation    => 33.8 33.8 33.8
# conditional => 55.6 33.7 33.7
# comparative => 33.8 33.8 33.8 
