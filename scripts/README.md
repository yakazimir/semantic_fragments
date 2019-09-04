This file describes the different scripts needed for reproducing the results from our semantic fragments paper.

Step 1: set up environments
==================

I use conda throughout[https://docs.conda.io/en/latest/miniconda.html] (see setup instructions there), and set up the following two environments: 

For setting up BERT related dependencies (all of which uses Huggingface[https://github.com/huggingface/pytorch-transformers]) do (note that I am using a slightly older version of `pytorch-transformers`):

    conda create -n nli python=3.6.7
    conda activate nli
    conda install -c anaconda numpy=1.15.4
    conda install -c anaconda scipy=1.2.1
    conda install scikit-learn=0.20.3 
    #pip install pytorch-pretrained-bert==0.6.1 ## (old version of huggingface before name change)
    pip install pytorch-transformers==0.6.1 # (will install pytorch, my version is 1.1.0)

For allennlp[https://allennlp.org/], do the following:

    conda create -n allennlp python=3.6.7
    conda activate allennlp 
    pip install allennlp==0.8.3 tqdm # (will install pytorch, my version is 1.1.0, like above)


Step 2: prepare everything (get associated data, models,etc..)
==================

Download the following BERT models, unpack then and put into `path_to_semantic/fragments/bert_scripts/` [https://drive.google.com/open?id=1n-8leJWmBE6gZcA7Dh1GB51hyNSS9qEH]

Download the associated experiment files (largely data files chopped up for inoculation, etc..)
here[https://drive.google.com/file/d/1-kSKE95uP92YM_Bw1qdRjrbeYQxwR0-X/view?usp=sharing]
and unpack at the top directory. 


hypothesis/premise-only baselines
==================

`./scripts/hypothesis_only.sh`

notes: requires modified version of Poliak et al.'s original code[https://github.com/azpoliak/hypothesis-only-NLI], included in the top directory: `hypothesis-only-NLI_mod.zip`; follow instructions inside to download the associated data and word embeddings. Given that the code is a bit old now (i.e., it uses Python2 and an older version of pytorch), I ran all experiments on CPU since I had difficulty getting it set up for GPU.

SNLI, breaking NLI and fragments test results
==================

`./scripts/snli_results.sh`

notes: Our BERT model was trained on the version of SNLI and MNLI distributed in the GLUE benchmark[https://gluebenchmark.com/]. In the case of SNLI, some of the training instances were not parsed correctly (see [https://github.com/PetrochukM/PyTorch-NLP/issues/26#issuecomment-445452286]), so there is some noise here (though we didn't see any difference in results from other versions of SNLI). We then used the SNLI test v1.0, which is identical to this one here[https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_test.jsonl] (testing examples without labels are removed and not considered in final results, as it standard).

The original breaking NLI datase, which is distributed with our data collection, can be found here: [https://github.com/BIU-NLP/Breaking_NLI]

MNLI dev/challenge final tests  and inoculation 
-------------------------------

`./scripts/test_inoculation.sh`

notes: runs through all models (i.e., BERT, esim, decomposable-attention); shows how to reproduce the final results and how to do inoculation. 
