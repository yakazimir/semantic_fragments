This file describes the different scripts needed for reproducing the
results from the probing multiple-choice models (for the main
transformer models, other models can be provided as needed). 

Step 1: set up environments
==================

I use [*conda*](https://docs.conda.io/en/latest/miniconda.html) throughout (see setup instructions there), and set up the following two environments: 

I set up two separate environments for `RoBERTa` and `BERT`
respectively.

Starting with `RoBERTa`,
```
conda create --name roberta --file etc/env/roberta_env.txt
pip install -r etc/env/requirements-roberta.txt
cd etc/
unzip allennlp-transf-exp1.zip
cd allennlp-transf-exp1
pip install --editable .
export allennlp=`pwd`
cd ../../
```
In the last step, it seps up a special branch of
[*AllenNLP*](https://allennlp.org/ that we) used throughout, and I set
up an environment variable `allennlp` that will get used in some of
the scriptst. 

Now the environment for `BERT` (which uses [*Huggingface*](https://github.com/huggingface/transformers))
```
conda create -n mcqa --file etc/env/mcqa_env.txt
pip install -r etc/env/requirements-mcqa.txt
```

Step 2: prepare everything (get associated data, models,etc..)
==================

A generic version of the data is included in
`/path/to/this_repo/data_mcqa/` in a json format (see details there
for the structure of these files). We also include the raw files
listed below, and the original science pre-training data.

| dataset | link | where to put here |
|:-------------:|:------------------------------------------------------------:|:-------------------------:|
| science training|[*here*](https://drive.google.com/file/d/1UviPzsUWRT4mnBC-su8ucMgJUCYitB5y/view?usp=sharing) | `path/to/repo/scripts_mcqa/etc`
| full wordnet data |[*here*](https://drive.google.com/file/d/1doWpEtglL3wH2Nl_yh7S9OvI8uOto59i/view?usp=sharing) | `path/to/repo/scripts_mcqa/etc`


Below are some of the checkpoints/pre-trained models used in the
scripts:

| model | description | link | location |
|:-------------:|:------------------------------------------------------------:|:-------------------------:|:-------------------------:|
| roberta weights | RACE roberta weights used to pre-train science  | [*here*](https://drive.google.com/file/d/1coZL9i8vIL-wBwa0JFgtTEnU-7wcMLw_/view?usp=sharing) | `path/to/repo/scripts_mcqa/etc/models` |
| bert science       | BERT modele trained on science | [*here*](https://drive.google.com/drive/folders/1-oSSNisgSZOM1aWKGwglLk_Tnc1WhL1r?usp=sharing) | `path/to/repo/scripts_mcqa/etc/models`|
| robert science | roberta science model trained on science |  [*here*](https://drive.google.com/drive/folders/17tqnNNrFtlQQgvqojXl9gDovOkmmGgpR?usp=sharing) | `path/to/repo/script_mcqa/etc/models`|

Step 3: run some of the scripts 
==================

`{bert,roberta}_science.sh` includes the details of how to implement
the science models and how to re-produce the main results from
the paper. Details also about how the inoculation process works are
included there. 
