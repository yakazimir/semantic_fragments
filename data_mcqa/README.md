Probing Using Expert Knowledge
================================

This sub-directory contains the *complete* 5 challenge datasets reported in our paper here[https://arxiv.org/abs/1912.13337], which were built automically from various knowledge resources such as [WordNet](https://wordnet.princeton.edu/) (via the `nltk` [library](https://www.nltk.org/howto/wordnet.html)) and dictionary resources such as the [GNU Collaborative International Dictionary of English (GCIDE)](https://gcide.gnu.org.ua/)

Each datasets contains a `{train,dev,test}.jsonl`, each containing the rough JSON format:

```json
{"question" : {"stem" : ".." ## question stem
               "choices" : "..." ## multiple choices
            }
            "answerKey" : ".." ## correct answer
            "notes" : ".." ## meta information about instance 
}
```

An exact version of the datasets used in the experiments can be accessed using the information in `etc/mcqa_scripts/README`. The following are built from WordNet: `definitions` (testing word definitions), `hypernymy/hyponymy` (testing ISA reasoning), `synonymy` (testing word sets + definitional reasoning), and `dictionary_qa` (testing word sense + definitions) are built from GCIDE. The latter 4 are what we call in the paper `WordNetQA` and the last dataset is what we call `DictionaryQA`. 

How Models were Trained
-----------------------

We largely used these datasets to do the following two types of experiments (as detailed in `scripts_mcqa/README.md`, .. similar story for the semantic fragments work detailed in `scripts/README.md`):

1. **Zero-shot evaluation** on QA models i.e., given models trained on standard QA benchmark, we tested to see how well do these models solve each `dev` and `test` set in our different datasets.  

2. **Model inoculation** (see [here](https://github.com/nelson-liu/inoculation-by-finetuning), i.e., given the same standard NLI models trained on QA benchmarks, can we continue training the models to perform well on these new tasks (without hurting their performance on the benchmarks) by giving them small **dosages** of these new tasks? In other words, can we teach these models to be good at our new tasks without totally re-training them from scratch? (see `scripts_mcqa/{bert,roberta}_science.sh`)

In the latter experiments, the number of training examples we use for inoculation is an additional hyper-parameter adjusted during the tuning phase. In other words, training on the full 3k training examples for each dataset does not necessarily yield the best results, and might seriously overfit, especially on the datasets that involve non-iid testing (as mentioned above). See all details in `scripts_mcqa/README` with pointers to the experiment files. 

**note** In virtue of these datasets being built from KBs and dictionaries, they are quite noisy. We dealt with this in two ways: 1) we created large amounts of dev data to look at general model trends and behavior; 2) to compare more directly against human performance, we also created in some cases (e.g., for `Hypernymy/Hyponymy` and `DictionaryQA`) smaller gold-standard test sets with high human agreement. 
