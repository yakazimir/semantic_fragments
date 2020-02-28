Semantic Fragment datasets
================================

This sub-directory contains the *complete* 8 challenge datasets reported in our paper here[https://arxiv.org/pdf/1909.07521.pdf].

Each datasets contains a `train` and `test` sub-directory, in which the datasets sit both in SNLI (allenNLP json) format (`{train,dev,test}.json`) and the format used by our BERT models (`challenge_{train,dev}.tsv`, Note that `challenge_dev.tsv` is the cover name for all held-out sets in our BERT implementation, so in each `test` directory `challenge_dev.tsv` is actually a test set).

NOTE: In the 6 *logic fragments* (i.e., boolean, compatative, conditional, counting, negation, quantifer), the tests sets have a disjoint vocabulary (this was intentional to to look at model performance in a non-iid/out-of-domain testing scenario), so results might vary dramatically from `dev` to `test` (see details of how we dealt with this is described below; see also `scripts/README.md` for details about all of our experiments). 

How Models were Trained
-----------------------

We largely used these datasets to do the following two types of experiments (as detailed in `scripts/README.md`):

1. **Zero-shot evaluation** on NLI models (similar in spirit to the *breaking NLI dataset* and challenge here[https://www.aclweb.org/anthology/P18-2103/] or **NLI Stress testing** here[https://www.aclweb.org/anthology/C18-1198/]), i.e., given models trained on standard benchmarks such as SNLI/MNLI, we tested to see how well do these models solve each `dev` and `test` set in our difference datasets (in general, the models we tested performed quite poorly across the board, with the highest scores being on the monotonicity fragments). In this setting, you can therefore just use the held-out datasets to test a given model (see `scripts/snli_results.sh`) and ignore the train sets.  

2. **Model inoculation** (see here[https://github.com/nelson-liu/inoculation-by-finetuning]), i.e., given the same standard NLI models trained on SNLI/MNLI, can we continue training the models to perform well on these new tasks (without hurting their performance on the benchmarks) by giving them small **dosages** of these new tasks? In other words, can we teach these models to be good at our new tasks with totally re-training them from scratch? (see `scripts/test_inoculation.sh`)

In the latter experiments, the number of training examples we use for inoculation is an additional hyper-parameter adjusted during the tuning phase. In other words, training on the full 3k training examples for each dataset does not necessarily yield the best results, and might seriously overfit, especially on the datasets that involve non-iid testing (as mentioned above). In addition to the files and datasets in this directory, you can also find our full set of experimental materials (with the different inoculation sets cut up), here[https://drive.google.com/file/d/1-kSKE95uP92YM_Bw1qdRjrbeYQxwR0-X/view?usp=sh], as described in `scripts/README.md`.

We experimented with all datasets separately, i.e., we trained 8 different models and reported on their average in our paper (see scripts in `scripts` for some documentation about individual model performance).

Additional credits
-----------------------

The 6 datasets that involve logic were modified from the code here: https://github.com/felipessalvatore/ContraBERT (and their associated paper). In particular, we added more rules to some of their data constructors, and also changed the semantics to have 3-way entailment labels.
