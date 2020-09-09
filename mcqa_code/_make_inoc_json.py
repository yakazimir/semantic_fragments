import os
import sys
import json


JSON="""
local train_size = %d;
local batch_size = 1;
local gradient_accumulation_batch_size = 16;
local num_epochs = 3;
local learning_rate = %s;
local weight_decay = 0.1;
local warmup_ratio = 0.06;
local transformer_model = "roberta-large";
local transformer_weights_model = "";
local dataset_dir = "%s";

{
    "dataset_reader": {
	"type": "transformer_mc_qa",
	"sample": -1,
	"num_choices": 5,
	"pretrained_model": transformer_model,
	"context_syntax": "q#_a!",
	"max_pieces": 256
    },
    "train_data_path": dataset_dir + "/train.jsonl",
    "validation_data_path": dataset_dir + "/small_dev.jsonl",
    "test_data_path": "%s",
    "evaluate_on_test": true,
    "model": {
	"type": "roberta_mc_qa",
	"transformer_weights_model": transformer_weights_model,
	"pretrained_model": transformer_model
    },
    "iterator": {
	"type": "basic",
	"batch_size": batch_size
    },
    "trainer": {
	"optimizer": {
	    "type": "adam_w",
	    "weight_decay": weight_decay,
	    "parameter_groups": [[["bias", "LayerNorm\\\\.weight", "layer_norm\\\\.weight"], {"weight_decay": 0}]],
	    "lr": learning_rate
	},
	"learning_rate_scheduler": {
	    "type": "slanted_triangular",
	    "num_epochs": num_epochs,
	    "cut_frac": warmup_ratio,
	    "num_steps_per_epoch": std.ceil(train_size / gradient_accumulation_batch_size),
	},
	"validation_metric": "+accuracy",
	"num_serialized_models_to_keep": 1,
	"should_log_learning_rate": true,
	"gradient_accumulation_batch_size": gradient_accumulation_batch_size,
	"num_epochs": num_epochs,
	"cuda_device": 0
    }
} 
"""


if __name__ == "__main__":
    ## 
    if not sys.argv[1:] or len(sys.argv[1:]) < 6:
        print(sys.argv[1:])
        raise ValueError('Please specify dataset size, learning rate and output dir')

    ##
    dataset_size,learning_rate,dataset_dir,test_path,curr_out,number = sys.argv[1:]
    dataset_size = int(dataset_size)
    learning_rate = float(learning_rate)

    assert os.path.isfile(os.path.join(dataset_dir,"train.jsonl")),dataset_dir
    assert os.path.isfile(os.path.join(dataset_dir,"small_dev.jsonl")),dataset_dir

    with open(os.path.join(curr_out,"_tmp_json_%s.json" % number),'w') as tmp_json:
        print(JSON % (dataset_size,learning_rate,dataset_dir,test_path),file=tmp_json)
