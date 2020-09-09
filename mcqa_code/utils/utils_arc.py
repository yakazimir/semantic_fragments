import json
import logging
import os

logger = logging.getLogger(__name__)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {"acc": simple_accuracy(preds, labels)}


class ARCExample(object):
    """A single training/test example for the ARC dataset."""

    def __init__(self,
                 arc_id,
                 question,
                 para,
                 choices,
                 num_choices,
                 label=None):
        self.arc_id = arc_id
        self.question = question
        self.para = para
        if len(choices) > num_choices:
            raise ValueError("More choices: {} in question: {} than allowed: {}".format(
                choices, question, num_choices
            ))
        self.choices = [choice["text"] for choice in choices]
        self.choice_paras = [choice.get("para") for choice in choices]
        if len(choices) < num_choices:
            add_num = num_choices - len(choices)
            self.choices.extend([""] * add_num)
            self.choice_paras.extend([None] * add_num)
        label_id = None
        if label is not None:
            for (idx, ch) in enumerate(choices):
                if ch["label"] == label:
                    label_id = idx
                    break
            if label_id is None:
                raise ValueError("No answer found matching the answer key:{} in {}".format(
                    label, choices
                ))
        self.label = label_id

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"arc_id: {self.arc_id}",
            f"para: {self.para}",
            f"question: {self.question}",
            f"choices: {self.choices}"
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def convert_examples_to_features(examples, max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % ex_index)

        choices_features = []
        for choice_idx, choice in enumerate(example.choices):
            context = example.para
            if example.choice_paras[choice_idx] is not None:
                context += " " + example.choice_paras[choice_idx]
            context += " " + example.question
            context_tokens = tokenizer.tokenize(context)
            context_tokens_choice = context_tokens[:]
            choice_tokens = tokenizer.tokenize(choice)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(context_tokens_choice, choice_tokens, max_seq_length - 3)
            tokens_a = context_tokens_choice
            tokens_b = choice_tokens

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = tokens_a + [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label_id = example.label
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info(f"arc_id: {example.arc_id}")
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(
                    choices_features):
                logger.info(f"choice: {choice_idx}")
                logger.info(f"tokens: {' '.join(tokens)}")
                logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
                logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
                logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
                logger.info(f"label: {label_id}")

        features.append(
            InputFeatures(
                example_id=example.arc_id,
                choices_features=choices_features,
                label=label_id
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class ARCExampleReader:
    """Reader for ARC dataset format"""

    ## older versions 
    # def get_train_examples(self, data_dir, max_choices,other_name=""):
    #     return self._create_examples(
    #         self._read_jsonl(os.path.join(data_dir, "train.jsonl")), for_training=True,
    #         max_choices=max_choices)

    # def get_dev_examples(self, data_dir, max_choices,other_name=""):
    #     return self._create_examples(
    #         self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), for_training=False,
    #         max_choices=max_choices)

    ## reads files directly 
    def get_train_examples(self, data_dir, max_choices,exclusion=""):
            return self._create_examples(
            self._read_jsonl(data_dir,exclusion), for_training=True,
            max_choices=max_choices)

    def get_dev_examples(self, data_dir, max_choices,exclusion=""):
        return self._create_examples(
            self._read_jsonl(data_dir,exclusion), for_training=False,
            max_choices=max_choices)

    def _read_jsonl(self, filepath,exclusion):
        excluded_set = set([d.strip() for d in exclusion.split(',')]) if exclusion else None
        
        with open(filepath, 'r') as fp_reader:
            for line in fp_reader:
                ## exclude something?
                if excluded_set:
                    json_line = json.loads(line.strip())
                    dataset = json_line.get("notes",{})
                    try: 
                        if dataset["source"].strip() not in excluded_set:
                            #print(dataset["source"])
                            continue
                    except KeyError:
                        raise ValueError('Included dataset does not appear to contain multiple datasets!')
                    #
                    #print(dataset["source"])
                yield json.loads(line.strip())

    def _create_examples(self, json_stream, for_training, max_choices):
        """Creates examples for the training and dev sets."""
        for input_json in json_stream:
            if "answerKey" not in input_json and for_training:
                raise ValueError("No answer key provided for training in {}".format(input_json))

            ### 
            
            arc_example = ARCExample(
                arc_id=input_json["id"],
                question=input_json["question"]["stem"],
                para=input_json.get("para", ""),
                choices=input_json["question"]["choices"],
                num_choices=max_choices,
                label=input_json.get("answerKey")
            )
            yield arc_example
