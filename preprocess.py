import argparse
import torch
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from seqtag.tokenizer import get_tokenizer
from seqtag.config import PAD_LABEL, SUBTOKEN_LABEL, PAD_LABEL_ID, SUBTOKEN_LABEL_ID
from seqtag import utils

FILES = ["train_atis.txt", "dev_atis.txt", "test_atis.txt"]

log = utils.get_logging()


def parse_file(file_path, tokenizer, label2id, args):
    cls_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0
    sep_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 0
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    all_tokens, all_masks, all_labels = [], [], []

    with open(file_path, "r", encoding='utf-8') as fp:
        sent_tokens, sent_labels = [], []

        for line in fp:
            if line != "\n":
                token, label = line.strip().split(args.separator)

                subtokens = tokenizer.encode(token, add_special_tokens=False)

                sent_tokens += subtokens
                # only the first subtoken retains the label, the rest are marked with the subtoken label
                sent_labels += [label] + [SUBTOKEN_LABEL] * (len(subtokens) - 1)
            else:
                if len(sent_tokens) == 0:
                    continue

                assert len(sent_tokens) == len(sent_labels)
                max_len = args.max_len - 2      # due to CLS and SEP tokens
                sent_tokens, sent_labels = sent_tokens[:max_len], sent_labels[:max_len]

                # appends [CLS] and [SEP] tokens and pads the tensor
                sent_tokens = [cls_token_id] + sent_tokens + [sep_token_id]
                all_tokens.append(torch.tensor(sent_tokens))

                # creates attention mask: 1's until [SEP] and 0's onwards
                sent_mask = [1] * len(sent_tokens)
                all_masks.append(torch.tensor(sent_mask))

                # appends pad label due to CLS and SEP tokens and pads the tensor
                sent_labels = [PAD_LABEL] + sent_labels + [PAD_LABEL]
                all_labels.append(sent_labels)

                assert len(sent_tokens) == len(sent_labels) == len(sent_mask)

                sent_tokens, sent_labels = [], []

    assert len(all_tokens) == len(all_masks) == len(all_labels)

    for lab in sum(all_labels, []):
        if lab not in label2id:
            label2id[lab] = len(label2id)

    label_ids = [torch.tensor([label2id[li] for li in ls]) for ls in all_labels]

    all_tokens = pad_sequence(all_tokens, batch_first=True, padding_value=pad_token_id)
    all_masks = pad_sequence(all_masks, batch_first=True, padding_value=0)
    label_ids = pad_sequence(label_ids, batch_first=True, padding_value=label2id[PAD_LABEL])

    return all_tokens, all_masks, label_ids


def load_data(args):
    data_path = Path(args.data_path)
    tokenizer = get_tokenizer(args.language_model)
    label2id = {PAD_LABEL: PAD_LABEL_ID, SUBTOKEN_LABEL: SUBTOKEN_LABEL_ID}
    result = {}
    for filename in FILES:
        log.info(f"Processing {filename}")
        tokens, masks, label_ids = parse_file(data_path / filename, tokenizer, label2id, args)

        log.info(f"\tSentences: {len(tokens)}, seq_length: {len(tokens[0])}")
        result[filename.split("_")[0]] = {
            "tokens": tokens,
            "masks": masks,
            "labels": label_ids
        }

    return result, label2id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, help="Name of preprocessed file to save")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data folder")
    parser.add_argument("--language_model", type=str, default="bert-base-uncased",
                        help="Language Model for tokenization")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum length of line.")
    parser.add_argument("--separator", type=str, default=None, help="Separator of the tokens in the train/dev files.")
    parser.add_argument("--seed", type=int, default=42, help="Maximum length of the files.")

    args = parser.parse_args()
    utils.set_seed(42)
    log.info(f"Preprocessing {args.data_path} for {args.language_model}")

    data_to_save, label2id = load_data(args)
    data_to_save["lang_model_name"] = args.language_model
    data_to_save["id2label"] = {v: k for k, v in label2id.items()}
    log.info(f"Labels: {len(label2id)}")

    lang_model_name = args.language_model.replace("_", "-").replace("/", "-")
    path_to_save = Path(args.data_path) / (args.run_id + f"_{lang_model_name}.pt")
    log.info(f"Saving to {path_to_save}")
    torch.save(data_to_save, path_to_save)
