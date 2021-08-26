import argparse
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from seqtag import config, utils
from seqtag.model import SequenceTaggingModel
from seqtag.tokenizer import get_tokenizer
from preprocess import parse_file

log = utils.get_logging()


def load_model(args, saved_data):
    model = SequenceTaggingModel(args)
    model.to(config.DEVICE)
    model = DistributedDataParallel(model, device_ids=None, find_unused_parameters=True)
    model.load_state_dict(saved_data["model"])
    return model


def get_data_loader(tokenizer, label2id, args):
    log.info(f"Loading data file: '{args.file}'")
    tokens, masks, _ = parse_file(args.file, tokenizer, label2id, args)
    if args.debug:
        n = 3
        tokens, masks = tokens[:n], masks[:n]
    dataset = TensorDataset(tokens.to(config.DEVICE), masks.to(config.DEVICE))
    return DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.batch_size)


def predict_labels(model, data_loader, id2label):
    log.info("Running inference")
    model.eval()
    all_labels = []
    for batch in data_loader:
        tokens, masks = batch
        with torch.no_grad():
            logits = model(tokens, masks)  # 1, seq_len, num_labels
            labels = torch.argmax(logits, dim=-1)  # 1, seq_len

            valid_tokens = (masks.sum() - 2).item()
            labels = labels[0][1:valid_tokens + 1].tolist()

            all_labels.append([id2label[li] for li in labels])
    return all_labels


def join_tokens_and_labels(all_labels, tokenizer, args):
    result_lines = []
    with open(args.file, "r", encoding='utf-8') as fp:
        sentence_idx = 0
        label_idx = 0
        for line in fp:
            if line != "\n":
                if sentence_idx >= len(all_labels) and args.debug:
                    break

                token = line.split(args.separator)[0]
                subtokens = tokenizer.encode(token, add_special_tokens=False)
                label = all_labels[sentence_idx][label_idx]

                result_lines.append(f"{token}\t{label}\n")
                label_idx += len(subtokens)
            else:
                result_lines.append(line)
                if result_lines[-2] != "\n":
                    sentence_idx += 1
                    label_idx = 0

    return result_lines


def infer(model, id2label, args):
    tokenizer = get_tokenizer(args.language_model)
    label2id = {v: k for k, v in id2label.items()}

    data_loader = get_data_loader(tokenizer, label2id, args)
    all_labels = predict_labels(model, data_loader, id2label)

    result_lines = join_tokens_and_labels(all_labels, tokenizer, args)

    out_file = args.file + ".out"
    log.info(f"Writing to {out_file}")
    with open(out_file, "w", encoding='utf-8') as fp:
        fp.write("".join(result_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train.py")
    parser.add_argument("--file", type=str, help="Path to file to process")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum length of line.")
    parser.add_argument("--separator", type=str, default=None, help="Separator of the tokens in the file.")

    parser.add_argument("--load_model", type=str, help="Path to model ckpt")
    parser.add_argument("--dropout", default=0, type=float, help="Dropout value.")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")

    parser.add_argument("--local_rank", type=int, help="Local process rank assigned by torch.distributed.launch")
    parser.add_argument("--n_procs", default=1, type=int, help="Number of process to create")
    parser.add_argument("--debug", dest='debug', action='store_true', default=False, help="Debug mode")
    args = parser.parse_args()

    log.info(args)

    # because we train with a distributed model, then we need to load it
    # also using a distributed schema, even if we run everything in only one process
    dist.init_process_group(backend=config.BACKEND, init_method='env://')

    log.info(f"Loading data from '{args.load_model}'")
    saved_data = torch.load(args.load_model, map_location=config.DEVICE)
    id2label = saved_data["id2label"]
    args.language_model = saved_data["lang_model_name"]
    args.num_labels = len(id2label)

    model = load_model(args, saved_data)

    infer(model, id2label, args)
