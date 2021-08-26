import argparse
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import optimization
from seqtag.runner import Runner
from seqtag.model import SequenceTaggingModel
from seqtag import config
from seqtag import utils


def config_parser(parser):
    # Data options
    parser.add_argument("--data", type=str, help="Path to preprocessed file")
    parser.add_argument("--run_id", type=str, help="Name of model/run to export")

    # optim and config
    parser.add_argument("--optim", default="AdamW", type=str, help="Optimization method: AdamW or Adafactor")
    parser.add_argument("--learning_rate", default=3e-4, type=float, help="Starting learning rate.")
    parser.add_argument("--weight_decay", default=0.00, type=float, help="L2 Regularization.")
    parser.add_argument("--val_every", default=5, type=int, help="Runs validation every n epochs.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout value.")
    parser.add_argument("--patience", default=50, type=int, help="Epochs of patience for scheduler and early stop.")
    parser.add_argument("--max_grad_norm", default=50.0, type=float, help="Max gradient norm.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs.")
    parser.add_argument("--grad_accum_steps", default=1, type=int,
                        help="Number of update steps to acum before backward.")
    parser.add_argument("--weigh_loss", dest='weigh_loss', action='store_true', default=False,
                        help="Weigh loss by label frequency")

    # Others
    parser.add_argument("--local_rank", type=int, help="Local process rank assigned by torch.distributed.launch")
    parser.add_argument("--job_id", default=-1, type=int, help="Slurm job id to be logged")
    parser.add_argument("--n_procs", default=1, type=int, help="Number of process to create")
    parser.add_argument("--load_model", default="", type=str, help="Load model from this file")
    parser.add_argument("--results_file", default="out/results.csv", type=str, help="Exports final results to this file")
    parser.add_argument("--save_epochs", default=10001, type=int, help="Exports every n epochs")
    parser.add_argument("--seed", default=42, type=int, help="Seed")
    parser.add_argument("--debug", dest='debug', action='store_true', default=False, help="Debug mode")


def get_model(args):
    model = SequenceTaggingModel(args)
    model.to(config.DEVICE)
    model = DistributedDataParallel(model, device_ids=None, find_unused_parameters=True)
    if args.load_model:
        saved_data = torch.load(args.load_model)
        model.load_state_dict(saved_data["model"])
    return model


def get_scheduler(optimizer, args):
    """Encapsulate scheduler if needed"""
    from transformers import get_constant_schedule
    return get_constant_schedule(optimizer)


def create_loader(data_dict, args, is_train=False):
    """
    :param data_dict: data_dict: dict where "tokens", "masks" and "labels" are (b, seq_len) tensors
    :param args:
    :param is_train: boolean
    :return: DataLoader
    """
    tokens, masks, labels = data_dict["tokens"], data_dict["masks"], data_dict["labels"]
    if args.debug:
        n = 20
        tokens, masks, labels = tokens[:n], masks[:n], labels[:n]

    device = config.DEVICE
    dataset = TensorDataset(tokens.to(device), masks.to(device), labels.to(device))
    if is_train:
        train_batch_size = args.batch_size // args.n_procs
        sampler = DistributedSampler(dataset, num_replicas=args.n_procs, rank=args.local_rank)
        return DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=False, num_workers=0,
                          pin_memory=False, sampler=sampler)

    return DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.batch_size)


def load_training_data(args, log):
    log.info(f"Loading data from '{args.data}'")
    data = torch.load(args.data)
    args.language_model = data["lang_model_name"]

    train_loader = create_loader(data["train"], args, is_train=True)
    dev_loader = create_loader(data["dev"], args)
    test_loader = create_loader(data["test"], args)
    id2label = data["id2label"]

    label_weights = utils.count_labels(data["train"]["labels"], id2label)

    return train_loader, dev_loader, test_loader, id2label, label_weights


def get_loss_function(args, label_weights):
    if not args.weigh_loss:
        weights = torch.ones_like(label_weights)
        # zeros weights for "artificial" labels
        weights[config.PAD_LABEL_ID] = 0
        weights[config.SUBTOKEN_LABEL_ID] = 0
        label_weights = weights
    return torch.nn.CrossEntropyLoss(weight=label_weights.to(config.DEVICE))


def main():
    parser = argparse.ArgumentParser("train.py")
    config_parser(parser)
    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(args.debug)

    # sets random seed
    seed = args.seed if args.seed > 0 else random.randint(1, 1000000)
    utils.set_seed(seed)

    log = utils.get_logging()
    if args.local_rank == 0:
        log.info(args)
        log.info(f"Job ID: {args.job_id}")

    dist.init_process_group(backend=config.BACKEND, init_method='env://')

    # correct parameters due to distributed training
    args.learning_rate *= args.n_procs

    train_loader, dev_loader, test_loader, id2label, label_weights = load_training_data(args, log)
    args.num_labels = len(id2label)

    model = get_model(args)
    optimizer = getattr(optimization, args.optim)(model.parameters(),
                                                  lr=args.learning_rate,
                                                  weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args)
    loss_fn = get_loss_function(args, label_weights)

    if args.local_rank == 0:
        log.info(model)
        log.info(f"GPU's available: {torch.cuda.device_count()}")
        n_params = sum([p.nelement() for p in model.parameters() if p.requires_grad])
        log.info(f"number of parameters: {n_params}")
        log.info(f"Train: {len(train_loader.dataset)}, Dev: {len(dev_loader.dataset)}, "
                 f"Test: {len(test_loader.dataset)}, labels: {len(id2label)}")

    runner = Runner(model, optimizer, scheduler=scheduler, loss_fn=loss_fn, id2label=id2label, args=args,
                    train_loader=train_loader, dev_loader=dev_loader, test_loader=test_loader)
    runner.run()
    log.info("Done!")


if __name__ == "__main__":
    main()
