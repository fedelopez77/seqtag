
import copy
import time
import torch
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
from seqtag import config
from seqtag.metrics import compute_metrics
from seqtag.utils import get_logging, write_results_to_file


class Runner(object):
    def __init__(self, model, optimizer, scheduler, loss_fn, id2label, train_loader, dev_loader, test_loader, args):
        self.ddp_model = model
        self.model = model.module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.id2label = id2label
        self.num_labels = len(id2label)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.args = args
        self.log = get_logging()
        self.is_main_process = args.local_rank == 0
        if self.is_main_process:
            self.writer = SummaryWriter(config.TENSORBOARD_PATH / args.run_id)

    def run(self):
        best_macro_f1, best_epoch = -1, -1
        best_model_state = None
        for epoch in range(1, self.args.epochs + 1):
            self.train_loader.sampler.set_epoch(epoch)      # sets epoch for shuffling
            start = time.perf_counter()
            train_loss = self.train_epoch(self.train_loader, epoch)
            exec_time = time.perf_counter() - start

            if self.is_main_process:
                self.log.info(f'Epoch {epoch} | train loss: {train_loss:.4f} | total time: {int(exec_time)} secs')
                self.writer.add_scalar("train/loss", train_loss, epoch)
                # self.writer.add_scalar("train/lr", self.get_lr(), epoch)

            if epoch % self.args.save_epochs == 0 and self.is_main_process:
                self.save_model(epoch)

            if epoch % self.args.val_every == 0:
                metrics = self.evaluate(self.dev_loader)
                if self.is_main_process:
                    self.writer.add_scalars("val/macro", metrics["macro"], epoch)
                    self.writer.add_scalars("val/micro", metrics["micro"], epoch)
                self.log.info(f"RANK {self.args.local_rank}: Results ep {epoch}: tr loss: {train_loss:.1f}, "
                              f"macroF1: {metrics['macro']['f1'] * 100:.2f}, "
                              f"microF1: {metrics['micro']['f1'] * 100:.2f}")

                macro_f1 = metrics['macro']['f1']
                self.scheduler.step()

                if macro_f1 > best_macro_f1:
                    if self.is_main_process:
                        self.log.info(f"Best val macro-F1: {macro_f1 * 100:.3f}, at epoch {epoch}")
                    best_macro_f1 = macro_f1
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(self.ddp_model.state_dict())

                # early stopping
                if epoch - best_epoch >= self.args.patience:
                    self.log.info(f"RANK {self.args.local_rank}: Early stopping at epoch {epoch}!!!")
                    break

        self.log.info(f"RANK {self.args.local_rank}: Final evaluation on best model from epoch {best_epoch}")
        self.ddp_model.load_state_dict(best_model_state)

        metrics = self.evaluate(self.test_loader)

        if self.is_main_process:
            self.export_results(metrics)
            macro, micro = metrics["macro"], metrics["micro"]
            self.log.info(f"Final Results:\n"
                          f"Macro: P: {macro['p'] * 100:.2f}, R: {macro['r'] * 100:.2f}, F1: {macro['r'] * 100:.2f}\n"
                          f"Micro: P: {micro['p'] * 100:.2f}, R: {micro['r'] * 100:.2f}, F1: {micro['r'] * 100:.2f}\n")
            self.save_model(best_epoch)
            self.writer.close()

    def train_epoch(self, train_loader, epoch_num):
        tr_loss = 0.0
        avg_grad_norm = 0.0
        self.ddp_model.train()
        self.ddp_model.zero_grad()
        self.optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            tokens, masks, labels = batch

            logits = self.ddp_model(tokens, masks)      # b, seq_len, num_labels

            logits = logits.reshape(-1, self.num_labels)
            labels = labels.reshape(-1)
            loss = self.loss_fn(logits, labels)
            loss = loss / self.args.grad_accum_steps
            loss.backward()

            # stats
            tr_loss += loss.item()
            gradient = self.model.linear.weight.grad.detach()
            grad_norm = gradient.data.norm(2).item()
            avg_grad_norm += grad_norm

            # update
            if (step + 1) % self.args.grad_accum_steps == 0:
                clip_grad_norm_(self.ddp_model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.ddp_model.zero_grad()
                self.optimizer.zero_grad()

        if self.is_main_process:
            self.writer.add_scalar("grad_norm/avg", avg_grad_norm / len(train_loader), epoch_num)
        return tr_loss / len(train_loader)

    def evaluate(self, eval_loader):
        """
        :param eval_loader: DataLoader
        :return: dict with micro and macro precision, recall and f1
        """
        self.ddp_model.eval()
        all_logits, all_labels = [], []
        for batch in eval_loader:
            tokens, masks, labels = batch
            with torch.no_grad():
                logits = self.ddp_model(tokens, masks)  # b, seq_len, num_labels

                all_logits.append(logits.reshape(-1, self.num_labels).detach())
                all_labels.append(labels.reshape(-1, 1).detach())

        all_logits = torch.cat(all_logits, dim=0)   # n, num_labels
        all_labels = torch.cat(all_labels, dim=0)   # n, 1
        metrics = compute_metrics(all_logits, all_labels)

        return metrics

    def save_model(self, epoch):
        # TODO save optimizer and scheduler
        save_path = config.CKPT_PATH / f"{self.args.run_id}-best-{epoch}ep"
        self.log.info(f"Saving model checkpoint to {save_path}")
        torch.save({
            "model": self.ddp_model.state_dict(),
            "lang_model_name": self.args.language_model,
            "id2label": self.id2label
        }, save_path)

    def export_results(self, metrics):
        result_data = {"lang_model": self.args.language_model, "run_id": self.args.run_id}
        for avg, mets in metrics.items():
            for met, val in mets.items():
                result_data[f"{avg}_{met}"] = val * 100
        write_results_to_file(self.args.results_file, result_data)
