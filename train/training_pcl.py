import math

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from model.Adversarial import DAANLoss
from model.PCL_TDGCN import Discriminator, PCL
from utils.metric import Metric


def _dbg_tstat(name: str, t: torch.Tensor) -> str:
    t = t.detach().float().cpu()
    return (f"{name}: shape={tuple(t.shape)} sum={t.sum().item():.6f} "
            f"norm={t.norm().item():.6f} mean={t.mean().item():.6f} std={t.std().item():.6f}")


def _dbg_params(name: str, module: torch.nn.Module) -> str:
    s = 0.0
    n = 0.0
    cnt = 0
    for p in module.parameters():
        pf = p.detach().float().cpu()
        s += pf.sum().item()
        n += pf.pow(2).sum().item()
        cnt += pf.numel()
    return f"{name}_params: count={cnt} sum={s:.6f} norm={n**0.5:.6f}"


class _LabelSmoothingCE(torch.nn.Module):
    def __init__(self, classes: int, epsilon: float = 0.0005):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.epsilon / (input.size(-1) - 1.0)
        weight.scatter_(-1, target.unsqueeze(-1), (1.0 - self.epsilon))
        return (-weight * log_prob).sum(dim=-1).mean()


class _StepwiseLR:
    def __init__(self, optimizer, init_lr: float, gamma: float = 10.0,
                 decay_rate: float = 0.75, max_iter: int = 1000):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.max_iter = max_iter
        self.iter_num = 0

    def step(self):
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num / self.max_iter)) ** self.decay_rate
        for pg in self.optimizer.param_groups:
            pg.setdefault("lr_mult", 1.0)
            pg["lr"] = lr * pg["lr_mult"]
        self.iter_num += 1


def train(
    model: PCL,
    metric: Metric,
    train_data: np.ndarray,   # (N, 310) float32, band-major
    train_labels: np.ndarray, # (N,) int64
    test_data: np.ndarray,    # (M, 310) float32
    test_labels: np.ndarray,  # (M,) int64
    batch_size: int,
    num_classes: int,
    device: str,
    epochs: int,
    subject_id: int,
    session_id: int,
    learning_rate: float,
    weight_decay: float,
    eval_interval: int,
    early_stop_patience: int,
) -> None:
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    train_labels = train_labels.astype(np.int64).reshape(-1)
    test_labels = test_labels.astype(np.int64).reshape(-1)

    source_num = train_data.shape[0]
    target_num = test_data.shape[0]

    # --- Datasets and loaders ---
    source_dataset = TensorDataset(
        torch.from_numpy(train_data),
        torch.arange(source_num).long(),
        torch.from_numpy(train_labels),
    )
    target_dataset = TensorDataset(
        torch.from_numpy(test_data),
        torch.arange(target_num).long(),
        torch.from_numpy(test_labels),
    )
    source_loader = DataLoader(source_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=2, pin_memory=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(target_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    # --- Loss, discriminator, optimizer ---
    criterion = _LabelSmoothingCE(classes=num_classes).to(device)
    discriminator = Discriminator(model.encoder.fc2.out_features).to(device)
    dann_loss = DAANLoss(discriminator, num_class=num_classes).to(device)

    optimizer = torch.optim.RMSprop(
        list(model.parameters()) + list(discriminator.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    lr_scheduler = _StepwiseLR(optimizer, init_lr=learning_rate,
                                gamma=10, decay_rate=0.75, max_iter=epochs)

    # === DBG[A] post-init ===
    print(f"[DBG-A] subj={subject_id} sess={session_id} src_n={source_num} tgt_n={target_num}")
    print(f"[DBG-A] {_dbg_params('model', model)}")
    print(f"[DBG-A] {_dbg_params('disc', discriminator)}")
    print(f"[DBG-A] {_dbg_tstat('GGCN.A', model.encoder.GGCN.A)}")

    # === DBG[A2] dataset[0] byte-fingerprint (BEFORE any DataLoader iteration) ===
    s0_feat, s0_idx, s0_lbl = source_dataset[0]
    t0_feat, t0_idx, t0_lbl = target_dataset[0]
    s0_bytes = s0_feat.detach().cpu().contiguous().view(torch.uint8)
    t0_bytes = t0_feat.detach().cpu().contiguous().view(torch.uint8)
    print(f"[DBG-A2] src_dataset[0] idx={s0_idx.item()} lbl={s0_lbl.item()} "
          f"bytes_sum={s0_bytes.sum().item()} first16={s0_bytes.flatten()[:16].tolist()} "
          f"feat[:8]={s0_feat[:8].tolist()}")
    print(f"[DBG-A2] tar_dataset[0] idx={t0_idx.item()} lbl={t0_lbl.item()} "
          f"bytes_sum={t0_bytes.sum().item()} first16={t0_bytes.flatten()[:16].tolist()} "
          f"feat[:8]={t0_feat[:8].tolist()}")
    s_last = source_dataset[source_num - 1]
    t_last = target_dataset[target_num - 1]
    print(f"[DBG-A2] src_dataset[-1] idx={s_last[1].item()} lbl={s_last[2].item()} "
          f"feat[:8]={s_last[0][:8].tolist()}")
    print(f"[DBG-A2] tar_dataset[-1] idx={t_last[1].item()} lbl={t_last[2].item()} "
          f"feat[:8]={t_last[0][:8].tolist()}")                      

    # --- Initialize memory banks ---
    model.eval()
    with torch.no_grad():
        for src_feat, src_idx, _ in source_loader:
            model.get_init_banks(src_feat.to(device), src_idx.to(device))
        for tgt_feat, tgt_idx, _ in target_loader:
            model.get_init_banks_tgt(tgt_feat.to(device), tgt_idx.to(device))

    # === DBG[B] post-bank-init ===
    print(f"[DBG-B] {_dbg_tstat('source_f_bank', model.source_f_bank)}")
    print(f"[DBG-B] {_dbg_tstat('target_f_bank', model.target_f_bank)}")
    print(f"[DBG-B] {_dbg_tstat('source_score_bank', model.source_score_bank)}")
    print(f"[DBG-B] {_dbg_tstat('target_score_bank', model.target_score_bank)}")

    best_acc = 0.0
    patience_counter = 0
    num_batches = len(target_loader.dataset) // batch_size

    for epoch in range(epochs):

        # --- Evaluation ---
        if epoch % eval_interval == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for feat, _, label in test_loader:
                    feat, label = feat.to(device), label.to(device)
                    out = model.target_predict(feat)
                    pred = out.argmax(dim=1)
                    correct += pred.eq(label.view_as(pred)).sum().item()
            accuracy = correct / len(test_loader.dataset)

            metric.update(subject_id, session_id, accuracy)

            if accuracy > best_acc:
                best_acc = accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % (eval_interval * 50) == 0 :
                logger.info(
                    "Epoch {}/{} subj {} sess {} | acc={:.4f} best={:.4f}",
                    epoch, epochs, subject_id, session_id, accuracy, best_acc,
                )

            if accuracy >= 1.0:
                logger.info("Perfect accuracy, stopping.")
                break
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                logger.info("Early stop at epoch {} best={:.4f}", epoch, best_acc)
                break

        # --- Training ---
        model.train()
        dann_loss.train()

        src_iter = iter(source_loader)
        tar_iter = iter(target_loader)

        total_loss_sum = 0.0
        total_cls_loss = 0.0
        total_transfer_loss = 0.0
        total_source_loss = 0.0
        total_target_loss = 0.0
        total_domain_loss = 0.0
        for batch_idx in range(num_batches):
            src_feat, src_idx, src_label = next(src_iter)
            tar_feat, tar_idx, _ = next(tar_iter)

            src_feat = src_feat.to(device)
            src_idx = src_idx.to(device)
            src_label = src_label.to(device).view(-1)
            tar_feat = tar_feat.to(device)
            tar_idx = tar_idx.to(device)

            # === DBG[C] per-batch (first 3 of each epoch) ===
            if batch_idx < 3:
                print(f"[DBG-C] ep={epoch} b={batch_idx} "
                      f"src_idx[:8]={src_idx[:8].cpu().tolist()} "
                      f"tar_idx[:8]={tar_idx[:8].cpu().tolist()} "
                      f"src_label[:8]={src_label[:8].cpu().tolist()}")
                print(f"[DBG-C] ep={epoch} b={batch_idx} "
                      f"{_dbg_tstat('src_feat', src_feat)}")
                print(f"[DBG-C] ep={epoch} b={batch_idx} "
                      f"{_dbg_tstat('tar_feat', tar_feat)}")

            # === DBG[E] gate the model only at ep=0 b=0 ===
            if epoch == 0 and batch_idx == 0:
                model._dbg = True
                # byte-level fingerprint of the very first input batch
                src_bytes = src_feat.detach().cpu().contiguous().view(torch.uint8)
                tar_bytes = tar_feat.detach().cpu().contiguous().view(torch.uint8)
                print(f"[DBG-E0] src_feat bytes_sum={src_bytes.sum().item()} "
                      f"first16={src_bytes.flatten()[:16].tolist()}")
                print(f"[DBG-E0] tar_feat bytes_sum={tar_bytes.sum().item()} "
                      f"first16={tar_bytes.flatten()[:16].tolist()}")
                print(f"[DBG-E0] src_feat[0,:8]={src_feat[0,:8].detach().cpu().tolist()}")
                print(f"[DBG-E0] tar_feat[0,:8]={tar_feat[0,:8].detach().cpu().tolist()}")
                print(f"[DBG-E0] src_label[:8]={src_label[:8].detach().cpu().tolist()}")
                print(f"[DBG-E0] src_idx[:8]={src_idx[:8].detach().cpu().tolist()}")
                print(f"[DBG-E0] tar_idx[:8]={tar_idx[:8].detach().cpu().tolist()}")
            else:
                model._dbg = False

            (src_out, src_f, tar_out, tar_f,
             _src_att, _tar_att,
             src_sim, tgt_sim, tgt_cluster_label,
             s2t_pro, t2s_pro, s2s_pro, t2t_pro) = model(
                src_feat, tar_feat, src_label, src_idx, tar_idx, epoch, epochs
            )

            cls_loss = criterion(src_out, src_label)

            src_prob = F.softmax(src_out, dim=1)
            mask = src_prob.max(dim=1).values > 0.7
            if mask.any():
                source_loss = criterion(src_prob[mask], src_label[mask])
            else:
                source_loss = torch.tensor(0.0, device=device)

            target_loss = criterion(tgt_sim, tgt_cluster_label.long())

            global_transfer_loss = dann_loss(
                src_f + 0.005 * torch.randn_like(src_f),
                tar_f + 0.005 * torch.randn_like(tar_f),
                src_prob, F.softmax(tar_out, dim=1),
            )

            boost_factor = 2.0 * (2.0 / (1.0 + math.exp(-epoch / 1000)) - 1)

            s2t_entropy = -(s2t_pro * torch.log(s2t_pro + 1e-10)).sum(dim=1).mean()
            t2s_entropy = -(t2s_pro * torch.log(t2s_pro + 1e-10)).sum(dim=1).mean()
            cross_domain_loss = s2t_entropy + t2s_entropy

            s2s_entropy = -(s2s_pro * torch.log(s2s_pro + 1e-10)).sum(dim=1).mean()
            t2t_entropy = -(t2t_pro * torch.log(t2t_pro + 1e-10)).sum(dim=1).mean()
            in_domain_loss = s2s_entropy + t2t_entropy

            loss = (cls_loss + global_transfer_loss + source_loss
                    + boost_factor * target_loss
                    + 0.2 * (cross_domain_loss + in_domain_loss))

            if epoch == 0 and batch_idx == 0:
                print(f"[DBG-E0] cls_loss={cls_loss.item():.8f} "
                      f"source_loss={source_loss.item():.8f} "
                      f"target_loss={target_loss.item():.8f}")
                print(f"[DBG-E0] global_transfer_loss={global_transfer_loss.item():.8f} "
                      f"cross_domain_loss={cross_domain_loss.item():.8f} "
                      f"in_domain_loss={in_domain_loss.item():.8f}")
                print(f"[DBG-E0] boost_factor={boost_factor:.8f} total_loss={loss.item():.8f}")
                print(f"[DBG-E0] src_prob_max_mean={src_prob.max(dim=1).values.mean().item():.6f} "
                      f"mask_count={mask.sum().item()}")

            if torch.isnan(loss):
                logger.warning("NaN loss at epoch {} batch {}, skipping.", epoch, batch_idx)
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_sum += loss.item()
            total_cls_loss += cls_loss.item()
            total_transfer_loss += global_transfer_loss.item()
            total_source_loss += source_loss.item()
            total_target_loss += target_loss.item()
            total_domain_loss += cross_domain_loss.item() + in_domain_loss.item()

        if epoch % (eval_interval) == 0:
            logger.info(
                "Epoch {}/{} | total_loss_sum={:.4f} | total_cls_loss={:.4f} | "
                "total_transfer_loss={:.4f}| total_domain_loss={:.4f} | total_source_loss={:.4f} | total_target_loss={:.4f}",
                epoch, epochs, total_loss_sum,
                total_cls_loss ,
                total_transfer_loss ,
                total_domain_loss,
                total_source_loss,
                total_target_loss
            )

        # === DBG[D] end-of-epoch (only first 30 epochs to keep logs small) ===
        if epoch < 30:
            print(f"[DBG-D] ep={epoch} {_dbg_params('model', model)}")
            print(f"[DBG-D] ep={epoch} {_dbg_params('disc', discriminator)}")
            print(f"[DBG-D] ep={epoch} {_dbg_tstat('source_f_bank', model.source_f_bank)}")
            print(f"[DBG-D] ep={epoch} {_dbg_tstat('target_f_bank', model.target_f_bank)}")
            print(f"[DBG-D] ep={epoch} {_dbg_tstat('source_score_bank', model.source_score_bank)}")
            print(f"[DBG-D] ep={epoch} {_dbg_tstat('target_score_bank', model.target_score_bank)}")
            print(f"[DBG-D] ep={epoch} lr={optimizer.param_groups[0]['lr']:.6e} "
                  f"iter_num={lr_scheduler.iter_num}")

        lr_scheduler.step()
