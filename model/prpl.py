import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class PairLoss(nn.Module):
    def __init__(self, max_iter=1000, eta=1e-5, upper_threshold=0.9, lower_threshold=0.5):
        super().__init__()
        self.max_iter = max_iter
        self.eta = eta
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.threshold = upper_threshold

    def forward(self, source_label, source_logits, target_logits):
        sim_matrix = self.get_cos_similarity_distance(source_logits)
        sim_matrix_target = self.get_cos_similarity_distance(target_logits)

        estimated_sim_truth = self.get_cos_similarity_distance(source_label)
        # Since target labels are not available, 
        # we use source labels that is larger than the upper threshold.
        estimated_sim_truth_target = self.get_cos_similarity_by_threshold(sim_matrix_target)

        bce_loss = (
            -(torch.log(sim_matrix + self.eta) * estimated_sim_truth)
            - (1 - estimated_sim_truth) * torch.log(1 - sim_matrix + self.eta)
        )
        cls_loss = torch.mean(bce_loss)

        bce_loss_target = (
            -(torch.log(sim_matrix_target + self.eta) * estimated_sim_truth_target)
            - (1 - estimated_sim_truth_target) * torch.log(1 - sim_matrix_target + self.eta)
        )

        indicator, nb_selected = self.compute_indicator(sim_matrix_target)
        cluster_loss = torch.sum(indicator * bce_loss_target) / nb_selected
        return cls_loss, cluster_loss

    def get_cos_similarity_distance(self, features):
        features_norm = torch.norm(features, dim=1, keepdim=True)
        features = features / features_norm
        cos_dist_matrix = torch.mm(features, features.transpose(0, 1))
        return cos_dist_matrix

    def get_cos_similarity_by_threshold(self, cos_dist_matrix):
        device = cos_dist_matrix.device
        dtype = cos_dist_matrix.dtype
        similar = torch.tensor(1, dtype=dtype, device=device)
        dissimilar = torch.tensor(0, dtype=dtype, device=device)
        sim_matrix = torch.where(cos_dist_matrix > self.threshold, similar, dissimilar)
        return sim_matrix

    def compute_indicator(self, cos_dist_matrix):
        device = cos_dist_matrix.device
        dtype = cos_dist_matrix.dtype
        selected = torch.tensor(1, dtype=dtype, device=device)
        not_selected = torch.tensor(0, dtype=dtype, device=device)
        w2 = torch.where(cos_dist_matrix < self.lower_threshold, selected, not_selected)
        w1 = torch.where(cos_dist_matrix > self.upper_threshold, selected, not_selected)
        w = w1 + w2
        nb_selected = torch.sum(w)
        return w, nb_selected

    def update_threshold(self, epoch):
        n_epochs = self.max_iter
        diff = self.upper_threshold - self.lower_threshold
        eta = diff / n_epochs
        if epoch != 0:
            self.upper_threshold = self.upper_threshold - eta
            self.lower_threshold = self.lower_threshold + eta
        self.threshold = (self.upper_threshold + self.lower_threshold) / 2


class LambdaSheduler(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1000):
        super().__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        return 2.0 / (1.0 + np.exp(-self.gamma * p)) - 1

    def step(self):
        self.curr_iter = self.curr_iter + 1


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class discriminator(nn.Module):
    def __init__(self, hidden_1=64):
        super().__init__()
        self.fc1 = nn.Linear(hidden_1, hidden_1)
        self.fc2 = nn.Linear(hidden_1, 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class AdversarialLoss(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1000, use_lambda_scheduler=True, hidden_1=64):
        super().__init__()
        self.domain_classifier = discriminator(hidden_1=hidden_1)
        self.use_lambda_scheduler = use_lambda_scheduler
        if self.use_lambda_scheduler:
            self.lambda_scheduler = LambdaSheduler(gamma, max_iter)

    def forward(self, source, target):
        lamb = 1.0
        if self.use_lambda_scheduler:
            lamb = self.lambda_scheduler.lamb()
            self.lambda_scheduler.step()
        return self.get_adversarial_result(source, target, lamb)

    def get_adversarial_result(self, source, target, lamb):
        f = ReverseLayerF.apply(torch.cat((source, target), dim=0), lamb)
        d = self.domain_classifier(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((source.size(0), 1), device=source.device)
        d_label_t = torch.zeros((target.size(0), 1), device=target.device)
        loss_fn = nn.BCELoss(reduction="mean")
        return 0.5 * (loss_fn(d_s, d_label_s) + loss_fn(d_t, d_label_t))


class TransferLoss(nn.Module):
    def __init__(self, loss_type, **kwargs):
        super().__init__()
        if loss_type == "dann":
            self.loss_func = AdversarialLoss(**kwargs)
        else:
            self.loss_func = lambda x, y: torch.tensor(0.0, device=x.device)

    def forward(self, source, target, **kwargs):
        return self.loss_func(source, target, **kwargs)


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=310, hidden_1=64, hidden_2=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x, torch.tensor(0.0, device=x.device)

    def get_parameters(self):
        return [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]


class LabelClassifier(nn.Module):
    """Bilinear transformation-based label classifier for PRPL"""
    def __init__(
        self,
        num_classes=3,
        lower_rank=32,
        max_iter=1000,
        upper_threshold=0.9,
        lower_threshold=0.5,
    ):
        super().__init__()
        self.U = nn.Parameter(torch.randn(lower_rank, 64), requires_grad=True)
        self.V = nn.Parameter(torch.randn(lower_rank, 64), requires_grad=True)
        self.register_buffer("P", torch.randn(num_classes, 64))
        self.register_buffer("stored_mat", torch.matmul(self.V, self.P.T).detach())

        self.max_iter = max_iter
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.threshold = upper_threshold
        self.num_classes = num_classes
        self.cluster_label = np.zeros(num_classes)

    def forward(self, feature):
        preds = torch.matmul(torch.matmul(self.U, feature.T).T, self.stored_mat)
        logits = F.softmax(preds, dim=1)
        return logits

    def update_P(self, source_feature, source_label):
        eye = torch.eye(self.num_classes, device=source_feature.device)
        self.P = torch.matmul(
            torch.inverse(torch.diag(source_label.sum(axis=0)) + eye),
            torch.matmul(source_label.T, source_feature),
        )
        self.stored_mat = torch.matmul(self.V, self.P.T)

    def update_cluster_label(self, source_feature, source_label):
        self.eval()
        with torch.no_grad():
            logits = self.forward(source_feature)
            source_cluster = np.argmax(logits.cpu().detach().numpy(), axis=1)
            source_label = np.argmax(source_label.cpu().numpy(), axis=1)
            for i in range(self.num_classes):
                samples_in_cluster_index = np.where(source_cluster == i)[0]
                label_for_samples = source_label[samples_in_cluster_index]
                if len(label_for_samples) == 0:
                    self.cluster_label[i] = 0
                else:
                    label_for_current_cluster = np.argmax(np.bincount(label_for_samples))
                    self.cluster_label[i] = label_for_current_cluster

    def predict(self, feature):
        self.eval()
        with torch.no_grad():
            logits = self.forward(feature)
            cluster = torch.argmax(logits, dim=1)
            cluster_label_tensor = torch.tensor(self.cluster_label, dtype=torch.long, device=cluster.device)
            preds = cluster_label_tensor[cluster]
        return preds

    def get_parameters(self):
        return [
            {"params": self.U, "lr_mult": 1},
            {"params": self.V, "lr_mult": 1},
        ]


class PRPL(nn.Module):
    def __init__(
        self,
        num_electrodes=62,
        num_features=5,
        num_classes=3,
        max_iter=1000,
        lower_rank=32,
        upper_threshold=0.9,
        lower_threshold=0.5,
        input_dim=None,
        **kwargs,
    ):
        super().__init__()
        if input_dim is None:
            input_dim = num_electrodes * num_features

        self.max_iter = max_iter
        self.feature_extractor = FeatureExtractor(input_dim, 64, 64)
        self.classifier = LabelClassifier(
            num_classes=num_classes,
            lower_rank=lower_rank,
            max_iter=max_iter,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
        )
        self.pair_loss = PairLoss(max_iter=max_iter)
        self.transfer_loss = TransferLoss(loss_type="dann", max_iter=max_iter)

    def forward(self, source, target, source_label):
        batch_size = source.size(0)
        source_feature, _ = self.feature_extractor(source)
        target_feature, _ = self.feature_extractor(target)

        self.classifier.update_P(self.feature_extractor(source)[0], source_label)

        source_logits = self.classifier(source_feature)
        target_logits = self.classifier(target_feature)

        clf_loss, cluster_loss = self.pair_loss(source_label, source_logits, target_logits)

        p_loss = torch.norm(
            torch.matmul(self.classifier.P.T, self.classifier.P) - torch.eye(64, device=source.device),
            "fro",
        )

        trans_loss = self.transfer_loss(
            source_feature + 0.005 * torch.randn((batch_size, 64), device=source_feature.device),
            target_feature + 0.005 * torch.randn((batch_size, 64), device=target_feature.device),
        )
        return clf_loss, cluster_loss, p_loss, trans_loss

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            feature, _ = self.feature_extractor(x)
            preds = self.classifier.predict(feature)
        return preds

    def predict_prob(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.classifier(self.feature_extractor(x)[0]).cpu().numpy()
            cluster_label = self.classifier.cluster_label.astype(np.int8)
            logits[:, cluster_label] = logits[:, [0, 1, 2]]
        return logits

    def get_parameters(self):
        return [
            *self.feature_extractor.get_parameters(),
            *self.classifier.get_parameters(),
            {"params": self.transfer_loss.loss_func.domain_classifier.parameters(), "lr_mult": 1},
        ]

    def epoch_end_hook(self, epoch, source_features, source_labels):
        self.pair_loss.update_threshold(epoch)
        self.classifier.update_cluster_label(self.feature_extractor(source_features)[0], source_labels)

    def get_state(self):
        return {
            "model": self.state_dict(),
            "cluster_label": self.classifier.cluster_label,
            "P": self.classifier.P.data.cpu().numpy(),
        }

    def load_state(self, state):
        self.load_state_dict(state["model"])
        self.classifier.cluster_label = state["cluster_label"]
        with torch.no_grad():
            self.classifier.P = torch.tensor(state["P"], dtype=torch.float32, device=self.classifier.P.device)
            self.classifier.stored_mat = torch.matmul(self.classifier.V, self.classifier.P.T)