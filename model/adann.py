import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class ADANN(nn.Module):
    def __init__(
        self,
        num_electrodes=62,
        num_features=5,
        num_classes=3,
        input_dim=None,
        hidden_dim=64,
        projected_dim=32,
        **kwargs,
    ):
        super().__init__()
        if input_dim is None:
            input_dim = num_electrodes * num_features

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.projected_dim = projected_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
        )

        self.domain_discriminator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.projector = nn.Linear(hidden_dim, projected_dim, bias=False)
        self.register_buffer("mu_s", torch.zeros(num_classes, hidden_dim))
        self.register_buffer("mu_t", torch.zeros(num_classes, hidden_dim))
        self.bce_loss = nn.BCELoss()

    def forward(self, source, target, source_label=None):
        f_s = self.feature_extractor(source)
        f_t = self.feature_extractor(target)

        p_s = self.projector(f_s)
        p_t = self.projector(f_t)

        alpha = 1.0
        reversed_f_s = GradientReversalLayer.apply(f_s, alpha)
        reversed_f_t = GradientReversalLayer.apply(f_t, alpha)

        d_s = self.domain_discriminator(reversed_f_s)
        d_t = self.domain_discriminator(reversed_f_t)

        loss_d_s = self.bce_loss(d_s, torch.ones_like(d_s))
        loss_d_t = self.bce_loss(d_t, torch.zeros_like(d_t))
        loss_dann = loss_d_s + loss_d_t

        return {
            "f_s": f_s,
            "p_s": p_s,
            "f_t": f_t,
            "p_t": p_t,
            "loss_dann": loss_dann,
        }

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            f = self.feature_extractor(x)
            phi_f = self.projector(f)
            phi_mu = self.projector(self.mu_t)

            phi_f = F.normalize(phi_f, p=2, dim=1)
            phi_mu = F.normalize(phi_mu, p=2, dim=1)

            sim = torch.mm(phi_f, phi_mu.t())
            preds = sim.argmax(dim=1)
        return preds

    def get_parameters(self):
        return [
            {"params": self.feature_extractor.parameters(), "lr_mult": 1},
            {"params": self.domain_discriminator.parameters(), "lr_mult": 1},
            {"params": self.projector.parameters(), "lr_mult": 1},
        ]

    def get_state(self):
        return {
            "model": self.state_dict(),
        }

    def load_state(self, state):
        self.load_state_dict(state["model"])