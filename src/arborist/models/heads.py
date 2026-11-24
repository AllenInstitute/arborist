"""
Created on Fri Nov 20 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Task heads for Arborist. A head owns everything specific to one task: how it
reads the tree encoding, its prediction layers, where its label lives on a
TreeSample, and its loss. MultiTaskArborist pairs one shared encoder with any
number of heads, so a new task is a new head and the training loop is unchanged.

A sample carries its labels in "TreeSample.target", either a single value or a
dict keyed by task name.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from arborist.models.arborist import READOUT_WIDTH, READOUTS, Arborist, readout


class TaskHead(nn.Module):
    """
    Base class for task heads.

    Subclasses set "kind", store their constructor arguments in
    "self.config" (used by MultiTaskArborist.load), and implement "forward"
    and "loss".
    """

    kind = None

    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, z_tree, z_curves, sample):
        """
        Scores one sample.

        Parameters
        ----------
        z_tree : torch.Tensor
            Shape (latent_dim,). Global tree embedding.
        z_curves : torch.Tensor
            Shape (n_curves, latent_dim). Per-curve embeddings.
        sample : TreeSample
            The encoded sample, for heads that need its structure.

        Returns
        -------
        torch.Tensor
            Prediction for this sample, without a batch dimension.
        """
        raise NotImplementedError

    def loss(self, preds, targets):
        """
        Parameters
        ----------
        preds : torch.Tensor
            Stacked predictions, shape (B, ...).
        targets : torch.Tensor
            Stacked targets, shape (B, ...).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        raise NotImplementedError

    def target(self, sample):
        """
        Reads this task's label from "sample.target", which is either the
        label itself or a dict keyed by task name.
        """
        target = sample.target
        if isinstance(target, dict):
            if self.name not in target:
                raise KeyError(
                    f"Sample has no target for task '{self.name}'; "
                    f"it has {sorted(target)}"
                )
            target = target[self.name]
        if target is None:
            raise ValueError(f"Sample has no target for task '{self.name}'")
        return torch.as_tensor(target, dtype=torch.float32)


class BinaryHead(TaskHead):
    """
    MLP classifier with one logit per sample, trained with
    BCEWithLogitsLoss.

    "readout" is one of ``arborist.models.arborist.READOUTS``; see
    ``readout`` there.
    """

    kind = "binary"

    def __init__(
        self,
        name,
        latent_dim,
        readout="cls+root",
        hidden_dim=None,
        n_layers=2,
        dropout=0.1,
        pos_weight=None,
    ):
        super().__init__(name)
        if readout not in READOUTS:
            raise ValueError(f"Unknown readout: {readout}")
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        self.config = {
            "name": name,
            "latent_dim": latent_dim,
            "readout": readout,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "dropout": dropout,
            "pos_weight": pos_weight,
        }
        self.readout = readout

        in_dim = latent_dim * READOUT_WIDTH[readout]
        hidden_dim = hidden_dim or latent_dim
        layers, dim = [], in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))

    def forward(self, z_tree, z_curves, sample):
        x = readout(z_tree, z_curves, sample, self.readout)
        return self.net(x).squeeze(-1)

    def loss(self, preds, targets):
        return F.binary_cross_entropy_with_logits(
            preds, targets, pos_weight=self.pos_weight
        )


HEAD_KINDS = {BinaryHead.kind: BinaryHead}


class MultiTaskArborist(nn.Module):
    """
    One Arborist encoder shared by named task heads.

    Example
    -------
    >>> model = MultiTaskArborist(
    ...     Arborist(latent_dim=64), [BinaryHead("merge", latent_dim=64)]
    ... )
    >>> preds, targets, loss = model.step(samples, "merge")
    """

    def __init__(self, arborist, heads):
        super().__init__()
        names = [head.name for head in heads]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate head names: {names}")
        self.arborist = arborist
        self.heads = nn.ModuleDict({head.name: head for head in heads})

    def forward(self, samples, task):
        """
        Parameters
        ----------
        samples : List[TreeSample]
            Batch of samples, as returned by collate_trees.
        task : str
            Name of the head to run.

        Returns
        -------
        torch.Tensor
            Predictions stacked over the batch, shape (B, ...).
        """
        head = self.heads[task]
        preds = []
        for sample in samples:
            z_tree, z_curves = self.arborist.encode(sample)
            preds.append(head(z_tree, z_curves, sample))
        return torch.stack(preds)

    def step(self, samples, task):
        """
        Runs "task" on the batch and computes its loss from the samples' own
        targets.

        Returns
        -------
        Tuple[torch.Tensor]
            Predictions, targets, and scalar loss.
        """
        head = self.heads[task]
        preds = self(samples, task)
        targets = torch.stack([head.target(s) for s in samples]).to(preds.device)
        return preds, targets, head.loss(preds, targets)

    def save(self, path):
        torch.save(
            {
                "arborist_config": self.arborist.config,
                "heads": [
                    {"kind": h.kind, "config": h.config}
                    for h in self.heads.values()
                ],
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        heads = [HEAD_KINDS[h["kind"]](**h["config"]) for h in ckpt["heads"]]
        model = cls(Arborist(**ckpt["arborist_config"]), heads)
        model.load_state_dict(ckpt["state_dict"])
        return model
