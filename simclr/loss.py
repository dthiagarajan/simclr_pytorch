import torch
import torch.nn as nn


class NTXEntCriterion(nn.Module):
    """Normalized, temperature-scaled cross-entropy criterion, as suggested in the SimCLR paper.

    Parameters:
        temperature (float, optional): temperature to scale the confidences. Defaults to 0.5.
    """
    criterion = nn.CrossEntropyLoss(reduction="sum")
    similarity = nn.CosineSimilarity(dim=2)

    def __init__(self, temperature=0.5):
        super(NTXEntCriterion, self).__init__()
        self.temperature = temperature
        self.batch_size = None
        self.mask = None

    def mask_correlated_samples(self, batch_size):
        """Masks examples in a batch and it's augmented pair for computing the valid summands for
            the criterion.

        Args:
            batch_size (int): batch size of the individual batch (not including it's augmented pair)

        Returns:
            torch.Tensor: a mask (tensor of 0s and 1s), where 1s indicates a pair of examples in a
                batch that will contribute to the overall batch loss
        """
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def compute_similarities(self, z_i, z_j, temperature):
        """Computes the similarities between two projections `z_i` and `z_j`, scaling based on
            `temperature`.

        Args:
            z_i (torch.Tensor): projection of a batch
            z_j (torch.Tensor): projection of the augmented pair for the batch
            temperature (float): temperature to scale the similarity by

        Returns:
            torch.Tensor: tensor of similarities for the positive and negative pairs
        """
        batch_size = len(z_i)
        mask = self.mask_correlated_samples(batch_size)

        p1 = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity(p1.unsqueeze(1), p1.unsqueeze(0)) / temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            batch_size * 2, 1
        )
        negative_samples = sim[mask].reshape(batch_size * 2, -1)

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits

    def forward(self, z):
        """Computes the loss for a batch and its augmented pair.

        Args:
            z (torch.Tensor): tensor of a batch and it's augmented pair, concatenated

        Returns:
            torch.Tensor: loss for the given batch
        """
        double_batch_size = len(z)
        batch_size = double_batch_size // 2
        z_i, z_j = z[:double_batch_size // 2], z[double_batch_size // 2:]
        if self.batch_size is None or batch_size != self.batch_size:
            self.batch_size = batch_size
            self.mask = None

        if self.mask is None:
            self.mask = self.mask_correlated_samples(self.batch_size)

        logits = self.compute_similarities(z_i, z_j, self.temperature)
        labels = torch.zeros(self.batch_size * 2).long()
        logits, labels = logits.to(z.device), labels.to(z.device)
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        return loss
