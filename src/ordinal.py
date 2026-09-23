"""Utilidades ordinales que los alumnos deben implementar."""

import torch


@torch.no_grad()
def logits_to_ordinal_predictions(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:

    # convierte logits CORAL a probabilidades: (batch_size, num_classes-1)
    probs = torch.sigmoid(logits)

    # True/False segun si supera el umbral, luego se suma como enteros
    # cuenta cuantos "si" seguidos hay -> esa es la clase predicha
    predictions = (probs > threshold).sum(dim=1)

    return predictions

