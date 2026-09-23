"""Perdidas y ponderaciones que los alumnos deben implementar."""

import numpy as np
import torch


#Convierte clases enteras a umbrales binarios acumulativos.
def labels_to_levels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:

    thresholds = torch.arange(num_classes - 1, device=labels.device)
    levels = (labels.unsqueeze(1) > thresholds).float()
    
    return levels


def coral_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    
    # convierte las etiquetas enteras en los K-1 umbrales binarios (0/1)
    levels = labels_to_levels(labels, num_classes)

    # BCE elemento a elemento, sin reducir todavía: (batch_size, num_classes-1)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, levels, reduction="none")

    
    if class_weights is not None:
        # peso por observacion, segun su clase real: (batch_size,)
        sample_weights = class_weights[labels]
        # aplicarlo a cada fila (se expande a las K-1 columnas)
        losses = losses * sample_weights.unsqueeze(1)

    return losses.mean()



def effective_number_weights(
    labels: np.ndarray,
    num_classes: int,
    beta: float = 0.99,
) -> torch.Tensor:

    # cuantas observaciones hay de cada clase
    counts = np.array(
        [np.sum(labels == c) for c in range(num_classes)], dtype=np.float64
    )
    counts = np.maximum(counts, 1)  # evita division por 0 si una clase no aparece

    # numero efectivo de muestras por clase
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num

    # normalizar para que la media de los pesos sea 1
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)
    
"""
Pesos por numero efectivo de muestras:

    w_c = (1 - beta) / (1 - beta ** n_c)

Normalizar los pesos para que su media sea 1.

Formas:
- labels: (N,)
- salida: (num_classes,)
"""


