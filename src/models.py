"""Modelos del laboratorio."""

import torch
import torch.nn as nn


class ShallowMultiClassNet(nn.Module):
    """
    Red neuronal poco profunda para clasificacion multiclase.

    Tiene:
    - una capa de entrada,
    - una capa oculta,
    - dropout,
    - y una capa de salida con tantas neuronas como clases tenga el experimento.

    El forward devuelve logits. No se agrega Softmax aqui porque
    CrossEntropyLoss lo aplica internamente.
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 32,
        dropout: float = 0.15,
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(inputs)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        return logits


class CoralLayer(nn.Module):
    """
    Capa de salida CORAL.

    Debe producir K-1 logits acumulativos a partir de un vector de
    caracteristicas de tamano input_size.

    Pistas:
    - un peso lineal compartido hacia un unico puntaje latente,
    - K-1 sesgos ordenados,
    - las diferencias entre sesgos pueden construirse con softplus y cumsum
      para forzar b0 >= b1 >= ... >= b_{K-2}.

    Formas esperadas:
    - x: (batch_size, input_size)
    - salida: (batch_size, num_classes - 1)
    """
    

    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.score = nn.Linear(input_size, 1)
        self.bias_deltas = nn.Parameter(torch.zeros(num_classes - 1)) 

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        s = self.score(inputs)
        ordered_biases = torch.cumsum(torch.nn.functional.softplus(self.bias_deltas), dim=0)
        logits = s + ordered_biases
        return logits
    


class MLPCoral(nn.Module):
    """
    MLP ordinal poco profunda con cabeza CORAL.

    Arquitectura sugerida:
    15 -> Linear(32) -> ReLU -> BatchNorm1d -> Dropout
       -> Linear(16) -> ReLU
       -> CoralLayer(16, K)

    El forward debe devolver logits de forma (batch_size, K-1).
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout

        self.fc1 = nn.Linear(num_features, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()

        self.coral = CoralLayer(16, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.fc1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.relu2(x)

        logits = self.coral(x)
        return logits
