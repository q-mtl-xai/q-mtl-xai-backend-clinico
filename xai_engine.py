
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _force_train_for_lstm(model):
    """LSTM cuDNN necesita train() para que fluyan los gradientes."""
    was_training = model.training
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.BatchNorm1d)):
            m.eval()
    return was_training


def compute_saliency_dynamic(model, signal, task, device='cpu', top_k=20):
    """
    Calcula la explicación XAI de UN registro específico.
    """
    signal = np.asarray(signal, dtype=np.float32)
    was_training = _force_train_for_lstm(model)

    if signal.ndim == 1:
        x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    else:
        x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
    x.requires_grad = True

    out = model(x, task=task)
    probs = F.softmax(out.detach(), dim=1).cpu().numpy()[0]
    pred_class = int(probs.argmax())

    score = out[0, pred_class]
    score.backward()

    saliency = x.grad.detach().abs().cpu().numpy()[0]

    if not was_training:
        model.eval()

    if saliency.ndim == 2:
        saliency_1d = saliency.mean(axis=0)
    else:
        saliency_1d = saliency

    smax = saliency_1d.max()
    if smax > 0:
        saliency_norm = saliency_1d / smax
    else:
        saliency_norm = saliency_1d

    top_idx = np.argsort(saliency_1d)[-top_k:][::-1].tolist()

    return {
        "pred_class": pred_class,
        "probabilities": probs.tolist(),
        "saliency": saliency_norm.tolist(),
        "top_timesteps": top_idx,
    }
