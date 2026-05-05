import torch
import numpy as np


def saliency(model_forward, x, target_fn):
    """
    x: torch tensor, shape (B, P, N)
    target_fn: function mapping prediction tensor to scalar
    """
    x_attr = x.clone().detach().requires_grad_(True)

    pred = model_forward(x_attr)
    target = target_fn(pred)

    target.backward()

    return x_attr.grad.detach()


def input_x_gradient(model_forward, x, target_fn):
    grad = saliency(model_forward, x, target_fn)
    return x.detach() * grad


def integrated_gradients(model_forward, x, target_fn, baseline, n_steps=20):
    total_grad = torch.zeros_like(x)

    alphas = torch.linspace(0.0, 1.0, n_steps, device=x.device)

    for alpha in alphas:
        x_step = baseline + alpha * (x - baseline)
        x_step = x_step.clone().detach().requires_grad_(True)

        pred = model_forward(x_step)
        target = target_fn(pred)

        target.backward()

        total_grad += x_step.grad.detach()

    avg_grad = total_grad / n_steps
    return (x - baseline) * avg_grad


@torch.no_grad()
def lag_occlusion(model_forward, x, target_pred, mask_value):
    """
    Measures prediction change from masking each lag.

    x: (B, P, N)
    target_pred: original prediction, (B, H, N)
    mask_value: tensor broadcastable to (B, N), usually lag-wise or global mean

    Returns:
        effects: numpy array, shape (P,)
    """
    B, P, N = x.shape
    effects = []

    for p in range(P):
        x_masked = x.clone()

        if mask_value.ndim == 0:
            x_masked[:, p, :] = mask_value
        elif mask_value.ndim == 1:
            x_masked[:, p, :] = mask_value.reshape(1, -1)
        else:
            x_masked[:, p, :] = mask_value[:, p, :]

        pred_masked = model_forward(x_masked)

        effect = torch.mean(torch.abs(target_pred - pred_masked)).item()
        effects.append(effect)

    return np.array(effects, dtype=np.float32)


@torch.no_grad()
def poi_occlusion(model_forward, x, target_pred, poi_indices, mask_values):
    """
    Occludes selected POIs one by one.

    poi_indices: list of node indices
    mask_values: tensor shape (N,) or scalar
    """
    effects = []

    for node_idx in poi_indices:
        x_masked = x.clone()

        if mask_values.ndim == 0:
            x_masked[:, :, node_idx] = mask_values
        else:
            x_masked[:, :, node_idx] = mask_values[node_idx]

        pred_masked = model_forward(x_masked)
        effect = torch.mean(torch.abs(target_pred - pred_masked)).item()

        effects.append({
            "poi_index": int(node_idx),
            "effect": effect,
        })

    return effects


@torch.no_grad()
def group_occlusion(model_forward, x, target_pred, groups, mask_values):
    """
    groups: dict mapping group name to list of POI indices.
    """
    rows = []

    for group_name, node_indices in groups.items():
        if len(node_indices) == 0:
            continue

        x_masked = x.clone()

        for node_idx in node_indices:
            if mask_values.ndim == 0:
                x_masked[:, :, node_idx] = mask_values
            else:
                x_masked[:, :, node_idx] = mask_values[node_idx]

        pred_masked = model_forward(x_masked)

        effect = torch.mean(torch.abs(target_pred - pred_masked)).item()

        rows.append({
            "group": group_name,
            "num_pois": len(node_indices),
            "effect": effect,
        })

    return rows


@torch.no_grad()
def lag_permutation(model_forward, x, y_true, metric_fn, random_state=0):
    """
    Permutes each lag across samples and measures degradation.
    """
    rng = np.random.default_rng(random_state)

    base_pred = model_forward(x).detach().cpu().numpy()
    base_score = metric_fn(y_true, base_pred)

    B, P, N = x.shape
    rows = []

    for p in range(P):
        perm = rng.permutation(B)
        x_perm = x.clone()
        x_perm[:, p, :] = x_perm[perm, p, :]

        pred_perm = model_forward(x_perm).detach().cpu().numpy()
        perm_score = metric_fn(y_true, pred_perm)

        rows.append({
            "lag": p + 1,
            "base_score": base_score,
            "permuted_score": perm_score,
            "delta": perm_score - base_score,
        })

    return rows