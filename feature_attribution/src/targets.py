def global_target(pred, horizons="all"):
    """
    pred: torch tensor, shape (B, H, N)

    Returns scalar target.
    """
    if horizons == "all" or horizons is None:
        return pred.mean()

    if isinstance(horizons, int):
        return pred[:, horizons, :].mean()

    return pred[:, horizons, :].mean()


def node_target(pred, node_idx, horizon_idx=None):
    """
    pred: torch tensor, shape (B, H, N)
    node_idx: integer POI index
    horizon_idx: zero-based horizon index, or None for all horizons
    """
    if horizon_idx is None:
        return pred[:, :, node_idx].mean()

    return pred[:, horizon_idx, node_idx].mean()


def category_target(pred, node_indices, horizon_idx=None):
    """
    pred: torch tensor, shape (B, H, N)
    node_indices: list or array of POI indices
    """
    if horizon_idx is None:
        return pred[:, :, node_indices].mean()

    return pred[:, horizon_idx, node_indices].mean()


def build_target_fn(target_type, **kwargs):
    if target_type == "global":
        horizons = kwargs.get("horizons", "all")
        return lambda pred: global_target(pred, horizons=horizons)

    if target_type == "node":
        node_idx = kwargs["node_idx"]
        horizon_idx = kwargs.get("horizon_idx", None)
        return lambda pred: node_target(
            pred,
            node_idx=node_idx,
            horizon_idx=horizon_idx,
        )

    if target_type == "category":
        node_indices = kwargs["node_indices"]
        horizon_idx = kwargs.get("horizon_idx", None)
        return lambda pred: category_target(
            pred,
            node_indices=node_indices,
            horizon_idx=horizon_idx,
        )

    raise ValueError(f"Unknown target_type: {target_type}")