import torch


def compute_default_rope_parameters(config, device=None):
    """Compute unscaled RoPE frequencies for configs using the default rope type."""
    rope_parameters = getattr(config, "rope_parameters", None)
    rope_theta = (rope_parameters or {}).get("rope_theta")
    if rope_theta is None:
        rope_theta = getattr(config, "rope_theta", getattr(config, "default_theta", 10000.0))

    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim)
    )
    return inv_freq, 1.0
