import torch


def compute_default_rope_parameters(config, device=None):
    """Compute unscaled RoPE frequencies for configs using the default rope type."""
    rope_parameters = getattr(config, "rope_parameters", None)
    rope_theta = (rope_parameters or {}).get("rope_theta")
    if rope_theta is None:
        rope_theta = getattr(config, "rope_theta", getattr(config, "default_theta", 10000.0))

    # Keep this tiny calculation on CPU for compatibility with meta loading
    # and backends with limited integer tensor support, then move it as needed.
    target_device = None if device is None else torch.device(device)
    if target_device is not None and target_device.type == "meta":
        target_device = None

    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    partial_rotary_factor = (rope_parameters or {}).get(
        "partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0)
    )
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
    )
    if target_device is not None:
        inv_freq = inv_freq.to(target_device)
    return inv_freq, 1.0


def initialize_default_rope_parameters(module, device):
    """Materialize default RoPE buffers after Transformers has loaded the model."""
    if module.rope_type != "default" or getattr(module, "_rope_initialized", False):
        return

    # Transformers 5 may materialize a model from meta tensors after __init__.
    # Since inv_freq is non-persistent, it is absent from the checkpoint and
    # can be left uninitialized by that process. Recompute this tiny buffer on
    # the actual model device once, on first use.
    if torch.device(device).type == "meta":
        return

    module.inv_freq, module.attention_scaling = compute_default_rope_parameters(module.config, device)
    module.original_inv_freq = module.inv_freq
    module._rope_initialized = True
