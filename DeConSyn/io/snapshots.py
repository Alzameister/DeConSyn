
import torch


@torch.no_grad()
def state_dict_snapshot(module: torch.nn.Module, device: str = "cpu") -> dict[str, torch.Tensor]:
    snap: dict[str, torch.Tensor] = {}
    for k, v in module.state_dict().items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            snap[k] = v.detach().to(device).clone()
    return snap

@torch.no_grad()
def snapshot_state_dict_pair(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    device: str = "cpu",
) -> dict[str, dict[str, torch.Tensor]]:
    g = state_dict_snapshot(generator, device)
    d = state_dict_snapshot(discriminator, device)
    return {"generator": g, "discriminator": d}