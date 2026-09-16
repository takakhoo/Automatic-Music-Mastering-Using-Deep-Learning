"""CPU-friendly shape and range checks for the Token U-Net forward pass."""

from __future__ import annotations

import torch

from token_unet import TokenUNet


def main() -> None:
    torch.manual_seed(7)
    model = TokenUNet(
        n_q=2,
        k=16,
        base_dim=16,
        depth=1,
        dropout=0.0,
    )
    model.eval()
    tokens = torch.randint(0, 16, (1, 2, 32))

    with torch.no_grad():
        output = model(tokens)

    expected_shapes = {
        "logits": (1, 16, 2, 32),
        "mask": (1, 2, 32),
        "perceptual": (1, 8),
        "gain": (1, 1),
        "stereo": (1, 2),
        "compression": (1, 2),
    }
    actual_shapes = {
        name: tuple(tensor.shape) for name, tensor in output.items()
    }
    assert actual_shapes == expected_shapes, actual_shapes
    assert torch.isfinite(output["logits"]).all()
    assert torch.all((0 <= output["mask"]) & (output["mask"] <= 1))
    print("Token U-Net smoke test passed:", actual_shapes)


if __name__ == "__main__":
    main()
