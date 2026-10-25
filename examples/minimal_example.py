"""Download-free example for Unconstrained EQO and FEI-VM."""

import torch
from torch import nn

from fei import generate_attribution, zero_reference


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(8, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs).flatten(1)
        return self.classifier(features)


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyClassifier().to(device).eval()
    images = torch.rand(2, 3, 16, 16, device=device)
    with torch.no_grad():
        targets = model(images).argmax(dim=1)

    for variant in ("NONE", "VM"):
        maps = generate_attribution(
            model,
            images,
            targets,
            variant=variant,
            areas=(0.25, 0.5),
            iterations=5,
            reference_fn=zero_reference,
        )
        print(
            f"{variant}: shape={tuple(maps.shape)}, "
            f"range=({maps.min().item():.3f}, {maps.max().item():.3f})"
        )


if __name__ == "__main__":
    main()

