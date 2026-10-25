"""Download-free example: Unconstrained EQO and FEI-VM on a toy detector.

The toy classifier scores image brightness through a ReLU layer, and each image
holds one bright 4x4 square on a dim background. A useful attribution should
rank the square's 16 pixels first.
"""

import torch
from torch import nn

from fei import generate_attribution, zero_reference

SIZE, SQUARE = 16, 4


class BrightSquareDetector(nn.Module):
    """Class 0 responds to bright pixels; class 1 is a constant baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(1, 2)
        with torch.no_grad():
            self.features[0].weight.fill_(1.0)
            self.features[0].bias.zero_()
            self.classifier.weight.copy_(torch.tensor([[30.0], [0.0]]))
            self.classifier.bias.copy_(torch.tensor([0.0, 1.0]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


def make_images(corners):
    images = torch.full((len(corners), 3, SIZE, SIZE), 0.1)
    squares = torch.zeros(len(corners), 1, SIZE, SIZE, dtype=torch.bool)
    for index, (row, col) in enumerate(corners):
        images[index, :, row : row + SQUARE, col : col + SQUARE] = 1.0
        squares[index, :, row : row + SQUARE, col : col + SQUARE] = True
    return images, squares


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrightSquareDetector().to(device).eval()
    images, squares = make_images([(2, 9), (10, 3)])
    images, squares = images.to(device), squares.to(device)
    with torch.no_grad():
        targets = model(images).argmax(dim=1)

    for variant in ("NONE", "VM"):
        maps = generate_attribution(
            model, images, targets, variant=variant, reference_fn=zero_reference
        )
        top = maps.flatten(1).topk(SQUARE * SQUARE, dim=1).indices
        hits = squares.flatten(1).gather(1, top).float().mean(dim=1)
        print(
            f"{variant}: maps {tuple(maps.shape)}, "
            f"top-16 pixels inside the square: {[f'{h:.0%}' for h in hits.tolist()]}"
        )


if __name__ == "__main__":
    main()
