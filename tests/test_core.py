import unittest
from unittest.mock import patch

import torch
from torch import nn

from fei import (
    ChannelSplit,
    TokenSplit,
    activation_trajectory_similarity,
    evaluate_black_input_induction,
    imagenet_black_input_like,
    deit_cls_attention_map,
    generate_attribution,
    pathway_ratio,
    select_clean_tokens,
    timm_deit_token_split,
    token_pathway_ratio,
    zero_reference,
)
from fei.clipgrad import ClipGrad


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(4, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


class TinyTokenBlock(nn.Module):
    """A minimal token-mixing block standing in for a timm transformer block."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + torch.tanh(self.linear(tokens))


class TinyPatchEmbed(nn.Module):
    """Returns patch tokens shaped [B, N, D], like timm's PatchEmbed."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=4, stride=4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.proj(inputs).flatten(2).transpose(1, 2)


class TinyTokenTransformer(nn.Module):
    """A minimal ViT-shaped model with the layout ``timm_deit_token_split`` expects."""

    def __init__(self, dim: int = 6, num_patches: int = 4, num_classes: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.patch_embed = TinyPatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Identity()
        self.blocks = nn.Sequential(*(TinyTokenBlock(dim) for _ in range(3)))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def _patch_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self._patch_tokens(inputs)
        cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1) + self.pos_embed
        tokens = self.blocks(self.pos_drop(tokens))
        return self.head(self.norm(tokens)[:, 0])


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = TinyClassifier().eval()
        self.inputs = torch.rand(2, 3, 8, 8)
        with torch.no_grad():
            self.targets = self.model(self.inputs).argmax(dim=1)

    def run_variant(self, variant: str):
        return generate_attribution(
            self.model,
            self.inputs,
            self.targets,
            variant=variant,
            areas=(0.25, 0.5),
            iterations=2,
            learning_rate=0.05,
            area_weight=0.1,
            reference_fn=zero_reference,
            return_stages=True,
        )

    def test_unconstrained_and_vm_execute(self) -> None:
        for variant in ("NONE", "VM"):
            with self.subTest(variant=variant):
                attribution, stages = self.run_variant(variant)
                self.assertEqual(tuple(attribution.shape), (2, 1, 8, 8))
                self.assertTrue(torch.isfinite(attribution).all())
                self.assertGreaterEqual(attribution.min().item(), 0.0)
                self.assertLessEqual(attribution.max().item(), 1.0)
                self.assertEqual(len(stages), 2)
                self.assertTrue(torch.all(stages[1] >= stages[0]))

    def test_all_clipgrad_variants_execute(self) -> None:
        for variant in ("IBM", "IVM", "AVM"):
            with self.subTest(variant=variant):
                attribution, _ = self.run_variant(variant)
                self.assertTrue(torch.isfinite(attribution).all())

    def test_model_state_is_restored(self) -> None:
        self.model.train()
        requires_grad = [p.requires_grad for p in self.model.parameters()]
        _ = self.run_variant("VM")
        self.assertTrue(self.model.training)
        self.assertEqual(
            [p.requires_grad for p in self.model.parameters()], requires_grad
        )
        self.assertTrue(
            all(not module._forward_hooks for module in self.model.modules())
        )
        self.assertTrue(
            all(not module._backward_hooks for module in self.model.modules())
        )

    def test_clipgrad_coordinate_rules(self) -> None:
        gradient = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        activation_mask = torch.tensor([False, True, False, True])
        expected = {
            "VM": torch.tensor([-2.0, 0.0, 0.0, 2.0]),
            "IBM": torch.tensor([0.0, -1.0, 1.0, 2.0]),
            "IVM": torch.tensor([0.0, -1.0, 1.0, 2.0]),
            "AVM": torch.tensor([-2.0, -1.0, 0.0, 2.0]),
        }
        for variant, wanted in expected.items():
            with self.subTest(variant=variant):
                clipgrad = ClipGrad(self.model, variant)
                actual = clipgrad._apply_rule(gradient, activation_mask)
                self.assertTrue(torch.equal(actual, wanted))

    def test_pathway_ratio_executes(self) -> None:
        prefix = self.model.features[:2]

        def suffix(activation):
            value = self.model.features[2:](activation)
            return self.model.classifier(value.flatten(1))

        split = ChannelSplit(self.model, prefix, suffix)
        attribution = torch.rand(1, 1, 8, 8)
        result = pathway_ratio(
            split,
            self.inputs[:1],
            attribution,
            int(self.targets[0]),
            insertion_step=16,
            candidate_batch_size=2,
            forward_batch_size=2,
        )
        self.assertTrue(torch.isfinite(torch.tensor(result.rho)))
        self.assertLess(result.split_max_abs_error, 1e-6)
        self.assertGreater(len(result.selected_channels), 0)

    def test_activation_trajectory_similarity_matches_identical_inputs(self) -> None:
        relu = self.model.features[1]
        result = activation_trajectory_similarity(
            self.model, self.inputs, self.inputs, layers=[relu], layer_names=["relu0"]
        )
        self.assertEqual(result.layer_names, ("relu0",))
        self.assertAlmostEqual(result.mse[0], 0.0, places=5)
        self.assertAlmostEqual(result.cosine[0], 1.0, places=5)

    def test_activation_trajectory_similarity_detects_perturbation(self) -> None:
        relu = self.model.features[1]
        perturbed = torch.rand_like(self.inputs)
        result = activation_trajectory_similarity(
            self.model, self.inputs, perturbed, layers=[relu]
        )
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.mse))
        self.assertTrue(all(-1.0 <= v <= 1.0 for v in result.cosine))

    def test_black_input_evaluation_executes(self) -> None:
        black_input = imagenet_black_input_like(self.inputs)
        result = evaluate_black_input_induction(
            self.model,
            black_input,
            self.targets,
            variant="NONE",
            areas=(0.25, 0.5),
            iterations=2,
        )
        self.assertEqual(result.induced.shape, (2,))
        self.assertEqual(result.predicted.shape, (2,))
        self.assertEqual(tuple(result.perturbed.shape), tuple(black_input.shape))
        self.assertEqual(tuple(result.attribution.shape), (2, 1, 8, 8))
        self.assertTrue(torch.isfinite(result.perturbed).all())

    def test_black_input_protocol_uses_zero_reference_and_ensemble_map(self) -> None:
        black_input = torch.full_like(self.inputs[:1], -2.0)
        ensemble = torch.full((1, 1, 8, 8), 0.25)
        with patch("fei.black_input.generate_attribution", return_value=ensemble) as generate:
            result = evaluate_black_input_induction(
                self.model, black_input, int(self.targets[0]), variant="NONE"
            )
        kwargs = generate.call_args.kwargs
        self.assertIs(kwargs["reference_fn"], zero_reference)
        self.assertNotIn("return_stages", kwargs)
        self.assertTrue(torch.equal(result.attribution, ensemble))
        self.assertTrue(torch.equal(result.perturbed, black_input * ensemble))

    def test_imagenet_black_input_is_not_zero_reference(self) -> None:
        black_input = imagenet_black_input_like(self.inputs[:1])
        expected = self.inputs.new_tensor(
            [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
        ).view(1, 3, 1, 1).expand_as(black_input)
        self.assertTrue(torch.allclose(black_input, expected))
        self.assertFalse(torch.equal(black_input, zero_reference(black_input)))


class TokenPathwayTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)
        self.model = TinyTokenTransformer().eval()
        self.image = torch.rand(1, 3, 8, 8)
        with torch.no_grad():
            self.target = int(self.model(self.image).argmax(dim=1).item())

    def _split(self) -> TokenSplit:
        def prefix_fn(inputs: torch.Tensor) -> torch.Tensor:
            tokens = self.model._patch_tokens(inputs)
            cls_token = self.model.cls_token.expand(tokens.shape[0], -1, -1)
            return torch.cat((cls_token, tokens), dim=1) + self.model.pos_embed

        def suffix_fn(tokens: torch.Tensor) -> torch.Tensor:
            value = self.model.blocks(self.model.pos_drop(tokens))
            return self.model.head(self.model.norm(value)[:, 0])

        return TokenSplit(model=self.model, prefix_fn=prefix_fn, suffix_fn=suffix_fn, cls_offset=1)

    def test_select_clean_tokens_excludes_cls(self) -> None:
        split = self._split()
        clean_probability, tokens, probabilities, split_error = select_clean_tokens(
            split, self.image, self.target, candidate_batch_size=2
        )
        self.assertLess(split_error, 1e-5)
        self.assertGreater(len(tokens), 0)
        self.assertTrue(all(0 <= t < 4 for t in tokens))  # 4 patch tokens, CLS excluded
        self.assertEqual(len(probabilities), len(tokens))
        self.assertGreater(clean_probability, 0.0)

    def test_token_pathway_ratio_executes(self) -> None:
        split = self._split()
        attribution = torch.rand(1, 1, 8, 8)
        result = token_pathway_ratio(
            split,
            self.image,
            attribution,
            self.target,
            insertion_step=16,
            candidate_batch_size=2,
            forward_batch_size=2,
        )
        self.assertTrue(torch.isfinite(torch.tensor(result.rho_token)))
        self.assertLess(result.split_max_abs_error, 1e-5)
        self.assertGreater(len(result.selected_tokens), 0)

    def test_timm_deit_token_split_matches_forward(self) -> None:
        with torch.no_grad():
            expected = self.model(self.image)
            for split_block in (-1, 0, 1):
                split = timm_deit_token_split(self.model, split_block=split_block)
                tokens = split.prefix(self.image)
                self.assertEqual(tuple(tokens.shape), (1, 5, self.model.dim))
                self.assertLess((split.suffix(tokens) - expected).abs().max().item(), 1e-5)
        with self.assertRaises(ValueError):
            timm_deit_token_split(self.model)  # default block 8 needs at least 10 blocks

    def test_deit_cls_attention_map_shape(self) -> None:
        attention = torch.rand(2, 3, 5, 5)  # [B, heads, tokens, tokens]
        cls_map = deit_cls_attention_map(attention)
        self.assertEqual(tuple(cls_map.shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
