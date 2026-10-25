# Beyond Output Faithfulness: Diagnosing and Mitigating Pathway Rerouting in Feature Attribution

Siyu Zhang and Kenneth McMillan, The University of Texas at Austin

*IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2027*

This is the official implementation of Ensemble Quantile Optimization (EQO),
the ClipGrad gradient filter, and the clean-pathway diagnostics from the paper.

## Overview

Insertion and deletion test whether the pixels an attribution selects control
the model output. They do not test whether those pixels act through the
computation that supports the clean prediction. EQO optimizes the insertion
sequence directly. Without constraints, it attains the highest insertion
scores while departing from the clean activations, and it keeps its insertion
score after the channels that support the clean prediction are ablated: the
attribution reroutes the prediction through a different pathway. ClipGrad is a
layer-wise gradient filter that keeps the optimized inputs on the clean
activation trajectory. EQO with ClipGrad is FEI (Faithfulness-guided Ensemble
Interpretation).

| Paper | Code |
|---|---|
| EQO and Unconstrained EQO (Sec. 3.2, Algorithm 1) | `generate_attribution(..., variant="NONE")` |
| ClipGrad variants VM, IBM, IVM, AVM (Sec. 5) | `generate_attribution(..., variant="VM")`, `fei/clipgrad.py` |
| Clean-support intervention $\rho$ (Sec. 4.1, Table 3, Supp. B) | `pathway_ratio`, `torchvision_channel_split` |
| Activation-trajectory similarity (Sec. 4.2, Table 4) | `activation_trajectory_similarity` |
| All-black-input target induction (Sec. 6.2, Table 2, Supp. K) | `evaluate_black_input_induction` |
| DeiT-Small token intervention $\rho_{\mathrm{token}}$ and last-attention match (Sec. 6.5, Table 5, Supp. G) | `token_pathway_ratio`, `deit_cls_attention_map` |

## Contents

- `fei/eqo.py`: nested quantile-mask optimization and attribution generation.
- `fei/clipgrad.py`: clean-activation capture and the ClipGrad backward rules.
- `fei/pathway.py`: clean-channel selection, the pathway ratio $\rho$, and the
  DeiT token-level analogue $\rho_{\mathrm{token}}$.
- `fei/activation.py`: activation-trajectory similarity (cosine and MSE
  against the clean trajectory) and the DeiT CLS-attention extraction used
  for last-attention activation matching.
- `fei/black_input.py`: the all-black-input target-induction evaluation.
- `examples/minimal_example.py`: a self-contained run on a small CNN.
- `tests/test_core.py`: CPU checks for mask shape, range, nesting, hook
  cleanup, the unconstrained and VM execution paths, activation similarity,
  the black-input evaluation, and the token-level pathway diagnostic.
- `data_index_imagenet.txt`, `data_index_cub.txt`: the 3,000 evaluation images
  per dataset, one index per line, into `torchvision.datasets.ImageFolder` over
  the ImageNet `val` folder and the CUB-200-2011 `images/test` folder.

The implementation supports the five configurations used in the paper:
`NONE` (Unconstrained EQO), `VM`, `IBM`, `IVM`, and `AVM`. Standard CNNs are
handled through ReLU/GELU module hooks. The `convnext-tiny` and `deit-small`
model kinds use output-tensor hooks.

## Install and verify

Python 3.9 or newer and PyTorch 2.0 or newer are required.

```bash
git clone https://github.com/zhangsiyu1103/fei.git
cd fei
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/minimal_example.py
```

The example uses a toy network and synthetic input, so it downloads no data or
weights. A CUDA model and CUDA input can be used without changing the API.

## Use with a model

```python
import torch
from fei import generate_attribution, random_solid_reference

model.eval()
image = image.to(next(model.parameters()).device)  # [B, C, H, W], ImageNet-normalized
with torch.no_grad():
    target = model(image).argmax(dim=1)

attribution = generate_attribution(
    model,
    image,
    target,
    variant="VM",                 # NONE, VM, IBM, IVM, or AVM
    areas=(0.1, 0.3, 0.5, 0.7, 0.9),
    iterations=100,
    learning_rate=0.05,
    area_weight=0.1,
    reference_fn=random_solid_reference,
)
```

`generate_attribution` returns one map per input with shape `[B, 1, H, W]` and
values in `[0, 1]`. Each optimized mask starts from the preceding area level,
so the sequence is spatially nested; the returned attribution is their mean.

The optimizer maximizes the selected entry of the model output, so the model
should return the score used by the intended protocol. Pass a model whose
output is class probability (for example, `torch.nn.Sequential(model,
torch.nn.Softmax(dim=1))`) when optimizing probability, or the plain model when
optimizing a logit.

## Clean-pathway diagnostic ($\rho$)

The diagnostic takes a model split into a channel-bearing prefix and the
remaining suffix, greedily selects the channels that support the clean
prediction, and compares intact and channel-ablated insertion AUC:

```python
from fei import ChannelSplit, pathway_ratio

split = ChannelSplit(
    model=model,
    prefix_fn=lambda x: model.features(x),
    suffix_fn=lambda h: model.classifier(model.avgpool(h).flatten(1)),
)
result = pathway_ratio(split, image, attribution, target)
print(result.rho, result.selected_channels)
```

`torchvision_channel_split` provides the VGG16, ResNet50, and ConvNeXt-Tiny
split locations used in the paper. The routine is single-image by design,
because its clean-supporting channel set is image-specific; candidate channel
ablations and insertion frames are evaluated in batches.

## Activation-trajectory similarity

```python
from fei import activation_trajectory_similarity

relu_layers = [model.features[1], model.features[3]]
result = activation_trajectory_similarity(
    model, clean_images, perturbed_images, layers=relu_layers,
    layer_names=["block1", "block2"],
)
print(result.mse, result.cosine)
```

This hooks the given layers, runs one clean and one perturbed forward pass,
and reports per-layer cosine similarity (directional alignment) and MSE
(magnitude preservation) between the two, averaged over the batch. This is the
comparison reported in Table 4. Pass the same module used to build a
`ChannelSplit` or `TokenSplit` to compare at the diagnostic's own interface.

For DeiT, `deit_cls_attention_map` extracts the head-averaged CLS-to-patch row
from a post-softmax attention tensor (`[B, heads, tokens, tokens]`); feed the
clean and perturbed maps it returns into the same cosine/MSE comparison to
compute the last-attention activation match.

## All-black-input target induction

This evaluation follows the black-input test introduced by FGVis (Wagner et
al., "Interpretable and Fine-Grained Visual Explanations for Convolutional
Neural Networks," CVPR 2019), with the category-level protocol of the paper.

```python
from fei import evaluate_black_input_induction, imagenet_black_input_like

black = imagenet_black_input_like(image)  # raw RGB black, then normalized
result = evaluate_black_input_induction(
    model, black, target, variant="NONE",
)
print(result.induced)  # one Boolean per requested target
```

It runs the full attribution pipeline with the normalized black RGB canvas in
place of the clean image and a zero tensor in normalized input space as the
replacement reference. It applies the returned ensemble attribution to the
canvas and checks the resulting top-1 prediction, never a threshold on the
rendered heatmap. For the aggregate experiment in the paper, repeat the canvas
and pass every model category except the black canvas's initial prediction as
the target batch.

## DeiT token intervention ($\rho_{\mathrm{token}}$)

```python
from fei import timm_deit_token_split, token_pathway_ratio

split = timm_deit_token_split(deit_model)  # standard timm VisionTransformer layout
result = token_pathway_ratio(split, image, attribution, target)
print(result.rho_token, result.selected_tokens)
```

This mirrors `pathway_ratio` with the ablation unit changed from a
convolutional channel to a patch-token embedding: the same clean top-1 target,
the same greedy steepest-descent selection, and the same insertion-trajectory
construction. `timm_deit_token_split` assumes the standard timm
`VisionTransformer` module layout (`patch_embed`, `cls_token`, `pos_embed`,
`pos_drop`, `blocks`, `norm`, `head`); build a `TokenSplit` directly for a
different implementation.

## Options

**References.** `random_solid_reference` samples an ImageNet-normalized solid
RGB reference at each optimization step, as in the paper's attribution
benchmark. For a fixed zero reference, use `zero_reference`. Any function
mapping an input batch to a same-shaped reference batch can be supplied.

**Architectures.** Ordinary ReLU/GELU CNNs need no architecture name. Use
`model_kind="convnext-tiny"` for ConvNeXt-Tiny and `model_kind="deit-small"`
for DeiT-Small. For DeiT, `deit_mode` chooses where ClipGrad is applied:
`residual` (block outputs, the default), `attention` (attention-module
outputs, used for the paper's DeiT-Small VM runs), `gelu`, or `both`.

## Citation

```bibtex
@inproceedings{zhang2027beyond,
  title     = {Beyond Output Faithfulness: Diagnosing and Mitigating Pathway Rerouting in Feature Attribution},
  author    = {Zhang, Siyu and McMillan, Kenneth},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## License

This code is released under the MIT License; see `LICENSE`.

## Earlier version

The code for the earlier preprint version of this work ("Beyond Output
Faithfulness: Learning Attributions that Preserve Computational Pathways")
remains in the git history at commit `0060ba1`.
