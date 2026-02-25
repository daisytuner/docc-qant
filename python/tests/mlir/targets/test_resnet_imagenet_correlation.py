import pytest
import os
import numpy as np
import torch
import torchvision.models as models
from torchvision.datasets import Imagenette
from torch.utils.data import DataLoader

import docc.torch
from docc.qant import register_docc_plugin

register_docc_plugin()

# Imagenette local index → ImageNet-1K class index mapping.
# Imagenette is a 10-class subset of ImageNet. These are the well-known
# ImageNet-1K indices for the 10 Imagenette synsets (sorted by wnid).
_IMAGENETTE_TO_IMAGENET = {
    0: 0,  # n01440764 → tench
    1: 217,  # n02102040 → English springer
    2: 482,  # n02979186 → cassette player
    3: 491,  # n03000684 → chain saw
    4: 497,  # n03028079 → church
    5: 566,  # n03394916 → French horn
    6: 569,  # n03417042 → garbage truck
    7: 571,  # n03425413 → gas pump
    8: 574,  # n03445777 → golf ball
    9: 701,  # n03888257 → parachute
}


def _imagenette_dataset(preprocess):
    """Load the Imagenette validation set (auto-downloads ~100 MB on first use)."""
    data_dir = os.environ.get("IMAGENETTE_DIR", "/tmp/imagenette")
    return Imagenette(
        root=data_dir, split="val", size="320px", download=True, transform=preprocess
    )


def _compute_metrics(preds_ref, preds_docc, labels):
    """Compute classification agreement and accuracy metrics.

    Returns a dict with:
      - accuracy_ref:  top-1 accuracy of the reference (fp32) model
      - accuracy_docc: top-1 accuracy of the docc (bf16) model
      - agreement:     fraction of samples where both models predict the same class
    """
    accuracy_ref = (preds_ref == labels).mean()
    accuracy_docc = (preds_docc == labels).mean()
    agreement = (preds_ref == preds_docc).mean()
    return {
        "accuracy_ref": float(accuracy_ref),
        "accuracy_docc": float(accuracy_docc),
        "agreement": float(agreement),
    }


def _compute_logit_correlation(logits_ref, logits_docc):
    """Compute mean per-sample Pearson correlation between logit vectors.

    For each sample the 1000-dim logit vectors from the reference and docc
    models are correlated.  The mean correlation across all samples is
    returned.
    """
    # logits shape: (N, 1000)
    ref = logits_ref - logits_ref.mean(axis=1, keepdims=True)
    docc = logits_docc - logits_docc.mean(axis=1, keepdims=True)

    num = (ref * docc).sum(axis=1)
    denom = np.sqrt((ref**2).sum(axis=1) * (docc**2).sum(axis=1))
    correlations = num / np.clip(denom, a_min=1e-12, a_max=None)
    return float(correlations.mean())


@pytest.mark.skipif(not os.environ.get("SLOW_TESTS", ""), reason="slow test")
def test_resnet18_imagenet_classification_correlation():
    """Compare ResNet18 classification between docc (bf16) and reference (fp32).

    Uses the Imagenette validation set (a 10-class subset of ImageNet-1K) to
    evaluate whether the bf16 accelerator path produces classifications that
    are consistent with the fp32 reference model.

    Metrics reported (via stdout):
      - Reference top-1 accuracy
      - DOCC top-1 accuracy
      - Model agreement rate  (same predicted class)
      - Mean per-sample Pearson correlation of logit vectors

    Environment variables:
      SLOW_TESTS     – must be set (non-empty) to run this test
      IMAGENETTE_DIR – override dataset cache directory (default: /tmp/imagenette)
    """
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))

    weights = models.ResNet18_Weights.IMAGENET1K_V1

    # Reference model (fp32)
    model_ref = models.resnet18(weights=weights)
    model_ref.eval()

    # DOCC-compiled model (bf16 on accelerator)
    model = models.resnet18(weights=weights)
    model.eval()
    program = torch.compile(
        model, backend="docc", options={"target": "qant", "category": "server"}
    )

    preprocess = weights.transforms()
    dataset = _imagenette_dataset(preprocess)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    all_preds_ref = []
    all_preds_docc = []
    all_logits_ref = []
    all_logits_docc = []
    all_labels = []

    with torch.no_grad():
        total = len(dataloader)
        for i, (images, labels) in enumerate(dataloader, 1):
            print(f"\r  [{i}/{total}] Processing batch...", end="", flush=True)

            # Pad the last batch if it's smaller than the batch size
            current_batch_size = images.shape[0]
            if current_batch_size < batch_size:
                padding_size = batch_size - current_batch_size
                padding = torch.zeros(
                    (padding_size, *images.shape[1:]), dtype=images.dtype
                )
                images = torch.cat([images, padding], dim=0)

            ref_out = model_ref(images)
            docc_out = program(images)

            # Truncate results if padding was added
            if current_batch_size < batch_size:
                ref_out = ref_out[:current_batch_size]
                docc_out = docc_out[:current_batch_size]

            all_preds_ref.append(ref_out.argmax(dim=1))
            all_preds_docc.append(docc_out.argmax(dim=1))
            all_logits_ref.append(ref_out)
            all_logits_docc.append(docc_out)

            mapped = torch.tensor([_IMAGENETTE_TO_IMAGENET[l.item()] for l in labels])
            all_labels.append(mapped)
        print()  # newline after progress

    preds_ref = torch.cat(all_preds_ref).numpy()
    preds_docc = torch.cat(all_preds_docc).numpy()
    logits_ref = torch.cat(all_logits_ref).numpy()
    logits_docc = torch.cat(all_logits_docc).numpy()
    labels = torch.cat(all_labels).numpy()

    metrics = _compute_metrics(preds_ref, preds_docc, labels)
    logit_corr = _compute_logit_correlation(logits_ref, logits_docc)

    print(f"\n{'='*60}")
    print("ResNet18 ImageNet Classification Correlation Report")
    print(f"{'='*60}")
    print(f"Dataset           : Imagenette validation ({len(dataset)} samples)")
    print(f"Reference (fp32)  : Top-1 accuracy = {metrics['accuracy_ref']:.4f}")
    print(f"DOCC      (bf16)  : Top-1 accuracy = {metrics['accuracy_docc']:.4f}")
    print(f"Agreement rate    : {metrics['agreement']:.4f}")
    print(f"Logit correlation : {logit_corr:.4f}")
    print(f"{'='*60}")

    # The bf16 model should agree with the fp32 reference on the vast
    # majority of samples.
    assert (
        metrics["agreement"] > 0.85
    ), f"Agreement rate {metrics['agreement']:.4f} is below threshold 0.85"

    # The logit vectors should be highly correlated.
    assert (
        logit_corr > 0.90
    ), f"Mean logit correlation {logit_corr:.4f} is below threshold 0.90"

    # The bf16 model should not lose more than 15 percentage points of
    # accuracy compared to the fp32 reference.
    assert metrics["accuracy_docc"] > metrics["accuracy_ref"] - 0.15, (
        f"DOCC accuracy {metrics['accuracy_docc']:.4f} dropped too far "
        f"below reference {metrics['accuracy_ref']:.4f}"
    )
