#!/usr/bin/env python3
"""CPU smoke test for the ZFC/SE/FiLM comparison modules."""
import torch

from basicsr.models.archs.Omni_freq_deblur_arch import (
    DCFiLM,
    SEBlock,
    CrossScaleSE,
    VanillaFiLM,
    _find_hidden_for_param_budget,
    _linear_mlp_params,
)


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def main():
    torch.manual_seed(7)
    batch, source_channels, target_channels = 2, 48, 192
    source = torch.randn(batch, source_channels, 32, 32)
    target = torch.randn(batch, target_channels, 8, 8)

    zfc = DCFiLM(source_channels, target_channels, expansion=4)
    zfc_budget = count_params(zfc)
    gamma, beta = zfc(source)
    zfc_out = gamma * target + beta

    standard_se = SEBlock(target_channels, reduction=16)
    se_out = standard_se(target)

    se_hidden = _find_hidden_for_param_budget(
        target_channels, target_channels, zfc_budget
    )
    matched_se = SEBlock(target_channels, hidden_channels=se_hidden)
    matched_se_out = matched_se(target)

    cross_hidden = _find_hidden_for_param_budget(
        source_channels, target_channels, zfc_budget
    )
    cross_se = CrossScaleSE(
        source_channels, target_channels, hidden_channels=cross_hidden
    )
    cross_se_out = cross_se(source, target)

    film = VanillaFiLM(
        source_channels, target_channels, hidden_channels=source_channels * 4
    )
    film_out = film(source, target)

    modules = {
        "ZFC": (zfc, zfc_out),
        "SE-r16": (standard_se, se_out),
        "SE-matched": (matched_se, matched_se_out),
        "Cross-SE-matched": (cross_se, cross_se_out),
        "FiLM-same-source": (film, film_out),
    }

    for name, (module, output) in modules.items():
        assert output.shape == target.shape, (name, output.shape, target.shape)
        assert torch.isfinite(output).all(), f"{name} produced non-finite values"
        print(f"{name:20s} params={count_params(module):,} output={tuple(output.shape)}")

    assert zfc_budget == 83_520, zfc_budget
    assert count_params(standard_se) == 4_812, count_params(standard_se)
    assert count_params(matched_se) == 83_352, count_params(matched_se)
    assert count_params(cross_se) == 83_578, count_params(cross_se)
    assert count_params(film) == zfc_budget, (count_params(film), zfc_budget)
    assert _linear_mlp_params(source_channels, source_channels * 4,
                              2 * target_channels) == zfc_budget

    assert gamma.min().item() >= 0.9 - 1e-6
    assert gamma.max().item() <= 1.1 + 1e-6
    assert beta.min().item() >= -0.1 - 1e-6
    assert beta.max().item() <= 0.1 + 1e-6

    print("\nAll ZFC/SE/FiLM module smoke tests passed.")


if __name__ == "__main__":
    main()
