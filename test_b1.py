import numpy as np
import pytest
from yaslp.phantom import Ellipse, shepp_logan

from b1_corrected_t1_vfa import fit_t1_vfa, flash_equation

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    HAS_MPL = True
except ModuleNotFoundError:
    HAS_MPL = False

GRID_SHAPE = (32, 32, 32)
HALF_SIZE = 16


@pytest.fixture()
def ground_truth_maps() -> dict[str, np.ndarray]:
    ellipses = {
        "s0": [
            Ellipse(value=1, radius=(0.8, 0.9, 0.8), center=(0, 0, 0), phi=0),  # FG
            Ellipse(
                value=1, radius=(0.35, 0.12, 0.12), center=(-0.35, 0.1, 0), phi=1.9
            ),  # A
            Ellipse(
                value=2, radius=(0.44, 0.15, 0.15), center=(0.35, 0.1, 0), phi=1.2
            ),  # B
        ],
        "t1": [
            Ellipse(value=1, radius=(0.8, 0.9, 0.8), center=(0, 0, 0), phi=0),  # FG
            Ellipse(value=3, radius=(0.12, 0.12, 0.40), center=(0, 0.5, 0), phi=0),  # C
            Ellipse(
                value=4, radius=(0.18, 0.18, 0.18), center=(0, -0.6, 0), phi=0
            ),  # D
        ],
        "b1": [
            Ellipse(value=1, radius=(0.8, 0.9, 0.8), center=(0, 0, 0), phi=0),  # FG
        ],
    }
    return {k: shepp_logan(GRID_SHAPE, ell) for k, ell in ellipses.items()}


def test_b1(ground_truth_maps):
    # params
    fas = np.array([5, 11, 18, 24, 32])
    tr = 16
    # simulate the data (no nosie)
    signal = flash_equation(
        ground_truth_maps["b1"][..., None] * fas,
        tr,
        ground_truth_maps["s0"][..., None],
        ground_truth_maps["t1"][..., None],
    )
    assert signal.shape == GRID_SHAPE + (fas.size,)

    ## show the data
    if HAS_MPL:
        _, axes = plt.subplots(ncols=fas.size)
        vmin, vmax = signal.min(), signal.max()
        for i, ax in enumerate(axes):
            ax.imshow(
                signal[..., HALF_SIZE, i].T,
                origin="lower",
                interpolation="none",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"FA = {fas[i]}°")
            ax.axis("off")
        plt.show()

    # recover the maps
    recovered_maps = dict(
        zip(["s0", "t1"], fit_t1_vfa(signal, fas, tr, ground_truth_maps["b1"]))
    )

    # Check the relative residuals where the signal exists to begin with
    fg_mask = ground_truth_maps["s0"] > 0
    residual = {}
    for var in ["s0", "t1"]:
        residual[var] = np.where(
            fg_mask,
            (ground_truth_maps[var] - recovered_maps[var]) / ground_truth_maps[var],
            np.nan,
        )

        assert np.allclose(
            residual[var][fg_mask], 0, atol=2e-2
        ), f"{var}: residual too large"

    ## plot the result
    if HAS_MPL:
        _, axes = plt.subplots(ncols=3, nrows=2)
        for ax_row, var in zip(axes, ["s0", "t1"]):
            ax_row[0].text(-0.1, 0.5, var, transform=ax_row[0].transAxes, ha="right")

            # Share the same norm for the maps
            norm_map = Normalize(
                ground_truth_maps[var].min(), ground_truth_maps[var].max()
            )

            # Set up a fancy colormap for the residuals
            res_min, res_max = (
                np.nanmin(residual[var]) < 0,
                np.nanmax(residual[var]) > 0,
            )
            if res_min < 0 and res_max > 0:
                norm_resid = TwoSlopeNorm(0, res_min, res_max)
                cmap = "RdBu_r"
            else:
                norm_resid = Normalize(res_min, res_max)
                cmap = "Reds" if res_min > 0 else "Blues_r"

            for ax, (arr, norm, cmap) in zip(
                ax_row,
                [
                    (ground_truth_maps[var], norm_map, None),
                    (recovered_maps[var], norm_map, None),
                    (residual[var], norm_resid, cmap),
                ],
            ):
                vmin, vmax = ground_truth_maps[var].min(), ground_truth_maps[var].max()
                im = ax.imshow(
                    arr[..., HALF_SIZE].T,
                    origin="lower",
                    interpolation="none",
                    norm=norm,
                    cmap=cmap,
                )
                plt.colorbar(im, ax=ax)
                ax.axis("off")

        for ax, title in zip(axes[0], ["GT", "Reco", "GT-Reco"]):
            ax.set_title(title)
        plt.show()
