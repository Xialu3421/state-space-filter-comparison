import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_simulation(
    y: np.ndarray,
    mu: np.ndarray,
    mu_hat: dict[str, np.ndarray],
    out: Path | None = None,
    title: str | None = None,
) -> None:
    plt.figure(figsize=(10, 3))
    plt.plot(y, color="k", lw=0.7, label="y")
    plt.plot(mu, color="grey", lw=1.0, label="μ (truth)")
    for name, series in mu_hat.items():
        plt.plot(series, lw=1.0, label=name, alpha=0.8)
    plt.legend(ncol=4, fontsize=8)
    if title:
        plt.title(title)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
