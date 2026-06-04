"""
Diagnostic figure: E_link and E_est error metrics over elicitation steps.

Loads all error_scores.csv files for a given experiment configuration and
produces a 4-panel figure:
  - Top-left:    E_link_rel (all algorithms)
  - Top-right:   epsilon_t (FTRL only)
  - Bottom-left: rho_t – fraction of unresolved candidates (BAYES only)
  - Bottom-right: E_est_FTLT – LT estimation-error envelope (FTRL only)

Usage (from repo root):
    python -m experiments.plot_diagnostics
or pass keyword arguments to `make_diagnostic_figure`.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Palette / style
# ---------------------------------------------------------------------------
STYLE = {
    'FTRL-BT_BALD':   dict(color='#1f77b4', ls='-',  label='FTRL-BT'),
    'FTRL-LIN_BALD':  dict(color='#ff7f0e', ls='-',  label='FTRL-LIN'),
    'BAYES-BT_BALD':  dict(color='#2ca02c', ls='--', label='BAYES-BT'),
    'BAYES-LIN_BALD': dict(color='#d62728', ls='--', label='BAYES-LIN'),
}

FTRL_ALGS  = ['FTRL-BT_BALD', 'FTRL-LIN_BALD']
BAYES_ALGS = ['BAYES-BT_BALD', 'BAYES-LIN_BALD']
ALL_ALGS   = FTRL_ALGS + BAYES_ALGS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_alg(base_dir: str, alg: str) -> pd.DataFrame | None:
    """
    Concatenate error_scores.csv across all table_* sub-folders for `alg`.
    Returns a DataFrame with an extra 'table' column, or None if no data.
    """
    alg_dir = os.path.join(base_dir, alg)
    if not os.path.isdir(alg_dir):
        return None

    frames = []
    for entry in sorted(os.scandir(alg_dir), key=lambda e: e.name):
        if entry.is_dir() and entry.name.startswith('table_'):
            csv = os.path.join(entry.path, 'error_scores.csv')
            if os.path.isfile(csv):
                df = pd.read_csv(csv)
                df['table'] = entry.name
                frames.append(df)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_all(base_dir: str) -> dict[str, pd.DataFrame]:
    """Return {alg: DataFrame} for every algorithm that has data."""
    return {alg: df for alg in ALL_ALGS
            if (df := _load_alg(base_dir, alg)) is not None}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _mean_ci(data: pd.DataFrame, col: str):
    """Per-step mean and 95 % CI (2 * SEM) across tables."""
    grp = data.groupby('step')[col]
    mean = grp.mean()
    sem  = grp.sem().fillna(0)
    return mean.index.to_numpy(), mean.to_numpy(), sem.to_numpy()


def _plot_band(ax, steps, mean, sem, style: dict, alpha=0.15):
    ax.plot(steps, mean, **style)
    ax.fill_between(steps, mean - 2 * sem, mean + 2 * sem,
                    color=style['color'], alpha=alpha, linewidth=0)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_diagnostic_figure(
    base_dir: str,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Build and optionally save the 4-panel diagnostic figure.

    Parameters
    ----------
    base_dir  : path to the config-level output folder
                (e.g. 'samples/test_dataset/f1_10_f2_2_f3_100')
    save_path : if given, save to this file (PDF recommended)
    show      : call plt.show() after building
    """
    data = load_all(base_dir)
    if not data:
        raise FileNotFoundError(f"No error_scores.csv found under {base_dir}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Error Diagnostics — Active Elicitation', fontsize=14, y=1.01)

    ax_link  = axes[0, 0]
    ax_eps   = axes[0, 1]
    ax_rho   = axes[1, 0]
    ax_estft = axes[1, 1]

    # ── Panel 1: E_link_rel (all algorithms) ────────────────────────────────
    for alg, df in data.items():
        steps, mean, sem = _mean_ci(df, 'E_link_rel')
        _plot_band(ax_link, steps, mean, sem, STYLE[alg])

    ax_link.set_xlabel('Elicitation step')
    ax_link.set_ylabel(r'$\widehat{\mathcal{E}}_{\mathrm{link}}^{\mathrm{IG,rel}}$')
    ax_link.set_title('Link error (all algorithms)')
    ax_link.legend(handles=[
        plt.Line2D([0], [0], **{k: v for k, v in s.items() if k != 'label'},
                   label=s['label'])
        for alg, s in STYLE.items() if alg in data
    ], fontsize=9)
    ax_link.grid(True, linestyle='--', alpha=0.4)
    ax_link.set_ylim(bottom=0)

    # ── Panel 2: epsilon_t (FTRL) ────────────────────────────────────────────
    for alg in FTRL_ALGS:
        if alg not in data:
            continue
        steps, mean, sem = _mean_ci(data[alg], 'epsilon_t')
        _plot_band(ax_eps, steps, mean, sem, STYLE[alg])

    ax_eps.axhline(1.0, color='black', ls=':', lw=1.2, label=r'$\epsilon_t = 1$ (unreliable)')
    ax_eps.set_xlabel('Elicitation step')
    ax_eps.set_ylabel(r'$\widehat{\epsilon}_t^{\mathrm{coord}}$')
    ax_eps.set_title(r'Cubic distortion $\epsilon_t$ (FTRL only)')
    ax_eps.legend(handles=[
        plt.Line2D([0], [0], **{k: v for k, v in STYLE[alg].items() if k != 'label'},
                   label=STYLE[alg]['label'])
        for alg in FTRL_ALGS if alg in data
    ] + [plt.Line2D([0], [0], color='black', ls=':', lw=1.2, label=r'$\epsilon_t=1$')],
        fontsize=9)
    ax_eps.grid(True, linestyle='--', alpha=0.4)
    ax_eps.set_ylim(bottom=0)

    # ── Panel 3: rho_t (BAYES) ───────────────────────────────────────────────
    for alg in BAYES_ALGS:
        if alg not in data:
            continue
        steps, mean, sem = _mean_ci(data[alg], 'rho_t')
        _plot_band(ax_rho, steps, mean, sem, STYLE[alg])

    ax_rho.set_xlabel('Elicitation step')
    ax_rho.set_ylabel(r'$\rho_t^{\mathrm{MC}}$')
    ax_rho.set_title(r'Unresolved fraction $\rho_t$ (BAYES only)')
    ax_rho.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax_rho.legend(handles=[
        plt.Line2D([0], [0], **{k: v for k, v in STYLE[alg].items() if k != 'label'},
                   label=STYLE[alg]['label'])
        for alg in BAYES_ALGS if alg in data
    ], fontsize=9)
    ax_rho.grid(True, linestyle='--', alpha=0.4)
    ax_rho.set_ylim(0, 1.05)

    # ── Panel 4: E_est_FTLT (FTRL) ───────────────────────────────────────────
    floor = 1e-6   # log-scale floor to avoid log(0)
    for alg in FTRL_ALGS:
        if alg not in data:
            continue
        steps, mean, sem = _mean_ci(data[alg], 'E_est_FTLT')
        # replace inf (epsilon_t >= 1 steps) with NaN so the line breaks there
        mean = np.where(np.isfinite(mean), mean, np.nan)
        mean = np.clip(mean, floor, None)
        sem  = np.clip(sem,  0,     None)
        _plot_band(ax_estft, steps, mean, sem, STYLE[alg])

    ax_estft.set_yscale('log')
    ax_estft.set_xlabel('Elicitation step')
    ax_estft.set_ylabel(r'$\widehat{\mathcal{E}}_{\mathrm{est},t}^{\mathrm{LT}}$  (log scale)')
    ax_estft.set_title('LT estimation-error envelope (FTRL only)')
    ax_estft.legend(handles=[
        plt.Line2D([0], [0], **{k: v for k, v in STYLE[alg].items() if k != 'label'},
                   label=STYLE[alg]['label'])
        for alg in FTRL_ALGS if alg in data
    ], fontsize=9)
    ax_estft.grid(True, linestyle='--', alpha=0.4, which='both')

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f'Saved: {save_path}')

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot error diagnostics from a simulation run.')
    parser.add_argument('--base_dir', default='samples/test_dataset/f1_10_f2_2_f3_100',
                        help='Path to the config-level output folder (contains ALG sub-folders).')
    parser.add_argument('--save', default=None,
                        help='Output file path (e.g. figs/diagnostics.pdf). Omit to skip saving.')
    parser.add_argument('--no-show', action='store_true', help='Do not call plt.show().')
    args = parser.parse_args()

    make_diagnostic_figure(
        base_dir=args.base_dir,
        save_path=args.save,
        show=not args.no_show,
    )
