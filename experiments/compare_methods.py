import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import common.utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import common.utils as utils

def plot_comparison(metric, model_base, dataset, f1, f2, f3, save_dir):
    """
    Plots a comparison of BALD, US, BALD+US, and PASSIVE for a given model and metric.
    """
    num_dm_dec = int(np.round(f3 * (f1 * (f1 - 1) / 200)))
    acquisition_methods = ['BALD', 'US', 'BALD+US']
    
    # Colors and Styles
    styles = {
        'BALD': {'color': '#1f77b4', 'label': 'BALD', 'ls': '-'},     # Blue
        'US': {'color': '#2ca02c', 'label': 'US', 'ls': '-'},       # Green
        'BALD+US': {'color': '#ff7f0e', 'label': 'BALD+US', 'ls': '-'}, # Orange
        'PASSIVE': {'color': '#7f7f7f', 'label': 'PASSIVE', 'ls': '--'} # Gray Dashed
    }
    
    plt.figure(figsize=(10, 6))
    x = np.arange(1, num_dm_dec + 1)
    
    passive_data = None
    data_found = False
    
    for method in acquisition_methods:
        sub_fold = f"{model_base}_{method}"
        try:
            # We try to load the results
            # y is passive/random, y_active is the active method
            y, y_active = utils.load_test_results(metric, dataset, sub_fold, num_dm_dec, f1, f2, f3)
            
            if y_active.size == 0:
                continue
            
            data_found = True
            
            # Use the first valid y for the Passive baseline
            if passive_data is None and y.size > 0:
                passive_data = y
                
            # Plot Active Series
            n_samples = y_active.shape[1]
            mean_active = np.mean(y_active, axis=1)
            std_active = np.std(y_active, axis=1) * 2 / np.sqrt(n_samples) # 95% CI
            
            style = styles[method]
            plt.plot(x, mean_active, label=style['label'], color=style['color'], linestyle=style['ls'], linewidth=2)
            plt.fill_between(x, mean_active - std_active, mean_active + std_active, color=style['color'], alpha=0.15)
            
        except Exception as e:
            print(f"Warning: Could not load data for {sub_fold}: {e}")
            continue

    if not data_found:
        print(f"No data found for {model_base} / {metric}. Skipping plot.")
        plt.close()
        return

    # Plot Passive Baseline
    if passive_data is not None:
        n_samples_p = passive_data.shape[1]
        mean_p = np.mean(passive_data, axis=1)
        std_p = np.std(passive_data, axis=1) * 2 / np.sqrt(n_samples_p)
        
        style_p = styles['PASSIVE']
        plt.plot(x, mean_p, label=style_p['label'], color=style_p['color'], linestyle=style_p['ls'], linewidth=1.5, alpha=0.8)
        plt.fill_between(x, mean_p - std_p, mean_p + std_p, color=style_p['color'], alpha=0.1)

    # Aesthetics
    plt.title(f"{metric.upper()} Comparison: {model_base.replace('_', ' ')}", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Preferences collected", fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc='best', frameon=True, fontsize=10)
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{metric.upper()}_{model_base}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Settings
    F1 = [30]
    F2 = [4]
    F3 = [100]
    DATASET = 'default_dataset'
    
    METRICS = ['asrs', 'asps']
    MODELS = ['FTRL_LIN', 'FTRL_BT', 'BAYES_LIN', 'BAYES_BT']
    
    output_dir = os.path.join('figs', DATASET, 'comparisons')
    
    print("Starting Comparative Analysis...")
    for model in MODELS:
        for metric in METRICS:
            plot_comparison(
                metric=metric,
                model_base=model,
                dataset=DATASET,
                f1=F1[0],
                f2=F2[0],
                f3=F3[0],
                save_dir=output_dir
            )
    print("Done.")
