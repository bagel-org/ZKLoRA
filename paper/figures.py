import matplotlib.pyplot as plt
import os

# Create an output folder for the figures
os.makedirs("figs", exist_ok=True)

# Example data (dictionary of dictionaries):
# Each key is a model name, value is a dict with:
#   'params'   -> total LoRA params (integer)
#   'settings' -> total settings/compile time (sec)
#   'proof'    -> total proof generation time (sec)
#   'verify'   -> total verification time (sec)
data = {
    'distilgpt2':       {'params': 589824,    'settings': 911.99,  'proof': 759.33,  'verify': 16.56},
    'gpt2-lora1':       {'params': 2359296,   'settings': 2092.13, 'proof': 1675.58, 'verify': 32.79},
    'Llama-3.2-1B':     {'params': 851968,    'settings': 1189.01, 'proof': 991.93,  'verify': 24.91},
    'Llama-3.3-70B':    {'params': 11796480,  'settings': 4392.86, 'proof': 3749.76, 'verify': 123.11},
    'Llama-3.1-8B':     {'params': 5242880,   'settings': 1836.83, 'proof': 1527.40, 'verify': 35.79},
    'Mixtral-8x7B':     {'params': 10485760,  'settings': 2754.91, 'proof': 2357.61, 'verify': 44.30},
}

# Distinct colors for each model
colors = {
    'distilgpt2':    'blue',
    'gpt2-lora1':    'red',
    'gpt2-lora2':    'green',
    'Llama-3.2-1B':  'orange',
    'Llama-3.3-70B': 'purple',
    'Llama-3.1-8B':  'brown',
    'Mixtral-8x7B':  'magenta',
}

# Sort the model names by ascending number of LoRA params
model_names_sorted = sorted(data.keys(), key=lambda m: data[m]['params'])

def plot_graph(
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_file: str
):
    """Creates a dotted line graph comparing (LoRA params in millions) to
    some measured time metric, and saves as PDF."""
    
    plt.figure(figsize=(10, 6), dpi=120)
    plt.title(title, fontsize=16, fontweight='bold', color='darkorange')

    # We'll store the sorted x and y values in separate lists
    x_vals = []
    y_vals = []

    # Collect data in sorted order
    for name in model_names_sorted:
        params_millions = data[name][x_key] / 1e6
        x_vals.append(params_millions)
        y_vals.append(data[name][y_key])

    # Plot each model as a distinct color dot (larger markers, edges)
    for name in model_names_sorted:
        mx = data[name][x_key] / 1e6   # convert to millions
        my = data[name][y_key]
        plt.plot(
            mx, my,
            marker='o',
            markersize=12,
            markeredgecolor='black',
            markeredgewidth=1.5,
            color=colors[name],
            linestyle='None',
            label=name  # We'll create a manual legend below
        )

    # Connect points with a faint dotted line to show progression
    plt.plot(
        x_vals, y_vals,
        color='darkorange',
        linestyle='--',
        alpha=0.8,
        linewidth=2.5
    )

    plt.xlabel(xlabel, fontsize=14, color='darkorange')
    plt.ylabel(ylabel, fontsize=14, color='darkorange')

    # Create a manual legend with model-specific colors
    legend_handles = []
    for name in model_names_sorted:
        legend_handles.append(
            plt.Line2D(
                [0], [0],
                marker='o',
                markeredgecolor='black',
                markeredgewidth=1.5,
                markerfacecolor=colors[name],
                markersize=10,
                linestyle='None',
                label=name
            )
        )
    plt.legend(handles=legend_handles, loc='best', fontsize=12, frameon=False)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved figure: {output_file}")

# 1) Plot total LoRA Params (Millions) vs. Total Settings Time
plot_graph(
    x_key='params',
    y_key='settings',
    title='Total LoRA Params vs. Total Settings Time',
    xlabel='Total LoRA Params (Millions)',
    ylabel='Total Settings Time (sec)',
    output_file='figs/fig_settings.pdf'
)

# 2) Plot total LoRA Params (Millions) vs. Total Proof Generation Time
plot_graph(
    x_key='params',
    y_key='proof',
    title='Total LoRA Params vs. Total Proof Generation Time',
    xlabel='Total LoRA Params (Millions)',
    ylabel='Total Proof Time (sec)',
    output_file='figs/fig_proof.pdf'
)

# 3) Plot total LoRA Params (Millions) vs. Total Verification Time
plot_graph(
    x_key='params',
    y_key='verify',
    title='Total LoRA Params vs. Total Verification Time',
    xlabel='Total LoRA Params (Millions)',
    ylabel='Total Verification Time (sec)',
    output_file='figs/fig_verify.pdf'
)
