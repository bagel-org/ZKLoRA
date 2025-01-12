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

def plot_graph(x_key, y_key, title, xlabel, ylabel, output_file):
    plt.figure(figsize=(6,4), dpi=100)
    plt.title(title)

    # We'll store the sorted x and y values in separate lists
    x_vals = []
    y_vals = []

    # Populate lists using sorted model names
    for name in model_names_sorted:
        # Convert param count to millions
        params_millions = data[name][x_key] / 1e6
        x_vals.append(params_millions)
        y_vals.append(data[name][y_key])

    # Plot each model as a distinct color dot
    for i, name in enumerate(model_names_sorted):
        model_x = data[name][x_key] / 1e6  # convert to millions
        model_y = data[name][y_key]
        plt.plot(model_x, model_y, marker='o', color=colors[name], label=name)

    # Connect the points with a faint dotted line in sorted order
    plt.plot(x_vals, y_vals, 'k:', alpha=0.4)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Create a custom legend
    legend_handles = []
    for name in model_names_sorted:
        patch_color = colors[name]
        legend_handles.append(
            plt.Line2D([0], [0],
                       marker='o', color=patch_color, label=name,
                       markerfacecolor=patch_color, markersize=6,
                       linestyle='None')
        )
    plt.legend(handles=legend_handles, loc='best')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved figure: {output_file}")


# 1) Total LoRA Params (Millions) vs. Total Settings Time
plot_graph(
    x_key='params',
    y_key='settings',
    title='Total LoRA Params vs. Total Settings Time',
    xlabel='Total LoRA Params (Millions)',
    ylabel='Total Settings Time (sec)',
    output_file='figs/fig_settings.pdf'
)

# 2) Total LoRA Params (Millions) vs. Total Proof Generation Time
plot_graph(
    x_key='params',
    y_key='proof',
    title='Total LoRA Params vs. Total Proof Time',
    xlabel='Total LoRA Params (Millions)',
    ylabel='Total Proof Generation Time (sec)',
    output_file='figs/fig_proof.pdf'
)

# 3) Total LoRA Params (Millions) vs. Total Verification Time
plot_graph(
    x_key='params',
    y_key='verify',
    title='Total LoRA Params vs. Total Verification Time',
    xlabel='Total LoRA Params (Millions)',
    ylabel='Total Verification Time (sec)',
    output_file='figs/fig_verify.pdf'
)
