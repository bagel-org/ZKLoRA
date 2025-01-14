import matplotlib.pyplot as plt
import os

# Create a folder for saving figures
os.makedirs("figs", exist_ok=True)

# Example data:
#   'lora_count'   -> number_of_loras (x-axis)
#   'avg_params'   -> average LoRA params (float), used for marker size
#   'total_verify' -> total verification time (sec)
#   'total_proof'  -> total proof generation time (sec)
#   'total_settings' -> total settings time (sec)
data = {
    'distilgpt2': {
        'lora_count': 24,
        'avg_params': 24576.0,
        'total_verify': 16.56,
        'total_proof': 759.33,
        'total_settings': 911.99
    },
    'gpt2': {
        'lora_count': 48,
        'avg_params': 49152.0,
        'total_verify': 32.79,
        'total_proof': 1675.58,
        'total_settings': 2092.13
    },
    'Llama-3.2-1B': {
        'lora_count': 32,
        'avg_params': 26624.0,
        'total_verify': 24.91,
        'total_proof': 991.93,
        'total_settings': 1189.01
    },
    'Llama-3.3-70B': {
        'lora_count': 80,
        'avg_params': 147456.0,
        'total_verify': 123.11,
        'total_proof': 3749.76,
        'total_settings': 4392.86
    },
    'Llama-3.1-8B': {
        'lora_count': 32,
        'avg_params': 163840.0,
        'total_verify': 35.79,
        'total_proof': 1527.40,
        'total_settings': 1836.83
    },
    'Mixtral-8x7B': {
        'lora_count': 32,
        'avg_params': 327680.0,
        'total_verify': 44.30,
        'total_proof': 2357.61,
        'total_settings': 2754.91
    },
}

# Colors for each model
colors = {
    'distilgpt2':    'blue',
    'gpt2':          'red',
    'Llama-3.2-1B':  'orange',
    'Llama-3.3-70B': 'purple',
    'Llama-3.1-8B':  'brown',
    'Mixtral-8x7B':  'green',
}

def plot_graph(
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_file: str
):
    """
    Creates a dotted-line graph of y_key vs. x_key, with marker size
    proportional to 'avg_params'. In the legend, however, all dots are uniform size.
    The label shows 'Avg LoRA Size=XXK'.
    """
    plt.figure(figsize=(7,5), dpi=120)
    plt.title(title, fontsize=12, fontweight='bold')

    # Sort by ascending x_key
    sorted_models = sorted(data.keys(), key=lambda m: data[m][x_key])

    x_vals = []
    y_vals = []

    handles = []
    labels = []

    # Collect numeric arrays for the dotted line
    for model in sorted_models:
        x_val = data[model][x_key]
        y_val = data[model][y_key]
        x_vals.append(x_val)
        y_vals.append(y_val)

    # Plot each model as a distinct point
    for model in sorted_models:
        x_val = data[model][x_key]
        y_val = data[model][y_key]
        avg_p = data[model]['avg_params']

        # Make the marker size scale with sqrt of avg_p
        marker_size = (avg_p ** 0.5) / 25.0

        # Show average param in thousands
        avg_k = avg_p / 1000.0
        label_str = f"{model} (Avg LoRA Size={avg_k:.1f}K)"

        # Create the plot for each point
        handle, = plt.plot(
            x_val,
            y_val,
            marker='o',
            markersize=marker_size,
            markeredgecolor='black',
            markeredgewidth=0.7,
            color=colors[model],
            linestyle='None'
        )
        handles.append(handle)
        labels.append(label_str)

    # Connect points with a dotted line
    plt.plot(x_vals, y_vals, 'k:', alpha=0.6, linewidth=2)

    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)

    # Build a custom legend
    leg = plt.legend(handles, labels, loc='best', fontsize=9)

    # Make the legend dots a uniform size (e.g., 6),
    # ignoring the actual plotted marker sizes
    for lh in leg.legend_handles:
        lh.set_markersize(6.0)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved figure: {output_file}")

# 1) total verification time vs. #lora
plot_graph(
    x_key='lora_count',
    y_key='total_verify',
    title='Number of LoRA Modules vs. Total Verification Time',
    xlabel='Number of LoRA Modules',
    ylabel='Total Verification Time (sec)',
    output_file='figs/fig_verify_vs_numlora.pdf'
)

# 2) total proof time vs. #lora
plot_graph(
    x_key='lora_count',
    y_key='total_proof',
    title='Number of LoRA Modules vs. Total Proof Generation Time',
    xlabel='Number of LoRA Modules',
    ylabel='Total Proof Generation Time (sec)',
    output_file='figs/fig_proof_vs_numlora.pdf'
)

# 3) total settings time vs. #lora
plot_graph(
    x_key='lora_count',
    y_key='total_settings',
    title='Number of LoRA Modules vs. Total Settings Time',
    xlabel='Number of LoRA Modules',
    ylabel='Total Settings Time (sec)',
    output_file='figs/fig_settings_vs_numlora.pdf'
)
