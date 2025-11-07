import os
import sys
import subprocess

# Random seeds for reproducible experiments
# seeds = [1,2,3,4,5]
seeds = [1]

# Basic configuration
project = "base"
dataset = "cub200"
# dataset = 'FGVCAircraft'
# dataset = 'iNF200'
gpu_num = "0"

# Model hyperparameters
lr_Frequency_mask = 0.1
temperature = 2

# Learning rates and dropout for 30-branch network
lr_InsVP = 0.01
Dropout_Prompt = 0.3

# Learning rates and dropout for 9-12 layer 1D convolution
lr_Block = 0.1
Dropout_Block = 0.3

# Training epochs
epochs_bases = [80]
epochs_new = 5

# Learning rates for prompt tokens (base and novel sessions)
lr_PromptTokens_base = 0.02
lr_PromptTokens_novel = 0.003

# Learning rates for classifier (base and novel sessions)
lr_base = 0.01
lr_new = 0.06

# Alternative settings for quick testing
# epochs_bases = [1]
# epochs_new = 1

# Training schedule
milestones_list = ["20"]
Prompt_Token_num = 5
lr_prompt_l2p = 0.002

# Baseline configuration (commented out)
# epochs_bases = [19]
# epochs_new = 5
# lr_PromptTokens_base = 0.01
# lr_PromptTokens_novel = 0.001
# lr_base = 0.012
# lr_new = 0.03

# Data directory path
data_dir = "CUB_200_2011"

# Worker and batch defaults (tune here if RAM is tight)
num_workers = 0
# Reduce batches for low-VRAM GPUs (e.g., RTX 3050 Ti 4GB)
batch_size_base = 8
test_batch_size = 32

# Training loop
def main():
    for seed in seeds:
        print("Pretraining -- Seed{}".format(seed))
        for i, epochs_base in enumerate(epochs_bases):
            cmd = [
                sys.executable, "train.py",
                "-project", project,
                "-dataset", dataset,
                "-base_mode", "ft_dot",
                "-new_mode", "avg_cos",
                "-gamma", str(0.1),
                "-lr_base", str(lr_base),
                "-lr_new", str(lr_new),
                "-lr_InsVP", str(lr_InsVP),
                "-decay", str(0.0005),
                "-epochs_base", str(epochs_base),
                "-epochs_new", str(epochs_new),
                "-schedule", "Cosine",
                "-milestones", milestones_list[i],
                "-gpu", gpu_num,
                "-temperature", str(16),
                "-start_session", str(0),
                "-batch_size_base", str(batch_size_base),
                "-test_batch_size", str(test_batch_size),
                "-num_workers", str(num_workers),
                "-seed", str(seed),
                "-vit",
                "-comp_out", str(1),
                "-prefix",
                "-LT",
                "-out", "PriViLege",
                "-Prompt_Token_num", str(Prompt_Token_num),
                "-lr_PromptTokens_base", str(lr_PromptTokens_base),
                "-lr_PromptTokens_novel", str(lr_PromptTokens_novel),
                "-lr_Block", str(lr_Block),
                "-Dropout_Block", str(Dropout_Block),
                "-Dropout_Prompt", str(Dropout_Prompt),
                "-temperature", str(temperature),
                "-lr_prompt_l2p", str(lr_prompt_l2p),
                "-lr_Frequency_mask", str(lr_Frequency_mask),
                "-dataroot", data_dir,
            ]

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

# Note:
# -prefix: If enabled, uses prompt to concatenate intermediate layer features
# If disabled, uses the proposed method to concatenate MLP and MSA features
# in the first two layers (main contribution of this paper)
