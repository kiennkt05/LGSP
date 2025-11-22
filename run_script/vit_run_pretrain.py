import sys
import subprocess

pool_sizes = [24]
def main():
    for M in pool_sizes:
        print("Local Pool Size: {}".format(M))
        cmd = [
            sys.executable, "train.py",
            "-dataset", "cub200",
            "-base_mode", "ft_dot",
            "-new_mode", "avg_cos",
            "-gamma", str(0.1),
            "-lr_base", str(0.012),
            "-lr_new", str(0.05),
            "-lr_local", str(0.01),
            "-epochs_base", str(19),
            "-epochs_new", str(5),
            "-schedule", "Cosine",
            "-milestones", str(20),
            "-gpu", str(0),
            "-temperature", str(0.1),
            "-start_session", str(0),
            "-batch_size_base", str(32),
            "-test_batch_size", str(32),
            "-num_workers", str(2),
            "-seed", str(1),
            "-out", "PriViLege",
            "-Prompt_Token_num", str(5),
            "-lr_PromptTokens_base", str(0.01),
            "-lr_PromptTokens_novel", str(0.001),
            "-Dropout_Prompt", str(0.3),
            "-lr_Frequency_mask", str(0.1),
            "-dataroot", "CUB_200_2011",
            "-num_r", str(100),
            "-pool_size", str(M),
            "-adaptive_weighting", str(True),
        ]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()