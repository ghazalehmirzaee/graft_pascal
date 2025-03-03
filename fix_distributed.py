#!/usr/bin/env python3
import os
import sys
import re


def fix_train_py():
    with open("train.py", "r") as f:
        content = f.read()

    # Find the part where local_rank is initialized and device is set
    # Look for the main function or where the device is initialized
    # We need to make sure torch.cuda.set_device(local_rank) is called

    # Pattern to find where the device is set
    device_pattern = r"(device = torch\.device\([^)]+\))"

    # Replacement that ensures each process gets a unique GPU
    device_replacement = """if distributed:
        # Ensure each process gets its own GPU
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        print(f"Process {local_rank} using device: {device}")
    else:
        device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))"""

    # Apply the replacement
    modified_content = re.sub(device_pattern, device_replacement, content)

    # If the pattern wasn't found, we need to add the code to the right place
    if modified_content == content:
        print("Could not find device initialization pattern. Trying alternative approach...")

        # Find the beginning of the main function
        main_pattern = r"def main\(args\):"
        main_match = re.search(main_pattern, content)

        if main_match:
            # Insert after the first few lines of the main function
            main_start = main_match.end()
            # Find the first line after def main(args):
            next_lines = content[main_start:].strip().split('\n', 3)
            insert_pos = main_start + len(next_lines[0]) + len(next_lines[1]) + 2  # +2 for newlines

            device_code = """
    # Initialize distributed training
    local_rank = args.local_rank
    distributed = torch.cuda.device_count() > 1 and 'WORLD_SIZE' in os.environ
    
    if distributed:
        # Ensure each process gets its own GPU
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        print(f"Process {local_rank} using device: {device}")
    else:
        device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

"""
            modified_content = content[:insert_pos] + device_code + content[insert_pos:]
        else:
            print("Could not find main function. Please modify the code manually.")
            return False

    # Write the modified content back to train.py
    with open("train.py", "w") as f:
        f.write(modified_content)

    print("Successfully updated device initialization in train.py")
    return True


def fix_run_distributed_sh():
    with open("run_distributed.sh", "r") as f:
        content = f.read()

    # Look for module loading patterns and fix if needed
    module_pattern = r"module load"
    if re.search(module_pattern, content):
        modified_content = re.sub(r"module load (.+)",
                                  r"command -v module &> /dev/null && module load \1 || echo \"Module command not available for \1\"",
                                  content)
    else:
        modified_content = content

    # Properly set CUDA_VISIBLE_DEVICES
    launcher_pattern = r"(python -m torch\.distributed\.launch|torchrun)"
    if re.search(launcher_pattern, modified_content):
        # Replace the launch command with torchrun and ensure proper environment variables
        launch_section = re.search(r"# Run training script.+?experiment_name \"[^\"]+\"", modified_content, re.DOTALL)
        if launch_section:
            old_launch = launch_section.group(0)

            # Create a new launch command that ensures each process sees all GPUs
            new_launch = """# Run training script with torchrun
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Make all GPUs visible
torchrun \\
    --nproc_per_node=8 \\
    --rdzv_backend=c10d \\
    train.py \\
    --config config/default.yaml \\
    --start_phase phase2_finetune \\
    --experiment_name "finetune_distributed" \\
    --local_rank \${SLURM_LOCALID:-0}"""

            modified_content = modified_content.replace(old_launch, new_launch)

    # Write the modified content back to run_distributed.sh
    with open("run_distributed.sh", "w") as f:
        f.write(modified_content)

    print("Successfully updated run_distributed.sh")

    # Make it executable
    os.chmod("run_distributed.sh", 0o755)
    return True


if __name__ == "__main__":
    print("Fixing distributed training configuration...")
    train_fixed = fix_train_py()
    run_fixed = fix_run_distributed_sh()

    if train_fixed and run_fixed:
        print("All fixes applied successfully!")
        print("To run your distributed training, execute:\n  ./run_distributed.sh")
    else:
        print("Some fixes could not be applied automatically.")
        print("Please review the changes and make any necessary manual adjustments.")

