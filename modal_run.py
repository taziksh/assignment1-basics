import modal
import sys
import os
import subprocess

image = (
    modal.Image.debian_slim(python_version="3.12")
        .pip_install("uv")
        .add_local_dir("/Users/tazik/Projects/cs336/assignment1-basics/", remote_path="/root/assignment1-basics", copy=True, ignore=[".venv", ".git", "data", "runs", "wandb", "__pycache__", "*.pyc", "tests", "*.pdf", "exercises", "*.cast", ".ruff_cache"])
        .run_commands("uv pip install --system -e /root/assignment1-basics")
)

app = modal.App("owt", image=image)
volume = modal.Volume.from_name("owt-data")

@app.function(gpu="B200", cpu=4, volumes={"/data": volume},
secrets=[modal.Secret.from_name("wandb")], timeout=3600)

def train(args: list[str]):
    os.chdir("/data")
    try:
        subprocess.run(["python", "-u", "-m", "cs336_basics.scripts.training", *args], check=True)
    finally:
        volume.commit()

if __name__ == "__main__":
    call = modal.Function.from_name("owt", "train").spawn(sys.argv[1:])
