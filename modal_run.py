import modal
import sys
import os
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 168, 617)

image = (
    modal.Image.debian_slim(python_version="3.12")
        .pip_install("uv")
        .add_local_dir(ROOT, remote_path="/root/assignment1-basics", copy=True, ignore=[".venv", ".git", "data", "runs", "wandb", "__pycache__", "*.pyc", "tests", "*.pdf", "exercises", "*.cast", ".ruff_cache"])
        .run_commands("uv pip install --system -e /root/assignment1-basics")
)

app = modal.App("owt-train", image=image)
volume = modal.Volume.from_name("owt-data")
launches = modal.Dict.from_name("owt-launches", create_if_missing=True)

@app.function(gpu="B200", cpu=4, volumes={"/data": volume},
secrets=[modal.Secret.from_name("wandb")], timeout=5400)

def train(args: list[str], group: str, seed: int) -> int:
    os.chdir("/data")
    env = {**os.environ, "WANDB_RUN_GROUP": group, "WANDB_NAME": f"{group}-seed{seed}"}
    try:
        return subprocess.run(["python", "-u", "-m", "cs336_basics.scripts.training", *args, "--seed", str(seed)], env=env).returncode
    finally:
        volume.commit()


def git(*cmd: str) -> str:
    return subprocess.run(["git", "-C", ROOT, *cmd], capture_output=True, text=True, check=True).stdout.strip()


if __name__ == "__main__":
    args = sys.argv[1:]
    relaunch = "--relaunch" in args
    args = [a for a in args if a != "--relaunch"]

    if git("status", "--porcelain", "--untracked-files=no"):
        sys.exit("uncommitted changes to tracked files; commit first so every run maps to a commit")

    group = f"{git('rev-parse', '--abbrev-ref', 'HEAD')}-{git('rev-parse', '--short', 'HEAD')}"
    record = {"args": args, "seeds": list(SEEDS)}
    if relaunch:
        launches.put(group, record)
    elif not launches.put(group, record, skip_if_exists=True):
        sys.exit(f"{group} was already launched; pass --relaunch to run it again")

    with modal.enable_output(), app.run(detach=True):
        calls = {seed: train.spawn(args, group, seed) for seed in SEEDS}
        for seed, call in calls.items():
            print(f"[launch] {group} seed {seed}: {call.object_id}")
        for seed, call in calls.items():
            try:
                print(f"[done] seed {seed}: exit code {call.get()}")
            except Exception as e:
                print(f"[failed] seed {seed}: {e!r}")
