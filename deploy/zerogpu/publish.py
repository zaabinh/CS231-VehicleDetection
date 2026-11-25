"""Upload the vehicle demo to a free ZeroGPU Space without changing the project README."""
import argparse
import hashlib
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, SpaceHardware
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils import get_stored_tokens

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="zaabinh/vehicle-detection-benchmark")
    parser.add_argument("--token-name")
    args = parser.parse_args()
    api = HfApi(token=get_stored_tokens()[args.token_name] if args.token_name else None)
    files = {p: ROOT / p for p in ["demo/app.py", "demo/inference.py", "demo/space.py"]}
    files.update({p.relative_to(ROOT).as_posix(): p for p in (ROOT / "demo/examples").glob("*.jpg")})
    for name in ["yolov8.pt", "best_FasterRCNN.pth", "best_SSD.pth"]:
        p = ROOT / "models" / name
        with p.open("rb") as f:
            if f.read(100).startswith(b"version https://git-lfs"):
                raise RuntimeError(f"Run git lfs pull before publishing {p.name}")
        files[f"models/{name}"] = p
    files["README.md"] = ROOT / "deploy/zerogpu/README.md"
    files["requirements.txt"] = ROOT / "deploy/zerogpu/requirements.txt"
    files["packages.txt"] = ROOT / "deploy/zerogpu/packages.txt"
    try:
        api.space_info(args.repo)
    except RepositoryNotFoundError:
        api.create_repo(args.repo, repo_type="space", space_sdk="gradio",
                        space_hardware=SpaceHardware.ZERO_A10G, private=False)
    print(f"https://huggingface.co/spaces/{args.repo}", flush=True)
    runtime = api.get_space_runtime(args.repo)
    if (runtime.requested_hardware or runtime.hardware) != "zero-a10g":
        raise RuntimeError("Existing Space is not on free ZeroGPU; upload stopped.")
    info = api.space_info(args.repo, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    operations = []
    for target, path in files.items():
        old = remote.get(target)
        if old:
            if old.lfs:
                with path.open("rb") as f:
                    same = hashlib.file_digest(f, "sha256").hexdigest() == old.lfs.sha256
            else:
                data = path.read_bytes()
                same = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == old.blob_id
            if same:
                continue
        operations.append(CommitOperationAdd(path_in_repo=target, path_or_fileobj=path))
    if operations:
        result = api.create_commit(repo_id=args.repo, repo_type="space", operations=operations,
                                   parent_commit=info.sha, commit_message="Deploy trained vehicle detectors on free ZeroGPU")
        print(result.commit_url, flush=True)
    else:
        print("Space is already up to date.", flush=True)


if __name__ == "__main__":
    main()
