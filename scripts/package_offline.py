import argparse
from importlib import metadata
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import venv

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename


ROOT = Path(__file__).resolve().parents[1]


def installed_lock(requirements_path):
    pending = [Requirement(line.strip()) for line in requirements_path.read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    versions = {}
    while pending:
        requirement = pending.pop()
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        name = canonicalize_name(requirement.name)
        version = metadata.version(name)
        if not requirement.specifier.contains(version, prereleases=True):
            raise RuntimeError(f"Installed {name}=={version} does not satisfy {requirement}")
        if name in versions:
            continue
        versions[name] = version
        pending.extend(Requirement(value) for value in metadata.requires(name) or [])
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    versions = installed_lock(ROOT / "requirements.txt")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    wheels = args.output_dir / "wheels"
    wheels.mkdir()
    lock = args.output_dir / "requirements.txt"
    lock.write_text("".join(f"{name}=={version}\n" for name, version in sorted(versions.items())), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--no-deps",
                    "--dest", str(wheels), "-r", str(lock)], check=True)
    hashes = {}
    for path in sorted(wheels.glob("*.whl")):
        name, _version, _build, _tags = parse_wheel_filename(path.name)
        hashes[canonicalize_name(name)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if set(hashes) != set(versions):
        raise RuntimeError("Wheel bundle does not contain the complete locked dependency set")
    lock.write_text("".join(f"{name}=={version} --hash=sha256:{hashes[name]}\n"
                            for name, version in sorted(versions.items())), encoding="utf-8")
    manifest = {"python": sys.version, "platform": platform.platform(), "packages": versions,
                "wheel_sha256": hashes, "offline_verified": False}
    if args.verify:
        with tempfile.TemporaryDirectory(prefix="branchseed-offline-") as temporary:
            venv.EnvBuilder(with_pip=True).create(temporary)
            python = Path(temporary) / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            commands = [
                [str(python), "-m", "pip", "install", "--no-index", "--require-hashes", "--find-links", str(wheels.resolve()), "-r", str(lock.resolve())],
                [str(python), "-m", "pip", "check"],
                [str(python), "-c", "import numpy, scipy, SimpleITK, nibabel, matplotlib, psutil, trimesh, skimage; print('Offline imports passed')"],
            ]
            logs = []
            for command in commands:
                result = subprocess.run(command, capture_output=True, text=True)
                logs.append(result.stdout + result.stderr)
                (args.output_dir / "verification.log").write_text("\n".join(logs), encoding="utf-8")
                result.check_returncode()
            manifest["offline_verified"] = True
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
