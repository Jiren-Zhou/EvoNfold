#!/usr/bin/env python3
"""Install PKfold weights from a manifest, URL, tarball, or unpacked assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import warnings
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


DEFAULT_WEIGHT = "PKfold_test1.pt"
DEFAULT_TARGET = Path("weights/pkfold.pt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https", "file"}


def _download(url: str, target: Path) -> None:
    print(f"download: {url}")
    with urlopen(url) as response, target.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _safe_extract_tar(tar_path: Path, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive.getmembers():
            member_target = (target_dir / member.name).resolve()
            try:
                member_target.relative_to(target_root)
            except ValueError as exc:
                raise SystemExit(f"unsafe tar member path: {member.name}") from exc
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The default behavior of tarfile extraction",
                category=RuntimeWarning,
            )
            archive.extractall(target_dir)


def _find_assets_root(path: Path) -> Path:
    if (path / "weights").is_dir():
        return path
    for child in sorted(path.iterdir()):
        if child.is_dir() and (child / "weights").is_dir():
            return child
    raise SystemExit(f"could not find an assets directory with weights/ under {path}")


def _copy_file(source: Path, target: Path, expected_sha256: str | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if expected_sha256:
        actual = _sha256(target)
        if actual != expected_sha256:
            target.unlink(missing_ok=True)
            raise SystemExit(f"sha256 mismatch for {target}: expected {expected_sha256}, got {actual}")


def _install_assets_package(
    *,
    source: Path,
    root: Path,
    weight_name: str,
    target: Path,
    assets_dir: Path,
) -> None:
    package = _find_assets_root(source)
    installed_package = assets_dir / package.name
    assets_dir.mkdir(parents=True, exist_ok=True)
    if installed_package.exists():
        shutil.rmtree(installed_package)
    shutil.copytree(package, installed_package)

    selected_weight = installed_package / "weights" / weight_name
    if not selected_weight.exists():
        available = ", ".join(p.name for p in sorted((installed_package / "weights").glob("*.pt")))
        raise SystemExit(f"missing {weight_name}; available weights: {available}")
    _copy_file(selected_weight, target)
    print(f"installed assets: {installed_package.relative_to(root)}")
    print(f"installed checkpoint: {target.relative_to(root)}")


def _install_source(
    *,
    source_arg: str,
    root: Path,
    weight_name: str,
    target: Path,
    assets_dir: Path,
    expected_sha256: str | None,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        if _is_url(source_arg):
            parsed_name = Path(urlparse(source_arg).path).name or "asset"
            source = tmp_path / parsed_name
            _download(source_arg, source)
            if expected_sha256 and _sha256(source) != expected_sha256:
                raise SystemExit(f"sha256 mismatch for downloaded source: {source_arg}")
        else:
            source = Path(source_arg).expanduser().resolve()
            if not source.exists():
                raise SystemExit(f"source does not exist: {source}")
            if expected_sha256 and source.is_file() and _sha256(source) != expected_sha256:
                raise SystemExit(f"sha256 mismatch for source: {source}")

        if source.is_file() and source.suffix == ".pt":
            _copy_file(source, target)
            print(f"installed checkpoint: {target.relative_to(root)}")
            return

        if source.is_file() and tarfile.is_tarfile(source):
            extracted = tmp_path / "extracted"
            extracted.mkdir()
            _safe_extract_tar(source, extracted)
            _install_assets_package(
                source=extracted,
                root=root,
                weight_name=weight_name,
                target=target,
                assets_dir=assets_dir,
            )
            return

        if source.is_dir():
            _install_assets_package(
                source=source,
                root=root,
                weight_name=weight_name,
                target=target,
                assets_dir=assets_dir,
            )
            return

        raise SystemExit(f"unsupported source type: {source}")


def _install_manifest(root: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for asset in manifest["assets"]:
        target = root / asset["path"]
        expected = str(asset["sha256"])
        if target.exists() and _sha256(target) == expected:
            print(f"ok: {target}")
            continue
        url = str(asset["url"])
        if "TODO" in url:
            raise SystemExit("assets_manifest.json still contains a placeholder Zenodo URL.")
        _download(url, target)
        actual = _sha256(target)
        if actual != expected:
            target.unlink(missing_ok=True)
            raise SystemExit(f"sha256 mismatch for {target}: expected {expected}, got {actual}")
        print(f"ok: {target}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install a PKfold checkpoint. SOURCE may be a .pt file, a tarball "
            "containing data/ and weights/, an unpacked assets directory, or a URL."
        )
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="asset source; defaults to assets_manifest.json when present",
    )
    parser.add_argument(
        "--weight-name",
        default=DEFAULT_WEIGHT,
        help=f"weight to select from an assets package (default: {DEFAULT_WEIGHT})",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"checkpoint target (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("assets"),
        help="where to copy unpacked assets packages",
    )
    parser.add_argument("--sha256", help="optional SHA256 check for SOURCE")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / args.target
    assets_dir = root / args.assets_dir

    if args.source:
        _install_source(
            source_arg=str(args.source),
            root=root,
            weight_name=str(args.weight_name),
            target=target,
            assets_dir=assets_dir,
            expected_sha256=args.sha256,
        )
        return

    manifest_path = root / "assets_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            "No source was provided and assets_manifest.json is absent. "
            "Pass a local assets tarball, unpacked assets directory, .pt file, or URL."
        )
    _install_manifest(root, manifest_path)


if __name__ == "__main__":
    main()
