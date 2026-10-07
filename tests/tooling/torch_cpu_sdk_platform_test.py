"""SDK metadata rejection contracts; not model/provider execution evidence."""
from pathlib import Path
import os
import subprocess
import tempfile


def main():
    repo = Path(__file__).resolve().parents[2]
    scratch = repo / ".scratch/validation/torch-cpu-platform-contract"
    scratch.mkdir(parents=True, exist_ok=True)
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    cases = [
        ("Windows", "x86_64", "2.8.0+cpu", True),
        ("Windows", "x86_64", "2.8.0", False),
        ("Linux", "x86_64", "2.8.0+cu128", False),
        ("Darwin", "aarch64", "2.8.0", True),
        ("Darwin", "aarch64", "2.8.0+cpu", False),
        ("Darwin", "x86_64", "2.8.0", False),
    ]
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        root = Path(directory).resolve()
        assert root.is_relative_to(scratch.resolve())
        sdk = root / "sdk"
        config = sdk / "share/cmake/Torch/TorchConfig.cmake"
        config.parent.mkdir(parents=True)
        config.write_text("# metadata fixture only\n")
        script = root / "check.cmake"
        script.write_text(
            'cmake_minimum_required(VERSION 3.25)\n'
            f'include("{(repo / "src/compute_provider_native/torch_cpu_fetch.cmake").as_posix()}")\n'
            f'_hyle_torch_cpu_check_root("{sdk.as_posix()}")\n')
        for system, processor, version, expected in cases:
            (sdk / "build-version").write_text(version)
            result = subprocess.run(["cmake", "-DCMAKE_SYSTEM_NAME=" + system,
                                     "-DHYLE_TARGET_PROCESSOR=" + processor,
                                     "-P", str(script)], capture_output=True, timeout=10, **flags)
            assert (result.returncode == 0) == expected, (system, processor, version, result.stderr)
        assert config.read_text() == "# metadata fixture only\n"
    print("PASS: platform version/architecture contracts reject CUDA and foreign labels; external roots unchanged")


if __name__ == "__main__":
    main()
