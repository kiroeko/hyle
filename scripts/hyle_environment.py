"""Discover installed native tools for the three Hyle development scripts.

Only child-process environments are changed. CMake owns project dependencies;
this module neither installs software nor replaces an existing build toolchain.
"""
from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import shutil
import subprocess
import tempfile


class SetupError(RuntimeError):
    pass


@dataclass
class Environment:
    env: dict[str, str]
    system: str
    arch: str
    cmake: str
    ctest: str
    cc: str
    cxx: str
    generator: str
    definitions: dict[str, str]


_ROOT = Path(__file__).resolve().parents[1]
_MIN_CMAKE = (3, 26)


def _run(args: list[str], env: dict[str, str]) -> str:
    try:
        result = subprocess.run(args, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=30,
                                **({"creationflags": subprocess.CREATE_NO_WINDOW}
                                   if os.name == "nt" else {}))
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SetupError(f"Cannot run {args[0]}: {error}") from error
    if result.returncode:
        raise SetupError(f"{' '.join(args)} failed:\n{result.stdout}{result.stderr}".strip())
    return result.stdout + result.stderr


def _prepend(env: dict[str, str], directories: list[str | Path]) -> None:
    values = [str(path) for path in directories if Path(path).is_dir()]
    values.extend(env.get("PATH", "").split(os.pathsep))
    unique = []
    seen: set[str] = set()
    for value in values:
        key = os.path.normcase(value)
        if value and key not in seen:
            unique.append(value)
            seen.add(key)
    env["PATH"] = os.pathsep.join(unique)


def _which(name: str, env: dict[str, str]) -> str | None:
    found = shutil.which(name, path=env.get("PATH", ""))
    # clang++ may be a symlink to clang: its invocation name selects C++ driver
    # behavior and runtime linkage. Make the path absolute without dereferencing.
    return os.path.abspath(found) if found else None


def _restore_cached_compilers(cache: dict[str, str]) -> None:
    # VS generators keep the selected compiler in platform metadata rather
    # than CMakeCache.txt. Read the cache's own CMake version, never a newer
    # installed toolset or an unrelated CMakeFiles directory.
    directory = cache.get("CMAKE_CACHEFILE_DIR")
    parts = [cache.get(f"CMAKE_CACHE_{part}_VERSION") for part in ("MAJOR", "MINOR", "PATCH")]
    if not directory or not all(parts):
        return
    metadata = Path(directory) / "CMakeFiles" / ".".join(parts)
    for language in ("C", "CXX"):
        key = f"CMAKE_{language}_COMPILER"
        source = metadata / f"CMake{language}Compiler.cmake"
        if not cache.get(key) and source.is_file():
            match = re.search(rf'set\({key}\s+"([^"\n]+)"\)', source.read_text(encoding="utf-8"))
            if match:
                cache[key] = match[1]


def _compiler_pair(cc: str, cxx: str, env: dict[str, str]) -> tuple[str, str, str]:
    paths = [_which(name, env) for name in (cc, cxx)]
    if not all(paths):
        raise SetupError(f"Compiler pair not found: {cc}, {cxx}. See docs/developer/setup/.")
    probes = []
    for path in paths:
        banner = _run([path, "--version"], env)
        family = "clang" if "clang" in banner.lower() else "gcc"
        if family == "clang":
            match = re.search(r"(?:clang )?version\s+(\d+)", banner, re.I)
        else:
            match = re.match(r"\s*(\d+)", _run([path, "-dumpfullversion", "-dumpversion"], env))
        if not match or int(match[1]) < {"gcc": 14, "clang": 18}[family]:
            raise SetupError(f"{path} requires GCC 14+ or Clang/AppleClang 18+. See docs/developer/setup/.")
        probes.append((family, int(match[1])))
    if probes[0] != probes[1]:
        raise SetupError(f"CC/CXX must be a matching compiler family and major version: {paths}")
    return paths[0], paths[1], probes[0][0]


def _unix_compilers(compiler: str, cache: dict[str, str], env: dict[str, str],
                    system: str) -> tuple[str, str]:
    cached = cache.get("CMAKE_CXX_COMPILER")
    if cached:
        default_cc = re.sub(r"clang\+\+", "clang", cached)
        default_cc = re.sub(r"g\+\+", "gcc", default_cc)
        if Path(default_cc).name == "c++":
            default_cc = str(Path(default_cc).with_name("cc"))
        cc, cxx, family = _compiler_pair(cache.get("CMAKE_C_COMPILER", default_cc), cached, env)
        if compiler not in {"auto", family}:
            raise SetupError(f"This build uses {family}; choose another build directory for {compiler}.")
        return cc, cxx
    if compiler == "auto" and (env.get("CC") or env.get("CXX")):
        if not env.get("CC") or not env.get("CXX"):
            raise SetupError("Set both CC and CXX, or unset both for automatic discovery.")
        return _compiler_pair(env["CC"], env["CXX"], env)[:2]
    candidates: list[tuple[str, str]] = []
    if system == "Darwin" and compiler in {"auto", "clang"}:
        try:
            candidates.append((_run(["xcrun", "--find", "clang"], env).strip(),
                               _run(["xcrun", "--find", "clang++"], env).strip()))
        except SetupError:
            pass
    if compiler in {"auto", "gcc"}:
        candidates += [(f"gcc-{version}", f"g++-{version}") for version in range(14, 31)]
        candidates.append(("gcc", "g++"))
    if compiler in {"auto", "clang"}:
        candidates += [(f"clang-{version}", f"clang++-{version}") for version in range(18, 31)]
        candidates += [("clang", "clang++"), ("cc", "c++")]
    for cc, cxx in candidates:
        if not _which(cc, env) or not _which(cxx, env):
            continue
        try:
            pair = _compiler_pair(cc, cxx, env)
            if compiler == "auto" or pair[2] == compiler:
                return pair[:2]
        except SetupError:
            continue
    raise SetupError("No supported native C/C++ pair found. Install GCC 14+ or Clang/AppleClang 18+; "
                     "on macOS select the installed Xcode/Command Line Tools. See docs/developer/setup/.")


def _vs_environment(installation: Path, version: str, arch: str,
                    env: dict[str, str]) -> dict[str, str]:
    active = env.get("VSINSTALLDIR", "").rstrip("\\/")
    if (active and os.path.normcase(active) == os.path.normcase(str(installation))
            and env.get("VCTOOLSVERSION", "").rstrip("\\/") == version
            and env.get("VSCMD_ARG_TGT_ARCH") == arch and env.get("VSCMD_VER")
            and env.get("WINDOWSSDKDIR") and env.get("INCLUDE") and env.get("LIB")):
        return env
    commands = ["@echo off", "chcp 65001 >nul"]
    previous = Path(active) / "Common7/Tools/VsDevCmd.bat"
    if env.get("VSCMD_VER") and previous.is_file():
        commands.append(f'call "{previous}" -clean_env >nul')
    devcmd = installation / "Common7/Tools/VsDevCmd.bat"
    if not devcmd.is_file():
        raise SetupError(f"Visual Studio environment script is missing: {devcmd}. Repair the C++ workload.")
    commands += [f'call "{devcmd}" -no_logo -arch={arch} -host_arch=x64 -vcvars_ver={version} >nul',
                 "if errorlevel 1 exit /b %errorlevel%", "set"]
    scratch = _ROOT / ".scratch/environment"
    scratch.mkdir(parents=True, exist_ok=True)
    # A batch file avoids cmd/Python nested command-string quoting and PATH's
    # 8191-character command limit. /u preserves Unicode environment values.
    with tempfile.TemporaryDirectory(prefix="vs-", dir=scratch) as directory:
        script = Path(directory) / "environment.cmd"
        script.write_text("\r\n".join(commands) + "\r\n", encoding="utf-8")
        try:
            result = subprocess.run([env.get("COMSPEC", "cmd.exe"), "/u", "/d", "/c", str(script)],
                                    env=env, capture_output=True, timeout=60,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SetupError(f"Cannot initialize Visual Studio: {error}") from error
    if result.returncode:
        raise SetupError(f"Cannot initialize Visual Studio {version}. Repair the C++ workload and Windows SDK. "
                         "See docs/developer/setup/windows.md.")
    selected = {}
    for line in result.stdout.decode("utf-16-le", errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if key and separator:
            selected[key.upper()] = value
    if not selected.get("INCLUDE") or not selected.get("LIB") or not selected.get("WINDOWSSDKDIR"):
        raise SetupError("Visual Studio did not provide INCLUDE/LIB/Windows SDK. Repair its C++ workload.")
    return selected


def _windows_compilers(compiler: str, cache: dict[str, str], env: dict[str, str],
                       generator: str | None) -> tuple[dict[str, str], str, str, Path, str]:
    if compiler not in {"auto", "msvc", "clang", "clang-cl"}:
        raise SetupError("Windows supports msvc or clang-cl. See docs/developer/setup/windows.md.")
    cached = cache.get("CMAKE_CXX_COMPILER", "")
    requested = cached
    requested_cc = cache.get("CMAKE_C_COMPILER", "")
    if not cached and compiler == "auto" and (env.get("CC") or env.get("CXX")):
        if not env.get("CC") or not env.get("CXX"):
            raise SetupError("Set both CC and CXX, or unset both for automatic discovery.")
        requested, requested_cc = env["CXX"], env["CC"]
        if not re.search(r"(^|[/\\])(cl|clang-cl)(\.exe)?$", requested, re.I):
            raise SetupError("Windows CC/CXX must select MSVC or clang-cl. See docs/developer/setup/windows.md.")
    family = "clang-cl" if "clang" in Path(requested).name.lower() else "msvc"
    if cached and compiler != "auto" and compiler.replace("clang-cl", "clang") != family.replace("clang-cl", "clang"):
        raise SetupError(f"This build uses {family}; choose another build directory for {compiler}.")
    installation = cache.get("CMAKE_GENERATOR_INSTANCE", "").split(",version=", 1)[0]
    required = cache.get("CMAKE_VS_PLATFORM_TOOLSET_VERSION", "")
    arch = {"win32": "x86", "x86": "x86", "arm64": "arm64"}.get(
        cache.get("CMAKE_GENERATOR_PLATFORM", "").split(",", 1)[0].lower(), "x64")
    for path in (requested, requested_cc, cache.get("CMAKE_LINKER", "")):
        match = re.search(r"^(.*)[/\\]VC[/\\]Tools[/\\]MSVC[/\\]([^/\\]+)[/\\]bin[/\\]Host[^/\\]+[/\\]([^/\\]+)[/\\]", path, re.I)
        if match:
            installation = installation or match[1]
            required = match[2]
            arch = match[3].lower()
            break
    if not installation:
        # A standalone Ninja header check has only its clang-cl path, with no
        # generator instance or linker metadata. Keep that VS installation too.
        for path in (requested, requested_cc):
            match = re.search(r"^(.*)[/\\]VC[/\\]Tools[/\\]Llvm[/\\]", path, re.I)
            if match:
                installation = match[1]
                break
    toolset = cache.get("CMAKE_GENERATOR_TOOLSET", "")
    match = re.search(r"(?:^|,)version=([^,]+)", toolset)
    if match:
        required = match[1]
    major = re.match(r"Visual Studio (\d+) ", generator or "")
    active = env.get("VSINSTALLDIR", "")
    if not installation and not major and active and (Path(active) / "Common7/Tools/VsDevCmd.bat").is_file():
        installation = active
        required = required or env.get("VCTOOLSVERSION", "").rstrip("\\/")
    if not installation:
        vswhere = _which("vswhere", env)
        if not vswhere:
            vswhere = str(Path(env.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) /
                          "Microsoft Visual Studio/Installer/vswhere.exe")
        args = [vswhere, "-latest", "-products", "*", "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-format", "json", "-utf8"]
        if major:
            args += ["-version", f"[{major[1]}.0,{int(major[1]) + 1}.0)"]
        try:
            instances = json.loads(_run(args, env))
            installation = instances[0]["installationPath"]
        except (SetupError, ValueError, KeyError, IndexError) as error:
            raise SetupError("Install Visual Studio/Build Tools with Desktop development with C++, "
                             "MSVC 19.43+ and Windows SDK. See docs/developer/setup/windows.md.") from error
    install = Path(installation).resolve()
    tools = install / "VC/Tools/MSVC"
    versions = []
    for path in tools.glob("*"):
        if not re.fullmatch(r"\d+\.\d+\.\d+", path.name):
            continue
        version = tuple(map(int, path.name.split(".")))
        if version < (14, 43):
            continue
        if required and path.name != required and not path.name.startswith(required + "."):
            continue
        if not required and ("v143" in toolset or (major and major[1] == "17")) and version >= (14, 50):
            continue
        if not required and ("v145" in toolset or (major and major[1] == "18")) and version < (14, 50):
            continue
        if (path / f"bin/Hostx64/{arch}/cl.exe").is_file():
            versions.append((version, path.name))
    if not versions:
        raise SetupError(f"No matching MSVC 19.43+ toolset '{required}' in {tools}. "
                         "Install it with Visual Studio Installer; an existing cache will not switch toolchains.")
    version = max(versions)[1]
    env = _vs_environment(install, version, arch, env)
    _prepend(env, [install / "VC/Tools/Llvm/x64/bin", install / "VC/Tools/Llvm/bin"])
    use_clang = (family == "clang-cl" if requested else compiler in {"clang", "clang-cl"})
    cxx = requested or ("clang-cl.exe" if use_clang else str(tools / version / f"bin/Hostx64/{arch}/cl.exe"))
    cc = requested_cc or cxx
    cc, cxx = _which(cc, env), _which(cxx, env)
    if not cc or not cxx:
        raise SetupError("Selected compiler is missing; install the matching MSVC or LLVM component. "
                         "See docs/developer/setup/windows.md.")
    if use_clang:
        banner = _run([cxx, "--version"], env)
        match = re.search(r"clang version\s+(\d+)", banner)
        if not match or int(match[1]) < 18:
            raise SetupError("clang-cl 18+ is required. Install the Visual Studio LLVM component.")
    return env, cc, cxx, install, version


def windows_compiler_environment(cache: dict[str, str]) -> dict[str, str]:
    """Restore one selected VS compiler environment without probing build tools."""
    selected = dict(cache)
    _restore_cached_compilers(selected)
    env = {key.upper(): value for key, value in os.environ.items()}
    env, *_ = _windows_compilers("auto", selected, env, selected.get("CMAKE_GENERATOR"))
    return env


def _cmake_tools(env: dict[str, str], system: str) -> tuple[str, str]:
    candidates = [_which("cmake", env)]
    executable = "cmake.exe" if system == "Windows" else "cmake"
    candidates += [str(Path(directory) / executable) for directory in env.get("PATH", "").split(os.pathsep) if directory]
    if system == "Windows":
        candidates.append(str(Path(env.get("PROGRAMFILES", "C:/Program Files")) / "CMake/bin/cmake.exe"))
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        try:
            match = re.search(r"cmake version\s+(\d+)\.(\d+)", _run([candidate, "--version"], env))
        except SetupError:
            continue
        if match and tuple(map(int, match.groups())) >= _MIN_CMAKE:
            ctest = Path(candidate).with_name("ctest.exe" if system == "Windows" else "ctest")
            if ctest.is_file():
                return candidate, str(ctest)
    raise SetupError("CMake/CTest 3.26+ are required. Install current CMake and retry. "
                     "See docs/developer/setup/README.md.")


def _cuda(env: dict[str, str], cache: dict[str, str], system: str) -> dict[str, str]:
    if system not in {"Windows", "Linux"}:
        raise SetupError("CUDA requires Windows or Linux; on Apple Silicon choose the apple backend.")
    suffix = "nvcc.exe" if system == "Windows" else "nvcc"
    cached = cache.get("CMAKE_CUDA_COMPILER")
    candidates = [cached] if cached else []
    if not cached:
        candidates += [env["CUDACXX"]] if env.get("CUDACXX") else []
        candidates += [str(Path(root) / "bin" / suffix) for root in
                       (cache.get("CUDAToolkit_ROOT"), env.get("CUDAToolkit_ROOT"),
                        env.get("CUDATOOLKIT_ROOT"), env.get("CUDA_PATH"), env.get("CUDA_HOME")) if root]
        if system == "Windows":
            roots = glob.glob(str(Path(env.get("PROGRAMFILES", "C:/Program Files")) /
                                  "NVIDIA GPU Computing Toolkit/CUDA/v*"))
        else:
            roots = glob.glob("/usr/local/cuda*")
        candidates += [str(Path(root) / "bin" / suffix) for root in sorted(roots, reverse=True)]
        candidates += [_which(suffix, env)]
    for candidate in candidates:
        if not candidate or (system == "Linux" and (candidate.lower().endswith(".exe") or "\\" in candidate)):
            continue
        path = Path(candidate).resolve()
        if not path.is_file():
            continue
        try:
            match = re.search(r"release\s+(\d+)\.(\d+)", _run([str(path), "--version"], env))
        except SetupError:
            continue
        if match and tuple(map(int, match.groups())) >= (13, 3):
            # CUDA 13 Windows layouts keep nvcc in bin and runtime DLLs in
            # bin/x64. CMake's native-architecture probe executes a CUDA
            # program, so locating only the compiler can hide a working GPU.
            directories = [path.parent]
            if system == "Windows":
                directories.append(path.parent / "x64")
            _prepend(env, directories)
            return {"CMAKE_CUDA_COMPILER": str(path), "CUDAToolkit_ROOT": str(path.parent.parent)}
    message = "Cached CUDA compiler is missing or unsupported; restore that toolkit" if cached else "Install the host's CUDA Toolkit 13.3+"
    raise SetupError(message + ". WSL requires Linux nvcc, not nvcc.exe. See docs/developer/setup/README.md.")


def _apple(env: dict[str, str], system: str, arch: str) -> None:
    if system != "Darwin" or arch != "aarch64":
        raise SetupError("The apple backend requires macOS on Apple Silicon.")

    def available(candidate: dict[str, str]) -> bool:
        try:
            for tool in ("metal", "metallib"):
                _run(["xcrun", "-sdk", "macosx", tool, "--version"], candidate)
            return True
        except SetupError:
            return False

    if available(env):
        return
    if not env.get("TOOLCHAINS"):
        directories = [Path("/Library/Developer/Toolchains"),
                       Path.home() / "Library/Developer/Toolchains"]
        try:
            directories.append(Path(_run(["xcode-select", "-p"], env).strip()) / "Toolchains")
        except SetupError:
            pass
        for directory in directories:
            for toolchain in sorted(directory.glob("*Metal*.xctoolchain")):
                identifier = toolchain.stem
                info = toolchain / "ToolchainInfo.plist"
                if info.is_file():
                    try:
                        metadata = plistlib.loads(info.read_bytes())
                        identifier = metadata.get("Identifier", metadata.get("CFBundleIdentifier", identifier))
                    except (OSError, ValueError, plistlib.InvalidFileException):
                        continue
                candidate = env.copy()
                candidate["TOOLCHAINS"] = identifier
                if available(candidate):
                    env["TOOLCHAINS"] = identifier
                    return
    raise SetupError("Metal/metalLib are unavailable through xcrun. Select your installed Xcode "
                     "with xcode-select and install its Metal Toolchain component; then retry. "
                     "See docs/developer/setup/macos.md.")


def find_openssl(environment: Environment) -> dict[str, str]:
    """Hint CMake at an existing development package; never acquire OpenSSL.

    An empty result deliberately retains FindOpenSSL's normal platform search
    and its installation diagnostic. Do not substitute runtime-only DLLs for
    the headers and link libraries needed by TLS consumers.
    """
    env = environment.env
    roots = [Path(env["OPENSSL_ROOT_DIR"])] if env.get("OPENSSL_ROOT_DIR") else []
    if environment.system == "Darwin":
        prefixes = [Path("/opt/homebrew"), Path("/usr/local")]
        if env.get("HOMEBREW_PREFIX"):
            prefixes.insert(0, Path(env["HOMEBREW_PREFIX"]))
        roots += [prefix / "opt/openssl@3" for prefix in prefixes]
    elif environment.system == "Windows":
        if env.get("VCPKG_ROOT"):
            triplet = "arm64-windows" if environment.arch == "aarch64" else "x64-windows"
            roots.append(Path(env["VCPKG_ROOT"]) / "installed" / triplet)
        roots += [Path(env.get("PROGRAMFILES", "C:/Program Files")) / "OpenSSL-Win64",
                  Path("C:/OpenSSL-Win64")]
    for root in roots:
        if not (root / "include/openssl/ssl.h").is_file():
            continue
        libraries = [root / "lib", root / "lib64"]
        if environment.system == "Windows":
            arch = "arm64" if environment.arch == "aarch64" else "x64"
            libraries += [root / f"lib/VC/{arch}/{crt}" for crt in ("MD", "MT")]
        for directory in libraries:
            ssl = any(directory.glob("libssl.*")) or any(directory.glob("ssl.*"))
            crypto = any(directory.glob("libcrypto.*")) or any(directory.glob("crypto.*"))
            if ssl and crypto:
                return {"OPENSSL_ROOT_DIR": str(root.resolve())}
    return {}


def discover_environment(compiler: str = "auto", cache: dict[str, str] | None = None,
                         backends=(), generator: str | None = None) -> Environment:
    cache = dict(cache or {})
    _restore_cached_compilers(cache)
    system = platform.system()
    if system not in {"Windows", "Linux", "Darwin"}:
        raise SetupError(f"Unsupported host system: {system}. Use Windows, Linux or macOS.")
    arch = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}.get(
        platform.machine().lower(), platform.machine().lower())
    compiler = compiler.lower()
    if compiler not in {"auto", "msvc", "clang-cl", "gcc", "clang"}:
        raise SetupError(f"Unknown compiler '{compiler}'; use auto, msvc, clang-cl, gcc or clang.")
    env = dict(os.environ)
    if system == "Windows":
        env = {key.upper(): value for key, value in env.items()}
    else:
        _prepend(env, ["/opt/homebrew/bin", "/usr/local/bin"])
    cached_generator = cache.get("CMAKE_GENERATOR")
    if cached_generator and generator and cached_generator != generator:
        raise SetupError(f"This build uses '{cached_generator}'; choose another directory for '{generator}'.")
    selected_generator = cached_generator or generator
    definitions: dict[str, str] = {}
    if system == "Windows":
        env, cc, cxx, installation, version = _windows_compilers(compiler, cache, env, selected_generator)
        _prepend(env, [installation / "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin",
                       installation / "Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja"])
    else:
        cc, cxx = _unix_compilers(compiler, cache, env, system)
    cmake, ctest = _cmake_tools(env, system)
    _prepend(env, [Path(cmake).parent])
    if not selected_generator:
        selected_generator = "Ninja" if _which("ninja", env) else ("NMake Makefiles" if system == "Windows" else "Unix Makefiles")
    build_tool = cache.get("CMAKE_MAKE_PROGRAM")
    if not build_tool:
        name = "ninja" if selected_generator.startswith("Ninja") else {
            "NMake Makefiles": "nmake", "Unix Makefiles": "make"}.get(selected_generator)
        if name:
            build_tool = _which(name, env)
            if not build_tool:
                raise SetupError(f"Generator '{selected_generator}' needs {name}; install it or choose an available generator.")
    if build_tool:
        if not Path(build_tool).is_file():
            raise SetupError(f"Cached build tool is missing: {build_tool}. Restore it; do not switch an existing cache implicitly.")
        definitions["CMAKE_MAKE_PROGRAM"] = build_tool
    if system == "Windows" and selected_generator.startswith("Visual Studio "):
        definitions["CMAKE_GENERATOR_INSTANCE"] = str(installation)
        definitions["CMAKE_GENERATOR_TOOLSET"] = cache.get("CMAKE_GENERATOR_TOOLSET") or f"version={version}"
    definitions.update(CMAKE_C_COMPILER=cc, CMAKE_CXX_COMPILER=cxx)
    env.update(CC=cc, CXX=cxx)
    if "cuda" in backends:
        definitions.update(_cuda(env, cache, system))
    if "apple" in backends:
        _apple(env, system, arch)
    return Environment(env, system, arch, cmake, ctest, cc, cxx, selected_generator, definitions)
