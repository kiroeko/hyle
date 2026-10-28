"""Shared implementation of init.py, build.py and test.py (standard library only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Hyle scripts require Python 3.10+. Run them with an installed Python 3.10+ interpreter; "
                     "see docs/developer/setup/README.md for platform prerequisites.")

from hyle_environment import Environment, SetupError, discover_environment, find_openssl
from hyle_resources import find_glslang, prepare_resources


ROOT = Path(__file__).resolve().parents[1]
CONFIGURATIONS = ("Release", "Debug", "RelWithDebInfo", "MinSizeRel")
_BOOL_TRUE = {"1", "ON", "YES", "TRUE", "Y"}
_VERBOSE = False


def _uses_compute(components) -> bool:
    # CMake owns the dependency closure; only these public roots request a
    # native route. Selecting a utility library must not probe a GPU toolkit.
    return bool({"engine", "all", "compute", "graph", "compute_stream"}.intersection(components))


class CommandError(SetupError):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def read_cache(directory: Path) -> dict[str, str]:
    path = directory / "CMakeCache.txt"
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^([^#/:][^:=]*):[^=]*=(.*)$", line)
        if match:
            values[match[1]] = match[2]
    return values


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(str(Path(right).resolve()))


def _check_source(cache: dict[str, str]) -> None:
    home = cache.get("CMAKE_HOME_DIRECTORY")
    if cache and (not home or not _same_path(home, ROOT)):
        raise SetupError("This build directory belongs to another source tree. Choose another --name or --build-dir.")
    if cache.get("HYLE_DEV_FACTORS"):
        raise SetupError("This is an internal factor build. Use a product build with init.py, or continue using CMake directly.")


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except (OSError, ValueError) as error:
        raise SetupError(f"Cannot read {path}: {error}. Run scripts/init.py again.") from error


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _current() -> Path:
    return ROOT / ".scratch/hyle/current.json"


def _previous_initialization() -> dict:
    if _current().is_file():
        try:
            return _json(_current())
        except SetupError:
            pass  # init can repair a damaged selection; build/test report it.
    return {}


def _state_path(directory: Path) -> Path:
    return directory / "hyle-init.json"


def _name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value) or value in {".", ".."}:
        raise argparse.ArgumentTypeError("use a short name containing letters, digits, '_', '-' or '.'")
    return value


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def _selected_directory(args: argparse.Namespace, *, required: bool) -> Path | None:
    if args.build_dir:
        return args.build_dir.expanduser().resolve()
    if args.name:
        return ROOT / ".scratch/build" / args.name
    if required:
        if not _current().is_file():
            raise SetupError("No initialized build is selected. First run: python scripts/init.py --backends cpu")
        current = _json(_current())
        if current.get("error"):
            raise SetupError("The last initialization failed. Fix it and rerun scripts/init.py; "
                             "or select a previous configuration with --name or --build-dir.")
        path = current.get("build_dir")
        if not isinstance(path, str) or not path:
            raise SetupError("No build directory is selected. Run scripts/init.py again.")
        return Path(path).expanduser().resolve()
    return None


def parse_backends(value: str, system: str, arch: str, *, complete_math: bool = True) -> tuple[str, ...]:
    requested = [part.strip().lower() for part in re.split(r"[,;]", value)]
    if not requested or any(not part for part in requested):
        raise SetupError("--backends must be a nonempty comma-separated set: cpu,cuda,metal,vulkan or all.")
    if "all" in requested:
        if requested != ["all"]:
            raise SetupError("Use --backends all alone, or list the backends you need.")
        requested = ["vulkan"]
        if system in {"Windows", "Linux"} and (arch == "x86_64" or not complete_math):
            requested = ["cpu", "cuda", "vulkan"]
        elif system == "Darwin" and arch == "aarch64":
            requested = ["cpu", "metal", "vulkan"]
    requested = ["apple" if part == "metal" else part for part in requested]
    unknown = set(requested) - {"cpu", "cuda", "apple", "vulkan"}
    if unknown:
        raise SetupError(f"Unknown backend(s): {', '.join(sorted(unknown))}. Use cpu,cuda,metal,vulkan or all.")
    if "cuda" in requested:
        if system not in {"Windows", "Linux"}:
            raise SetupError("The CUDA backend requires Windows or Linux.")
        if complete_math and arch != "x86_64":
            raise SetupError("The complete CUDA backend supports Windows/Linux x86_64. Use Vulkan on this target.")
    if "apple" in requested and (system != "Darwin" or arch != "aarch64"):
        raise SetupError("The Metal backend requires macOS on Apple Silicon. Use CUDA or Vulkan on this target.")
    return tuple(item for item in ("cpu", "cuda", "apple", "vulkan") if item in requested)


def _host() -> tuple[str, str]:
    system = platform.system()
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    if system not in {"Windows", "Linux", "Darwin"}:
        raise SetupError("Hyle initialization supports Windows, Linux and macOS.")
    return system, arch


def _cache_directory(args: argparse.Namespace, cache: dict[str, str], environment: Environment) -> Path:
    if args.cache_dir:
        path = args.cache_dir
    elif cache.get("HYLE_DEP_SOURCE_CACHE"):
        path = Path(cache["HYLE_DEP_SOURCE_CACHE"])
    elif environment.env.get("HYLE_DEP_SOURCE_CACHE"):
        path = Path(environment.env["HYLE_DEP_SOURCE_CACHE"])
    elif environment.system == "Windows":
        path = Path(environment.env.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "hyle/dependencies/v1"
    elif environment.system == "Darwin":
        path = Path.home() / "Library/Caches/hyle/dependencies/v1"
    else:
        path = Path(environment.env.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "hyle/dependencies/v1"
    return path.expanduser().resolve()


def _boolean(args: argparse.Namespace, name: str, cache: dict[str, str], key: str) -> bool:
    value = getattr(args, name)
    return cache.get(key, "OFF").upper() in _BOOL_TRUE if value is None else value


def _selection(args: argparse.Namespace, cache: dict[str, str], saved_config: str | None = None) -> dict:
    system, arch = _host()
    backend_value = args.backends if args.backends is not None else cache.get("HYLE_COMPUTE_ROUTES", "cpu")
    components = args.components if args.components is not None else cache.get("HYLE_COMPONENTS", "engine")
    products = [part.strip().lower() for part in re.split(r"[,;]", components)]
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", part) for part in products):
        raise SetupError("--components must be product names separated by commas, or all.")
    if "all" in products and products != ["all"]:
        raise SetupError("Use --components all alone, or list the products you need.")
    aten = cache.get("HYLE_WITH_ATEN", "ON").upper() in _BOOL_TRUE if args.aten is None else args.aten
    routes = parse_backends(backend_value, system, arch,
                            complete_math=aten and bool({"engine", "all", "compute", "graph"}.intersection(products)))
    if not _uses_compute(products):
        routes = ()
    configuration = args.config or cache.get("CMAKE_BUILD_TYPE") or saved_config or "Release"
    if configuration not in CONFIGURATIONS:
        # Existing multi-config/custom CMake trees remain consumable by build/test.
        raise SetupError(f"Unsupported initialization configuration '{configuration}'. Use {', '.join(CONFIGURATIONS)}.")
    selected = {"routes": routes, "config": configuration, "aten": aten,
                "components": tuple(dict.fromkeys(products)),
                "sanitizer": args.sanitizer or cache.get("HYLE_SANITIZER", "off")}
    for name, key in (("tests", "HYLE_BUILD_TESTS"), ("static", "HYLE_STATIC"),
                      ("tls", "HYLE_IO_TLS"), ("http2", "HYLE_IO_HTTP2"),
                      ("trace", "HYLE_TRACE"), ("offline", "FETCHCONTENT_FULLY_DISCONNECTED")):
        selected[name] = _boolean(args, name, cache, key)
    if (selected["tls"] or selected["http2"]) and not {"all", "io"}.intersection(products):
        raise SetupError("TLS and HTTP/2 require the io product. Include io in --components, or omit that subset.")
    return selected


def _definitions(values: list[str]) -> dict[str, str]:
    result = {}
    managed = {"HYLE_COMPONENTS", "HYLE_COMPUTE_ROUTES", "HYLE_STATIC", "HYLE_BUILD_TESTS", "HYLE_WITH_ATEN",
               "HYLE_INSTALL", "HYLE_SANITIZER", "HYLE_IO_TLS", "HYLE_IO_HTTP2", "HYLE_TRACE",
               "HYLE_DEP_SOURCE_CACHE", "HYLE_DEP_BUILD_DIR", "CMAKE_BUILD_TYPE",
               "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_GENERATOR", "CMAKE_HOME_DIRECTORY",
               "FETCHCONTENT_FULLY_DISCONNECTED", "FETCHCONTENT_UPDATES_DISCONNECTED"}
    for value in values:
        key, separator, data = value.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise SetupError("-D/--define expects NAME=VALUE, for example -DCMAKE_POSITION_INDEPENDENT_CODE=ON.")
        if key in managed:
            raise SetupError(f"{key} is managed by the script's named options; use those instead of -D.")
        result[key] = data
    return result


def _command(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def _run(command: list[str], environment: Environment, log: Path | None = None) -> None:
    if _VERBOSE:
        print("+ " + _command(command), flush=True)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"Running {log.stem.removeprefix('hyle-')}; log: {log}", flush=True)
    try:
        # Stream CMake's original diagnostics and keep one bounded-purpose log
        # per action. Never capture or persist the process environment.
        with (log.open("w", encoding="utf-8") if log else open(os.devnull, "w")) as stream:
            stream.write("+ " + _command(command) + "\n")
            with subprocess.Popen(command, cwd=ROOT, env=environment.env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                  errors="replace") as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                code = process.wait()
    except OSError as error:
        raise SetupError(f"Cannot start {command[0]}: {error}. Rerun scripts/init.py to check installed tools.") from error
    if code:
        location = f" See {log}." if log else ""
        raise CommandError(code, f"Command failed (exit {code}).{location}")


def _display(directory: Path, selected: dict, environment: Environment, cache_dir: Path) -> None:
    routes = ",".join("metal" if route == "apple" else route for route in selected["routes"])
    print(f"Hyle: {environment.system}/{environment.arch}; components={','.join(selected['components'])}; "
          f"backends={routes or 'not requested'}; {selected['config']}; "
          f"{'static' if selected['static'] else 'shared'}; tests={'ON' if selected['tests'] else 'OFF'}; "
          f"ATen={'ON' if selected['aten'] else 'OFF'}")
    print(f"Compiler: {environment.cxx}\nBuild directory: {directory}\nShared dependency cache: {cache_dir}")


def initialize(args: argparse.Namespace) -> int:
    directory = _selected_directory(args, required=False)
    # Repeating init without choices reuses the last directory. Explicit choices
    # without --name get their own deterministic tree, leaving older modes reusable.
    choices = ("backends", "components", "config", "compiler", "generator", "tests", "static",
               "sanitizer", "tls", "http2", "trace", "offline", "cache_dir", "aten")
    if directory is None and not args.define and not any(getattr(args, key) is not None for key in choices):
        if _current().is_file():
            previous = _previous_initialization()
            if isinstance(previous.get("build_dir"), str):
                # Initialization may retry a failed tree; build/test may not.
                directory = Path(previous["build_dir"]).expanduser().resolve()
    cache = read_cache(directory) if directory else {}
    _check_source(cache)
    saved_config = None
    if directory and cache.get("CMAKE_CONFIGURATION_TYPES") and _state_path(directory).is_file():
        try:
            saved_config = _json(_state_path(directory)).get("config")
        except SetupError:
            pass  # Reinitialization can repair damaged script metadata.
    selected = _selection(args, cache, saved_config)
    extra = _definitions(args.define)
    if cache.get("CMAKE_CUDA_COMPILER"):
        cuda_root = cache.get("CUDAToolkit_ROOT") or str(Path(cache["CMAKE_CUDA_COMPILER"]).parent.parent)
        for key, previous in (("CMAKE_CUDA_COMPILER", cache["CMAKE_CUDA_COMPILER"]), ("CUDAToolkit_ROOT", cuda_root)):
            if extra.get(key) and not _same_path(extra[key], previous):
                raise SetupError(f"{key} differs from this build's CUDA toolchain. Choose another --name or --build-dir.")
    environment = discover_environment(compiler=args.compiler or "auto", cache=dict(cache, **extra),
                                       backends=selected["routes"], generator=args.generator)
    if "vulkan" in selected["routes"] and cache.get("HYLE_VULKAN_GLSLANG"):
        environment.definitions["HYLE_VULKAN_GLSLANG"] = cache["HYLE_VULKAN_GLSLANG"]
    environment.definitions.update(extra)
    if selected["tls"] and not (extra.get("OPENSSL_ROOT_DIR") or cache.get("OPENSSL_ROOT_DIR")):
        environment.definitions.update(find_openssl(environment))
    cache_dir = _cache_directory(args, cache, environment)
    if directory is None:
        identity = dict(selected, cc=environment.cc, cxx=environment.cxx,
                        cuda=environment.definitions.get("CMAKE_CUDA_COMPILER"),
                        generator=environment.generator, system=environment.system,
                        arch=environment.arch, extra=extra, cache=str(cache_dir))
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
        label = "-".join("metal" if item == "apple" else item for item in selected["routes"]) or "library"
        directory = ROOT / ".scratch/build" / f"{label}-{selected['config'].lower()}-{digest}"
        cache = read_cache(directory)
        _check_source(cache)
        # A deterministic directory can already exist from an earlier invocation.
        if cache and not _same_path(cache.get("CMAKE_CXX_COMPILER", environment.cxx), environment.cxx):
            raise SetupError("The selected build has another compiler. Use a new --name or --build-dir.")
    for key, value in (("CMAKE_C_COMPILER", environment.cc), ("CMAKE_CXX_COMPILER", environment.cxx),
                       ("CMAKE_CUDA_COMPILER", environment.definitions.get("CMAKE_CUDA_COMPILER")),
                       ("CMAKE_GENERATOR", environment.generator)):
        if value and cache.get(key) and (cache[key] != value if key == "CMAKE_GENERATOR" else not _same_path(cache[key], value)):
            raise SetupError(f"{key} differs from this build's toolchain. Choose another --name or --build-dir.")
    _display(directory, selected, environment, cache_dir)
    definitions = dict(environment.definitions)
    definitions.update({
        "CMAKE_C_COMPILER": environment.cc, "CMAKE_CXX_COMPILER": environment.cxx,
        "HYLE_COMPONENTS": ";".join(selected["components"]),
        "HYLE_SANITIZER": selected["sanitizer"],
        "HYLE_DEP_SOURCE_CACHE": str(cache_dir),
        "HYLE_BUILD_BENCH": extra.get("HYLE_BUILD_BENCH", cache.get("HYLE_BUILD_BENCH", "OFF")),
        "HYLE_INSTALL": cache.get("HYLE_INSTALL", "OFF"),
        "Python3_EXECUTABLE": sys.executable,
    })
    if not ("Visual Studio" in environment.generator or "Multi-Config" in environment.generator):
        definitions["CMAKE_BUILD_TYPE"] = selected["config"]
    if _uses_compute(selected["components"]):
        definitions["HYLE_COMPUTE_ROUTES"] = ";".join(selected["routes"])
        definitions["HYLE_WITH_ATEN"] = "ON" if selected["aten"] else "OFF"
    if {"all", "io"}.intersection(selected["components"]):
        definitions["HYLE_IO_TLS"] = "ON" if selected["tls"] else "OFF"
        definitions["HYLE_IO_HTTP2"] = "ON" if selected["http2"] else "OFF"
    for option, key in (("tests", "HYLE_BUILD_TESTS"), ("static", "HYLE_STATIC"), ("trace", "HYLE_TRACE"),
                        ("offline", "FETCHCONTENT_FULLY_DISCONNECTED"),
                        ("offline", "FETCHCONTENT_UPDATES_DISCONNECTED")):
        definitions[key] = "ON" if selected[option] else "OFF"
    if args.dry_run:
        if "vulkan" in selected["routes"]:
            shader = find_glslang(environment)
            if shader:
                definitions["HYLE_VULKAN_GLSLANG"] = str(shader)
            else:
                print("Vulkan: initialization will prepare the locked host glslang tool in the shared tools cache.")
    else:
        _write_json(_current(), {"build_dir": str(directory)})
        _write_json(_state_path(directory), {"ready": False, "source_dir": str(ROOT), "config": selected["config"]})
        if "vulkan" in selected["routes"]:
            definitions.update(prepare_resources(environment, cache_dir, selected["offline"], args.jobs))
    command = [environment.cmake, "-S", str(ROOT), "-B", str(directory), "-G", environment.generator]
    command += [f"-D{key}={value}" for key, value in definitions.items()]
    if args.dry_run:
        print("+ " + _command(command))
        return 0
    _run(command, environment, directory / "hyle-configure.log")
    _write_json(_state_path(directory), {"ready": True, "source_dir": str(ROOT), "config": selected["config"]})
    print("Initialized. Next: python scripts/build.py" + ("; then python scripts/test.py" if selected["tests"] else ""))
    return 0


def _configured(args: argparse.Namespace) -> tuple[Path, dict[str, str], str, Environment]:
    directory = _selected_directory(args, required=True)
    cache = read_cache(directory)
    if not cache:
        raise SetupError(f"No configured build at {directory}. First run scripts/init.py for this --name or --build-dir.")
    _check_source(cache)
    state = _json(_state_path(directory)) if _state_path(directory).is_file() else {}
    if state and (not state.get("ready") or not _same_path(state.get("source_dir", ""), ROOT)):
        raise SetupError(f"Initialization did not finish for {directory}. Rerun scripts/init.py with this --name or --build-dir.")
    configuration = args.config or state.get("config") or cache.get("CMAKE_BUILD_TYPE") or "Release"
    configurations = cache.get("CMAKE_CONFIGURATION_TYPES", "").split(";")
    if cache.get("CMAKE_BUILD_TYPE") and configuration != cache["CMAKE_BUILD_TYPE"]:
        raise SetupError(f"This is a {cache['CMAKE_BUILD_TYPE']} single-config build. "
                         f"Run init.py --config {configuration} with another --name to preserve both builds.")
    if cache.get("CMAKE_CONFIGURATION_TYPES") and configuration not in configurations:
        raise SetupError(f"Configuration '{configuration}' is unavailable; choose from {', '.join(configurations)}.")
    products = cache.get("HYLE_COMPONENTS", "engine").split(";")
    routes = (tuple(filter(None, cache.get("HYLE_COMPUTE_ROUTES", "").split(";")))
              if _uses_compute(products) else ())
    environment = discover_environment(cache=cache, backends=routes)
    return directory, cache, configuration, environment


def build(args: argparse.Namespace) -> int:
    directory, cache, configuration, environment = _configured(args)
    prefix = None
    if args.install is not None:
        if args.target:
            raise SetupError("--install builds and installs the complete selected project. Omit --target when installing.")
        prefix = (args.install if isinstance(args.install, Path)
                  else ROOT / ".scratch/install" / directory.name).expanduser().resolve()
        if cache.get("HYLE_TEST_INSTRUMENTATION", "OFF").upper() in _BOOL_TRUE:
            raise SetupError("Test-instrumented libraries cannot be installed. Reinitialize without test instrumentation.")
        if cache.get("HYLE_INSTALL", "OFF").upper() not in _BOOL_TRUE:
            command = [environment.cmake, "-S", str(ROOT), "-B", str(directory), "-DHYLE_INSTALL=ON"]
            if args.dry_run:
                print("+ " + _command(command))
            else:
                _write_json(_state_path(directory), {"ready": False, "source_dir": str(ROOT), "config": configuration})
                _run(command, environment, directory / "hyle-configure.log")
                _write_json(_state_path(directory), {"ready": True, "source_dir": str(ROOT), "config": configuration})
    command = [environment.cmake, "--build", str(directory), "--config", configuration,
               "--parallel", str(args.jobs)]
    if args.target:
        command += ["--target", *args.target]
    if args.dry_run:
        print("+ " + _command(command))
    else:
        _run(command, environment, directory / "hyle-build.log")
    if prefix:
        command = [environment.cmake, "--install", str(directory), "--config", configuration, "--prefix", str(prefix)]
        if args.dry_run:
            print("+ " + _command(command))
        else:
            _run(command, environment, directory / "hyle-install.log")
            products = cache.get("HYLE_COMPONENTS", "engine").split(";")
            if {"engine", "all"}.intersection(products):
                print(f"Installed Hyle project: {prefix}\nEngine: {prefix / 'bin' / ('hyle.exe' if environment.system == 'Windows' else 'hyle')}")
            else:
                print(f"Installed SDK: {prefix}")
            print("Library consumers: set CMAKE_PREFIX_PATH to this directory and find_package(Hyle).")
    return 0


def test(args: argparse.Namespace) -> int:
    directory, cache, configuration, environment = _configured(args)
    if cache.get("HYLE_BUILD_TESTS", "OFF").upper() not in _BOOL_TRUE:
        selector = f"--name {args.name}" if args.name else f'--build-dir "{directory}"'
        raise SetupError(f"Tests are disabled. Run: python scripts/init.py {selector} --tests; "
                         "then python scripts/build.py and python scripts/test.py.")
    command = [environment.ctest, "--test-dir", str(directory), "-C", configuration,
               "--output-on-failure", "--no-tests=error", "--parallel", str(args.jobs)]
    if args.filter:
        command += ["-R", args.filter]
    if args.list:
        command += ["--show-only"]
    if args.dry_run:
        print("+ " + _command(command))
        return 0
    try:
        _run(command, environment, directory / "hyle-test.log")
    except CommandError as error:
        raise CommandError(error.code, str(error) + " Tests do not build binaries automatically; "
                           "run scripts/build.py first if an executable is missing.") from error
    return 0


def parser(action: str) -> argparse.ArgumentParser:
    descriptions = {
        "init": "Discover installed tools, prepare selected Hyle dependencies and configure a build.",
        "build": "Build an initialized Hyle configuration and optionally install the selected project.",
        "test": "Run tests from an initialized and built Hyle configuration; never builds implicitly.",
    }
    result = argparse.ArgumentParser(description=descriptions[action],
                                     epilog="See docs/developer/setup/README.md. No global environment settings are changed.")
    selector = result.add_mutually_exclusive_group()
    selector.add_argument("--name", type=_name, help="short configuration name under .scratch/build")
    selector.add_argument("--build-dir", type=Path, help="advanced: use a specific existing or new build directory")
    result.add_argument("--config", choices=CONFIGURATIONS, help="build configuration (new builds: Release)")
    result.add_argument("--jobs", "-j", type=_positive, default=2, help="parallel jobs (default: 2)")
    result.add_argument("--dry-run", action="store_true", help="discover tools and print commands without configuring/building/testing")
    result.add_argument("--verbose", action="store_true", help="also print complete CMake commands")
    if action == "init":
        result.add_argument("--backends", help="comma-separated cpu,cuda,metal,vulkan or all; default: cpu")
        result.add_argument("--components", help="engine (new builds default), all, or comma-separated library products such as util or compute,graph; existing selection is preserved")
        result.add_argument("--compiler", choices=("auto", "msvc", "clang-cl", "gcc", "clang"), help="installed compiler family; default: auto")
        result.add_argument("--generator", help="advanced: CMake generator; defaults to available Ninja or platform make")
        result.add_argument("--cache-dir", type=Path, help="advanced: reuse a shared Hyle dependency source/SDK cache")
        for name, help_text in (("tests", "prepare test dependencies and build targets"),
                                ("aten", "include optional ATen CPU/CUDA operations (default: enabled)"),
                                ("static", "build static libraries (default: shared)"),
                                ("tls", "enable TLS with an installed OpenSSL 3 SDK"),
                                ("http2", "enable HTTP/2 (locked nghttp2 is prepared automatically)"),
                                ("trace", "enable tracing (locked Tracy is prepared automatically)"),
                                ("offline", "use cached dependencies only; fail clearly if anything is missing")):
            result.add_argument("--" + name, action=argparse.BooleanOptionalAction, default=None, help=help_text)
        result.add_argument("--sanitizer", choices=("off", "asan", "tsan", "ubsan"), help="optional compiler instrumentation; default: off")
        result.add_argument("-D", "--define", action="append", default=[], metavar="NAME=VALUE", help="advanced: additional CMake definition (repeatable)")
    elif action == "build":
        result.add_argument("--target", action="append", help="build a specific CMake target (repeatable)")
        result.add_argument("--install", type=Path, nargs="?", const=False, help="also install the selected project; optional prefix defaults to .scratch/install/<build-name>")
    else:
        result.add_argument("--filter", help="CTest test-name regular expression")
        result.add_argument("--list", action="store_true", help="list selected tests without executing them")
    return result


def main(action: str, argv: list[str] | None = None) -> int:
    global _VERBOSE
    arguments = list(sys.argv[1:] if argv is None else argv)
    if action == "init" and not arguments and _current().is_file():
        previous = _previous_initialization()
        if previous.get("error") and isinstance(previous.get("init_args"), list) and previous["init_args"]:
            # A failure before CMake has written a cache (e.g. a missing host
            # tool) still retries the same request, without a second profile.
            arguments = previous["init_args"]
    args = parser(action).parse_args(arguments)
    _VERBOSE = args.verbose
    try:
        return {"init": initialize, "build": build, "test": test}[action](args)
    except (SetupError, OSError, ValueError, RuntimeError) as error:
        if action == "init" and not args.dry_run:
            try:
                previous = _previous_initialization()
                _write_json(_current(), dict(previous, error=str(error), init_args=arguments))
            except (OSError, SetupError):
                pass  # Preserve the original actionable diagnostic on read-only checkouts.
        print(f"Hyle {action}: {error}", file=sys.stderr)
        return error.code if isinstance(error, CommandError) and error.code > 0 else 2
    except KeyboardInterrupt:
        if action == "init" and not args.dry_run:
            try:
                previous = _previous_initialization()
                _write_json(_current(), dict(previous, error="Initialization was interrupted", init_args=arguments))
            except (OSError, SetupError):
                pass
        print(f"Hyle {action}: interrupted. Rerun the same command to continue using cached resources.", file=sys.stderr)
        return 130
