"""OpenSSL runtime metadata contracts; no native build, DLL load or download."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
NOTICE = b"fixture OpenSSL supplier terms\r\nraw bytes: \x96\xff\r\n"
DLL_NAMES = {
    "Release": ("libssl-3-x64.dll", "libcrypto-3-x64.dll"),
    "Debug": ("libssl-3-x64d.dll", "libcrypto-3-x64d.dll"),
}


def quoted(path: Path) -> str:
    return '[==[' + path.as_posix() + ']==]'


class OpenSSLRuntimeTest(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / ".scratch/validation/openssl-runtime-contract"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()

    def sdk(self, name="vendor SDK", *, notice=True):
        prefix = self.directory / name
        for relative in ("include/openssl", "lib", "bin", "share/openssl"):
            (prefix / relative).mkdir(parents=True, exist_ok=True)
        (prefix / "include/openssl/ssl.h").write_bytes(b"fixture header\n")
        for config, names in DLL_NAMES.items():
            for name in names:
                (prefix / "bin" / name).write_bytes(
                    ("fixture " + config + " " + name).encode("ascii"))
        # A library's bytes are deliberately not evidence that it is a DLL.
        for name in ("ssl.lib", "crypto.lib", "libssl.a", "libcrypto.a"):
            (prefix / "lib" / name).write_bytes(
                b"not a COFF parser fixture\x00libssl-3-x64.dll\x00libcrypto-3-x64.dll\x00")
        if notice:
            (prefix / "share/openssl/copyright").write_bytes(NOTICE)
        return prefix

    def target_metadata(self, prefix, kind="SHARED"):
        body = [f"set(OPENSSL_INCLUDE_DIR {quoted(prefix / 'include')})"]
        for target, stem, index in (("SSL", "ssl", 0), ("Crypto", "crypto", 1)):
            body.append(f"add_library(OpenSSL::{target} {kind} IMPORTED)")
            if kind == "SHARED":
                body.extend((
                    f"set_target_properties(OpenSSL::{target} PROPERTIES",
                    '  IMPORTED_CONFIGURATIONS "RELEASE;DEBUG"',
                    '  MAP_IMPORTED_CONFIG_MINSIZEREL Release',
                    f"  IMPORTED_LOCATION_RELEASE {quoted(prefix / 'bin' / DLL_NAMES['Release'][index])}",
                    f"  IMPORTED_LOCATION_DEBUG {quoted(prefix / 'bin' / DLL_NAMES['Debug'][index])}",
                    f"  IMPORTED_IMPLIB_RELEASE {quoted(prefix / 'lib' / (stem + '.lib'))}",
                    f"  IMPORTED_IMPLIB_DEBUG {quoted(prefix / 'lib' / (stem + '.lib'))})",
                ))
            else:
                name = stem + ".lib" if kind == "UNKNOWN" else "lib" + stem + ".a"
                body.append(f"set_property(TARGET OpenSSL::{target} PROPERTY "
                            f"IMPORTED_LOCATION {quoted(prefix / 'lib' / name)})")
        return "\n".join(body) + "\n"

    def configure(self, name, body, *, config="Release", success=True):
        source = self.directory / (name + "-source")
        build = self.directory / (name + "-build")
        source.mkdir()
        (source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.25)\n'
            'project(OpenSSLRuntimeContract NONE)\n' + body,
            encoding="utf-8")
        result = subprocess.run(
            ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=" + config],
            capture_output=True, text=True, timeout=30, **NO_WINDOW)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode == 0, success, output)
        return build, output

    def producer(self, name, prefix, *, kind="SHARED", windows=True,
                 config="Release", success=True, include_prefix=None, install=False):
        body = f'''
set(WIN32 {"TRUE" if windows else "FALSE"})
set(HYLE_INSTALL {"ON" if install else "OFF"})
set(CMAKE_INSTALL_BINDIR bin)
set(CMAKE_INSTALL_DATADIR share)
include({quoted(ROOT / "cmake/HyleRuntimeStaging.cmake")})
include({quoted(ROOT / "src/io_tls/openssl_runtime.cmake")})
add_library(hyle_io_tls INTERFACE)
add_library(hyle::_factor_io_tls INTERFACE IMPORTED)
{self.target_metadata(prefix, kind)}
set(OPENSSL_INCLUDE_DIR {quoted((include_prefix or prefix) / "include")})
hyle_register_openssl_runtime(_bundled {"TRUE" if install else "FALSE"})
file(GENERATE OUTPUT "${{CMAKE_CURRENT_BINARY_DIR}}/bundled.txt"
  CONTENT "${{_bundled}}")
foreach(_owner IN ITEMS hyle_io_tls hyle::_factor_io_tls)
  _hyle_collect_runtime_files("${{_owner}}" _runtime)
  string(REPLACE "::" "_" _name "${{_owner}}")
  file(GENERATE OUTPUT "${{CMAKE_CURRENT_BINARY_DIR}}/${{_name}}.txt"
    CONTENT "$<JOIN:${{_runtime}},\n>")
endforeach()
'''
        return self.configure(name, body, config=config, success=success)

    def runtime(self, build, owner="hyle_io_tls"):
        return [Path(line) for line in (build / (owner + ".txt")).read_text().splitlines()
                if line]

    def test_unknown_and_static_libraries_do_not_infer_runtime_from_bytes_or_siblings(self):
        prefix = self.sdk(notice=False)
        for kind in ("UNKNOWN", "STATIC"):
            with self.subTest(kind=kind):
                build, _ = self.producer(kind.lower(), prefix, kind=kind)
                self.assertEqual(self.runtime(build), [])
                self.assertEqual(self.runtime(build, "hyle__factor_io_tls"), [])
                self.assertEqual((build / "bundled.txt").read_text(), "")

    def test_explicit_shared_targets_follow_imported_configuration_mapping(self):
        prefix = self.sdk()
        for config, selected in (("Release", "Release"), ("Debug", "Debug"),
                                 ("MinSizeRel", "Release")):
            with self.subTest(config=config):
                build, _ = self.producer("shared-" + config, prefix, config=config)
                files = set(self.runtime(build))
                self.assertEqual(files, set(self.runtime(build, "hyle__factor_io_tls")))
                dlls = {path for path in files if path.suffix.lower() == ".dll"}
                self.assertEqual(dlls, {prefix / "bin" / name
                                        for name in DLL_NAMES[selected]})
                notices = [path for path in files if path.suffix.lower() != ".dll"]
                self.assertEqual(len(notices), 1)
                self.assertEqual(notices[0].read_bytes(), NOTICE)
                self.assertEqual(notices[0].name,
                                 "OpenSSL-" + selected.upper() + "-"
                                 + hashlib.sha256(NOTICE).hexdigest()[:12] + "-copyright")
                bundled = [item for item in (build / "bundled.txt").read_text().split(";")
                           if item]
                self.assertTrue(bundled)
                self.assertTrue(all(not Path(item).is_absolute() and ".." not in Path(item).parts
                                    for item in bundled))
                self.assertIn("bin/" + DLL_NAMES[selected][0], bundled)
                self.assertIn("share/licenses/Hyle/openssl/" + notices[0].name, bundled)

    def test_additional_notice_keeps_its_own_raw_bytes_and_content_identity(self):
        prefix = self.sdk()
        additional = b"separate supplier notice\r\nraw bytes: \x80\xfe\r\n"
        (prefix / "NOTICE").write_bytes(additional)
        build, _ = self.producer("additional-notice", prefix)
        notices = {path for path in self.runtime(build) if path.suffix.lower() != ".dll"}
        expected = {
            "OpenSSL-RELEASE-" + hashlib.sha256(content).hexdigest()[:12] + "-" + name: content
            for name, content in (("copyright", NOTICE), ("NOTICE", additional))
        }
        self.assertEqual({path.name for path in notices}, set(expected))
        for path in notices:
            self.assertEqual(path.read_bytes(), expected[path.name])

    def test_explicit_shared_runtime_without_supplier_notice_fails(self):
        prefix = self.sdk(notice=False)
        _, output = self.producer("missing-notice", prefix, success=False)
        self.assertIn("openssl", output.lower())
        self.assertTrue("license" in output.lower() or "notice" in output.lower(), output)

    def test_shared_runtime_cannot_borrow_notice_from_sibling_include_prefix(self):
        matching_name_only = self.sdk("supplier SDK")
        actual_vendor = self.sdk("supplier SDK-other", notice=False)
        _, output = self.producer("mismatched-notice", actual_vendor,
                                  include_prefix=matching_name_only, success=False)
        self.assertIn("matching", output.lower())

    def test_linux_static_targets_do_not_acquire_windows_runtime_or_notice(self):
        prefix = self.sdk(notice=False)
        build, _ = self.producer("linux-static", prefix, kind="STATIC", windows=False)
        self.assertEqual(self.runtime(build), [])
        self.assertEqual((build / "bundled.txt").read_text(), "")

    def test_shared_install_copies_selected_dlls_and_original_notices(self):
        vendor = self.sdk()
        build, _ = self.producer("actual-install", vendor, config="MinSizeRel", install=True)
        for component in ("Hyle_Runtime", "Hyle_Development"):
            with self.subTest(component=component):
                prefix = self.directory / component
                result = subprocess.run(
                    ["cmake", "--install", str(build), "--prefix", str(prefix),
                     "--config", "MinSizeRel", "--component", component],
                    capture_output=True, text=True, timeout=30, **NO_WINDOW)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                notices = list((prefix / "share/licenses/Hyle/openssl").iterdir())
                self.assertEqual(len(notices), 1)
                self.assertEqual(notices[0].read_bytes(), NOTICE)
                self.assertIn("RELEASE", notices[0].name)
                if component == "Hyle_Runtime":
                    self.assertEqual({p.name for p in (prefix / "bin").iterdir()},
                                     set(DLL_NAMES["Release"]))
                    for name in DLL_NAMES["Release"]:
                        self.assertEqual((prefix / "bin" / name).read_bytes(),
                                         (vendor / "bin" / name).read_bytes())
                else:
                    self.assertFalse((prefix / "bin").exists())

    def configure_shim(self, destination):
        # Configure the production template, rather than substituting its body.
        shutil.copyfile(ROOT / "src/io_tls/openssl_runtime.cmake",
                        destination.parent / "openssl_runtime.cmake")
        body = f'''
set(HYLE_OPENSSL_PREFIX_FROM_CONFIG "../../..")
configure_file({quoted(ROOT / "src/io_tls/HyleOpenSSLRuntimeConfig.cmake.in")}
  {quoted(destination)} @ONLY)
'''
        self.configure("configure-shim-" + destination.parent.parent.parent.name, body)

    def test_shared_installed_shim_uses_sdk_payload_without_external_openssl(self):
        vendor = self.sdk("producer vendor SDK")
        build, _ = self.producer("shim-producer", vendor)
        prefix = self.directory / "relocated shared SDK"
        config_dir = prefix / "lib/cmake/HyleOpenSSLRuntime"
        config_dir.mkdir(parents=True)
        raw_relative_files = (build / "bundled.txt").read_text().split(";")
        relative_files = set(raw_relative_files) - {""}
        self.assertTrue(relative_files)
        # Model an installed SDK using only the helper's returned payload list.
        # This checks shim metadata, not the production install() instructions.
        for relative in relative_files:
            target = prefix / relative
            self.assertTrue(target.resolve().is_relative_to(prefix.resolve()))
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.suffix.lower() == ".dll":
                shutil.copyfile(vendor / "bin" / target.name, target)
            else:
                target.write_bytes(NOTICE)
        (config_dir / "HyleOpenSSLFiles.cmake").write_text(
            "set(_hyle_openssl_files [==[" + ";".join(raw_relative_files) + "]==])\n",
            encoding="utf-8")
        self.configure_shim(config_dir / "HyleOpenSSLRuntimeConfig.cmake")
        # The producer's vendor tree must not remain an available fallback.
        unavailable_vendor = vendor.with_name(vendor.name + " unavailable")
        vendor.rename(unavailable_vendor)
        body = f'''
set(WIN32 TRUE)
set(Hyle_STATIC FALSE)
set(CMAKE_DISABLE_FIND_PACKAGE_OpenSSL TRUE)
include({quoted(ROOT / "cmake/HyleRuntimeStaging.cmake")})
include({quoted(config_dir / "HyleOpenSSLRuntimeConfig.cmake")})
if(TARGET OpenSSL::SSL OR TARGET OpenSSL::Crypto)
  message(FATAL_ERROR "shared shim discovered an external OpenSSL SDK")
endif()
add_library(hyle::_factor_io_tls INTERFACE IMPORTED)
_hyle_collect_runtime_files(hyle::_factor_io_tls _runtime)
file(GENERATE OUTPUT "${{CMAKE_CURRENT_BINARY_DIR}}/runtime.txt"
  CONTENT "$<JOIN:${{_runtime}},\n>")
'''
        consumer, _ = self.configure("shared-shim-consumer", body)
        files = [Path(line) for line in (consumer / "runtime.txt").read_text().splitlines()
                 if line]
        self.assertEqual(set(files), {prefix / relative for relative in relative_files})
        for path in files:
            self.assertTrue(path.is_file(), str(path))
            if path.suffix.lower() != ".dll":
                self.assertEqual(path.read_bytes(), NOTICE)

    def test_static_shim_finds_consumers_openssl_and_registers_its_runtime(self):
        prefix = self.sdk("consumer vendor SDK")
        config_dir = self.directory / "static SDK/lib/cmake/HyleOpenSSLRuntime"
        config_dir.mkdir(parents=True)
        self.configure_shim(config_dir / "HyleOpenSSLRuntimeConfig.cmake")
        modules = self.directory / "consumer find modules"
        modules.mkdir()
        (modules / "FindOpenSSL.cmake").write_text(
            'if(NOT OpenSSL_FIND_VERSION STREQUAL "3.0")\n'
            '  message(FATAL_ERROR "static shim omitted its OpenSSL version requirement")\n'
            'endif()\n' + self.target_metadata(prefix) + '\nset(OpenSSL_FOUND TRUE)\n'
            'set(OPENSSL_FOUND TRUE)\nset(OPENSSL_VERSION "3.0.0")\n',
            encoding="utf-8")
        body = f'''
set(WIN32 TRUE)
set(Hyle_STATIC TRUE)
list(PREPEND CMAKE_MODULE_PATH {quoted(modules)})
include(CMakeFindDependencyMacro)
include({quoted(ROOT / "cmake/HyleRuntimeStaging.cmake")})
include({quoted(config_dir / "HyleOpenSSLRuntimeConfig.cmake")})
if(NOT TARGET OpenSSL::SSL OR NOT TARGET OpenSSL::Crypto)
  message(FATAL_ERROR "static shim omitted the consumer's OpenSSL dependency")
endif()
add_library(hyle::_factor_io_tls INTERFACE IMPORTED)
_hyle_collect_runtime_files(hyle::_factor_io_tls _runtime)
file(GENERATE OUTPUT "${{CMAKE_CURRENT_BINARY_DIR}}/runtime.txt"
  CONTENT "$<JOIN:${{_runtime}},\n>")
'''
        consumer, _ = self.configure("static-shim-consumer", body)
        files = {Path(line) for line in (consumer / "runtime.txt").read_text().splitlines()
                 if line}
        self.assertEqual({path for path in files if path.suffix.lower() == ".dll"},
                         {prefix / "bin" / name for name in DLL_NAMES["Release"]})
        notices = [path for path in files if path.suffix.lower() != ".dll"]
        self.assertEqual(len(notices), 1)
        self.assertEqual(notices[0].read_bytes(), NOTICE)
        self.assertTrue(notices[0].resolve().is_relative_to(consumer.resolve()))


if __name__ == "__main__":
    unittest.main()
