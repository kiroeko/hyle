"""Finite U2/U3 delivery commands. Not a public validation framework."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STATE = ROOT / '.scratch/validation/cuda-final-u3-execution'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def local(path, root):
    value = path.replace('{root}', root.as_posix())
    if os.name == 'nt' and value.startswith('/root/'):
        value = '//wsl.localhost/Ubuntu-24.04' + value
    return Path(value) if Path(value).is_absolute() else root / value


def child_logs(job, logs):
    value = logs.as_posix()
    if job.get('linux_output') and os.name == 'nt':
        if len(logs.drive) != 2 or logs.drive[1] != ':':
            raise ValueError('Linux logs require a declared absolute Windows drive path')
        value = '/mnt/' + logs.drive[0].lower() + value[2:]
    return value


def make_jobs(manifest):
    """Four remaining recipes share steps; no copied per-platform drivers."""
    jobs = []
    for profile in manifest['profiles']:
        linux = profile['system'] == 'Linux'
        p = manifest['linux'] if linux else manifest['windows']
        prefix = p['delivery'] + '/' + profile['directory']
        installed, relocated, sdk = (prefix + '-' + part for part in ('installed', 'relocated', 'consumer'))
        cpu, static = ('ON' if profile[k] else 'OFF' for k in ('cpu', 'static'))
        if linux:
            launch = ['wsl.exe', '-d', 'Ubuntu-24.04', '--', 'env', '-u', 'LD_LIBRARY_PATH', *p['environment']]
        else:
            launch = []
        previous = []

        def add(name, argv=None, **extra):
            nonlocal previous
            job = {'id': profile['id'] + '.' + name, 'profile': profile['id'],
                   'contract': {k: v for k, v in profile.items() if k != 'legacy_configure'},
                   'inputs': manifest['inputs'], 'deps': previous, 'timeout': 900,
                   'env': {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}, **extra}
            if argv is not None:
                job['argv'] = launch + argv
            if linux:
                job['linux_output'] = True
            if not linux and name in ('configure', 'build', 'install', 'sdk-configure', 'sdk-build'):
                job['vs_env'] = manifest['windows']['vs_env']
            jobs.append(job); previous = [job['id']]
            return job

        common = ['-DU3_REFERENCE_DIR=' + p['reference'], '-DU3_GTEST_SOURCE=' + p['gtest'],
                  '-DU3_LIB_PREFIX=' + ('lib' if linux else ''),
                  '-DU3_LIB_SUFFIX=' + ('.a' if linux else '.lib'), '-DFETCHCONTENT_FULLY_DISCONNECTED=ON']
        if linux:
            add('fixture', copy_fixture=[manifest['consumer'], p['consumer']],
                inputs=[manifest['consumer']])
            configure = [p['source'] + '/scripts/hyle-unix.sh', 'cmake', '--preset', 'compute-cuda',
                '-S', p['source'], '-B', p['build'], *common, '-DHYLE_COMPONENTS=compute;graph;simd',
                '-DHYLE_INSTALL=ON', '-DHYLE_BUILD_TESTS=OFF', '-DCMAKE_CUDA_ARCHITECTURES=89-real',
                '-DCMAKE_PROJECT_INCLUDE=' + p['overlay']]
        else:
            configure = ['cmake', '-S', '{root}', '-B', p['build']]
        configured = add('configure', configure + ['-DHYLE_COMPUTE_CPU=' + cpu,
            '-DHYLE_STATIC=' + static, '-DCMAKE_INSTALL_PREFIX=' + installed])
        if profile.get('legacy_configure'):
            configured['legacy'] = profile['legacy_configure']
        add('build', ['cmake', '--build', p['build'], '--target', 'u3_delivery', '--parallel', '2'])
        for name, target, flt in manifest['source_checks']:
            if target == 'u3_mixed_cpu' and not profile['cpu']:
                continue
            suffix = '' if linux else '.exe'
            xml = '{logs}/' + profile['id'] + '.' + name + '.xml'
            add(name, [p['build'] + '/bin/' + target + suffix, '--gtest_filter=' + flt,
                '--gtest_output=xml:' + xml], xml=xml, timeout=120,
                env={'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'HYLE_D5_QWEN35_FIXTURE': p['fixture']})
        add('install', ['cmake', '--install', p['build']])
        add('relocate', relocate=[installed, relocated])
        add('sdk-configure', ['cmake', '-S', p['consumer'], '-B', sdk, '-G', 'Ninja',
            '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_PREFIX_PATH=' + relocated, *common])
        add('sdk-build', ['cmake', '--build', sdk, '--target', 'u3_delivery', '--parallel', '2'])
        for group in manifest['runtime_groups']:
            xml = '{logs}/' + profile['id'] + '.' + group['name'] + '.xml'
            target = sdk + '/' + group['target'] + ('' if linux else '.exe')
            # New binary/runtime = a new delivery check, not a rerun of all mathematical shapes.
            add(group['name'], [target, '--gtest_filter=' + group['filter'], '--gtest_output=xml:' + xml],
                xml=xml, timeout=300, system_path=not linux,
                inputs=manifest['inputs'] + [target, sdk + ('/*.so*' if linux else '/*.dll')],
                env={'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'HYLE_D5_QWEN35_FIXTURE': p['fixture']})
        if profile['cpu']:
            xml = '{logs}/' + profile['id'] + '.cpu.xml'
            target = sdk + '/u3_mixed_cpu' + ('' if linux else '.exe')
            add('cpu', [target, '--gtest_filter=MixedCpuDelivery.*', '--gtest_output=xml:' + xml],
                xml=xml, system_path=not linux, inputs=manifest['inputs'] + [target])
    return jobs


class Runner:
    def __init__(self, root, jobs, state=STATE):
        self.root, self.state = root, state
        self.jobs = {job['id']: job for job in jobs}
        if len(self.jobs) != len(jobs):
            raise ValueError('Duplicate task ID')
        self.hashes = {}

    @contextlib.contextmanager
    def locked(self):
        self.state.mkdir(parents=True, exist_ok=True)
        path = self.state / 'writer.lock'
        # OS releases this lock after an interrupted process; no stale lock deletion.
        with path.open('a+b') as stream:
            if stream.tell() == 0:
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def digest(self, path):
        stat = path.stat(); key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.hashes:
            with path.open('rb') as stream:
                self.hashes[key] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return self.hashes[key]

    def inputs(self, job):
        result = {}
        for pattern in job['inputs']:
            path = local(pattern, self.root)
            files = sorted(path.parent.glob(path.name)) if '*' in path.name else [path]
            content = {}
            if not files:
                result[pattern] = 'MISSING'
                continue
            for item in files:
                if item.is_dir():
                    content.update({str(p): self.digest(p) for p in sorted(item.rglob('*'))
                                    if p.is_file() and '__pycache__' not in p.parts})
                else:
                    content[str(item)] = self.digest(item) if item.is_file() else 'MISSING'
            result[pattern] = ('MISSING' if 'MISSING' in content.values() else
                hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest())
        result['recipe'] = hashlib.sha256(json.dumps(
            {k: v for k, v in job.items() if k not in ('legacy', 'deps')},
            sort_keys=True).encode()).hexdigest()
        return result

    def history(self, task):
        if task not in self.jobs:
            raise ValueError('Unregistered task: ' + task)
        path = self.state / (task + '.json')
        return read(path) if path.exists() else []

    def seed(self):
        # One declared historical step, not an arbitrary PASS import interface.
        for job in self.jobs.values():
            prior = job.get('legacy')
            if not prior or self.history(job['id']):
                continue
            path = local(prior['receipt'], self.root)
            if self.digest(path) != prior['sha256'] or self.inputs(job) != prior['inputs']:
                raise ValueError('Historical step identity changed; cannot import')
            row = next(x for x in read(path) if x['name'] == prior['name'])
            if row['returncode'] != 0:
                raise ValueError('Cannot import a failed step')
            write(self.state / (job['id'] + '.json'), [{'status': 'PASS',
                'inputs': prior['inputs'], 'seconds': row['seconds'], 'attempt': 1,
                'reason': 'Existing unchanged successful configure; no process launched',
                'proofs': {str(path): prior['sha256']}}])

    def verdict(self, task):
        job = self.jobs[task]; history = self.history(task)
        if history and history[-1]['status'] == 'PASS' and history[-1]['inputs'] == self.inputs(job):
            previous = history[-1]
            if all(local(p, self.root).is_file() and self.digest(local(p, self.root)) == value
                   for p, value in previous.get('proofs', {}).items()):
                return 'REUSE', 'Matching inputs and preserved result'
            return 'STOP', 'Saved result is missing or changed'
        if len(history) >= 2:
            return 'STOP', 'Task budget exhausted; output directory/commit does not reset it'
        if history:
            return 'STOP', 'Prior result needs changed inputs and an explicit correction reason'
        return 'RUN', 'No result for this declared task'

    def perform(self, task, logs, reason=None):
        job = self.jobs.get(task)
        if job is None:
            raise ValueError('Unregistered task: ' + task)
        for dependency in job.get('deps', []):
            if self.verdict(dependency)[0] != 'REUSE':
                raise ValueError('Dependency is not qualified: ' + dependency)
        action, detail = self.verdict(task)
        if action == 'REUSE':
            return action
        history = self.history(task); inputs = self.inputs(job)
        if history:
            if len(history) >= 2 or not reason or history[-1]['inputs'] == inputs:
                raise ValueError(detail)
        if any(value == 'MISSING' for value in inputs.values()):
            raise ValueError('Required input is missing: ' + task)
        attempt = len(history) + 1; logs.mkdir(parents=True, exist_ok=True)
        log = logs / (task + '-' + str(attempt) + '.log')
        if log.exists():
            raise ValueError('Preserved log exists; do not overwrite: ' + str(log))
        row = {'status': 'RUNNING', 'inputs': inputs, 'reason': reason,
               'seconds': 0, 'proofs': {}, 'log': str(log), 'attempt': attempt}
        history.append(row); write(self.state / (task + '.json'), history)
        start = time.monotonic()
        try:
            environment = dict(os.environ, **{k: v.replace('{root}', self.root.as_posix())
                                               for k, v in job.get('env', {}).items()})
            flags = {}
            if os.name == 'nt':
                import ctypes
                ctypes.windll.kernel32.SetErrorMode(32771)
                flags['creationflags'] = subprocess.CREATE_NO_WINDOW
            if job.get('vs_env'):
                command = local(job['vs_env'], self.root)
                setup = subprocess.run(['cmd', '/d', '/c', str(command)], check=True,
                    capture_output=True, **flags)
                for line in setup.stdout.decode(errors='replace').splitlines():
                    if '=' in line and not line.startswith('='):
                        key, value = line.split('=', 1); environment[key] = value
            if job.get('system_path'):
                environment = {k: v for k, v in environment.items() if k.lower() != 'path'}
                environment['PATH'] = os.environ['SystemRoot'] + '/System32;' + os.environ['SystemRoot']
            with log.open('wb') as stream:
                if 'copy_fixture' in job:
                    before, after = (local(p, self.root) for p in job['copy_fixture'])
                    after.mkdir(parents=True, exist_ok=True)
                    for path in before.rglob('*'):
                        if path.is_file() and '__pycache__' not in path.parts:
                            target = after / path.relative_to(before)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            if not target.is_file() or self.digest(path) != self.digest(target):
                                shutil.copy2(path, target)
                    rc = 0
                elif 'relocate' in job:
                    before, after = (local(p, self.root) for p in job['relocate'])
                    if not before.is_dir() or after.exists():
                        raise ValueError('Relocation requires the declared fresh installed prefix')
                    before.rename(after); rc = 0
                else:
                    argv = [part.replace('{root}', self.root.as_posix()).replace('{logs}', child_logs(job, logs))
                            for part in job['argv']]
                    rc = subprocess.run(argv, cwd=self.root, env=environment, stdout=stream,
                        stderr=subprocess.STDOUT, timeout=job['timeout'], **flags).returncode
            row['status'] = 'PASS' if rc == 0 else 'FAIL'; row['returncode'] = rc
            if rc == 0 and job.get('xml'):
                xml = local(job['xml'].replace('{logs}', logs.as_posix()), self.root)
                tests = list(ET.parse(xml).getroot().iter('testcase'))
                if not tests or any(t.find('failure') is not None or t.find('skipped') is not None
                                    or t.attrib.get('status') != 'run' for t in tests):
                    row['status'] = 'FAIL'
                row['tests'] = len(tests); row['proofs'][str(xml)] = self.digest(xml)
        except BaseException as error:
            row['status'] = 'FAIL'; row['error'] = str(error)
        finally:
            row['seconds'] = round(time.monotonic() - start, 3)
            if log.exists():
                row['proofs'][str(log)] = self.digest(log)
            write(self.state / (task + '.json'), history)
        if row['status'] != 'PASS':
            raise ValueError('Stopped on failed/interrupted task: ' + task)
        return 'RUN'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run', 'status'))
    parser.add_argument('--task', action='append', default=[])
    parser.add_argument('--profile', action='append', default=[])
    parser.add_argument('--logs', type=Path, default=STATE / 'logs')
    parser.add_argument('--reason')
    args = parser.parse_args()
    manifest = read(HERE / 'delivery.json')
    if args.action == 'run' and not manifest['run_enabled']:
        parser.error('CUDA product goal is paused; only plan/status are enabled')
    jobs = make_jobs(manifest)
    runner = Runner(ROOT, jobs)
    if set(args.profile) - {p['id'] for p in manifest['profiles']}:
        parser.error('Unregistered delivery profile')
    selected = args.task or [job['id'] for job in jobs
                            if not args.profile or job['profile'] in args.profile]
    # Validation before any subprocess. A different log path never changes state.
    if set(selected) - runner.jobs.keys():
        parser.error('Unregistered task')
    with runner.locked():
        runner.seed()
        if args.action == 'run':
            if not (args.task or args.profile):
                parser.error('run needs explicit tasks/profiles; plan/status never launch builds')
            for task in selected:
                print(json.dumps({'task': task, 'action': runner.perform(task, args.logs, args.reason)}), flush=True)
        else:
            rows = [{'task': task, 'action': runner.verdict(task)[0],
                     'reason': runner.verdict(task)[1], 'attempts': len(runner.history(task)),
                     'seconds': sum(x['seconds'] for x in runner.history(task))} for task in selected]
            for cell in manifest['completed_cells']:
                for proof in cell['proofs']:
                    if runner.digest(local(proof['path'], ROOT)) != proof['sha256']:
                        raise ValueError('Retained cell evidence changed: ' + cell['id'])
            report = {'scope': 'U2/U3 delivery only; not overall goal percent',
                'run_enabled': manifest['run_enabled'],
                'completed': sum(row['action'] == 'REUSE' for row in rows), 'total': len(rows),
                'tasks': rows, 'retained_completed_cells': [c['id'] for c in manifest['completed_cells']]}
            write(STATE / 'plan.json', report)
            if args.action == 'status':
                report.pop('tasks')
                report['profiles'] = {p['id']: {
                    'completed': sum(r['action'] == 'REUSE' for r in rows if r['task'].startswith(p['id'] + '.')),
                    'remaining': sum(r['action'] != 'REUSE' for r in rows if r['task'].startswith(p['id'] + '.')),
                    'seconds': sum(r['seconds'] for r in rows if r['task'].startswith(p['id'] + '.'))}
                    for p in manifest['profiles'] if not args.profile or p['id'] in args.profile}
            print(json.dumps(report, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr); sys.exit(1)
