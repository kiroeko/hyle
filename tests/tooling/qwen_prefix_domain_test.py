"""Check trusted cache sharing domains on a local Qwen CUDA model."""
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--inputs', required=True)
    args = parser.parse_args()
    result = subprocess.run([args.probe, '--prefix-domain-report', args.inputs],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
    if result.returncode:
        raise RuntimeError(f'prefix-domain execution failed: {result.stderr[-4000:]}')
    if 'PASS actual prefix sharing domain' not in result.stdout:
        raise AssertionError('actual prefix-domain checks did not complete')
    print('PASS: a real complete prefix misses in another domain and hits in its own domain')


if __name__ == '__main__':
    main()
