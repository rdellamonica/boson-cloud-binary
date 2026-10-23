#!/usr/bin/env python3
"""Run isolated fixtures sequentially, compare states, and report median timings."""
import argparse
import math
from pathlib import Path
import re
import statistics
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('--baseline', type=Path, help='Optional unchanged mode_benchmark executable')
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--l-max', type=int, default=12)
    parser.add_argument('--nr', type=int, default=512)
    parser.add_argument('--steps', type=int, default=11, help='Includes the initial time sample')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--scenario', type=int, default=0, choices=range(3))
    parser.add_argument('--preconditioner', choices=['radial', 'ilut'], default='radial')
    args = parser.parse_args()
    if args.repeat < 1 or min(args.threads) < 1:
        parser.error('repeat and thread counts must be positive')
    candidates = []
    if args.baseline:
        candidates.append(('original', args.baseline.resolve(), 1))
    candidates += [(f'{n} threads', args.executable.resolve(), n) for n in args.threads]
    reference = None
    reference_time = None
    print('case,init_seconds,run_seconds,total_seconds,speedup,relative_state_error')
    with tempfile.TemporaryDirectory(prefix='gm-benchmark-') as temporary:
        for index, (name, executable, threads) in enumerate(candidates):
            runs = []
            max_error = 0.0
            for repetition in range(args.repeat):
                directory = Path(temporary) / f'{index}-{repetition}'
                directory.mkdir()
                command = [str(executable), str(args.l_max), str(args.nr), str(args.steps),
                           str(threads), str(args.scenario), args.preconditioner]
                result = subprocess.run(command, cwd=directory, capture_output=True, text=True, check=True)
                if result.stderr:
                    print(result.stderr, end='')
                summary = next(line for line in result.stdout.splitlines() if line.startswith('BENCHMARK'))
                fields = dict(re.findall(r'(\w+)=([\d.eE+-]+)', summary))
                runs.append((float(fields['init_seconds']), float(fields['run_seconds'])))
                state = [float(x) for x in (directory / 'state.txt').read_text().split()]
                if not all(map(math.isfinite, state)):
                    raise RuntimeError(f'{name}: nonfinite final state')
                if reference is None:
                    reference = state
                if len(state) != len(reference):
                    raise RuntimeError(f'{name}: state sizes differ')
                denominator = math.fsum(x*x for x in reference)
                difference = math.fsum((x-y)**2 for x, y in zip(state, reference))
                error = math.sqrt(difference / denominator) if denominator else math.sqrt(difference)
                max_error = max(max_error, error)
                if error > 1e-7:
                    raise RuntimeError(f'{name}: relative state error {error} exceeds 1e-7')
            init = statistics.median(run[0] for run in runs)
            evolution = statistics.median(run[1] for run in runs)
            total = statistics.median(sum(run) for run in runs)
            if reference_time is None:
                reference_time = evolution
            print(f'{name},{init:.6f},{evolution:.6f},{total:.6f},'
                  f'{reference_time/evolution:.2f},{max_error:.3e}', flush=True)


if __name__ == '__main__':
    main()
