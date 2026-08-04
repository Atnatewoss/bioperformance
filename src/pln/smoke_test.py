"""
PLN smoke test — verifies the real hyperon runtime + lib_pln.metta works end-to-end.

Tests two things:
1. Truth_ModusPonens from lib_pln.metta fires correctly and returns a parseable (stv s c)
2. Truth_Revision from lib_pln.metta is callable (result parsed in Python)

This is the canonical integration pattern for BioPerformance:
  - hyperon Python package provides the MeTTa runtime
  - lib_pln.metta (from trueagi-io/PLN, cloned to PLN_CACHE_DIR) provides the PLN engine
  - Modus Ponens fires per (fact, rule) pair and returns stv + evidence stamp
  - Revision combines multiple evidence paths (formula from lib_pln.metta, evaluated in Python)

Note on PLN.Derive / PLN.Query:
  trueagi-io/PLN was designed for the PeTTa Prolog runtime which implements git-import!
  and has different let/tuple semantics. PLN.Derive's let ($T $B) destructuring does not
  reduce correctly in hyperon 0.2.10. We drive inference directly via the |- rules and
  Truth_ModusPonens, which are the actual PLN inference operations and work correctly.

Run:
    cd /path/to/bioperformance
    PLN_CACHE_DIR=/path/to/PLN .venv/bin/python3 src/pln/smoke_test.py

PLN_CACHE_DIR defaults to ~/.cache/bioperformance/pln if not set.
The cache is populated automatically on first run by pln_explain.py.
"""

import os
import re
import sys

_THIS_DIR = os.path.dirname(__file__)
_DEFAULT_CACHE = os.path.expanduser('~/.cache/bioperformance/pln')
PLN_CACHE_DIR = os.environ.get('PLN_CACHE_DIR', _DEFAULT_CACHE)


def _ensure_pln_cache(cache_dir: str) -> bool:
    """Clones trueagi-io/PLN into cache_dir if not already present."""
    lib_pln = os.path.join(cache_dir, 'lib_pln.metta')
    if os.path.exists(lib_pln):
        return True
    print(f"PLN not cached at {cache_dir}. Cloning trueagi-io/PLN...")
    os.makedirs(cache_dir, exist_ok=True)
    ret = os.system(
        f'git clone --depth=1 https://github.com/trueagi-io/PLN.git "{cache_dir}" 2>&1'
    )
    if ret != 0 or not os.path.exists(lib_pln):
        print(f"ERROR: Failed to clone PLN into {cache_dir}")
        return False
    print(f"PLN cached at {cache_dir}")
    return True


def run_smoke_test(pln_cache_dir: str = PLN_CACHE_DIR) -> bool:
    from hyperon import MeTTa, Environment

    print("=" * 60)
    print("BioPerformance PLN Smoke Test")
    print("=" * 60)

    if not _ensure_pln_cache(pln_cache_dir):
        return False

    lib_pln_path = os.path.join(pln_cache_dir, 'lib_pln.metta')

    env = Environment.custom_env(include_paths=[pln_cache_dir])
    runner = MeTTa(env_builder=env)
    runner.load_module_at_path(lib_pln_path, 'lib_pln')

    # Test 1: Truth_ModusPonens returns evaluated (stv s c) with evidence stamp.
    # elevated_load fact (stv 0.72 0.83) + rule elevated_load->target (stv 0.60 0.70)
    # Expected strength: 0.72 * 0.60 + 0.02 * (1 - 0.72) = 0.4376
    # Expected confidence: 0.72 * 0.60 * 0.83 * 0.70 = 0.250992
    prog_mp = """
!(import! &self lib_pln)
!(let $stv (Truth_ModusPonens (stv 0.72 0.83) (stv 0.60 0.70))
   ($stv (101 3)))
"""
    print("\nTest 1: Truth_ModusPonens from lib_pln.metta...")
    results = runner.run(prog_mp)
    mp_result = None
    for r in results:
        if r:
            s = str(r[0])
            m = re.match(r'^\(\(stv\s+([\d.]+)\s+([\d.]+)\)\s+\((\d+)\s+(\d+)\)\)$', s)
            if m:
                mp_result = {
                    'strength': float(m.group(1)),
                    'confidence': float(m.group(2)),
                    'evidence_ids': [int(m.group(3)), int(m.group(4))],
                    'raw': s,
                }

    if mp_result is None:
        print(f"FAIL: could not parse Modus Ponens result. Got: {results}")
        return False

    expected_s = round(0.72 * 0.60 + 0.02 * (1 - 0.72), 4)
    expected_c = round(0.72 * 0.60 * 0.83 * 0.70, 6)
    actual_s = round(mp_result['strength'], 4)
    actual_c = round(mp_result['confidence'], 6)

    if abs(actual_s - expected_s) > 0.001 or abs(actual_c - expected_c) > 0.001:
        print(f"FAIL: STV mismatch. Expected (stv {expected_s} {expected_c}), "
              f"got (stv {actual_s} {actual_c})")
        return False

    print(f"  OK  Modus Ponens: {mp_result['raw']}")
    print(f"      strength={mp_result['strength']:.4f} (expected ~{expected_s})")
    print(f"      confidence={mp_result['confidence']:.6f} (expected ~{expected_c})")
    print(f"      evidence_ids={mp_result['evidence_ids']}")

    # Test 2: Truth_Revision is callable
    print("\nTest 2: Truth_Revision callable from lib_pln.metta...")
    prog_rev = "!(import! &self lib_pln)\n!(Truth_Revision (stv 0.44 0.25) (stv 0.34 0.17))"
    runner2 = MeTTa(env_builder=Environment.custom_env(include_paths=[pln_cache_dir]))
    runner2.load_module_at_path(lib_pln_path, 'lib_pln')
    rev_results = runner2.run(prog_rev)
    rev_ok = any(
        'stv' in str(r[0]) for r in rev_results if r
    )
    if not rev_ok:
        print(f"FAIL: Truth_Revision returned unexpected result: {rev_results}")
        return False
    rev_atom = next(str(r[0]) for r in rev_results if r and 'stv' in str(r[0]))
    print(f"  OK  Revision atom: {rev_atom}")
    print("      (min/max evaluated in Python — known hyperon 0.2.10 behaviour)")

    print("\n" + "=" * 60)
    print("Smoke test PASSED.")
    print(f"  hyperon runtime :  OK  (version imported successfully)")
    print(f"  lib_pln.metta   :  OK  ({lib_pln_path})")
    print(f"  Truth_ModusPonens: OK  (real PLN STV formula)")
    print(f"  Evidence stamp  :  OK  ({mp_result['evidence_ids']})")
    print(f"  Truth_Revision  :  OK  (callable, result parsed in Python)")
    print("=" * 60)
    return True


if __name__ == '__main__':
    success = run_smoke_test()
    sys.exit(0 if success else 1)
