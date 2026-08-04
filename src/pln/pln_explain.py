"""
PLN inference bridge — real hyperon runtime integration.

Sits after XGBoost + SHAP output. Takes one athlete-day's SHAP values,
converts them to PLN facts, runs Truth_ModusPonens from lib_pln.metta for
each matching (fact, rule) pair, combines derived STVs via Truth_Revision,
and returns a structured explanation dict.

Flow
----
shap_values dict
    │
    ▼  shap_to_metta.shap_to_facts()
PLN facts: [{concept, strength, confidence, evidence_id, ...}]
    │
    ▼  _build_mp_program()
MeTTa program string: import lib_pln + KB rules + Truth_ModusPonens calls
    │
    ▼  MeTTa.run()  [real hyperon runtime]
List of ((stv s c) (ev_fact ev_rule)) atoms
    │
    ▼  _parse_mp_results()
List of {strength, confidence, evidence_ids} dicts
    │
    ▼  _truth_revision_combine()
Final {strength, confidence, evidence_ids}  — the explanation

Why Truth_ModusPonens directly (not PLN.Derive / PLN.Query)
-----------------------------------------------------------
trueagi-io/PLN was designed for the PeTTa Prolog runtime. Its PLN.Derive
orchestrator uses `let ($T $B) ...` tuple destructuring that does not reduce
correctly in hyperon 0.2.10 (returns () instead of the tuple). We drive
inference directly via the |- rules and Truth_ModusPonens, which are the
actual PLN inference operations and work correctly. The result is identical:
same formulas, same evidence tracking, same STV semantics.
See docs/02_pln_integration_design.md for full details.
"""

from __future__ import annotations

import os
import re
import json
import logging

from .shap_to_metta import shap_to_facts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_THIS_DIR    = os.path.dirname(__file__)
_KB_PATH     = os.path.join(_THIS_DIR, 'bioperformance_kb.metta')
_DEFAULT_PLN = os.path.expanduser('~/.cache/bioperformance/pln')
PLN_CACHE_DIR = os.environ.get('PLN_CACHE_DIR', _DEFAULT_PLN)

# Map wellness item name to PLN query target
ITEM_TO_TARGET: dict[str, str] = {
    'fatigue':       'is_fatigue_elevated',
    'soreness':      'is_soreness_elevated',
    'mood':          'is_mood_low',
    'sleep_quality': 'is_sleep_poor',
}

# Static rules per query target: (rule_id, antecedent, strength, confidence)
# These mirror bioperformance_kb.metta exactly. Kept in Python so the bridge
# can select which (fact, rule) pairs to apply without parsing the .metta file.
_RULES: dict[str, list[tuple[int, str, float, float]]] = {
    'is_fatigue_elevated': [
        (1,  'elevated_load',     0.60, 0.70),
        (2,  'poor_sleep',        0.55, 0.65),
        (3,  'elevated_fatigue',  0.65, 0.75),
        (4,  'poor_recovery',     0.55, 0.65),
        (5,  'low_mood',          0.45, 0.60),
    ],
    'is_soreness_elevated': [
        (10, 'elevated_load',      0.60, 0.70),
        (11, 'elevated_soreness',  0.65, 0.75),
        (12, 'poor_recovery',      0.50, 0.65),
        (13, 'elevated_fatigue',   0.50, 0.60),
    ],
    'is_mood_low': [
        (20, 'elevated_fatigue',  0.55, 0.65),
        (21, 'low_mood',          0.60, 0.70),
        (22, 'poor_sleep',        0.50, 0.60),
        (23, 'elevated_load',     0.40, 0.55),
    ],
    'is_sleep_poor': [
        (30, 'poor_sleep',        0.65, 0.75),
        (31, 'elevated_load',     0.45, 0.60),
        (32, 'elevated_fatigue',  0.45, 0.55),
        (33, 'low_mood',          0.45, 0.55),
    ],
}


# ---------------------------------------------------------------------------
# PLN cache setup
# ---------------------------------------------------------------------------
def _ensure_pln_cache(cache_dir: str = PLN_CACHE_DIR) -> str:
    """
    Ensures lib_pln.metta is available at cache_dir.
    Clones trueagi-io/PLN on first call. Returns the path to lib_pln.metta.
    """
    lib_pln = os.path.join(cache_dir, 'lib_pln.metta')
    if os.path.exists(lib_pln):
        return lib_pln
    logger.info('PLN not cached at %s. Cloning trueagi-io/PLN...', cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    ret = os.system(
        f'git clone --depth=1 https://github.com/trueagi-io/PLN.git "{cache_dir}" 2>&1'
    )
    if ret != 0 or not os.path.exists(lib_pln):
        raise RuntimeError(
            f'Failed to clone PLN into {cache_dir}. '
            'Check your internet connection or set PLN_CACHE_DIR to an existing clone.'
        )
    return lib_pln


# ---------------------------------------------------------------------------
# MeTTa runner (singleton per process)
# ---------------------------------------------------------------------------
_runner = None
_lib_pln_path: str | None = None


def _get_runner(pln_cache_dir: str = PLN_CACHE_DIR):
    """Returns (and caches) a MeTTa runner with lib_pln loaded."""
    global _runner, _lib_pln_path
    if _runner is not None:
        return _runner, _lib_pln_path

    from hyperon import MeTTa, Environment

    lib_pln = _ensure_pln_cache(pln_cache_dir)
    env = Environment.custom_env(include_paths=[os.path.dirname(lib_pln)])
    runner = MeTTa(env_builder=env)
    runner.load_module_at_path(lib_pln, 'lib_pln')

    _runner = runner
    _lib_pln_path = lib_pln
    return _runner, _lib_pln_path


# ---------------------------------------------------------------------------
# Truth_Revision (verbatim formula from lib_pln.metta lines 136-143)
# ---------------------------------------------------------------------------
def _truth_revision(s1: float, c1: float, s2: float, c2: float) -> tuple[float, float]:
    """
    PLN Truth_Revision formula from lib_pln.metta.

    Combines two independently-derived STV pairs into one.
    Uses the weight-based formula:
        w  = c / (1 - c)
        s  = (w1*s1 + w2*s2) / (w1 + w2)
        c  = w / (w + 1)
    then clamps s and c to [0, 1] and takes max(c, c1, c2).

    Note: Truth_Revision in MeTTa returns `(stv (min 1.0 X) ...)` unevaluated
    because `stv` is a symbolic constructor that doesn't reduce its arguments.
    This Python function evaluates the same formula numerically.
    """
    def c2w(c: float) -> float:
        return c / (1.0 - c) if c < 1.0 else 1e9

    def w2c(w: float) -> float:
        return w / (w + 1.0)

    w1, w2 = c2w(c1), c2w(c2)
    w  = w1 + w2
    s  = (w1 * s1 + w2 * s2) / w
    c  = w2c(w)
    return min(1.0, s), min(1.0, max(max(c, c1), c2))


def _truth_revision_combine(stv_list: list[tuple[float, float]]) -> tuple[float, float]:
    """Applies Truth_Revision iteratively to combine a list of (s, c) pairs."""
    if not stv_list:
        return (0.0, 0.0)
    s, c = stv_list[0]
    for s2, c2 in stv_list[1:]:
        s, c = _truth_revision(s, c, s2, c2)
    return s, c


# ---------------------------------------------------------------------------
# MeTTa program builder
# ---------------------------------------------------------------------------
def _build_mp_program(
    facts: list[dict],
    query_target: str,
) -> str:
    """
    Builds a MeTTa program that applies Truth_ModusPonens for each
    (fact, rule) pair where fact.concept matches rule.antecedent.

    Each call returns: ((stv s c) (ev_fact ev_rule))

    Returns the program string (without the import header — added by caller).
    """
    rules = _RULES.get(query_target, [])
    # Build fact lookup: concept -> fact dict
    facts_by_concept = {f['concept']: f for f in facts}

    lines = []
    for rule_id, antecedent, rule_s, rule_c in rules:
        fact = facts_by_concept.get(antecedent)
        if fact is None:
            continue  # no SHAP evidence for this antecedent
        fs = fact['strength']
        fc = fact['confidence']
        ev_fact = fact['evidence_id']
        lines.append(
            f'!(let $stv (Truth_ModusPonens (stv {fs} {fc}) (stv {rule_s} {rule_c}))\n'
            f'   ($stv ({ev_fact} {rule_id})))'
        )

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Result parser
# ---------------------------------------------------------------------------
# Pattern: ((stv 0.4376 0.250992) (101 3))
_MP_RESULT_RE = re.compile(
    r'^\(\(stv\s+([\d.]+)\s+([\d.]+)\)\s+\((\d+)\s+(\d+)\)\)$'
)


def _parse_mp_results(results) -> list[dict]:
    """
    Extracts (stv, evidence_ids) from the list of result sets returned
    by MeTTa.run().

    Returns a list of dicts: {strength, confidence, evidence_ids: [int, int]}
    """
    parsed = []
    for result_set in results:
        if not result_set:
            continue
        atom_str = str(result_set[0])
        m = _MP_RESULT_RE.match(atom_str)
        if m:
            parsed.append({
                'strength':     float(m.group(1)),
                'confidence':   float(m.group(2)),
                'evidence_ids': [int(m.group(3)), int(m.group(4))],
            })
    return parsed


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def explain(
    prediction: float,
    shap_values: dict[str, float | None],
    days_of_history: int,
    item: str,
    top_n: int = 5,
    pln_cache_dir: str = PLN_CACHE_DIR,
) -> dict:
    """
    Produces a PLN-backed explanation for one athlete-day per-item prediction.

    Parameters
    ----------
    prediction : float
        Raw XGBoost prediction value (0-10 scale).
    shap_values : dict
        Feature name -> SHAP value for this prediction row.
    days_of_history : int
        Days of personal data available for this athlete.
    item : str
        Which wellness item: 'fatigue', 'soreness', 'mood', 'sleep_quality'.
    top_n : int
        Maximum number of SHAP features to convert to facts (default 5).
    pln_cache_dir : str
        Path to the local PLN clone. Auto-populated on first call.

    Returns
    -------
    dict with keys:
        item              : str   — wellness item
        query_target      : str   — PLN concept queried (e.g. 'is_fatigue_elevated')
        prediction        : float — raw XGBoost prediction
        strength          : float — combined PLN strength [0, 1]
        confidence        : float — combined PLN confidence [0, 1]
        evidence_ids      : list[int] — all evidence IDs used in derivation
        facts_used        : list[dict] — the SHAP facts that fired at least one rule
        n_rules_fired     : int   — number of Modus Ponens applications
        no_evidence       : bool  — True if no SHAP facts exceeded noise floor
    """
    if item not in ITEM_TO_TARGET:
        raise ValueError(
            f"item must be one of {sorted(ITEM_TO_TARGET)}. Got {item!r}"
        )

    query_target = ITEM_TO_TARGET[item]

    # Step 1: convert SHAP to PLN facts
    facts = shap_to_facts(shap_values, days_of_history=days_of_history, top_n=top_n)

    if not facts:
        return {
            'item':          item,
            'query_target':  query_target,
            'prediction':    prediction,
            'strength':      0.0,
            'confidence':    0.0,
            'evidence_ids':  [],
            'facts_used':    [],
            'n_rules_fired': 0,
            'no_evidence':   True,
        }

    # Step 2: build MeTTa program and run through hyperon
    mp_program = _build_mp_program(facts, query_target)

    if not mp_program.strip():
        # Facts exist but none match any rule antecedent for this item
        return {
            'item':          item,
            'query_target':  query_target,
            'prediction':    prediction,
            'strength':      0.0,
            'confidence':    0.0,
            'evidence_ids':  [],
            'facts_used':    facts,
            'n_rules_fired': 0,
            'no_evidence':   False,
        }

    runner, _ = _get_runner(pln_cache_dir)
    full_program = '!(import! &self lib_pln)\n' + mp_program
    results = runner.run(full_program)

    # Step 3: parse results
    derived = _parse_mp_results(results)

    if not derived:
        return {
            'item':          item,
            'query_target':  query_target,
            'prediction':    prediction,
            'strength':      0.0,
            'confidence':    0.0,
            'evidence_ids':  [],
            'facts_used':    facts,
            'n_rules_fired': 0,
            'no_evidence':   False,
        }

    # Step 4: combine all derived STVs via Truth_Revision
    stv_pairs = [(d['strength'], d['confidence']) for d in derived]
    final_s, final_c = _truth_revision_combine(stv_pairs)

    # Collect all unique evidence IDs
    all_evidence_ids = sorted({eid for d in derived for eid in d['evidence_ids']})

    # Identify which facts actually fired a rule
    fired_fact_ev_ids = {d['evidence_ids'][0] for d in derived}
    facts_used = [f for f in facts if f['evidence_id'] in fired_fact_ev_ids]

    return {
        'item':          item,
        'query_target':  query_target,
        'prediction':    round(prediction, 4),
        'strength':      round(final_s, 4),
        'confidence':    round(final_c, 4),
        'evidence_ids':  all_evidence_ids,
        'facts_used':    facts_used,
        'n_rules_fired': len(derived),
        'no_evidence':   False,
    }


def run_explain(sample: int = 3, pln_cache_dir: str = PLN_CACHE_DIR) -> None:
    """
    Reads shap_per_item.json, runs PLN for every unique (item, athlete, date),
    writes pln_explanations.json, and prints a compact sample to stdout.
    Called by main.py's `explain` subcommand.
    """
    shap_path = 'data/processed/shap_per_item.json'
    out_path  = 'data/processed/pln_explanations.json'

    if not os.path.exists(shap_path):
        raise SystemExit(
            f'{shap_path} not found. Run `python src/main.py shap` first.'
        )

    with open(shap_path) as f:
        rows = json.load(f)

    print(f'Running PLN explanations for {len(rows)} athlete-day rows...')
    print('(First run will clone trueagi-io/PLN — may take ~30s)\n')

    explanations = []
    seen: set[tuple] = set()

    for entry in rows:
        key = (entry['item'], entry['athlete_id'], entry['date'])
        if key in seen:
            continue
        seen.add(key)

        result = explain(
            prediction=entry['prediction'],
            shap_values=entry['shap_values'],
            days_of_history=entry['days_of_history'],
            item=entry['item'],
            pln_cache_dir=pln_cache_dir,
        )
        explanations.append({**entry, 'pln': result})

    os.makedirs('data/processed', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(explanations, f, indent=2)

    print(f'Saved {len(explanations)} explanations to {out_path}\n')
    _print_sample(explanations, sample)


def _print_sample(explanations: list[dict], sample: int) -> None:
    """Prints a compact per-item sample of PLN explanation results."""
    items_order = ['fatigue', 'soreness', 'mood', 'sleep_quality']
    for item in items_order:
        shown = 0
        for entry in explanations:
            if entry['item'] != item or shown >= sample:
                continue
            pln = entry['pln']
            print('=' * 62)
            print(
                f"item={item:<14} athlete={entry['athlete_id']:<4} "
                f"date={entry['date']}  (history={entry['days_of_history']} days)"
            )
            print(f"  prediction={entry['prediction']:.2f}  actual={entry['actual']:.2f}")
            print(
                f"  {pln['query_target']}: "
                f"strength={pln['strength']:.4f}  "
                f"confidence={pln['confidence']:.4f}"
            )
            if pln['evidence_ids']:
                print(f"  evidence used: rules {pln['evidence_ids']}")
            if pln['facts_used']:
                print('  facts fired:')
                for fact in pln['facts_used']:
                    print(
                        f"    {fact['concept']:<22} "
                        f"s={fact['strength']:.3f}  "
                        f"<- {fact['source_feat']} "
                        f"(SHAP {fact['shap_value']:+.4f})"
                    )
            if pln['no_evidence']:
                print('  [no SHAP evidence above noise floor]')
            elif pln['n_rules_fired'] == 0:
                print('  [no rules fired for this item]')
            print()
            shown += 1
