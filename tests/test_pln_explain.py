"""
Integration tests for src/pln/pln_explain.py

These tests run the real hyperon runtime with lib_pln.metta.
They require network access on first run (to clone trueagi-io/PLN).

Run:
    cd /path/to/bioperformance
    PLN_CACHE_DIR=/tmp/pln_cache .venv/bin/python3 -m pytest tests/test_pln_explain.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest

# Use a shared cache dir for tests so we only clone once
PLN_CACHE = os.environ.get('PLN_CACHE_DIR', os.path.expanduser('~/.cache/bioperformance/pln'))

from pln.pln_explain import explain, _truth_revision, _truth_revision_combine


class TestTruthRevision:
    """Unit tests for the Python Truth_Revision implementation (lib_pln.metta formula)."""

    def test_identical_sources_increase_confidence(self):
        """Combining two identical evidence sources should raise confidence."""
        s1, c1 = _truth_revision(0.6, 0.5, 0.6, 0.5)
        assert s1 == pytest.approx(0.6, abs=0.01)
        assert c1 > 0.5

    def test_stronger_source_dominates_strength(self):
        """Higher-strength source should pull combined strength toward it."""
        s, c = _truth_revision(0.8, 0.7, 0.3, 0.7)
        assert s > 0.5  # pulled toward the stronger source

    def test_output_bounded(self):
        """Strength and confidence must stay in [0, 1]."""
        s, c = _truth_revision(1.0, 1.0, 1.0, 1.0)
        assert 0.0 <= s <= 1.0
        assert 0.0 <= c <= 1.0

    def test_combine_empty_list(self):
        """Empty list returns (0.0, 0.0)."""
        assert _truth_revision_combine([]) == (0.0, 0.0)

    def test_combine_single(self):
        """Single-element list returns that element unchanged."""
        assert _truth_revision_combine([(0.6, 0.7)]) == (0.6, 0.7)

    def test_combine_multiple_increases_confidence(self):
        """Three independent paths should produce higher confidence than one."""
        single = _truth_revision_combine([(0.5, 0.3)])
        triple = _truth_revision_combine([(0.5, 0.3), (0.5, 0.3), (0.5, 0.3)])
        assert triple[1] > single[1]


class TestExplain:
    """Integration tests — require real hyperon runtime."""

    def test_explain_fatigue_returns_valid_structure(self):
        """explain() with a clear fatigue signal returns a well-formed dict."""
        shap = {
            'acwr_ratio':           0.30,
            'fatigue_28d_mean':     0.25,
            'sleep_quality_7d_mean': 0.20,
        }
        result = explain(
            prediction=6.5,
            shap_values=shap,
            days_of_history=40,
            item='fatigue',
            pln_cache_dir=PLN_CACHE,
        )
        assert result['item'] == 'fatigue'
        assert result['query_target'] == 'is_fatigue_elevated'
        assert 0.0 <= result['strength'] <= 1.0
        assert 0.0 <= result['confidence'] <= 1.0
        assert isinstance(result['evidence_ids'], list)
        assert result['n_rules_fired'] > 0
        assert result['no_evidence'] is False

    def test_explain_soreness_returns_valid_structure(self):
        shap = {'soreness_t0': 0.25, 'srpe_t0': 0.30}
        result = explain(
            prediction=5.0, shap_values=shap,
            days_of_history=30, item='soreness',
            pln_cache_dir=PLN_CACHE,
        )
        assert result['query_target'] == 'is_soreness_elevated'
        assert 0.0 <= result['strength'] <= 1.0

    def test_explain_mood_returns_valid_structure(self):
        shap = {'mood_7d_mean': -0.30, 'fatigue_28d_mean': -0.20}
        result = explain(
            prediction=4.0, shap_values=shap,
            days_of_history=60, item='mood',
            pln_cache_dir=PLN_CACHE,
        )
        assert result['query_target'] == 'is_mood_low'
        assert 0.0 <= result['strength'] <= 1.0

    def test_explain_sleep_quality_returns_valid_structure(self):
        shap = {'sleep_quality_7d_mean': 0.35, 'acwr_ratio': 0.15}
        result = explain(
            prediction=3.5, shap_values=shap,
            days_of_history=45, item='sleep_quality',
            pln_cache_dir=PLN_CACHE,
        )
        assert result['query_target'] == 'is_sleep_poor'
        assert 0.0 <= result['strength'] <= 1.0

    def test_evidence_ids_are_ints(self):
        """Evidence IDs must be integers for JSON serialization."""
        shap = {'acwr_ratio': 0.30}
        result = explain(
            prediction=5.0, shap_values=shap,
            days_of_history=30, item='fatigue',
            pln_cache_dir=PLN_CACHE,
        )
        for eid in result['evidence_ids']:
            assert isinstance(eid, int)

    def test_dynamic_fact_ids_in_evidence(self):
        """At least one evidence ID should be >= 101 (dynamic SHAP fact)."""
        shap = {'acwr_ratio': 0.30}
        result = explain(
            prediction=5.0, shap_values=shap,
            days_of_history=30, item='fatigue',
            pln_cache_dir=PLN_CACHE,
        )
        assert any(eid >= 101 for eid in result['evidence_ids'])

    def test_static_rule_id_in_evidence(self):
        """At least one evidence ID should be <= 99 (static KB rule)."""
        shap = {'acwr_ratio': 0.30}
        result = explain(
            prediction=5.0, shap_values=shap,
            days_of_history=30, item='fatigue',
            pln_cache_dir=PLN_CACHE,
        )
        assert any(eid <= 99 for eid in result['evidence_ids'])

    def test_no_evidence_flag_on_weak_shap(self):
        """All-zero (or None) SHAP values trigger no_evidence=True."""
        shap = {'acwr_ratio': 0.0, 'fatigue_t0': None}
        result = explain(
            prediction=5.0, shap_values=shap,
            days_of_history=30, item='fatigue',
            pln_cache_dir=PLN_CACHE,
        )
        assert result['no_evidence'] is True
        assert result['strength'] == 0.0

    def test_more_rules_fired_produces_higher_confidence(self):
        """
        More matching (fact, rule) pairs should yield higher confidence than one.

        Confidence comes from two sources:
        1. Evidence-count confidence on each fact: k/(k+1) — more facts = higher c
        2. Truth_Revision combining multiple Modus Ponens results — more paths = higher c
        """
        # Single strong signal — 1 fact → fact confidence = 0.5
        shap_one = {'acwr_ratio': 0.35}
        result_one = explain(
            prediction=6.0, shap_values=shap_one,
            days_of_history=40, item='fatigue',
            pln_cache_dir=PLN_CACHE,
        )
        # Multiple signals — 4 facts → fact confidence = 4/5 = 0.8, more rules fire
        shap_many = {
            'acwr_ratio':            0.35,
            'sleep_quality_7d_mean': 0.28,
            'fatigue_28d_mean':      0.22,
            'days_since_last_rest':  0.18,
        }
        result_many = explain(
            prediction=6.0, shap_values=shap_many,
            days_of_history=40, item='fatigue',
            pln_cache_dir=PLN_CACHE,
        )
        assert result_many['n_rules_fired'] > result_one['n_rules_fired']
        assert result_many['confidence'] >= result_one['confidence']

    def test_invalid_item_raises(self):
        """Invalid item name raises ValueError."""
        with pytest.raises(ValueError):
            explain(5.0, {}, 30, item='invalid_item', pln_cache_dir=PLN_CACHE)

    def test_real_pmdata_shap_row_fatigue(self):
        """Smoke test with actual values from shap_per_item.json."""
        shap = {
            'fatigue_28d_sd':         -0.1790,
            'sleep_cv':               -0.1541,
            'fatigue_28d_mean':       -0.0912,
            'sleep_quality_28d_mean': +0.0409,
            'fatigue_zscore_28d':     -0.0358,
        }
        result = explain(
            prediction=2.28,
            shap_values=shap,
            days_of_history=78,
            item='fatigue',
            pln_cache_dir=PLN_CACHE,
        )
        assert result['strength'] > 0.0
        assert result['n_rules_fired'] >= 1
        assert len(result['evidence_ids']) >= 2  # at least 1 fact + 1 rule


class TestNegativeCases:
    """
    Adversarial / bad-mimicry tests.

    These verify that the pipeline does NOT produce falsely confident output
    when fed garbage, spoofed, or mismatched signals. A system that assigns
    high strength/confidence to bad input is more dangerous than one that
    simply predicts wrong — it would mislead coaching staff with apparent
    certainty.

    Each test documents the *expected failure mode* so it is clear what
    "correct rejection" looks like.
    """

    def test_all_noise_no_evidence_flag(self):
        """
        All SHAP values below the noise floor (0.06).
        Expected: no_evidence=True, strength=0.0, zero rules fired.
        The system must refuse to reason rather than fabricate a conclusion.
        """
        shap = {
            'acwr_ratio':            0.01,
            'fatigue_t0':            0.02,
            'sleep_quality_t0':      0.03,
            'soreness_t0':           0.04,
            'mood_t0':               0.05,
        }
        result = explain(
            prediction=5.0, shap_values=shap,
            days_of_history=30, item='fatigue',
            pln_cache_dir=PLN_CACHE,
        )
        assert result['no_evidence'] is True, (
            "All-noise input should set no_evidence=True"
        )
        assert result['strength'] == 0.0
        assert result['confidence'] == 0.0
        assert result['n_rules_fired'] == 0

    def test_empty_shap_dict_no_evidence(self):
        """
        Completely empty SHAP dict (e.g. upstream pipeline failed to produce values).
        Expected: no_evidence=True, no crash, safe zero output.
        """
        result = explain(
            prediction=5.0, shap_values={},
            days_of_history=30, item='fatigue',
            pln_cache_dir=PLN_CACHE,
        )
        assert result['no_evidence'] is True
        assert result['strength'] == 0.0

    def test_none_shap_values_no_evidence(self):
        """
        All SHAP values are None (malformed upstream output).
        Expected: no_evidence=True, no crash.
        """
        shap = {'acwr_ratio': None, 'fatigue_t0': None, 'sleep_quality_7d_mean': None}
        result = explain(
            prediction=5.0, shap_values=shap,
            days_of_history=30, item='fatigue',
            pln_cache_dir=PLN_CACHE,
        )
        assert result['no_evidence'] is True
        assert result['strength'] == 0.0

    def test_wrong_item_features_no_rules_fire(self):
        """
        SHAP features all map to a concept that has no rules for the queried item.
        Specifically: only soreness features passed, querying sleep_quality.
        The KB has no rule 'elevated_soreness -> is_sleep_poor'.
        Expected: strength=0.0, n_rules_fired=0.
        The pipeline must not invent a spurious connection.
        """
        shap = {
            'soreness_t0':       0.40,
            'soreness_28d_mean': 0.35,
            'soreness_7d_mean':  0.30,
        }
        result = explain(
            prediction=5.0, shap_values=shap,
            days_of_history=30, item='sleep_quality',
            pln_cache_dir=PLN_CACHE,
        )
        assert result['n_rules_fired'] == 0, (
            "Soreness features have no rules for is_sleep_poor — "
            "no inference should fire"
        )
        assert result['strength'] == 0.0

    def test_uniform_shap_low_strength(self):
        """
        Attacker floods all features with identical SHAP values — the classic
        'uniform noise disguised as signal' pattern. When every feature looks
        equally important, no single concept dominates.
        Expected: relative attribution gives each fact equal low strength (1/N),
        which after Modus Ponens produces low combined strength — NOT high confidence.
        The system should not be fooled into certainty by volume alone.
        """
        shap = {f: 0.30 for f in [
            'acwr_ratio', 'sleep_quality_7d_mean', 'fatigue_28d_mean',
            'soreness_t0', 'mood_7d_mean',
        ]}
        result = explain(
            prediction=5.0, shap_values=shap,
            days_of_history=30, item='fatigue',
            pln_cache_dir=PLN_CACHE,
        )
        # With 5 equal features, each gets strength = 0.20. After MP with
        # rules of strength ~0.60, derived STV strength < 0.20.
        # Revised combination of 4 such weak paths still stays well below 0.5.
        assert result['strength'] < 0.50, (
            f"Uniform-signal flood should not produce high strength "
            f"(got {result['strength']:.4f})"
        )

    def test_extreme_outlier_shap_stays_bounded(self):
        """
        Corrupted or spoofed SHAP values with extreme magnitudes (e.g. 50x normal).
        Expected: relative attribution absorbs the extreme value as a ratio,
        so strength stays in [0, 1]. The system must not overflow or produce NaN.
        """
        shap = {
            'acwr_ratio':            50.0,   # ~350x normal max
            'sleep_quality_7d_mean': -30.0,  # ~200x normal max
        }
        result = explain(
            prediction=5.0, shap_values=shap,
            days_of_history=30, item='fatigue',
            pln_cache_dir=PLN_CACHE,
        )
        assert 0.0 <= result['strength'] <= 1.0, (
            f"Extreme SHAP values must not overflow: strength={result['strength']}"
        )
        assert 0.0 <= result['confidence'] <= 1.0, (
            f"Extreme SHAP values must not overflow: confidence={result['confidence']}"
        )
        assert result['strength'] == result['strength'], "strength must not be NaN"
        assert result['confidence'] == result['confidence'], "confidence must not be NaN"

    def test_invalid_item_name_raises_not_silently_wrong(self):
        """
        Passing an unknown item name should raise ValueError immediately,
        not silently produce a result for the wrong query target.
        A silent wrong answer is worse than a loud error.
        """
        with pytest.raises(ValueError, match="item must be one of"):
            explain(
                prediction=5.0,
                shap_values={'acwr_ratio': 0.30},
                days_of_history=30,
                item='heart_rate_variability',  # not a valid item
                pln_cache_dir=PLN_CACHE,
            )

    def test_high_confidence_requires_multiple_independent_facts(self):
        """
        High confidence (> 0.5 in the final combined output) should require
        multiple independent pieces of evidence. A single fact, no matter how
        strong, cannot alone produce high confidence because:
        - fact confidence = 1/(1+1) = 0.50 (one fact)
        - Modus Ponens multiplies by rule confidence (~0.65-0.75)
        - final confidence < 0.50

        This guards against a fake single-fact injection claiming high certainty.
        """
        shap_single = {'acwr_ratio': 0.40}   # one strong signal only
        result = explain(
            prediction=7.0, shap_values=shap_single,
            days_of_history=90, item='fatigue',
            pln_cache_dir=PLN_CACHE,
        )
        assert result['confidence'] < 0.50, (
            f"A single fact should not produce confidence >= 0.50 "
            f"(got {result['confidence']:.4f}). "
            "High confidence requires multiple independent evidence sources."
        )

    def test_stv_never_exceeds_one_regardless_of_input(self):
        """
        Property test: no matter what valid (non-adversarial) SHAP input
        is provided, strength and confidence must always stay in [0, 1].
        Tests a range of realistic input patterns.
        """
        test_inputs = [
            {'acwr_ratio': 0.50},
            {'acwr_ratio': 0.30, 'sleep_quality_7d_mean': 0.25},
            {'acwr_ratio': 0.30, 'sleep_quality_7d_mean': 0.25, 'fatigue_28d_mean': 0.20},
            {'acwr_ratio': 0.30, 'sleep_quality_7d_mean': 0.25,
             'fatigue_28d_mean': 0.20, 'days_since_last_rest': 0.18},
        ]
        for shap in test_inputs:
            for item in ['fatigue', 'soreness', 'mood', 'sleep_quality']:
                result = explain(
                    prediction=5.0, shap_values=shap,
                    days_of_history=30, item=item,
                    pln_cache_dir=PLN_CACHE,
                )
                assert 0.0 <= result['strength'] <= 1.0, (
                    f"Overflow: item={item} shap={shap} "
                    f"strength={result['strength']}"
                )
                assert 0.0 <= result['confidence'] <= 1.0, (
                    f"Overflow: item={item} shap={shap} "
                    f"confidence={result['confidence']}"
                )


if __name__ == '__main__':
    import pytest as pt
    pt.main([__file__, '-v'])
