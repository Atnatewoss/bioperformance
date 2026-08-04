"""
Unit tests for src/pln/shap_to_metta.py

Run:
    cd /path/to/bioperformance
    .venv/bin/python3 -m pytest tests/test_shap_to_metta.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from pln.shap_to_metta import (
    shap_to_facts, NOISE_FLOOR, EVIDENCE_PRIOR, _evidence_count_confidence
)


class TestEvidenceCountConfidence:
    """Unit tests for the PLN evidence-count confidence formula."""

    def test_zero_facts_gives_zero(self):
        assert _evidence_count_confidence(0) == 0.0

    def test_one_fact_gives_half(self):
        """k=1, k0=1 → 1/2 = 0.5"""
        assert _evidence_count_confidence(1) == pytest.approx(0.5, abs=0.001)

    def test_two_facts(self):
        """k=2, k0=1 → 2/3 ≈ 0.667"""
        assert _evidence_count_confidence(2) == pytest.approx(2/3, abs=0.001)

    def test_three_facts(self):
        """k=3, k0=1 → 3/4 = 0.75"""
        assert _evidence_count_confidence(3) == pytest.approx(0.75, abs=0.001)

    def test_approaches_one_with_many_facts(self):
        assert _evidence_count_confidence(100) > 0.99

    def test_bounded(self):
        """Must never exceed 1.0."""
        assert _evidence_count_confidence(1000) <= 1.0


class TestShapToFacts:

    def test_basic_conversion(self):
        """Top feature maps to correct concept with non-zero strength."""
        shap = {'acwr_ratio': 0.30, 'fatigue_t0': 0.05}
        facts = shap_to_facts(shap, days_of_history=30)
        # fatigue_t0 is at noise floor so only acwr_ratio should survive
        assert len(facts) == 1
        assert facts[0]['concept'] == 'elevated_load'
        assert 0.0 < facts[0]['strength'] <= 1.0

    def test_noise_floor_filters_weak_signals(self):
        """Features at or below NOISE_FLOOR produce no facts."""
        shap = {'acwr_ratio': NOISE_FLOOR, 'fatigue_28d_mean': NOISE_FLOOR - 0.001}
        facts = shap_to_facts(shap, days_of_history=30)
        assert facts == []

    def test_single_feature_gets_full_strength(self):
        """One feature above floor → it gets 100% of relative attribution = 1.0."""
        shap = {'acwr_ratio': 0.30}
        facts = shap_to_facts(shap, days_of_history=30)
        assert len(facts) == 1
        assert facts[0]['strength'] == pytest.approx(1.0, abs=0.001)

    def test_relative_attribution_sums_to_one(self):
        """Strength values across all facts sum to 1.0."""
        shap = {
            'acwr_ratio':            0.30,
            'sleep_quality_7d_mean': 0.20,
            'fatigue_28d_mean':      0.10,
        }
        facts = shap_to_facts(shap, days_of_history=30)
        total = sum(f['strength'] for f in facts)
        assert total == pytest.approx(1.0, abs=0.001)

    def test_dominant_feature_gets_majority(self):
        """A feature with 3x the magnitude of others gets ~75% of strength."""
        shap = {
            'acwr_ratio':            0.30,   # 3x
            'sleep_quality_7d_mean': 0.10,   # 1x
        }
        facts = shap_to_facts(shap, days_of_history=30)
        load_fact = next(f for f in facts if f['concept'] == 'elevated_load')
        # 0.30 / (0.30 + 0.10) = 0.75
        assert load_fact['strength'] == pytest.approx(0.75, abs=0.01)

    def test_confidence_is_evidence_count_based(self):
        """Confidence = k/(k+1) where k = number of facts above noise floor."""
        shap = {
            'acwr_ratio':            0.30,   # elevated_load
            'sleep_quality_7d_mean': 0.25,   # poor_sleep
        }
        facts = shap_to_facts(shap, days_of_history=30)
        # 2 facts → confidence = 2/(2+1) ≈ 0.667
        assert facts[0]['confidence'] == pytest.approx(2/3, abs=0.01)

    def test_confidence_three_facts(self):
        """Three facts → confidence = 3/4 = 0.75."""
        shap = {
            'acwr_ratio':            0.30,
            'sleep_quality_7d_mean': 0.25,
            'fatigue_28d_mean':      0.20,
        }
        facts = shap_to_facts(shap, days_of_history=30)
        assert facts[0]['confidence'] == pytest.approx(0.75, abs=0.01)

    def test_confidence_same_for_all_facts_in_prediction(self):
        """All facts in one prediction share the same confidence value."""
        shap = {
            'acwr_ratio':            0.30,
            'sleep_quality_7d_mean': 0.25,
        }
        facts = shap_to_facts(shap, days_of_history=30)
        confidences = {f['confidence'] for f in facts}
        assert len(confidences) == 1

    def test_days_of_history_recorded_for_tracing(self):
        """days_of_history is stored in the fact for tracing, not for confidence."""
        shap = {'acwr_ratio': 0.30}
        facts_10  = shap_to_facts(shap, days_of_history=10)
        facts_100 = shap_to_facts(shap, days_of_history=100)
        # History depth is stored
        assert facts_10[0]['days_of_history'] == 10
        assert facts_100[0]['days_of_history'] == 100
        # But confidence is the same (1 fact in both cases → 0.5)
        assert facts_10[0]['confidence'] == facts_100[0]['confidence']

    def test_top_n_limit(self):
        """At most top_n facts are returned."""
        shap = {
            'acwr_ratio':             0.40,
            'sleep_quality_7d_mean':  0.35,
            'fatigue_28d_mean':       0.30,
            'soreness_t0':            0.25,
            'mood_7d_mean':           0.20,
            'days_since_last_rest':   0.18,
        }
        facts = shap_to_facts(shap, days_of_history=30, top_n=3)
        assert len(facts) == 3

    def test_deduplication_keeps_strongest(self):
        """When two features map to the same concept, only the stronger survives."""
        shap = {
            'fatigue_7d_mean':   0.35,   # elevated_fatigue, stronger
            'fatigue_t0':        0.20,   # elevated_fatigue, weaker
        }
        facts = shap_to_facts(shap, days_of_history=30)
        fatigue_facts = [f for f in facts if f['concept'] == 'elevated_fatigue']
        assert len(fatigue_facts) == 1
        assert fatigue_facts[0]['source_feat'] == 'fatigue_7d_mean'

    def test_multiple_concepts(self):
        """Multiple distinct concepts are all captured."""
        shap = {
            'acwr_ratio':            0.30,
            'sleep_quality_7d_mean': 0.25,
            'fatigue_28d_mean':      0.20,
        }
        facts = shap_to_facts(shap, days_of_history=30)
        concepts = {f['concept'] for f in facts}
        assert 'elevated_load'    in concepts
        assert 'poor_sleep'       in concepts
        assert 'elevated_fatigue' in concepts

    def test_none_values_skipped(self):
        """Features with None SHAP value are skipped."""
        shap = {'acwr_ratio': None, 'sleep_quality_7d_mean': 0.25}
        facts = shap_to_facts(shap, days_of_history=30)
        assert len(facts) == 1
        assert facts[0]['concept'] == 'poor_sleep'

    def test_evidence_ids_unique_and_sequential(self):
        """Each fact gets a unique, sequential evidence ID starting at 101."""
        shap = {
            'acwr_ratio':            0.35,
            'sleep_quality_7d_mean': 0.28,
            'fatigue_28d_mean':      0.22,
        }
        facts = shap_to_facts(shap, days_of_history=30)
        ids = [f['evidence_id'] for f in facts]
        assert ids == list(range(101, 101 + len(facts)))

    def test_unmapped_features_ignored(self):
        """Features not in FEATURE_TO_CONCEPT are silently ignored."""
        shap = {'unknown_feature_xyz': 0.50, 'acwr_ratio': 0.30}
        facts = shap_to_facts(shap, days_of_history=30)
        assert len(facts) == 1
        assert facts[0]['concept'] == 'elevated_load'

    def test_empty_shap_returns_empty(self):
        """Empty SHAP dict returns empty list."""
        facts = shap_to_facts({}, days_of_history=30)
        assert facts == []

    def test_strength_ordering(self):
        """Facts are ordered by descending |SHAP| magnitude."""
        shap = {
            'sleep_quality_7d_mean': 0.20,
            'acwr_ratio':            0.40,   # higher — should be first
        }
        facts = shap_to_facts(shap, days_of_history=30)
        assert facts[0]['concept'] == 'elevated_load'

    def test_real_pmdata_row_strength_meaningful(self):
        """
        Real PMData row: dominant feature should have strength >= 0.40
        (it contributes ~47% of total attribution).
        """
        shap = {
            'fatigue_28d_sd':          -0.1790,   # elevated_fatigue — dominant
            'sleep_cv':                -0.1541,   # poor_sleep
            'fatigue_28d_mean':        -0.0912,   # elevated_fatigue (weaker)
            'sleep_quality_28d_mean':  +0.0409,   # below noise floor
            'fatigue_zscore_28d':      -0.0358,   # below noise floor
        }
        facts = shap_to_facts(shap, days_of_history=78)
        assert len(facts) >= 1
        # Dominant feature (fatigue_28d_sd, |0.179|) should be first
        assert facts[0]['concept'] == 'elevated_fatigue'
        # 0.179 / (0.179 + 0.154) ≈ 0.537 — clearly meaningful strength
        assert facts[0]['strength'] >= 0.40

    def test_real_pmdata_row_confidence_meaningful(self):
        """
        2 facts above noise floor → confidence = 2/3 ≈ 0.667.
        This is the kind of value that represents meaningful signal to a coach.
        """
        shap = {
            'fatigue_28d_sd':  -0.1790,
            'sleep_cv':        -0.1541,
        }
        facts = shap_to_facts(shap, days_of_history=78)
        assert facts[0]['confidence'] == pytest.approx(2/3, abs=0.01)


if __name__ == '__main__':
    import pytest as pt
    pt.main([__file__, '-v'])
