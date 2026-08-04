"""
SHAP -> PLN fact translation.

Converts a SHAP value dict (one athlete-day, one wellness item) into a list of
PLN facts ready to be used with lib_pln.metta's Truth_ModusPonens rule.

Each fact has the shape:
    {"concept": str, "strength": float, "confidence": float, "evidence_id": int}

Calibration approach (production-standard)
------------------------------------------
SHAP values are *relative* within a single prediction — they sum to the
difference between the prediction and the model's base value. Scaling them
against an absolute ceiling misrepresents their meaning: a |SHAP| = 0.14
may be the dominant driver for one prediction and a minor contributor for
another. Using absolute scaling both predictions would get the same low
strength, losing the signal of *dominance*.

The standard production approach (see arXiv:2405.11766 "From SHAP Scores to
Feature Importance Scores") is to express each feature's contribution as its
share of the total attribution in that prediction:

    strength_i = |SHAP_i| / Σ|SHAP_j|  for j in {features above noise floor}

This gives the dominant driver strength ≈ 0.6–0.8, regardless of the absolute
scale of the SHAP values.

Confidence follows the PLN evidence-count formula (Goertzel et al., PLN book
Ch. 5):

    confidence = k / (k + k0)

where k = number of independent facts above the noise floor, and k0 = 1.0
(the prior weight, representing one "virtual" observation at the base rate).
This is the semantically correct PLN interpretation: confidence measures how
much independent evidence is available, not how many training days the athlete
has. Athlete history depth is already encoded in the rule confidences
(bioperformance_kb.metta) via the literature-cited population confidence values.

With these choices, a typical PMData row with 2–3 features above the noise
floor produces strength ≈ 0.55–0.75 for the dominant fact and confidence ≈
0.67–0.75 for the combined explanation — values that represent meaningful
signal to a coach.

Other design decisions
----------------------
- Only features with |SHAP| above NOISE_FLOOR contribute (below this the
  signal is indistinguishable from noise on PMData, where median |SHAP| ~0.12).
- Multiple features can map to the same concept. Only the highest-magnitude
  instance per concept is kept (avoids redundant evidence from correlated
  features like fatigue_t0 and fatigue_7d_mean).
- Evidence IDs for dynamic facts start at 101 so they never collide with
  static KB rule IDs (1–99).
- days_of_history is recorded in the fact for tracing but does not set
  confidence.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Noise floor — calibrated against PMData SHAP distributions
# p25 of top-1 SHAP across all items: ~0.07–0.10
# Features below this threshold contribute negligible variance and are noise.
# ---------------------------------------------------------------------------
NOISE_FLOOR = 0.06

# Evidence-count prior weight (PLN book Ch. 5, k0 parameter).
# k0 = 1.0 means "one virtual observation at the base rate" — standard PLN default.
EVIDENCE_PRIOR = 1.0


# ---------------------------------------------------------------------------
# Feature -> PLN concept mapping
# ---------------------------------------------------------------------------
# Each entry: (feature_name_substring, pln_concept)
# First match wins. Ordering matters: more specific patterns first.
#
# Concepts used:
#   elevated_load     — high ACWR or acute load spike (Gabbett 2016)
#   poor_sleep        — low sleep quality or high sleep variance (Taber 2024)
#   elevated_fatigue  — chronic high fatigue readings (Rossi 2022)
#   elevated_soreness — chronic high soreness readings (Rossi 2022)
#   low_mood          — consistently low mood readings (Rossi 2022 / Taber 2024)
#   poor_recovery     — many consecutive training days without rest (Gabbett 2016)
FEATURE_TO_CONCEPT: list[tuple[str, str]] = [
    # --- Training load ---
    ('acwr_ratio',           'elevated_load'),
    ('acute_load_7d',        'elevated_load'),
    ('chronic_load_28d',     'elevated_load'),
    ('srpe_28d_mean',        'elevated_load'),
    ('srpe_28d_sd',          'elevated_load'),
    ('srpe_zscore_28d',      'elevated_load'),
    ('srpe_7d_mean',         'elevated_load'),
    ('srpe_t0',              'elevated_load'),
    ('srpe_t1',              'elevated_load'),
    ('srpe_t2',              'elevated_load'),
    # --- Recovery ---
    ('days_since_last_rest', 'poor_recovery'),
    # --- Sleep ---
    ('sleep_quality_7d_mean',  'poor_sleep'),
    ('sleep_quality_28d_mean', 'poor_sleep'),
    ('sleep_quality_28d_sd',   'poor_sleep'),
    ('sleep_quality_zscore',   'poor_sleep'),
    ('sleep_cv',               'poor_sleep'),
    ('sleep_quality_t0',       'poor_sleep'),
    ('sleep_quality_t1',       'poor_sleep'),
    ('sleep_quality_t2',       'poor_sleep'),
    # --- Fatigue ---
    ('fatigue_7d_mean',    'elevated_fatigue'),
    ('fatigue_28d_mean',   'elevated_fatigue'),
    ('fatigue_28d_sd',     'elevated_fatigue'),
    ('fatigue_zscore_28d', 'elevated_fatigue'),
    ('fatigue_t0',         'elevated_fatigue'),
    ('fatigue_t1',         'elevated_fatigue'),
    ('fatigue_t2',         'elevated_fatigue'),
    # --- Soreness ---
    ('soreness_7d_mean',    'elevated_soreness'),
    ('soreness_28d_mean',   'elevated_soreness'),
    ('soreness_28d_sd',     'elevated_soreness'),
    ('soreness_zscore_28d', 'elevated_soreness'),
    ('soreness_t0',         'elevated_soreness'),
    ('soreness_t1',         'elevated_soreness'),
    ('soreness_t2',         'elevated_soreness'),
    # --- Mood ---
    ('mood_7d_mean',    'low_mood'),
    ('mood_28d_mean',   'low_mood'),
    ('mood_28d_sd',     'low_mood'),
    ('mood_zscore_28d', 'low_mood'),
    ('mood_t0',         'low_mood'),
    ('mood_t1',         'low_mood'),
    ('mood_t2',         'low_mood'),
]


def _feature_to_concept(feature_name: str) -> str | None:
    """Returns the PLN concept for a feature name, or None if unmapped."""
    for pattern, concept in FEATURE_TO_CONCEPT:
        if pattern in feature_name:
            return concept
    return None


def _evidence_count_confidence(n_facts: int, k0: float = EVIDENCE_PRIOR) -> float:
    """
    PLN evidence-count confidence formula (Goertzel et al., PLN book Ch. 5):
        confidence = k / (k + k0)

    k  = number of independent facts (above noise floor)
    k0 = prior weight (default 1.0)

    Semantics: confidence measures how much independent evidence is available.
    - 0 facts  → 0.0  (no evidence at all)
    - 1 fact   → 0.50 (one piece of evidence, balanced against prior)
    - 2 facts  → 0.67 (two independent pieces)
    - 3 facts  → 0.75
    - 10 facts → 0.91 (approaching certainty with many independent pieces)
    """
    if n_facts <= 0:
        return 0.0
    return n_facts / (n_facts + k0)


def shap_to_facts(
    shap_values: dict[str, float | None],
    days_of_history: int,
    top_n: int = 5,
    start_evidence_id: int = 101,
) -> list[dict]:
    """
    Converts a SHAP value dict into a list of PLN facts.

    Strength uses relative attribution (share of total evidence in this
    prediction). Confidence uses the PLN evidence-count formula.

    Parameters
    ----------
    shap_values : dict
        Feature name -> SHAP value for one athlete-day prediction.
        Values may be None (missing features are skipped).
    days_of_history : int
        Days of personal data available for this athlete.
        Recorded in each fact for tracing but does not set confidence
        (confidence is set by evidence count, not history depth).
    top_n : int
        Maximum number of facts to produce (default 5).
        Only the top-N features by |SHAP| magnitude are considered.
    start_evidence_id : int
        First evidence ID to assign. Defaults to 101 so dynamic facts
        never collide with static KB rules (IDs 1-99).

    Returns
    -------
    list[dict]
        Each dict: {
            "concept":           str,   # PLN concept name
            "strength":          float, # [0, 1] — relative attribution share
            "confidence":        float, # [0, 1] — evidence-count confidence
            "evidence_id":       int,   # unique ID for this fact
            "source_feat":       str,   # original feature name (for tracing)
            "shap_value":        float, # raw SHAP value (for tracing)
            "days_of_history":   int,   # athlete history depth (for tracing)
        }
        Empty if no features exceed the noise floor.
    """
    # Sort all features by |SHAP| descending
    sorted_features = sorted(
        [(k, v) for k, v in shap_values.items() if v is not None],
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )

    # --- Pass 1: collect candidates above noise floor (per concept, keep max) ---
    # Map concept -> (feature_name, shap_value) for the strongest instance
    best_per_concept: dict[str, tuple[str, float]] = {}
    n_considered = 0

    for feat_name, shap_val in sorted_features:
        if n_considered >= top_n * 3:  # examine generously before dedup
            break
        if abs(shap_val) <= NOISE_FLOOR:
            break  # sorted descending — everything after this is noise too

        concept = _feature_to_concept(feat_name)
        if concept is None:
            continue

        if concept not in best_per_concept or abs(shap_val) > abs(best_per_concept[concept][1]):
            best_per_concept[concept] = (feat_name, shap_val)
        n_considered += 1

    if not best_per_concept:
        return []

    # --- Pass 2: sort retained concepts by magnitude and apply top_n cap ---
    retained = sorted(
        best_per_concept.items(),
        key=lambda kv: abs(kv[1][1]),
        reverse=True,
    )[:top_n]

    # --- Relative attribution: strength_i = |SHAP_i| / Σ|SHAP_j| ---
    total_magnitude = sum(abs(v) for _, (_, v) in retained)
    if total_magnitude == 0:
        return []

    # --- Evidence-count confidence: k / (k + k0) where k = len(retained) ---
    confidence = _evidence_count_confidence(len(retained))

    # --- Build fact list ---
    facts: list[dict] = []
    evidence_id = start_evidence_id

    for concept, (feat_name, shap_val) in retained:
        strength = abs(shap_val) / total_magnitude
        facts.append({
            'concept':         concept,
            'strength':        round(strength, 4),
            'confidence':      round(confidence, 4),
            'evidence_id':     evidence_id,
            'source_feat':     feat_name,
            'shap_value':      round(float(shap_val), 4),
            'days_of_history': days_of_history,
        })
        evidence_id += 1

    return facts
