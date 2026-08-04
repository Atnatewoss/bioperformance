;; =============================================================================
;; BioPerformance Knowledge Base
;; For use with trueagi-io/PLN lib_pln.metta via the hyperon Python package.
;;
;; This file is loaded by pln_explain.py via:
;;     runner.load_module_at_path(lib_pln_path, 'lib_pln')
;;     runner.run('!(import! &self lib_pln)\n' + kb_contents + dynamic_facts + query)
;;
;; Structure
;; ---------
;; 1. STV base rates  — prior probabilities for each concept.
;;    Required by PLN's Deduction formula (Truth_Deduction uses STV of intermediate
;;    terms to compute conditional probabilities correctly).
;;
;; 2. Implication rules — (Implication antecedent consequent) with (stv s c).
;;    Each rule is a Sentence with a unique evidence ID (1-99).
;;    Dynamic athlete-day facts use IDs 101+.
;;
;; 3. Per-item query targets — four concepts, one per wellness item:
;;    is_fatigue_elevated, is_soreness_elevated, is_mood_low, is_sleep_poor
;;
;; Rule strengths and confidences are drawn from the sports science literature
;; cited inline. All strengths are capped at 0.85 — no single published finding
;; justifies higher given population-specificity limitations noted in each paper.
;;
;; This file does NOT contain dynamic facts. Those are injected at runtime by
;; pln_explain.py using the concept names defined here.
;;
;; This file is NOT executed directly. It is read as a string and prepended to
;; a per-call MeTTa program. The kb() function pattern is not used here because
;; pln_explain.py drives inference directly via Truth_ModusPonens calls for each
;; (fact, rule) pair — see docs/02_pln_integration_design.md.
;; =============================================================================


;; =============================================================================
;; STV BASE RATES
;; Prior probability that each concept is true for a typical athlete.
;; Values are rough population estimates; confidence reflects how well-studied
;; each concept is. These are used by Truth_Deduction for multi-hop inference.
;; =============================================================================

(= (STV elevated_load)     (stv 0.35 0.80))
(= (STV poor_sleep)        (stv 0.30 0.80))
(= (STV elevated_fatigue)  (stv 0.30 0.75))
(= (STV elevated_soreness) (stv 0.28 0.75))
(= (STV low_mood)          (stv 0.25 0.70))
(= (STV poor_recovery)     (stv 0.25 0.70))

;; Query targets
(= (STV is_fatigue_elevated)  (stv 0.30 0.75))
(= (STV is_soreness_elevated) (stv 0.28 0.75))
(= (STV is_mood_low)          (stv 0.25 0.70))
(= (STV is_sleep_poor)        (stv 0.30 0.75))


;; =============================================================================
;; IMPLICATION RULES — is_fatigue_elevated
;; =============================================================================

;; Rule 1: Elevated training load -> fatigue elevated tomorrow
;; Gabbett (2016) — high ACWR (>1.3) is the strongest single workload predictor.
;; Rossi et al. (2022) — srpe_t0 predicts next-day fatigue in Serie A players.
(= (bp-rule 1)
   (Sentence ((Implication elevated_load is_fatigue_elevated)
              (stv 0.60 0.70)) (1)))

;; Rule 2: Poor sleep -> fatigue elevated tomorrow
;; Taber et al. (2024) — sleep consistency was the #2 feature across all wellness
;; items in NCAA Division I athletes.
(= (bp-rule 2)
   (Sentence ((Implication poor_sleep is_fatigue_elevated)
              (stv 0.55 0.65)) (2)))

;; Rule 3: Chronic fatigue elevation -> fatigue elevated tomorrow
;; Rossi et al. (2022) — fatigue_7d_mean deviation is a leading indicator of
;; next-day fatigue in professional soccer.
(= (bp-rule 3)
   (Sentence ((Implication elevated_fatigue is_fatigue_elevated)
              (stv 0.65 0.75)) (3)))

;; Rule 4: Poor recovery (days since rest) -> fatigue elevated tomorrow
;; Gabbett (2016) — consecutive training days without rest elevates fatigue.
;; Fast bowlers with <2 days recovery had 2.4x injury risk.
(= (bp-rule 4)
   (Sentence ((Implication poor_recovery is_fatigue_elevated)
              (stv 0.55 0.65)) (4)))

;; Rule 5: Low mood -> fatigue elevated tomorrow (cross-signal)
;; Taber (2024) / Rossi (2022) — mood and fatigue are strongly correlated;
;; low mood is a secondary predictor of next-day fatigue.
(= (bp-rule 5)
   (Sentence ((Implication low_mood is_fatigue_elevated)
              (stv 0.45 0.60)) (5)))


;; =============================================================================
;; IMPLICATION RULES — is_soreness_elevated
;; =============================================================================

;; Rule 10: Elevated load -> soreness elevated tomorrow
;; Gabbett (2016) — load spikes precede soreness peaks (lag 1-2 days).
;; Rossi (2022) — srpe_t0 predicts next-day soreness in Serie A.
(= (bp-rule 10)
   (Sentence ((Implication elevated_load is_soreness_elevated)
              (stv 0.60 0.70)) (10)))

;; Rule 11: Elevated soreness -> soreness elevated tomorrow (persistence)
;; Schliep & Schafer (2021) — soreness has the longest temporal lag signature
;; of the four wellness items; yesterday's soreness is the strongest predictor.
(= (bp-rule 11)
   (Sentence ((Implication elevated_soreness is_soreness_elevated)
              (stv 0.65 0.75)) (11)))

;; Rule 12: Poor recovery -> soreness elevated tomorrow
;; Gabbett (2016) — insufficient rest between sessions extends soreness duration.
(= (bp-rule 12)
   (Sentence ((Implication poor_recovery is_soreness_elevated)
              (stv 0.50 0.65)) (12)))

;; Rule 13: Elevated fatigue -> soreness elevated tomorrow (cross-signal)
;; Rossi (2022) — fatigue and soreness are jointly elevated after load spikes;
;; chronic fatigue elevation predicts soreness persistence.
(= (bp-rule 13)
   (Sentence ((Implication elevated_fatigue is_soreness_elevated)
              (stv 0.50 0.60)) (13)))


;; =============================================================================
;; IMPLICATION RULES — is_mood_low
;; =============================================================================

;; Rule 20: Chronic fatigue -> mood low tomorrow
;; Rossi (2022) — fatigue_7d_mean and mood are inversely correlated in
;; professional soccer; chronic fatigue elevation predicts mood decline.
(= (bp-rule 20)
   (Sentence ((Implication elevated_fatigue is_mood_low)
              (stv 0.55 0.65)) (20)))

;; Rule 21: Low mood -> mood low tomorrow (persistence)
;; Schliep & Schafer (2021) — mood recovers fastest of the four items but
;; still has significant day-to-day autocorrelation.
(= (bp-rule 21)
   (Sentence ((Implication low_mood is_mood_low)
              (stv 0.60 0.70)) (21)))

;; Rule 22: Poor sleep -> mood low tomorrow
;; Taber (2024) — sleep consistency and mood co-appear in the top predictors.
(= (bp-rule 22)
   (Sentence ((Implication poor_sleep is_mood_low)
              (stv 0.50 0.60)) (22)))

;; Rule 23: Elevated load -> mood low tomorrow (overreaching signal)
;; Gabbett (2016) — load spikes > 15% week-over-week correlate with mood decline
;; in athletes. The relationship is weaker than for fatigue/soreness.
(= (bp-rule 23)
   (Sentence ((Implication elevated_load is_mood_low)
              (stv 0.40 0.55)) (23)))


;; =============================================================================
;; IMPLICATION RULES — is_sleep_poor
;; =============================================================================

;; Rule 30: Poor sleep -> sleep poor tomorrow (persistence)
;; Taber (2024) — sleep consistency was the #2 feature; poor sleep has strong
;; day-to-day autocorrelation in athlete populations.
(= (bp-rule 30)
   (Sentence ((Implication poor_sleep is_sleep_poor)
              (stv 0.65 0.75)) (30)))

;; Rule 31: Elevated load -> sleep poor tomorrow
;; Gabbett (2016) — high training load disrupts sleep architecture. The
;; relationship is moderate; load explains less sleep variance than sleep
;; history itself.
(= (bp-rule 31)
   (Sentence ((Implication elevated_load is_sleep_poor)
              (stv 0.45 0.60)) (31)))

;; Rule 32: Elevated fatigue -> sleep poor tomorrow (cross-signal)
;; Rossi (2022) — chronic fatigue and sleep disruption co-occur; elevated
;; fatigue baseline predicts worse sleep quality the following night.
(= (bp-rule 32)
   (Sentence ((Implication elevated_fatigue is_sleep_poor)
              (stv 0.45 0.55)) (32)))

;; Rule 33: Low mood -> sleep poor tomorrow
;; Taber (2024) / Rossi (2022) — mood and sleep quality are bidirectionally
;; linked; low mood predicts poorer sleep the following night.
(= (bp-rule 33)
   (Sentence ((Implication low_mood is_sleep_poor)
              (stv 0.45 0.55)) (33)))


;; =============================================================================
;; RULE INDEX — maps query target to relevant rule IDs
;; Used by pln_explain.py to select which rules to apply for each item.
;; =============================================================================

(= (rules-for is_fatigue_elevated)  (1 2 3 4 5))
(= (rules-for is_soreness_elevated) (10 11 12 13))
(= (rules-for is_mood_low)          (20 21 22 23))
(= (rules-for is_sleep_poor)        (30 31 32 33))


;; =============================================================================
;; CONCEPT -> QUERY TARGET LINKS
;; Which rule IDs can fire for a given (concept, query_target) pair.
;; Used to build the set of (fact, rule) pairs for Modus Ponens.
;; =============================================================================

;; is_fatigue_elevated antecedents
(= (links-to elevated_load    is_fatigue_elevated)  (1))
(= (links-to poor_sleep       is_fatigue_elevated)  (2))
(= (links-to elevated_fatigue is_fatigue_elevated)  (3))
(= (links-to poor_recovery    is_fatigue_elevated)  (4))
(= (links-to low_mood         is_fatigue_elevated)  (5))

;; is_soreness_elevated antecedents
(= (links-to elevated_load     is_soreness_elevated) (10))
(= (links-to elevated_soreness is_soreness_elevated) (11))
(= (links-to poor_recovery     is_soreness_elevated) (12))
(= (links-to elevated_fatigue  is_soreness_elevated) (13))

;; is_mood_low antecedents
(= (links-to elevated_fatigue is_mood_low) (20))
(= (links-to low_mood         is_mood_low) (21))
(= (links-to poor_sleep       is_mood_low) (22))
(= (links-to elevated_load    is_mood_low) (23))

;; is_sleep_poor antecedents
(= (links-to poor_sleep       is_sleep_poor) (30))
(= (links-to elevated_load    is_sleep_poor) (31))
(= (links-to elevated_fatigue is_sleep_poor) (32))
(= (links-to low_mood         is_sleep_poor) (33))
