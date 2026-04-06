"""E-040: Three-valued vs binary verdict comparison experiment.

Compares false positive/false negative rates between:
- Binary: majority-vote (pass_rate >= 0.5 → PASS, else FAIL)
- Three-valued: PASS/FAIL/INCONCLUSIVE with configurable inconclusive band

Simulates non-deterministic agent evaluations at 11 quality levels (0.0–1.0)
with 100 Monte Carlo runs per level, K=5 trials each.
"""

import json
import random

import pytest

from checkagent.core.types import AgentInput, AgentRun
from checkagent.judge.judge import RubricJudge
from checkagent.judge.types import Criterion, Rubric, Verdict
from checkagent.judge.verdict import compute_verdict

# --- Helpers ---

def _make_run() -> AgentRun:
    return AgentRun(input=AgentInput(query="test"), final_output="result")


def _make_noisy_judge(true_quality: float, noise_std: float = 0.15) -> RubricJudge:
    """Create a judge that returns noisy scores around true_quality.

    Simulates non-deterministic LLM evaluation: each trial returns
    true_quality + N(0, noise_std), clipped to [0, 1].
    """
    rubric = Rubric(
        name="noisy",
        criteria=[Criterion(name="quality", description="overall quality")],
    )

    async def noisy_llm(system: str, user: str) -> str:
        score = true_quality + random.gauss(0, noise_std)
        score = max(0.0, min(1.0, score))
        raw = 1 + score * 4  # Map [0,1] → [1,5] for rubric scale
        return json.dumps({
            "scores": [{"criterion": "quality", "value": raw, "reasoning": ""}],
            "overall_reasoning": "",
        })

    return RubricJudge(rubric=rubric, llm=noisy_llm, model_name="mock-noisy")


def binary_verdict_from_pass_rate(pass_rate: float) -> str:
    """Binary majority vote: >= 0.5 pass rate → PASS, else FAIL."""
    return "PASS" if pass_rate >= 0.5 else "FAIL"


def classify_outcome(
    predicted: str, true_quality: float, quality_threshold: float = 0.7
) -> str:
    """Classify a verdict as TP, TN, FP, or FN.

    Ground truth: true_quality >= quality_threshold means the agent is "good".
    """
    actually_good = true_quality >= quality_threshold
    predicted_pass = predicted == "PASS"

    if actually_good and predicted_pass:
        return "TP"
    elif actually_good and not predicted_pass:
        return "FN"
    elif not actually_good and predicted_pass:
        return "FP"
    else:
        return "TN"


# --- Experiment ---

QUALITY_LEVELS = [round(q * 0.1, 1) for q in range(11)]  # 0.0 to 1.0
NUM_SIMULATIONS = 100
NUM_TRIALS = 5
THRESHOLD = 0.7
NOISE_STD = 0.15
QUALITY_THRESHOLD = 0.7  # Ground truth: agent is "good" if true_quality >= this


class TestThreeValuedVsBinary:
    """Compare three-valued vs binary verdicts across quality levels."""

    @pytest.mark.asyncio
    async def test_binary_has_higher_error_rate_near_boundary(self):
        """Near the decision boundary, binary forces wrong decisions that
        three-valued routes to INCONCLUSIVE."""
        random.seed(42)
        boundary_quality = 0.7  # Right at the threshold

        binary_errors = 0
        three_valued_errors = 0
        inconclusive_count = 0
        n_sims = 50

        for _ in range(n_sims):
            judge = _make_noisy_judge(boundary_quality, noise_std=NOISE_STD)
            run = _make_run()

            v = await compute_verdict(
                judge, run,
                num_trials=NUM_TRIALS,
                threshold=THRESHOLD,
                min_pass_rate=0.5,
                inconclusive_band=0.15,
            )

            # Binary: just use pass_rate with majority vote
            binary_decision = binary_verdict_from_pass_rate(v.pass_rate)
            binary_outcome = classify_outcome(binary_decision, boundary_quality)
            if binary_outcome in ("FP", "FN"):
                binary_errors += 1

            # Three-valued: INCONCLUSIVE is not an error
            if v.verdict == Verdict.INCONCLUSIVE:
                inconclusive_count += 1
            else:
                tv_outcome = classify_outcome(v.verdict.value.upper(), boundary_quality)
                if tv_outcome in ("FP", "FN"):
                    three_valued_errors += 1

        # Three-valued should have fewer hard errors (routes borderline to INCONCLUSIVE)
        assert three_valued_errors <= binary_errors, (
            f"Three-valued ({three_valued_errors}) should have <= errors than "
            f"binary ({binary_errors})"
        )
        # Should have some INCONCLUSIVE verdicts near boundary
        assert inconclusive_count > 0, "Expected some INCONCLUSIVE near boundary"

    @pytest.mark.asyncio
    async def test_clear_pass_both_agree(self):
        """At high quality (0.9+), both methods should agree on PASS."""
        random.seed(42)
        judge = _make_noisy_judge(0.95, noise_std=NOISE_STD)

        pass_count = 0
        n_sims = 30
        for _ in range(n_sims):
            v = await compute_verdict(
                judge, _make_run(),
                num_trials=NUM_TRIALS, threshold=THRESHOLD,
            )
            if v.verdict == Verdict.PASS:
                pass_count += 1

        # At quality=0.95, nearly all should pass
        assert pass_count >= n_sims * 0.8

    @pytest.mark.asyncio
    async def test_clear_fail_both_agree(self):
        """At low quality (0.2), both methods should agree on FAIL."""
        random.seed(42)
        judge = _make_noisy_judge(0.2, noise_std=NOISE_STD)

        fail_count = 0
        n_sims = 30
        for _ in range(n_sims):
            v = await compute_verdict(
                judge, _make_run(),
                num_trials=NUM_TRIALS, threshold=THRESHOLD,
            )
            if v.verdict == Verdict.FAIL:
                fail_count += 1

        assert fail_count >= n_sims * 0.8

    @pytest.mark.asyncio
    async def test_full_comparison_matrix(self):
        """Run full Monte Carlo comparison across all quality levels.

        This is the core experiment: collect FP/FN rates for binary vs
        three-valued across 11 quality levels.
        """
        random.seed(42)
        results = []

        for quality in QUALITY_LEVELS:
            binary_fp = 0
            binary_fn = 0
            tv_fp = 0
            tv_fn = 0
            tv_inconclusive = 0

            for _ in range(NUM_SIMULATIONS):
                judge = _make_noisy_judge(quality, noise_std=NOISE_STD)
                v = await compute_verdict(
                    judge, _make_run(),
                    num_trials=NUM_TRIALS,
                    threshold=THRESHOLD,
                    min_pass_rate=0.5,
                    inconclusive_band=0.15,
                )

                # Binary classification
                binary_decision = binary_verdict_from_pass_rate(v.pass_rate)
                binary_out = classify_outcome(binary_decision, quality)
                if binary_out == "FP":
                    binary_fp += 1
                elif binary_out == "FN":
                    binary_fn += 1

                # Three-valued classification
                if v.verdict == Verdict.INCONCLUSIVE:
                    tv_inconclusive += 1
                else:
                    tv_out = classify_outcome(
                        v.verdict.value.upper(), quality
                    )
                    if tv_out == "FP":
                        tv_fp += 1
                    elif tv_out == "FN":
                        tv_fn += 1

            results.append({
                "quality": quality,
                "binary_fp_rate": binary_fp / NUM_SIMULATIONS,
                "binary_fn_rate": binary_fn / NUM_SIMULATIONS,
                "binary_error_rate": (binary_fp + binary_fn) / NUM_SIMULATIONS,
                "tv_fp_rate": tv_fp / NUM_SIMULATIONS,
                "tv_fn_rate": tv_fn / NUM_SIMULATIONS,
                "tv_inconclusive_rate": tv_inconclusive / NUM_SIMULATIONS,
                "tv_error_rate": (tv_fp + tv_fn) / NUM_SIMULATIONS,
            })

        # --- Assertions on the collected data ---

        # 1. Three-valued total error rate <= binary total error rate at every level
        for r in results:
            assert r["tv_error_rate"] <= r["binary_error_rate"] + 0.01, (
                f"At quality={r['quality']}: three-valued error "
                f"({r['tv_error_rate']:.2f}) > binary ({r['binary_error_rate']:.2f})"
            )

        # 2. INCONCLUSIVE concentrates near the boundary (0.5-0.8)
        boundary_inconclusive = sum(
            r["tv_inconclusive_rate"]
            for r in results
            if 0.5 <= r["quality"] <= 0.8
        )
        extreme_inconclusive = sum(
            r["tv_inconclusive_rate"]
            for r in results
            if r["quality"] <= 0.2 or r["quality"] >= 0.95
        )
        assert boundary_inconclusive > extreme_inconclusive, (
            "INCONCLUSIVE should concentrate near the decision boundary"
        )

        # 3. At extremes (0.0, 0.1, 0.9, 1.0), both methods should have low error
        for r in results:
            if r["quality"] <= 0.1 or r["quality"] >= 0.9:
                assert r["binary_error_rate"] <= 0.15, (
                    f"High binary error at extreme quality={r['quality']}"
                )
                assert r["tv_error_rate"] <= 0.15, (
                    f"High three-valued error at extreme quality={r['quality']}"
                )

        # 4. Store results for logging (read by experiment logging)
        self._results = results

    @pytest.mark.asyncio
    async def test_inconclusive_band_width_tradeoff(self):
        """Wider inconclusive band → fewer errors but more INCONCLUSIVE."""
        random.seed(42)
        quality = 0.65  # Near boundary

        narrow_errors = 0
        narrow_inconclusive = 0
        wide_errors = 0
        wide_inconclusive = 0
        n_sims = 50

        for _ in range(n_sims):
            judge = _make_noisy_judge(quality, noise_std=NOISE_STD)
            run = _make_run()

            # Narrow band (0.05)
            v_narrow = await compute_verdict(
                judge, run, num_trials=NUM_TRIALS, threshold=THRESHOLD,
                min_pass_rate=0.5, inconclusive_band=0.05,
            )
            if v_narrow.verdict == Verdict.INCONCLUSIVE:
                narrow_inconclusive += 1
            else:
                out = classify_outcome(v_narrow.verdict.value.upper(), quality)
                if out in ("FP", "FN"):
                    narrow_errors += 1

            # Wide band (0.25)
            judge2 = _make_noisy_judge(quality, noise_std=NOISE_STD)
            v_wide = await compute_verdict(
                judge2, run, num_trials=NUM_TRIALS, threshold=THRESHOLD,
                min_pass_rate=0.5, inconclusive_band=0.25,
            )
            if v_wide.verdict == Verdict.INCONCLUSIVE:
                wide_inconclusive += 1
            else:
                out = classify_outcome(v_wide.verdict.value.upper(), quality)
                if out in ("FP", "FN"):
                    wide_errors += 1

        # Wider band should have more INCONCLUSIVE and fewer hard errors
        assert wide_inconclusive >= narrow_inconclusive, (
            f"Wide band ({wide_inconclusive}) should have >= INCONCLUSIVE "
            f"than narrow ({narrow_inconclusive})"
        )
        assert wide_errors <= narrow_errors + 2, (
            f"Wide band ({wide_errors}) should have <= errors "
            f"than narrow ({narrow_errors})"
        )

    @pytest.mark.asyncio
    async def test_trial_count_and_granularity_interaction(self):
        """With fixed band, more trials gives finer pass_rate granularity,
        which can INCREASE INCONCLUSIVE rate because more discrete values
        fall within the band.

        K=3: pass_rate ∈ {0, 0.33, 0.67, 1.0} → none in [0.35, 0.65]
        K=9: pass_rate ∈ {0, 0.11, ..., 0.44, 0.56, ...} → 0.44, 0.56 in band

        This is a key finding: users must consider K and band together.
        """
        random.seed(42)
        quality = 0.65
        n_sims = 50

        inconclusive_k3 = 0
        inconclusive_k9 = 0
        errors_k3 = 0
        errors_k9 = 0

        for _ in range(n_sims):
            judge3 = _make_noisy_judge(quality, noise_std=NOISE_STD)
            v3 = await compute_verdict(
                judge3, _make_run(), num_trials=3, threshold=THRESHOLD,
                min_pass_rate=0.5, inconclusive_band=0.15,
            )
            if v3.verdict == Verdict.INCONCLUSIVE:
                inconclusive_k3 += 1
            else:
                out = classify_outcome(v3.verdict.value.upper(), quality)
                if out in ("FP", "FN"):
                    errors_k3 += 1

            judge9 = _make_noisy_judge(quality, noise_std=NOISE_STD)
            v9 = await compute_verdict(
                judge9, _make_run(), num_trials=9, threshold=THRESHOLD,
                min_pass_rate=0.5, inconclusive_band=0.15,
            )
            if v9.verdict == Verdict.INCONCLUSIVE:
                inconclusive_k9 += 1
            else:
                out = classify_outcome(v9.verdict.value.upper(), quality)
                if out in ("FP", "FN"):
                    errors_k9 += 1

        # K=3 with band [0.35, 0.65]: no pass_rate falls in band → 0 INCONCLUSIVE
        # K=9 with same band: 4/9=0.44, 5/9=0.56 can fall in band
        assert inconclusive_k3 == 0, (
            f"K=3 should have 0 INCONCLUSIVE (discrete pass_rate misses band), "
            f"got {inconclusive_k3}"
        )
        assert inconclusive_k9 > 0, (
            "K=9 should have some INCONCLUSIVE (finer granularity hits band)"
        )
        # K=9 routes borderline cases to INCONCLUSIVE → fewer hard errors
        assert errors_k9 <= errors_k3, (
            f"K=9 ({errors_k9} errors) should have <= errors than "
            f"K=3 ({errors_k3} errors) due to INCONCLUSIVE routing"
        )
