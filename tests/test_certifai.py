"""
Tests for the CERTIFAI state-leak bug and related behavior.
"""

import numpy as np
import pytest

from certifai.certifai import CERTIFAI
from tests.conftest import N_FEATURES

# ---------------------------------------------------------------------------
# Test 1: Bug reproduction — fit() with trained_with_columns=True on a
# model trained with named DataFrame columns must not raise ValueError.
# ---------------------------------------------------------------------------


class TestBugReproduction:
    def test_fit_trained_with_columns_no_value_error(
        self, synthetic_data, trained_sklearn_model
    ):
        X_train, _ = synthetic_data
        explainer = CERTIFAI(pandas_dataset=X_train)
        explainer.fit(
            model=trained_sklearn_model,
            trained_with_columns=True,
            target_name="target",
            model_type="sklearn",
            classification=True,
            generations=3,
            final_k=1,
        )
        assert explainer.results is not None
        for sample, cfs, pred_targets in explainer.results:
            for cf in cfs:
                assert len(cf) == N_FEATURES, (
                    f"Counterfactual has {len(cf)} features, expected {N_FEATURES}"
                )


# ---------------------------------------------------------------------------
# Test 2: Multiple fit() calls on the same instance — no state leaks.
# ---------------------------------------------------------------------------


class TestMultipleFitCalls:
    def test_repeated_fit_no_state_leak(self, synthetic_data, trained_sklearn_model):
        X_train, _ = synthetic_data
        explainer = CERTIFAI(pandas_dataset=X_train)
        n_calls = 4
        for call_idx in range(n_calls):
            explainer.fit(
                model=trained_sklearn_model,
                trained_with_columns=True,
                target_name="target",
                model_type="sklearn",
                classification=True,
                generations=3,
                final_k=1,
            )
            if explainer.results:
                for sample, cfs, pred_targets in explainer.results:
                    for cf in cfs:
                        assert len(cf) == N_FEATURES, (
                            f"Call {call_idx + 1}: counterfactual has {len(cf)} "
                            f"features, expected {N_FEATURES}"
                        )


# ---------------------------------------------------------------------------
# Test 3: trained_with_columns=False (numpy model)
# ---------------------------------------------------------------------------


class TestNumpyModel:
    def test_fit_numpy_model_no_columns(self, synthetic_data, numpy_trained_model):
        X_train, _ = synthetic_data
        explainer = CERTIFAI(pandas_dataset=X_train)
        explainer.fit(
            model=numpy_trained_model,
            trained_with_columns=False,
            target_name="target",
            model_type="sklearn",
            classification=True,
            generations=3,
            final_k=1,
        )
        assert explainer.results is not None
        for sample, cfs, pred_targets in explainer.results:
            for cf in cfs:
                assert len(cf) == N_FEATURES


# ---------------------------------------------------------------------------
# Test 4: trained_with_columns=True with explicit model_input
# ---------------------------------------------------------------------------


class TestExplicitModelInput:
    def test_fit_with_explicit_model_input(self, synthetic_data, trained_sklearn_model):
        X_train, _ = synthetic_data
        explainer = CERTIFAI(pandas_dataset=X_train)
        explainer.fit(
            model=trained_sklearn_model,
            model_input=X_train,
            trained_with_columns=True,
            target_name="target",
            model_type="sklearn",
            classification=True,
            generations=3,
            final_k=1,
        )
        assert explainer.results is not None
        for sample, cfs, pred_targets in explainer.results:
            for cf in cfs:
                assert len(cf) == N_FEATURES


# ---------------------------------------------------------------------------
# Test 5: Crossover output integrity
# ---------------------------------------------------------------------------


class TestCrossoverIntegrity:
    def test_crossover_shape_and_values(self, synthetic_data):
        X_train, _ = synthetic_data
        explainer = CERTIFAI(pandas_dataset=X_train)
        data = X_train.values.tolist()

        result_df = explainer.crossover(data, return_df=True)

        assert result_df.shape == (len(data), N_FEATURES), (
            f"Crossover output shape {result_df.shape} != expected "
            f"({len(data)}, {N_FEATURES})"
        )

        copy_shape = result_df.copy().to_numpy().shape
        direct_shape = result_df.to_numpy().shape
        assert copy_shape == direct_shape, (
            "Block corruption detected in crossover output"
        )

        assert not np.any(np.isnan(result_df.to_numpy().astype(float))), (
            "Crossover introduced NaN values"
        )


# ---------------------------------------------------------------------------
# Test 6: Mutate output integrity
# ---------------------------------------------------------------------------


class TestMutateIntegrity:
    def test_mutate_preserves_feature_count(self, synthetic_data):
        X_train, _ = synthetic_data
        explainer = CERTIFAI(pandas_dataset=X_train)
        data = X_train.values.tolist()

        mutated = explainer.mutate(data)

        assert len(mutated) == len(data), (
            f"Mutate changed row count: {len(mutated)} != {len(data)}"
        )
        for i, row in enumerate(mutated):
            assert len(row) == N_FEATURES, (
                f"Mutated row {i} has {len(row)} features, expected {N_FEATURES}"
            )


# ---------------------------------------------------------------------------
# Test 7: Mixed dtype preservation (categorical + continuous)
# ---------------------------------------------------------------------------


class TestMixedDtypePreservation:
    @pytest.fixture
    def mixed_data(self):
        np.random.seed(42)
        n = 50
        return [
            [
                np.random.choice(["A", "B", "C"]),
                np.random.choice(["X", "Y"]),
                float(np.random.randn()),
                float(np.random.randn()),
                float(np.random.randn()),
            ]
            for _ in range(n)
        ]

    def test_mutate_preserves_dtypes(self, mixed_data):
        explainer = CERTIFAI()
        mutated = explainer.mutate(mixed_data)

        for i, row in enumerate(mutated):
            assert isinstance(row[0], str), (
                f"Row {i} col 0: expected str, got {type(row[0])}"
            )
            assert isinstance(row[1], str), (
                f"Row {i} col 1: expected str, got {type(row[1])}"
            )
            for col_idx in (2, 3, 4):
                assert isinstance(row[col_idx], (int, float, np.floating)), (
                    f"Row {i} col {col_idx}: expected numeric, got {type(row[col_idx])}"
                )

    def test_crossover_preserves_dtypes(self, mixed_data):
        explainer = CERTIFAI()
        crossed = explainer.crossover(mixed_data, return_df=False)

        for i, row in enumerate(crossed):
            assert isinstance(row[0], str), (
                f"Row {i} col 0: expected str, got {type(row[0])}"
            )
            assert isinstance(row[1], str), (
                f"Row {i} col 1: expected str, got {type(row[1])}"
            )
            for col_idx in (2, 3, 4):
                assert isinstance(row[col_idx], (int, float, np.floating)), (
                    f"Row {i} col {col_idx}: expected numeric, got {type(row[col_idx])}"
                )
