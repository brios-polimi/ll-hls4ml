from ll_hls4ml.reporting.accounting import (
    hurdle_calibration_rows,
    hurdle_confusion_rows,
    macro_metric_rows,
    paired_delta_rows,
    split_sha256,
)


def test_split_hash_ignores_non_membership_fields_and_row_order():
    left = {
        "test": [
            {"kernel_family": "b", "tensor_path": "b/archive_1/y.pt", "dataset_index": 2},
            {"kernel_family": "a", "tensor_path": "a/archive_1/x.pt", "dataset_index": 1},
        ]
    }
    right = {
        "test": [
            {"kernel_family": "a", "tensor_path": "a/archive_1/x.pt"},
            {"kernel_family": "b", "tensor_path": "b/archive_1/y.pt"},
        ]
    }
    assert split_sha256(left) == split_sha256(right)


def test_hurdle_confusion_uses_positive_presence():
    rows = [
        {
            "split": "test",
            "kernel_family": "dense",
            "target_dsp": truth,
            "prediction_dsp": prediction,
            "target_bram": truth,
            "prediction_bram": prediction,
        }
        for truth, prediction in ((0, 0), (0, 2), (3, 0), (3, 2))
    ]
    overall = [
        row for row in hurdle_confusion_rows(rows) if row["cohort"] == "all"
    ]
    assert len(overall) == 2
    assert all(
        (row["tn"], row["fp"], row["fn"], row["tp"]) == (1, 1, 1, 1)
        for row in overall
    )


def test_hurdle_reporting_prefers_presence_probabilities_and_calibrates():
    rows = [
        {
            "split": "test",
            "kernel_family": "dense",
            "target_dsp": truth,
            "prediction_dsp": 1,
            "presence_probability_dsp": probability,
            "target_bram": truth,
            "prediction_bram": 1,
            "presence_probability_bram": probability,
        }
        for truth, probability in ((0, 0.1), (0, 0.2), (3, 0.8), (3, 0.9))
    ]
    overall = [
        row for row in hurdle_confusion_rows(rows) if row["cohort"] == "all"
    ]
    assert all(
        (row["tn"], row["fp"], row["fn"], row["tp"]) == (2, 0, 0, 2)
        for row in overall
    )
    calibration = hurdle_calibration_rows(rows, bins=2)
    assert calibration
    assert all(
        abs(row["expected_calibration_error"] - 0.15) < 1e-12
        for row in calibration
    )


def test_macro_metrics_separate_resource_and_timing():
    rows = [
        {
            "split": "test",
            "kernel_family": "all",
            "target": target,
            "n_samples": 2,
            "r2": index,
            "smape": 10 + index,
            "rmse": 20 + index,
        }
        for index, target in enumerate(
            ("lut", "ff", "dsp", "bram", "cycles_max", "interval_max")
        )
    ]
    macros = {row["scope"]: row for row in macro_metric_rows(rows)}
    assert macros["resource"]["smape"] == 11.5
    assert macros["timing"]["smape"] == 14.5


def test_paired_delta_is_negative_when_candidate_improves():
    def prediction(value):
        row = {
            "split": "test",
            "kernel_family": "dense",
            "tensor_path": "dense/archive_1/sample.pt",
        }
        for target in ("lut", "ff", "dsp", "bram", "cycles_max", "interval_max"):
            row[f"target_{target}"] = 10
            row[f"prediction_{target}"] = value
        return row

    per_sample, summary = paired_delta_rows([prediction(9)], [prediction(5)])
    assert per_sample[0]["delta_smape_lut"] < 0
    assert all(row["fraction_improved"] == 1 for row in summary)
