# Eval & Assertions

Metrics, structured assertions, datasets, and cost tracking.

## Assertions

### assert_tool_called

::: checkagent.eval.assertions.assert_tool_called

### assert_output_schema

::: checkagent.eval.assertions.assert_output_schema

### assert_output_matches

::: checkagent.eval.assertions.assert_output_matches

### assert_json_schema

::: checkagent.eval.assertions.assert_json_schema

### StructuredAssertionError

::: checkagent.eval.assertions.StructuredAssertionError

## Metrics

::: checkagent.eval.metrics
    options:
      show_root_heading: false
      members:
        - task_completion
        - tool_correctness
        - step_efficiency
        - trajectory_match
        - aggregate_scores
        - compute_step_stats
        - detect_regressions

## Datasets

### GoldenDataset

::: checkagent.datasets.GoldenDataset

### EvalCase

::: checkagent.datasets.schema.EvalCase

### load_dataset

::: checkagent.datasets.load_dataset

### load_cases

::: checkagent.datasets.load_cases

## Cost Tracking

### CostTracker

::: checkagent.core.cost.CostTracker

### CostReport

::: checkagent.core.cost.CostReport

### CostBreakdown

::: checkagent.core.cost.CostBreakdown

### calculate_run_cost

::: checkagent.core.cost.calculate_run_cost

### BudgetExceededError

::: checkagent.core.cost.BudgetExceededError
