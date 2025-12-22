"""
Metrics are measures of a model's performance, such as loss, accuracy, or squared error.

Metrics are updated incrementally as new data arrives, making them suitable for online learning scenarios.

Each metric must implement the following interface:
- [`name`](@ref): Returns the human-readable name of the metric.
- [`initial_state`](@ref): Returns the initial state of the metric.
- [`batch_state`](@ref): Computes the metric's state for a single batch of predictions and labels.
- [`merge_state`](@ref): Merges two metric states into a single state.
- [`metric_value`](@ref): Computes the metric's value from its current state.

# Optional Methods
- [`data_format`](@ref): Returns the data format expected by the metric, or `nothing` if no specific format is required. Defaults to `nothing`.
"""
abstract type AbstractMetric end

"""
    name(m::AbstractMetric)

Return the human readable name of the metric.
"""
function name end

"""
    initial_state(m::AbstractMetric) -> state

Return the initial state of the metric.
"""
function initial_state end

"""
    batch_state(m::AbstractMetric, y_pred, y_true) -> state

Compute the metric's state for a single batch of predictions and labels.

The type of `y_pred` and `y_true` should not be specialized. Data formatting 
and validaton should instead handled by the `format` and `validate` methods 
defined by the metric's data format. This ensures that users receive more
informative error messages when providing invalid data.
"""
function batch_state end

"""
    merge_state(m::AbstractMetric, state1, state2) -> merged_state

Merge two metric states into a single state.
"""
function merge_state end

"""
    metric_value(m::AbstractMetric, state)

Compute the metric's value from its current state.
"""
function metric_value end

"""
    data_format(m::AbstractMetric) -> Union{AbstractDataFormat, Nothing}

Return the data format expected by the metric, or `nothing` if no specific format is required.
"""
data_format(::AbstractMetric) = nothing

"""
    step(m::AbstractMetric, y_pred, y_true, oldstate) -> newstate

Update the metric state for the given batch of labels and predictions.

# Parameters
- `m::AbstractMetric`: The metric to be updated.
- `y_pred`: The model predictions for the current batch.
- `y_true`: The true labels for the current batch.
- `oldstate`: The previous state of the metric.

# Returns
- `newstate`: The updated state of the metric.
"""
function step(m::AbstractMetric, y_pred, y_true, oldstate) 
    # Get appropriate data format
    df = data_format(m)

    # Validate inputs
    validate(df, y_pred)
    validate(df, y_true)

    # Format inputs
    y_pred_formatted = format(df, y_pred)
    y_true_formatted = format(df, y_true)

    # Update metric state
    newstate = batch_state(m, y_pred_formatted, y_true_formatted)
    return merge_state(m, oldstate, newstate)
end

"""
    merge_state(m::AbstractMetric, states...) -> merged_state

Merge multiple metric states into a single state.
"""
function merge_state(m::AbstractMetric, states...)
    return reduce((s1, s2) -> merge_state(m, s1, s2), states)
end