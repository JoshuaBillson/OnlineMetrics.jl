```@meta
CurrentModule = OnlineMetrics
```

# OnlineMetrics

```@docs
OnlineMetrics
```

## Metric Tracking
```@docs
Metric
MetricCollection
step!
merge
value
```

## Metric Interface
```@docs
AbstractMetric
ClassificationMetric
name
initial_state
batch_state
merge_state
metric_value
data_format
step
```

## Classification Metrics

```@docs
Accuracy
IoU
mIoU
Precision
BinaryPrecision
Recall
BinaryRecall
F1Score
ConfusionMatrix
```

## Other Metrics
```@docs
AverageMeasure
```

## Data Formats
```@docs
AbstractDataFormat
OneHot
format
validate
```

## Index

```@index
```