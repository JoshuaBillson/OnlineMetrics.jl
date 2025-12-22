module OnlineMetrics

@doc read(joinpath(dirname(@__DIR__), "README.md"), String) OnlineMetrics

using Statistics, Match, AbstractTrees
using Pipe: @pipe
using ArgCheck: @argcheck, @check

include("utils.jl")

include("format.jl")
export AbstractDataFormat, OneHot, format, validate

include("interface.jl")
export AbstractMetric, name, initial_state, merge_state, batch_state, current_value, data_format, step

include("classification.jl")
export ClassificationMetric, Accuracy, mIoU, ConfusionMatrix, BinaryPrecision, Precision, BinaryRecall, Recall

include("misc.jl")
export AverageMeasure

include("metric_collection.jl")
export MetricCollection, Metric, step!, merge, value

end