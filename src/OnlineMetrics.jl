module OnlineMetrics

using AbstractTrees
using Statistics, Match
using Pipe: @pipe
using ArgCheck: @argcheck

include("utils.jl")

include("interface.jl")
export AbstractMetric, name, step!, value

include("collection.jl")
export MetricCollection

include("classification.jl")
export ClassificationMetric, Accuracy, MIoU, ConfusionMatrix, Precision, Recall

include("misc.jl")
export AverageMeasure

end