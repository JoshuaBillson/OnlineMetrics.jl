module BatchedMetrics

using Statistics, Match
using OrderedCollections: OrderedDict
using Pipe: @pipe

include("utils.jl")

include("interface.jl")
export AbstractMetric, Metric, name, init, update, step!, compute, reset!

include("collection.jl")
export MetricCollection, scores

include("classification.jl")
export ClassificationMetric, Accuracy, MIoU, Precision, Recall

include("misc.jl")
export AverageMeasure

end