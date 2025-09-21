module OnlineMetrics

using AbstractTrees
using Statistics, Match, OneHotArrays
using Pipe: @pipe
using ArgCheck: @argcheck

include("utils.jl")
export one_hot, weighted_average

include("interface.jl")
export AbstractMetric, step!, value, params

include("collection.jl")
export MetricCollection

include("classification.jl")
export ClassificationMetric, Accuracy, MIoU, ConfusionMatrix, BinaryPrecision, Precision, BinaryRecall, Recall

include("misc.jl")
export AverageMeasure, MAE, MSE

end