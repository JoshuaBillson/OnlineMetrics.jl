function _confusion_matrix(y_pred::AbstractMatrix{<:Real}, y_true::AbstractMatrix{<:Real})
    @assert size(y_pred) == size(y_true)
    return y_pred * transpose(y_true) 
end

_tfpn(y_pred::AbstractMatrix{<:Real}, y_true::AbstractMatrix{<:Real}) = _confusion_matrix(y_pred, y_true) |> _tfpn
function _tfpn(confusion_matrix::AbstractMatrix)
    nclasses = size(confusion_matrix, 1)
    TP = zeros(Int, nclasses)
    TN = zeros(Int, nclasses)
    FP = zeros(Int, nclasses)
    FN = zeros(Int, nclasses)
    for c in 1:nclasses
        TP[c] = confusion_matrix[c,c]
        FP[c] = sum(confusion_matrix[c,:]) - TP[c]
        FN[c] = sum(confusion_matrix[:,c]) - TP[c]
        TN[c] = sum(confusion_matrix) - TP[c] - FP[c] - FN[c]
    end
    return TP, TN, FP, FN
end