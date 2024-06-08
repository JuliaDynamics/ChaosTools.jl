function storefp!(set, fp, abstol)
    previously_detected(set, fp, abstol) ? nothing : push!(set, fp)
end

function previously_detected(set, fp, abstol)
    for elem in set
        if norm(fp - elem) <= abstol
            return true
        end
    end
    return false
end