function lyapunovs_convergence(integ, N, dt::Real, Ttr::Real = 0.0)

    T = stateeltype(integ)
    t0 = integ.t
    if Ttr > 0
        while integ.t < t0 + Ttr
            step!(integ, dt)
            qrdec = LinearAlgebra.qr(get_deviations(integ))
            set_deviations!(integ, _get_Q(qrdec))
        end
    end
    k = size(get_deviations(integ))[2]
    t0 = integ.t; t = zeros(T, N); t[1] = t0
    λs = [zeros(T, k) for i in 1:N];

    for i in 2:N
        step!(integ, dt)
        qrdec = LinearAlgebra.qr(get_deviations(integ))
        for j in 1:k
            @inbounds λs[i][j] = λs[i-1][j] + log(abs(qrdec.R[j,j]))
        end
        t[i] = integ.t
        set_deviations!(integ, _get_Q(qrdec))
    end
    popfirst!(λs); popfirst!(t)
    for j in eachindex(t)
        λs[j] ./= (t[j] - t0)
    end
    return λs, t
end
