using LinearAlgebra, StaticArrays
import ProgressMeter
using DynamicalSystemsBase: MinimalDiscreteIntegrator

"""
    lyapunovspectrum(ds::DynamicalSystem, N, k = dimension(ds); kwargs...) -> λs

Calculate the spectrum of Lyapunov exponents [^Lyapunov1992] of `ds` by applying
a QR-decomposition on the parallelepiped defined by the deviation vectors, in total for
`N` evolution steps. Return the spectrum sorted from maximum to minimum.
The third argument `k` is optional, and dictates how many lyapunov exponents
to calculate (defaults to `dimension(ds)`).

See also [`lyapunov`](@ref), [`local_growth_rates`](@ref).

**Note:** This function simply initializes a [`TangentDynamicalSystem`](@ref) and calls
the method below. This means that the automatic Jacobian is used by default.
Initialize manually a [`TangentDynamicalSystem`](@ref) if you have a hand-coded Jacobian.

## Keyword arguments

* `u0 = current_state(ds)`: State to start from.
* `Ttr = 0`: Extra transient time to evolve the system before application of the
  algorithm. Should be `Int` for discrete systems. Both the system and the
  deviation vectors are evolved for this time.
* `Δt = 1`: Time of individual evolutions
  between successive orthonormalization steps. For continuous systems this is approximate.
* `show_progress = false`: Display a progress bar of the process.

## Description

The method we employ is "H2" of [^Geist1990], originally stated in [^Benettin1980],
and explained in educational form in [^DatserisParlitz2022].

The deviation vectors defining a `D`-dimensional parallepiped in tangent space
are evolved using the tangent dynamics of the system (see [`TangentDynamicalSystem`](@ref)).
A QR-decomposition at each step yields the local growth rate for each dimension
of the parallepiped. At each step the parallepiped is re-normalized to be orthonormal.
The growth rates are then averaged over `N` successive steps,
yielding the lyapunov exponent spectrum.

[^Lyapunov1992]: A. M. Lyapunov, *The General Problem of the Stability of Motion*, Taylor & Francis (1992)

[^Geist1990]: K. Geist *et al.*, Progr. Theor. Phys. **83**, pp 875 (1990)

[^Benettin1980]: G. Benettin *et al.*, Meccanica **15**, pp 9-20 & 21-30 (1980)

[^DatserisParlitz2022]:
    Datseris & Parlitz 2022, _Nonlinear Dynamics: A Concise Introduction Interlaced with Code_,
    [Springer Nature, Undergrad. Lect. Notes In Physics](https://doi.org/10.1007/978-3-030-91032-7)
"""
function lyapunovspectrum(ds::DS{IIP, S, D}, N, k = dimension(ds); u0 = current_state(ds), kwargs...)
    tands = TangentDynamicalSystem(ds, k; u0)
    λ = lyapunovspectrum(tands, N, Δt, Ttr, show_progress)
    return λ
end

"""
    lyapunovspectrum(tands::TangentDynamicalSystem, N::Int; Ttr, Δt, show_progress)

The low-level method that is called by `lyapunovspectrum(ds::DynamicalSystem, ...)`.
Use this method for looping over different initial conditions or parameters by
calling [`reinit!`](@ref) to `tands`.

Also use this method if you have a hand-coded Jacobian to pass when creating `tands`.
"""
function lyapunovspectrum(tands::TangentDynamicalSystem, N;
        Δt::Real = 1, Ttr::Real = 0.0, show_progress = false
    )
    progress = ProgressMeter.Progress(N;
        desc = "Lyapunov Spectrum: ", dt = 1.0, enabled = show_progress
    )
    B = copy(current_deviations(tands)) # for use in buffered QR
    if Ttr > 0 # This is useful to start orienting the deviation vectors
        t0 = current_time(tands)
        while current_time(tands) < t0 + Ttr
            step!(tands, Δt)
            Q, R = _buffered_qr(B, current_deviations(tands))
            set_deviations!(tands, Q)
        end
    end

    k = size(current_deviations(tands))[2]
    λ = zeros(eltype(current_deviations(tands)), k)
    t0 = current_time(tands)
    for i in 1:N
        step!(tands, Δt)
        Q, R = _buffered_qr(B, current_deviations(tands))
        for j in 1:k
            @inbounds λ[j] += log(abs(R[j,j]))
        end
        set_deviations!(tands, Q)
        ProgressMeter.update!(progress, i)
    end
    λ ./= (current_time(tands) - t0)
    return λ
end

# For out-of-place systems, this is just standard QR decomposition.
# For in-place systems, this is a more performant buffered version.
function _buffered_qr(B::SMatrix, Y) # Y are the deviations
    Q, R = LinearAlgebra.qr(Y)
    return Q, R
end
function _buffered_qr(B::Matrix, Y) # Y are the deviations
    B .= Y
    Q, R = LinearAlgebra.qr!(B)
    return Q, R
end


lyapunovspectrum(ds::DiscreteDynamicalSystem{IIP, T, 1}, a...; kw...) where {IIP, T} =
error("For discrete 1D systems, only method with state type = number is implemented.")

function lyapunovspectrum(ds::DDS{false, T, 1}, N; Ttr = 0) where {T}
    x = current_state(ds); f = ds.f
    p = ds.p; t0 = ds.t0
    if Ttr > 0
        for i in t0:(Ttr+t0)
            x = f(x, p, i)
        end
    end
    λ = zero(T)
    for i in (t0+Ttr):(t0+Ttr+N)
        x = f(x, p, i)
        λ += log(abs(ds.jacobian(x, p, i)))
    end
    return λ/N
end

#####################################################################################
#                           Lyapunov Spectrum Convergence                           #
#####################################################################################
"""
    lyapunovspectrum_convergence(ds::DynamicalSystem, N [, k::Int | Q0]; kwargs...) -> λs, t

Computes the convergence of the Lyapunov exponents, calculated exactly as in `lyapunovspectrum`.
(See [`lyapunovspectrum`](@ref) for information on the parameters and implementation).
Returns `λs`, a vector of `N` vectors, wherein vector of index `i` contains the Lyapunov
spectrum evaluated up to the time `t[i]`. The final vector is equal to the returned vector
of `lyapunovspectrum`.

## Performance Notes
This function uses a [`tangent_integrator`](@ref).
For loops over initial conditions and/or
parameter values one should use the low level method that accepts
an integrator, and `reinit!` it to new initial conditions.
See the "advanced documentation" for info on the integrator object.
The low level method is
```julia
lyapunovspectrum_convergence(tinteg, N, Δt::Real, Ttr::Real)
```
"""
lyapunovspectrum_convergence(ds::DS, N, k::Int = dimension(ds); kwargs...) =
lyapunovspectrum_convergence(ds, N, orthonormal(dimension(ds), k); kwargs...)

function lyapunovspectrum_convergence(ds::DS{IIP, S, D}, N, Q0::AbstractMatrix;
        Ttr::Real = 0, Δt::Real = 1, u0 = current_state(ds), show_progress = false, diffeq...
    ) where {IIP, S, D}

    @warn "This function is deprecated."
    if typeof(ds) <: DDS
        @assert typeof(Ttr) == Int
        tands = tangent_integrator(ds, Q0; u0)
    else
        tands = tangent_integrator(ds, Q0; u0, diffeq...)
    end
    λ = lyapunovspectrum_convergence(tands, N, Δt, Ttr, show_progress)
    return λ
end

function lyapunovspectrum_convergence(tands, N, Δt::Real, Ttr::Real = 0.0, show_progress=false)
    if show_progress
        progress = ProgressMeter.Progress(N; desc = "Lyapunov Spectrum: ", dt = 1.0)
    end
    B = copy(current_deviations(tands)) # for use in buffer
    if Ttr > 0
        t0 = current_time(tands)
        while current_time(tands) < t0 + Ttr
            step!(tands, Δt)
            Q, R = _buffered_qr(B, current_deviations(tands))
            set_deviations!(tands, Q)
        end
    end

    k = size(current_deviations(tands))[2]
    T = stateeltype(tands)
    t0 = current_time(tands); t = zeros(T, N); t[1] = t0
    λs = [zeros(T, k) for i in 1:N];

    for i in 2:N
        step!(tands, Δt)
        Q, R = _buffered_qr(B, current_deviations(tands))
        for j in 1:k
            @inbounds λs[i][j] = λs[i-1][j] + log(abs(R[j,j]))
        end
        t[i] = current_time(tands)
        set_deviations!(tands, Q)
        show_progress && ProgressMeter.update!(progress, i)
    end
    popfirst!(λs); popfirst!(t)
    for j in eachindex(t)
        λs[j] ./= (t[j] - t0)
    end
    return λs, t
end

lyapunovspectrum_convergence(ds::DiscreteDynamicalSystem{IIP, T, 1}, a...; kw...) where {IIP, T} =
error("For discrete 1D systems, only method with state type = number is implemented.")

function lyapunovspectrum_convergence(ds::DDS{false, T, 1}, N; Ttr = 0) where {T}
    x = current_state(ds); f = ds.f
    p = ds.p; t0 = ds.t0
    if Ttr > 0
        for i in t0:(Ttr+t0)
            x = f(x, p, i)
        end
    end
    t = (t0+Ttr):(t0+Ttr+N)
    λs = zeros(T, length(t));
    for idx = 2:length(t)
        i = t[idx]
        x = f(x, p, i)
        @inbounds λs[idx] = λs[idx-1] + log(abs(ds.jacobian(x, p, i)))
    end
    t = collect(t)
    popfirst!(λs); popfirst!(t)
    λs ./= t .- t0
    return λs, t
end
