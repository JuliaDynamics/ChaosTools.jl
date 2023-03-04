import LinearAlgebra
import ProgressMeter
export lyapunovspectrum

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
function lyapunovspectrum(ds::CoreDynamicalSystem, N, k = dimension(ds); kwargs...)
    tands = TangentDynamicalSystem(ds; k)
    λ = lyapunovspectrum(tands, N; kwargs...)
    return λ
end

"""
    lyapunovspectrum(tands::TangentDynamicalSystem, N::Int; Ttr, Δt, show_progress)

The low-level method that is called by `lyapunovspectrum(ds::DynamicalSystem, ...)`.
Use this method for looping over different initial conditions or parameters by
calling [`reinit!`](@ref) to `tands`.

Also use this method if you have a hand-coded Jacobian to pass when creating `tands`.
"""
function lyapunovspectrum(tands::TangentDynamicalSystem, N::Int;
        Δt::Real = 1, Ttr::Real = 0, show_progress = false,
        u0 = current_state(tands),
    )
    reinit!(tands, u0)
    progress = ProgressMeter.Progress(N;
        desc = "Lyapunov spectrum: ", dt = 1.0, enabled = show_progress
    )
    B = copy(current_deviations(tands)) # for use in buffered QR
    if Ttr > 0 # This is useful to start orienting the deviation vectors
        t0 = current_time(tands)
        while current_time(tands) < t0 + Ttr
            step!(tands, Δt)
            Q, R = _buffered_qr(B, current_deviations(tands))
            set_Q_as_deviations!(tands, Q)
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
        set_Q_as_deviations!(tands, Q)
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

# By definition QR decomposition will return a DxD matrix as Q,
# but the deviation vectors are a Dxk if k < D. We need to efficiently
# utilize only the first k columns of Q, however, how to do this depends
# strongly on the storage type (iip/oop)

# TODO: check if `if` statements make it more performant

function set_Q_as_deviations!(tands::TangentDynamicalSystem{true}, Q)
    devs = current_deviations(tands) # it is a view
    if size(Q) ≠ size(devs)
        copyto!(devs, LinearAlgebra.I)
        LinearAlgebra.lmul!(Q, devs)
        set_deviations!(tands, devs)
    else
        set_deviations!(tands, Q)
    end
end

function set_Q_as_deviations!(tands::TangentDynamicalSystem{false}, Q)
    # here `devs` is a static vector
    devs = current_deviations(tands)
    ks = axes(devs, 2) # it is a `StaticArrays.SOneTo(k)`
    set_deviations!(tands, Q[:, ks])
end
