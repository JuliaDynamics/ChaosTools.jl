import LinearAlgebra
import ProgressMeter
export clv

"""
    clv(ds::DynamicalSystem, N, k = dimension(ds); kwargs...) → (V, λ, x, t)

Compute the Covariant Lyapunov Vectors (CLVs) of `ds` using the Ginelli algorithm [^Ginelli2007].
This is a convenience wrapper that initializes a [`TangentDynamicalSystem`](@ref) with
automatic Jacobian computation and then calls `clv(tands, N; kwargs...)`.

To use a hand-coded Jacobian, create a [`TangentDynamicalSystem`](@ref) manually and call
the lower-level method directly.

See [`clv(::TangentDynamicalSystem, ...)`](@ref) for the full documentation, keyword arguments,
and algorithm description.

See also [`lyapunovspectrum`](@ref), [`lyapunov`](@ref).

[^Ginelli2007]: F. Ginelli *et al.*, Phys. Rev. Lett. **99**, 130601 (2007)
"""
function clv(ds::CoreDynamicalSystem, N, k=dimension(ds); kwargs...)
    tands = TangentDynamicalSystem(ds; k)
    return clv(tands, N; kwargs...)
end

"""
    clv(tands::TangentDynamicalSystem, N::Int; kwargs...) → (V, λ, x, t)

Compute the Covariant Lyapunov Vectors (CLVs) using the Ginelli algorithm [^Ginelli2007].
CLVs are intrinsic, norm-independent vectors that characterize the local stability directions
of a dynamical system, in contrast to Gram-Schmidt vectors which depend on the chosen norm.

Use this method for looping over different initial conditions or parameters by
calling [`reinit!`](@ref) on `tands`, or when you have a hand-coded Jacobian.

## Returns

A named tuple `(V, λ, x, t)` where:
- `V::Vector{Matrix}`: CLVs at each stored time step. `V[i]` is a `D×k` matrix whose columns
  are the `k` CLVs at time `t[i]`, ordered from most expanding to most contracting.
- `λ::Vector`: Lyapunov exponents (growth rates along CLVs), ordered from largest to smallest.
- `x::Vector`: State vectors at each stored time step.
- `t::Vector`: Time stamps corresponding to each stored CLV snapshot.

## Keyword arguments

* `u0 = current_state(tands)`: Initial state.
* `Ttr = 0`: Forward transient time to evolve before storing CLVs.
  Both the system and deviation vectors are evolved during this time.
* `Ttr_bkw = N`: Backward transient steps. More steps improve convergence of
  the backward pass but increase computation time.
* `Δt = 1`: Time between successive orthonormalization steps.
  For continuous systems this is approximate.
* `show_progress = false`: Display a progress bar.

## Algorithm

The Ginelli algorithm computes CLVs via a two-pass procedure:

1. **Forward pass**: Evolve the tangent dynamics, applying QR decomposition at each step
   to obtain the Gram-Schmidt (GS) orthonormal basis Q and upper triangular R matrices.
   To ensure uniqueness, sign corrections are applied so that R has positive diagonal
   entries: if `R[j,j] < 0`, the j-th *row* of R is negated, and correspondingly the
   j-th *column* of Q is negated. Store Q and R for `N` steps (plus `Ttr_bkw` additional
   steps for backward convergence).

2. **Backward pass**: Initialize a coefficient matrix C = I. Iterate backward through
   the stored R matrices, computing `C ← R⁻¹ C` and normalizing columns at each step.
   The physical CLVs are then `V = Q C`.

The CLVs have the key property of being *covariant*: if V(t) is a CLV at time t,
then after evolution by the tangent dynamics, the resulting vector is parallel to V(t+Δt).

[^Ginelli2007]: F. Ginelli *et al.*, Phys. Rev. Lett. **99**, 130601 (2007)
"""
function clv(tands::TangentDynamicalSystem, N::Int;
    Δt::Real=1, Ttr::Real=0, Ttr_bkw::Int=N,
    show_progress=false, u0=current_state(tands),
)
    reinit!(tands, u0)
    k = size(current_deviations(tands), 2)

    # Set up progress bar
    total_steps = (Ttr > 0 ? ceil(Int, Ttr / Δt) : 0) + N + Ttr_bkw + N
    progress = ProgressMeter.Progress(total_steps;
        desc="CLV computation: ", dt=1.0, enabled=show_progress
    )
    step_count = 0

    # --- Phase 1: Forward transient (converge GS directions, discard R) ---
    D = size(current_deviations(tands), 1)
    T = eltype(current_deviations(tands))
    Q_transient = Matrix{T}(undef, D, k)
    R_transient = Matrix{T}(undef, k, k)
    if Ttr > 0
        t0 = current_time(tands)
        while current_time(tands) < t0 + Ttr
            step!(tands, Δt)
            _thin_qr_positive_diagonal!(Q_transient, R_transient, current_deviations(tands), k)
            set_deviations!(tands, Q_transient)
            step_count += 1
            ProgressMeter.update!(progress, step_count)
        end
    end

    # --- Phase 2: Forward pass (store Q and R for N + Ttr_bkw steps) ---
    total_store = N + Ttr_bkw
    T = eltype(current_deviations(tands))
    StateType = typeof(current_state(tands))
    D = size(current_deviations(tands), 1)
    Q_history = Vector{Matrix{T}}(undef, N)           # Only store Q for kept window
    R_history = Vector{Matrix{T}}(undef, total_store) # Store R for all
    x_history = Vector{StateType}(undef, N)           # Store states for kept window
    t_history = Vector{Float64}(undef, N)

    # Pre-allocate buffers for QR decomposition
    Q_buffer = Matrix{T}(undef, D, k)
    R_buffer = Matrix{T}(undef, k, k)

    t_start = current_time(tands)
    for i in 1:total_store
        step!(tands, Δt)
        # Compute thin QR with positive diagonal (sign-corrected Q and R)
        _thin_qr_positive_diagonal!(Q_buffer, R_buffer, current_deviations(tands), k)

        # Set the sign-corrected Q as the new deviations
        set_deviations!(tands, Q_buffer)

        # Store R for backward pass (must copy since we reuse buffer)
        R_history[i] = copy(R_buffer)

        # Store Q, state, and time for the kept window (first N steps)
        if i <= N
            Q_history[i] = copy(Q_buffer)
            x_history[i] = current_state(tands)
            t_history[i] = current_time(tands)
        end

        step_count += 1
        ProgressMeter.update!(progress, step_count)
    end

    # Compute Lyapunov exponents from R diagonal elements
    λ = zeros(T, k)
    for i in 1:total_store
        for j in 1:k
            @inbounds λ[j] += log(R_history[i][j, j])  # Already positive from extraction
        end
    end
    t_total = current_time(tands) - t_start
    λ ./= t_total

    # --- Phase 3: Backward pass (compute CLVs as V = Q * C) ---
    C = Matrix{T}(LinearAlgebra.I, k, k)  # Coefficient matrix
    V_history = Vector{Matrix{T}}(undef, N)

    # First, traverse the backward transient (discard, just evolve C)
    for i in total_store:-1:(N+1)
        C = LinearAlgebra.UpperTriangular(R_history[i]) \ C
        _normalize_columns!(C)
        step_count += 1
        ProgressMeter.update!(progress, step_count)
    end

    # Now traverse the kept window, computing physical CLVs
    for i in N:-1:1
        V_history[i] = Q_history[i] * C

        # Prepare C for the next (earlier) time step
        C = LinearAlgebra.UpperTriangular(R_history[i]) \ C
        _normalize_columns!(C)
        step_count += 1
        ProgressMeter.update!(progress, step_count)
    end

    return (V=V_history, λ=λ, x=x_history, t=t_history)
end

# --- Helper functions ---

"""
Compute thin QR decomposition with positive diagonal on R, storing results in pre-allocated buffers.
This ensures uniqueness of the QR decomposition for CLV computation.
Q_out is m×k and R_out is k×k.
Sign flips are applied to both Q columns and R rows to ensure R[j,j] > 0.
"""
function _thin_qr_positive_diagonal!(Q_out::AbstractMatrix, R_out::AbstractMatrix, Z::AbstractMatrix, k::Int)
    F = LinearAlgebra.qr(Z)
    m = size(Z, 1)
    # Copy into pre-allocated buffers
    Q_full = F.Q
    R_full = F.R
    @inbounds for j in 1:k
        for i in 1:m
            Q_out[i, j] = Q_full[i, j]
        end
        for i in 1:k
            R_out[i, j] = R_full[i, j]
        end
    end
    # Apply sign corrections
    @inbounds for j in 1:k
        if real(R_out[j, j]) < 0
            @views R_out[j, :] .*= -1
            @views Q_out[:, j] .*= -1
        end
    end
    return nothing
end

"""
Normalize each column of matrix C to unit length.
"""
function _normalize_columns!(C::AbstractMatrix)
    @inbounds for j in axes(C, 2)
        col = @view C[:, j]
        s = LinearAlgebra.norm(col)
        col ./= s
    end
    return nothing
end
