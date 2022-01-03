using Revise
using DynamicalSystems
using ChaosTools
using StaticArrays

function attractor_id_init(grid, integ, Δt, T, idxs::SVector, complete_state)
    complete_and_reinit! = ChaosTools.CompleteAndReinit(complete_state, idxs, length(get_state(integ)))
    get_projected_state = (integ) -> view(get_state(integ), idxs)
    MDI = DynamicalSystemsBase.MinimalDiscreteIntegrator
    if !isnothing(T)
        iter_f! = (integ) -> step!(integ, T, true)
    elseif (integ isa PoincareMap) || (integ isa MDI) || fixed_solver
        iter_f! = step!
    else # generic case
        iter_f! = (integ) -> step!(integ, Δt) # we don't have to step _exactly_ `Δt` here
    end
    bsn_nfo = ChaosTools.init_bsn_nfo(grid, integ, iter_f!, complete_and_reinit!, get_projected_state; sparse = true)
    return bsn_nfo
end

ds = Systems.henon_iip(zeros(2); a = 1.4, b = 0.3)
xg = yg = range(-2.,2.,length=100)
integ = integrator(ds)
idxs = SVector(1:2...)

bsn_nfo = attractor_id_init((xg,yg), integ, 1, 1, idxs, nothing)

y0 = [-1., 1.]

n = ChaosTools.basin_cell_index(y0, bsn_nfo)
bsn_nfo.basin[n] = ChaosTools.get_color_point!(bsn_nfo, integ, y0)
