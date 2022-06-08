using LightGraphs
using Random
#using DifferentialEquations
using OrdinaryDiffEq:Vern9
using DynamicalSystems
using JLD2
using Statistics

struct second_order_kuramoto_parameters
    systemsize::Int
    damping::Float64
    coupling::Float64
    incidence
    drive::Array{Float64}
end

"""
    second_order_kuramoto(du, u, p::second_order_kuramoto_parameters, t)

Second order Kuramoto system on the adjacency matrix ``A_{ij} = E'_{ie} E_{ej}``.

``\\dot{\\theta}_i = w_i``
``\\dot{\\omega} = \\Omega_i - \\alpha\\omega + \\lambda\\sum_{j=1}^N A_{ij} sin(\\theta_j - \\theta_i)``

"""


function second_order_kuramoto(du, u, p::second_order_kuramoto_parameters, t)
    du[1:p.systemsize] .= u[1 + p.systemsize:2*p.systemsize]
    du[p.systemsize+1:end] .= p.drive .- p.damping .* u[1 + p.systemsize:2*p.systemsize] .- p.coupling .* (p.incidence * sin.(p.incidence' * u[1:p.systemsize]))
    nothing
end


seed = 5386748129040267798
Random.seed!(seed)
# Set up the parameters for the network
N = 30 # in this case this is the number of oscillators, the system dimension is twice this value
g = random_regular_graph(N, 3)
E = incidence_matrix(g, oriented=true)
drive = [isodd(i) ? +1. : -1. for i = 1:N]
par = second_order_kuramoto_parameters(N, 0.1, 2., E, drive)
T = 5000.


#knp = ODEProblem(second_order_kuramoto, zeros(2*N), (0.,T), par)

ds = ContinuousDynamicalSystem(second_order_kuramoto, zeros(2*N), par, (J,z0, p, n) -> nothing)

#u = trajectory(ds, 30)

diffeq = (reltol = 1e-9,  alg = Vern9())

xg = range(-11, 11; length = 40)
yg = range(-1, 1; length = 100)
#grid = (ntuple(x -> xg, N)..., ntuple(x -> yg, N)...)
psys = projected_integrator(ds, N+1:2*N, pi*(rand(N) .- 0.5))
pgrid = ntuple(x -> xg, N)
mapper = AttractorsViaRecurrences(psys, pgrid; Δt = .2, diffeq, sparse = true, mx_chk_fnd_att = 20000,
mx_chk_loc_att = 20000,
mx_chk_hit_bas = 100)
#u = trajectory(ds, 100, rand(2*N))

for k = 1:100
    @show mapper(rand(N))
end

#collect attractors
m_att = []
v_att = []
for att in mapper.bsn_nfo.attractors
    u = Matrix(att[2])
    push!(m_att, mean(u; dims = 1))
    push!(v_att, mean(u; dims = 1))
end

m = vcat(m_att...)
v = vcat(v_att...)



function haussdorff_dist(v01,v02)
    kdtree1 = KDTree(v01)
    idxs12, dists12 = knn(kdtree1, v02, 1, true)
    kdtree2 = KDTree(v02)
    idxs21, dists21 = knn(kdtree2, v01, 1, true)
    max12 = maximum(dists12);
    max21 = maximum(dists21);
    hd =max(max12[1],max21[1])
    return hd
end

m_dist = zeros(length(m_att),length(m_att))
for (i,vi) in enumerate(m_att)
    for (j,vj) in enumerate(m_att)
        #m_dist[i,j] = haussdorff_dist(vi,vj)
        m_dist[i,j] = norm(vi -vj)
    end
end


