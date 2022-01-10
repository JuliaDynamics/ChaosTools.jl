using Revise
using DynamicalSystems
using BenchmarkTools
using ProgressMeter

function f2(bsn_nfo, integ, grid)
	nthreads = Threads.nthreads()
	bsn = zeros(Int16, map(length, grid))
	bsn_nfo_v = [deepcopy(bsn_nfo) for i in 1:nthreads]
	integ_v = [deepcopy(integ) for i in 1:nthreads]
	I = CartesianIndices(bsn)
	Threads.@sync for ind in I
		u0 = ChaosTools.generate_ic_on_grid(grid, ind)
		Threads.@spawn bsn[ind] = get_label_ic!(bsn_nfo_v[Threads.threadid()], integ_v[Threads.threadid()], u0)
	end
	return (bsn .- 1) .÷ 2
end


function f3(bsn_nfo, integ, grid)
	nthreads = Threads.nthreads()
	bsn = zeros(Int16, map(length, grid))
	bsn_nfo_v = [deepcopy(bsn_nfo) for i in 1:nthreads]
	integ_v = [deepcopy(integ) for i in 1:nthreads]
	I = CartesianIndices(bsn)
	Threads.@threads for ind in I
		u0 = ChaosTools.generate_ic_on_grid(grid, ind)
		bsn[ind] = get_label_ic!(bsn_nfo_v[Threads.threadid()], integ_v[Threads.threadid()], u0)
	end
	return (bsn .- 1) .÷ 2
end

function f1(bsn_nfo, integ, grid)
	nthreads = Threads.nthreads()
	bsn = zeros(Int16, map(length, grid))
	bsn_nfo_v = [deepcopy(bsn_nfo) for i in 1:nthreads]
	integ_v = [deepcopy(integ) for i in 1:nthreads]
	I = CartesianIndices(bsn)
	nbatches = length(bsn) ÷ nthreads
	 for k in 1:nbatches
		Threads.@threads  for n in 1:nthreads
			ind = I[(k-1)*nthreads + n]
			u0 = ChaosTools.generate_ic_on_grid(grid, ind)
			bsn[ind] = get_label_ic!(bsn_nfo_v[n], integ_v[n], u0)
		end
	end
	return (bsn .- 1) .÷ 2
end


res = 2000
ds = Systems.henon(zeros(2); a = 1.4, b = 0.3)
xg = yg = range(-2.,2.,length = res)
grid = (xg,yg)
@time basins, att = basins_of_attraction(grid, ds)

bsn_nfo, integ = ic_labelling(ds; attractors = att)

nthreads = Threads.nthreads()
bsn = zeros(Int16, map(length, grid))
bsn_nfo_v = [deepcopy(bsn_nfo) for i in 1:nthreads]
integ_v = [deepcopy(integ) for i in 1:nthreads]
I = CartesianIndices(bsn)
progress = ProgressMeter.Progress(
	length(I); desc = "Basins of attraction: ", dt = 1.0
)
@time for (j,ind) in enumerate(I)
	u0 = ChaosTools.generate_ic_on_grid(grid, ind)
	#ProgressMeter.update!(progress, j)
	bsn[ind] = get_label_ic!(bsn_nfo, integ, u0)
end
bsn = (bsn .- 1) .÷ 2
@show sum(bsn .!= basins)



progress = ProgressMeter.Progress(
	length(I); desc = "Basins of attraction: ", dt = 1.0
)
bsn_nfo, integ = ic_labelling(ds; grid = grid)
bsn_nfo_v = [deepcopy(bsn_nfo) for i in 1:nthreads]
integ_v = [deepcopy(integ) for i in 1:nthreads]
I = CartesianIndices(bsn)
@time for (j,ind) in enumerate(I)
	u0 = ChaosTools.generate_ic_on_grid(grid, ind)
	ProgressMeter.update!(progress, j)
	bsn[ind] = get_label_ic!(bsn_nfo_v[1], integ_v[1], u0)
end



bsn_nfo, integ = ic_labelling(ds; attractors = att)

@time bsn2 = f2(bsn_nfo, integ, grid)
@time bsn1 = f1(bsn_nfo, integ, grid)
@time bsn3 = f3(bsn_nfo, integ, grid)

@show sum(bsn3 .!= basins)
#@time basins, att2 = basins_of_attraction(grid, ds; attractors = att, show_progress = false)
