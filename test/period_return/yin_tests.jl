using ChaosTools, Test, DelimitedFiles

@testset "exponential-chirp" begin 
	file = "exponential-chirp-440-880-5.csv" #generated initially with y = librosa.chirp(440, 880, duration=5.0) 
	println("Downloading 2MB file into $(file) in current dir.")
	download("https://raw.githubusercontent.com/JuliaDynamics/JuliaDynamics/master/timeseries/$(file)", file)
	y = readdlm(file)[:,1]
	sr = Int64(length(y)/5)
	F0s, time_scale = yin(y, sr, f0_min=200, f0_max=1000, f_step=512, w_len=1024)
	t = range(0,5,length=length(F0s))
	chirp_freqs = 440 .*exp.(log(2)/5 .* t)
	@test all(abs.(F0s .- chirp_freqs) .< 5)
	println("Removing the file.")
	rm(file)
end
