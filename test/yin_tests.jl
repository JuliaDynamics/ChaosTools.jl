# # using ChaosTools #use this once the code is in a working branch
# include("../src/periodicity/yin.jl")
# using WAV

# audiofile = "whereIam.wav"
# sig, sr = wavread(audiofile, format="native")
# sig = sig[:,1]
# sig = convert(Array{Int64,1}, sig)


# pitches, harmonic_rates, argmins, times = yin(sig, sr; w_len=1024, w_step=256, f0_min=70, f0_max=200, harmonic_threshold=0.85)

# #Plot results, comparing with python's version. If we use the differenceFunction_original there, the results match perfectly!
# harmonic_threshold=0.85
# duration = length(sig)/float(sr)

# using PyPlot
# ax1 = subplot(4, 1, 1)
# ax1.plot(collect(1:length(sig)).*(duration/length(sig)), sig)
# ax1.set_title("Audio data")
# ax1.set_ylabel("Amplitude")
# ax2 = subplot(4, 1, 2)
# ax2.plot(collect(1:length(pitches)).*(duration/length(pitches)), pitches)
# ax2.set_title("F0")
# ax2.set_ylabel("Frequency (Hz)")
# ax3 = subplot(4, 1, 3, sharex=ax2)
# ax3.plot(collect(1:length(harmonic_rates)).*(duration/length(harmonic_rates)), harmonic_rates)
# ax3.axhline(y=harmonic_threshold, color="r")
# ax3.set_title("Harmonic rate")
# ax3.set_ylabel("Rate")
# ax4 = subplot(4, 1, 4, sharex=ax2)
# ax4.plot(collect(1:length(argmins)).*(duration/length(argmins)), argmins)
# ax4.set_title("Index of minimums of CMND")
# ax4.set_ylabel("Frequency (Hz)")
# ax4.set_xlabel("Time (seconds)")
# savefig("yin-results-jl.png")