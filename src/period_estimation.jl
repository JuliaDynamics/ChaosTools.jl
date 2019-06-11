import LombScargle, DSP

export find_period

function find_period(ds::DynamicalSystem, T::Real, alg = :pg_fft) # algs in {:periodogram, :lombscargle, :hilbert, ...}
