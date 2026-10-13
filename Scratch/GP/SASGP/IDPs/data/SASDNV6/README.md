# Experimental E1A SEC-SAXS candidate

Source: https://www.sasbdb.org/data/SASDNV6/
Paper: https://www.nature.com/articles/s41594-022-00811-w

The raw downloaded curve and full-entry archive are preserved. `curve_native.csv`
contains all numeric q, intensity, and reported error rows without filtering,
normalization, noise addition, or unit conversion. `source.json` records checks
and the raw-file checksum. This dataset has not been fitted.

SASBDB reports Guinier Rg = 3.6 nm and Dmax = 17.5 nm. The native DAT file does
not specify q units, while the website plots inverse nm. A rough low-q Guinier
slope gives Rg about 36 native inverse-q units, consistent with the file using
inverse angstroms. This is an inference; confirm the native units before fitting.

The GP now infers a constant observation-noise SD from q and intensity.
Reported per-point errors are optional reference data and are not required
by this model. Do not add synthetic noise to the experimental intensities. The 10 nm
pair-distance display also does not span the deposited Dmax.

Local alternatives in ../IDPdatabank/Data/Experiments/saxs: SASDQK7 has 101
positive error rows out of 618; SASDQJ7 has 69 out of 618. SASBDB flags negative
experimental errors for both. The two DOI-named local datasets lack errors.
