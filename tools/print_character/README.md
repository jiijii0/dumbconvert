# Lab print: how the numbers were made

`simpleneg/engine.py` `PRINT_CHARACTER` holds, per film stock and colour-mixer band, the hue turn
and chroma factor between DumbConvert's straight conversion and the same negative printed on photo
paper. They come from a spectral simulation built only on manufacturer data:

1. **Scene:** the 24 ColorChecker patches (BabelColor average spectra) under D55, the films' reference daylight.
2. **Film:** each layer's spectral sensitivity, its characteristic curve and its dye spectra, from Kodak and
   Fujifilm data sheets as digitised by [spektrafilm](https://github.com/andreavolpato/spektrafilm)
   (profiles © Andrea Volpato, CC BY-SA 4.0). Interlayer (DIR coupler) effects use spektrafilm's
   default matrix: the data sheets don't publish them, so they are the same for every stock.
3. **Scan:** negative x a high-CRI white LED (CIE LED-B4, 5100 K) x a real camera's sensitivities
   (Canon 5D Mark II, Nikon Z6), white balanced on the light, then the real DumbConvert engine. The
   stock's colour is fixed from a grey-scale-only negative first, as a lab calibrates a film.
4. **Print:** the same negative on the paper its maker pairs it with (Kodak EDGE for Gold and Ultra Max,
   per data sheets E-7022 / E-7023; Supra Endura for Ektar; Portra Endura for Portra; Crystal Archive
   for Fujifilm; Supra Endura for Vision3, which as stills goes through a C-41 lab onto RA-4 paper rather
   than onto cinema print film), tungsten enlarger, filtration solved so mid grey prints neutral, viewed under D50.

Compared in Oklab, the space of the colour mixer. The Kodak stocks agree closely (yellows turn
11-17° toward amber, blues toward teal, greens about 0.7x); Fujifilm keeps lemon yellows, turns greens
toward cyan and magentas toward red. DumbConvert applies 70 % of the hue turns and leaves out the
print's overall ~9 % lower colourfulness (`PRINT_STRENGTH`, `PRINT_CHROMA_REF`).

## Re-running

`spk/`, `specdata/` and `simout/` are git-ignored.

The data is not in this repository. Put it next to the scripts:

- `spk/<stock>.json`: spektrafilm profiles (`src/spektrafilm/data/profiles/`), the films and papers above.
- `specdata/`: `cmfs.py`, `sds_d_illuminant_series.py`, `illum.py` (colour-science's
  `colorimetry/datasets/illuminants/sds.py`) and `checker.py` (`characterisation/datasets/colour_checkers/sds.py`)
  from [colour-science](https://github.com/colour-science/colour), as text; then
  `python parse_spec.py specdata` (reads the tables, runs nothing from them).
- `specdata/cam_*.csv`: camera sensitivities from [ssf-data](https://github.com/butcherg/ssf-data) (CC BY-NC-SA 4.0).

Then from the repository root: `python tools/print_character/run_sim.py` and
`PYTHONPATH=vendor/negpy python tools/print_character/analyse.py` (prints the per-band table).
