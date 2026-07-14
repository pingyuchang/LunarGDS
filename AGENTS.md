# LunarGDS

Raw data and Python processing scripts for Apollo-mission lunar electromagnetic sounding
(magnetometer time series → Z/H transfer function → colatitude → 1D conductivity model → plots).

## Cursor Cloud specific instructions

### Nature of the project
This repo is a collection of standalone Python analysis scripts plus CSV data and PNG
outputs. There is **no application server, API, database, build step, lint config, or test
suite**. "Running the app" means executing an individual script with `python3 <script>.py`.
Dependencies (`numpy`, `pandas`, `scipy`, `matplotlib`, `statsmodels`) are installed
system-wide by the startup update script.

### Two gotchas that block scripts from running as-is
The scripts were authored on Windows and were not parameterized, so before running any
script you must work around both of these (do **not** edit the scripts):

1. **Hardcoded input paths.** Scripts read/write via absolute-looking Windows paths that on
   Linux resolve to *relative* paths under the current working directory. Run scripts from a
   staging directory that contains the expected files. Examples of what to stage there:
   - `ZH_estimation/*` and `Colatitude_estimation/A12_*` read `D:/Sat_MV/AP12_1969_1min.csv`
     → create a `D:/Sat_MV/` subfolder and copy the matching file from `Raw_Time_series/`.
     (AP15/AP16 variants expect `Apollo15_0527.csv` / `Apollo16_0527.csv` there.)
   - `plot_out/Compare_Res_T_VpVs.py` reads literal backslash filenames such as
     `D:\Users\pingy.LAPTOP-9PCQEDK5\Downloads\apollo12_1D_cond.csv`. On Linux the backslashes
     are part of the filename, so create files with those exact names by copying the
     corresponding CSVs from `Inverted_model/` (e.g. `apollo12_1D_cond.csv`,
     `apollo15+12_1D.csv`, `apollo16+12_1D.csv`).
   Scripts write their output CSV/JPEG into the current working directory, so run them from a
   scratch dir (not the repo) to avoid polluting the tree.

2. **Interactive matplotlib.** Scripts call `plt.show()`. The VM is headless, so set
   `MPLBACKEND=Agg` (e.g. `MPLBACKEND=Agg python3 <script>.py`); `plt.show()` then becomes a
   harmless no-op (you'll see benign "FigureCanvasAgg is non-interactive" warnings). Scripts
   that use `plt.savefig(...)` (e.g. `plot_out/Compare_Res_T_VpVs.py`) write figures directly;
   for scripts that only call `plt.show()`, run them through a small wrapper that monkeypatches
   `plt.show` to `savefig` if you need the figure images.
