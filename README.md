# AMOC Analysis Assignment 1 **|Remake|**

> 📊 Data analysis for physical oceanography - analyzing AMOC transport data using Python


## Project Structure

```
amoc-analysis/
├── notebooks/                  # Jupyter notebooks for analysis
│   └──  Assignment1.ipynb
├── amoc_analysis/             # Main Python package
│   ├── data.py                # Data loading functions
│   ├── analysis.py            # Analysis tools and utilities
│   └── plotting.py            # Visualization functions
├── figures/     
├── tests/                     # Test suite
│   ├── test_data.py           # Tests for data loading
│   ├── test_analysis.py       # Tests for analysis functions
│   └── test_plotting.py       # Tests for plotting functions
├── data/                      # no data folder, using read to laod dataset from AMOCatals
├── pyproject.toml             # Package configuration
├── requirements.txt           # Core dependencies
├── requirements-dev.txt       # Development dependencies
├── GETTING_STARTED.md         # Step-by-step guide for beginners
├── INSTRUCTIONS.md            # Detailed project structure explanation
├── writeup.md                 # write-up
└── ⭐Assignment1_Report_Remake.pdf      # pdf version of writeup.md
```

## ✅ Aim

- Load the DSO transport timeseries and fill its NaN positions. Plot a Frequency distribution figure to illustrate the volume distribution.

- Compute and plot a power spectrum of DSO transport using Welch's overlapped-segment averaging. Furthermore, verify the variance budget with Parseval and apply a low-pass filter on the original timeseries, plot a timeseries and a power spectrum to compare the original one and the filtered one.

- Apply two different windows and see which one has a better ability to suppress the sidelobes. Add a chi-squared confidence band to the spectrum with EDF.


## ⏳ Data 

- Denmark Strait Overflow, DSO, hourly, from AMOCatlas.
- Timespan: 1996-05-01 to 2021-08-07
- Record length: 221514

---

## 📚 Useful Resources

- [RAPID-MOCHA Array](https://rapid.ac.uk/rapidmoc)
- [Xarray Documentation](https://xarray.pydata.org/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [Physical Oceanography Concepts](https://www.whoi.edu/know-your-ocean/)


