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
├── data/                      # Data files (downloaded automatically)
├── pyproject.toml             # Package configuration
├── requirements.txt           # Core dependencies
├── requirements-dev.txt       # Development dependencies
├── GETTING_STARTED.md         # Step-by-step guide for beginners
├── INSTRUCTIONS.md            # Detailed project structure explanation
├── writeup.md                 # write-up
└── ⭐Assignment1_Report.pdf      # pdf version of writeup.md
```

## ✅ Aim

- Load DSO timeseris from AMOCatlas and describe it quantitatively — its basic time-domain statistics and its spectrum. 

- Reuse and extend the starter code from the spectra & filtering lecture, and submit a small, tested, reproducible analysis.


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


