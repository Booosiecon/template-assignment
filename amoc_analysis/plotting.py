import matplotlib.pyplot as plt
import xarray as xr
import scipy.signal as signal
import scipy.stats
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.io.formats.style import Styler

from pathlib import Path

from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from urllib.parse import urlparse

import requests







def plot_monthly_transport(
    ds: xr.Dataset, var: str = "moc_mar_hc10"
) -> Tuple[Any, Any]:
    """Plot original and monthly averaged transport time series.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a time dimension and a transport variable.
    var : str, optional
        Name of the variable to plot. Default is "moc_mar_hc10".

    Returns
    -------
    tuple
        Figure and axis objects from matplotlib.
    """
    here = Path(__file__).resolve().parent
    style_file = here / "amoc_analysis.mplstyle"
    if style_file.exists():
        plt.style.use(style_file)

    da = ds[var]
    ds_monthly = ds.resample(TIME="ME").mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ds.TIME, da, color="grey", alpha=0.5, linewidth=0.5, label="Original")
    ax.plot(
        ds_monthly.TIME,
        ds_monthly[var],
        color="red",
        linewidth=1.0,
        label="Monthly Avg",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)

    ax.set_title("RAPID 26°N - AMOC Transport")

    # Use variable attributes if present
    label = da.attrs.get("long_name", var)
    units = da.attrs.get("units", "")
    ax.set_ylabel(f"{label} [{units}]" if units else label)
    ax.set_xlabel("Time")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    plt.tight_layout()

    return fig, ax


def show_variables(data: Union[str, xr.Dataset]) -> Styler:
    """
    Processes an xarray Dataset or a netCDF file, extracts variable information,
    and returns a styled DataFrame with details about the variables.

    Parameters
    ----------
    data : str or xr.Dataset
        The input data, either a file path to a netCDF file or an xarray Dataset.

    Returns
    -------
    pandas.io.formats.style.Styler
        A styled DataFrame containing variable information including dimensions,
        names, units, and comments.
    """
    if isinstance(data, str):
        print(f"Information is based on file: {data}")
        dataset = xr.open_dataset(data)
        variables = dataset.variables
    elif isinstance(data, xr.Dataset):
        print("Information is based on xarray Dataset")
        variables = data.variables
    else:
        raise TypeError("Input data must be a file path (str) or an xarray Dataset")

    info = {}
    for i, key in enumerate(variables):
        var = variables[key]
        if isinstance(data, str):
            dims = var.dims[0] if len(var.dims) == 1 else "multiple"
            units = var.attrs.get("units", "")
            comment = var.attrs.get("comment", "")
        else:
            dims = var.dims[0] if len(var.dims) == 1 else "multiple"
            units = var.attrs.get("units", "")
            comment = var.attrs.get("comment", "")

        info[i] = {
            "name": key,
            "dims": dims,
            "units": units,
            "comment": comment,
            "standard_name": var.attrs.get("standard_name", ""),
            "dtype": str(var.dtype),
        }

    vars_df = DataFrame(info).T

    # Clean up dimensions display
    dims = vars_df.dims
    dims[dims.str.startswith("str")] = "string"
    vars_df["dims"] = dims

    vars_styled = (
        vars_df.sort_values(["dims", "name"])
        .reset_index(drop=True)
        .loc[:, ["dims", "name", "units", "comment", "standard_name", "dtype"]]
        .set_index("name")
        .style
    )

    return vars_styled


def show_attributes(data: Union[str, xr.Dataset]) -> DataFrame:
    """
    Processes an xarray Dataset or a netCDF file, extracts attribute information,
    and returns a DataFrame with details about the attributes.

    Parameters
    ----------
    data : str or xr.Dataset
        The input data, either a file path to a netCDF file or an xarray Dataset.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing attribute names, values, and data types.
    """
    if isinstance(data, str):
        print(f"Information is based on file: {data}")
        dataset = xr.open_dataset(data)
        attributes = dataset.attrs.keys()
        get_attr = lambda key: dataset.attrs[key]
    elif isinstance(data, xr.Dataset):
        print("Information is based on xarray Dataset")
        attributes = data.attrs.keys()
        get_attr = lambda key: data.attrs[key]
    else:
        raise TypeError("Input data must be a file path (str) or an xarray Dataset")

    info = {}
    for i, key in enumerate(attributes):
        dtype = type(get_attr(key)).__name__
        info[i] = {"Attribute": key, "Value": get_attr(key), "DType": dtype}

    attrs_df = DataFrame(info).T

    return attrs_df


def plot_time_series(
    ds: xr.Dataset, 
    var: str, 
    title: str = None,
    ylabel: str = None,
    color: str = "blue",
    figsize: Tuple[int, int] = (12, 6)
) -> Tuple[Any, Any]:
    """Plot a simple time series from an xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a time dimension and the variable to plot.
    var : str
        Name of the variable to plot.
    title : str, optional
        Plot title. If None, uses variable's long_name or the variable name.
    ylabel : str, optional
        Y-axis label. If None, uses variable's long_name and units.
    color : str, optional
        Line color. Default is "blue".
    figsize : tuple, optional
        Figure size as (width, height). Default is (12, 6).

    Returns
    -------
    tuple
        Figure and axis objects from matplotlib.
    """
    da = ds[var]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ds.TIME, da, color=color, linewidth=1.0)
    
    # Set title
    if title is None:
        title = da.attrs.get("long_name", var)
    ax.set_title(title)
    
    # Set ylabel
    if ylabel is None:
        label = da.attrs.get("long_name", var)
        units = da.attrs.get("units", "")
        ylabel = f"{label} [{units}]" if units else label
    ax.set_ylabel(ylabel)
    
    ax.set_xlabel("Time")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, ax


# plot the long-term linearized trend for AMOC annual mean data

def calculate_and_plot_trend(amoc_annual_series: xr.DataArray, figsize=(15, 5.5)):
 
    from scipy.stats import linregress
    
    y_data = amoc_annual_series.values
    years = pd.to_datetime(amoc_annual_series.TIME.values).year
    x_data = years - years[0]
    
    slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
    trend_line = slope * x_data + intercept
    
    # print results
    print('------- AMOC long-term trend----------')
    print(f"annual mean change rate (slope): {slope:.4f} Sv/year")
    print(f"Total change in 20 years: {slope * len(x_data):.2f} Sv")
    print(f"p-value: {p_value: .5f}")

    if p_value < 0.05:
        print("This weakening trend is significant at 95% confidence.")
    else:
        print("The trend did't pass significance testing, it may be a random fluctuation.")
    
    print("----"*5)
    
    plt.figure(figsize=figsize)
    plt.plot(years, y_data, color='crimson', marker='^', linewidth=2, label='Annual Mean')
    plt.plot(years, trend_line, color='royalblue', linestyle='--', linewidth=2, 
             label=f'Linear Trend ({slope:.3f} Sv/year)')
    
    plt.xticks(years, rotation=0) 
    plt.title('AMOC Long-term Linear Trend at 26.5°N', fontsize=14)
    plt.ylabel('Transport [Sv]', fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.show()
    
    return {"slope": slope, "p_value": p_value, "total_change": slope * len(x_data)}



# Various conversions from the key to units_name with the multiplicative conversion factor
unit_conversion = {
    "cm/s": {"units_name": "m/s", "factor": 0.01},
    "cm s-1": {"units_name": "m s-1", "factor": 0.01},
    "m/s": {"units_name": "cm/s", "factor": 100},
    "m s-1": {"units_name": "cm s-1", "factor": 100},
    "S/m": {"units_name": "mS/cm", "factor": 0.1},
    "S m-1": {"units_name": "mS cm-1", "factor": 0.1},
    "mS/cm": {"units_name": "S/m", "factor": 10},
    "mS cm-1": {"units_name": "S m-1", "factor": 10},
    "dbar": {"units_name": "Pa", "factor": 10000},
    "Pa": {"units_name": "dbar", "factor": 0.0001},
    "degrees_Celsius": {"units_name": "Celsius", "factor": 1},
    "Celsius": {"units_name": "degrees_Celsius", "factor": 1},
    "m": {"units_name": "cm", "factor": 100},
    "cm": {"units_name": "m", "factor": 0.01},
    "km": {"units_name": "m", "factor": 1000},
    "g m-3": {"units_name": "kg m-3", "factor": 0.001},
    "kg m-3": {"units_name": "g m-3", "factor": 1000},
}

# Specify the preferred units, and it will convert if the conversion is available in unit_conversion
preferred_units = ["m s-1", "dbar", "S m-1"]

# String formats for units.  The key is the original, the value is the desired format
unit_str_format = {
    "m/s": "m s-1",
    "cm/s": "cm s-1",
    "S/m": "S m-1",
    "meters": "m",
    "degrees_Celsius": "Celsius",
    "g/m^3": "g m-3",
    "m^3/s": "Sv",
}


def reformat_units_var(
    ds: xr.Dataset, var_name: str, unit_format: Dict[str, str] = unit_str_format
) -> str:
    """
    Renames units in the dataset based on the provided dictionary.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing variables with units to be renamed.
    var_name : str
        Name of the variable to check units for.
    unit_format : dict
        A dictionary mapping old unit strings to new formatted unit strings.

    Returns
    -------
    str
        The formatted unit string.
    """
    old_unit = ds[var_name].attrs["units"]
    if old_unit in unit_format:
        new_unit = unit_format[old_unit]
    else:
        new_unit = old_unit
    return new_unit


def convert_units_var(
    var_values: Any,
    current_unit: str,
    new_unit: str,
    unit_conversion: Dict[str, Dict[str, Union[str, float]]] = unit_conversion,
) -> Any:
    """
    Convert the units of variables to new units.

    Parameters
    ----------
    var_values : Any
        The variable values to convert.
    current_unit : str
        Current unit of the variable.
    new_unit : str
        Target unit for conversion.
    unit_conversion : dict
        A dictionary mapping current units to conversion information.

    Returns
    -------
    Any
        The converted variable values.
    """
    if (
        current_unit in unit_conversion
        and new_unit in unit_conversion[current_unit]["units_name"]
    ):
        conversion_factor = unit_conversion[current_unit]["factor"]
        new_values = var_values * conversion_factor
    else:
        new_values = var_values
        print(f"No conversion information found for {current_unit} to {new_unit}")
    return new_values


def get_default_data_dir() -> Path:
    """Get the default data directory for the project."""
    return Path(__file__).resolve().parent.parent / "data"


def apply_defaults(default_source: str, default_files: List[str]) -> Callable:
    """Decorator to apply default values for 'source' and 'file_list' parameters if they are None.

    Parameters
    ----------
    default_source : str
        Default source URL or path.
    default_files : list of str
        Default list of filenames.

    Returns
    -------
    Callable
        A wrapped function with defaults applied.

    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(
            source: Optional[str] = None,
            file_list: Optional[List[str]] = None,
            *args,
            **kwargs,
        ) -> Callable:
            if source is None:
                source = default_source
            if file_list is None:
                file_list = default_files
            return func(source=source, file_list=file_list, *args, **kwargs)

        return wrapper

    return decorator


def _is_valid_url(url: str) -> bool:
    """Validate if a given string is a valid URL with supported schemes.

    Parameters
    ----------
    url : str
        The URL string to validate.

    Returns
    -------
    bool
        True if the URL is valid and uses a supported scheme ('http', 'https', 'ftp'),
        otherwise False.

    """
    try:
        result = urlparse(url)
        return all(
            [
                result.scheme in ("http", "https", "ftp"),
                result.netloc,
                result.path,  # Ensure there's a path, not necessarily its format
            ],
        )
    except Exception:
        return False


def resolve_file_path(
    file_name: str,
    source: Union[str, Path, None],
    download_url: Optional[str],
    local_data_dir: Path,
    redownload: bool = False,
) -> Path:
    """Resolve the path to a data file, using local source, cache, or downloading if necessary.

    Parameters
    ----------
    file_name : str
        The name of the file to resolve.
    source : str or Path or None
        Optional local source directory.
    download_url : str or None
        URL to download the file if needed.
    local_data_dir : Path
        Directory where downloaded files are stored.
    redownload : bool, optional
        If True, force redownload even if cached file exists.

    Returns
    -------
    Path
        Path to the resolved file.

    """
    # Use local source if provided
    if source and not _is_valid_url(source):
        source_path = Path(source)
        candidate_file = source_path / file_name
        if candidate_file.exists():
            return candidate_file
        else:
            raise FileNotFoundError(f"Local file not found: {candidate_file}")

    # Use cached file if available and redownload is False
    cached_file = local_data_dir / file_name
    if cached_file.exists() and not redownload:
        return cached_file

    # Download if URL is provided
    if download_url:
        try:
            return download_file(download_url, local_data_dir, redownload=redownload)
        except Exception as e:
            raise FileNotFoundError(f"Failed to download {download_url}: {e}")

    # If no options succeeded
    raise FileNotFoundError(
        f"File {file_name} could not be resolved from local source, cache, or remote URL.",
    )


def download_file(url: str, dest_folder: str, redownload: bool = False) -> str:
    """Download a file from HTTP(S) or FTP to the specified destination folder.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    dest_folder : str
        Local folder to save the downloaded file.
    redownload : bool, optional
        If True, force re-download of the file even if it exists.

    Returns
    -------
    str
        The full path to the downloaded file.

    Raises
    ------
    ValueError
        If the URL scheme is unsupported.

    """
    dest_folder_path = Path(dest_folder)
    dest_folder_path.mkdir(parents=True, exist_ok=True)

    local_filename = dest_folder_path / Path(url).name
    if local_filename.exists() and not redownload:
        # File exists and redownload not requested
        return str(local_filename)

    parsed_url = urlparse(url)

    if parsed_url.scheme in ("http", "https"):
        # HTTP(S) download
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    elif parsed_url.scheme == "ftp":
        # FTP download
        from ftplib import FTP
        with FTP(parsed_url.netloc) as ftp:
            ftp.login()  # anonymous login
            with open(local_filename, "wb") as f:
                ftp.retrbinary(f"RETR {parsed_url.path}", f.write)

    else:
        raise ValueError(f"Unsupported URL scheme in {url}")

    return str(local_filename)


def safe_update_attrs(
    ds: xr.Dataset,
    new_attrs: Dict[str, str],
    overwrite: bool = False,
    verbose: bool = True,
) -> xr.Dataset:
    """Safely update attributes of an xarray Dataset without overwriting existing keys,
    unless explicitly allowed.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset whose attributes will be updated.
    new_attrs : dict of str
        Dictionary of new attributes to add.
    overwrite : bool, optional
        If True, allow overwriting existing attributes. Defaults to False.
    verbose : bool, optional
        If True, emit a warning when skipping existing attributes. Defaults to True.

    Returns
    -------
    xr.Dataset
        The dataset with updated attributes.

    """
    for key, value in new_attrs.items():
        if key in ds.attrs:
            if not overwrite:
                if verbose:
                    print(
                        f"Attribute '{key}' already exists in dataset attrs and will not be overwritten.",
                    )
                continue  # Skip assignment
        ds.attrs[key] = value

    return ds




# calcualte the annual mean from AMOC daily data
def calculate_annual_mean(amoc_daily_data: xr.DataArray) -> xr.DataArray:

    if "time" in amoc_daily_data.dims:
        amoc_daily_data = amoc_daily_data.rename({"time": "TIME"})
    return amoc_daily_data.resample(TIME='YE').mean()



def calculate_statistics(series: xr.DataArray) -> Tuple[float, float]:
    """Calculate the temporal mean and standard deviation of the given AMOC time series.

    Parameters
    ----------
    series : xr.DataArray
        The time series data array containing physical values.

    Returns
    -------
    Tuple[float, float]
        A tuple containing (mean_value, std_value) as standard floats.
    """
    # Calculate the mean and standard deviation along the temporal dimension
    mean_val = float(series.mean().item())
    std_val = float(series.std().item())
    
    return mean_val, std_val




def apply_tukey_filter(series: xr.DataArray, window_size: int = 12) -> xr.DataArray:
    """Apply a low-pass filter to the series using a Tukey window convolution.

    Parameters
    ----------
    series : xr.DataArray
        The cleaned input time series.
    window_size : int
        The size of the moving window (default 12 quarters, roughly 3 years).

    Returns
    -------
    xr.DataArray
        The filtered time series with boundary NaNs preserved.
    """
    # Convert to pandas Series to leverage advanced window types
    df_series = series.to_series()
    
    # Perform rolling mean with a Tukey window
    series_filtered = df_series.rolling(window=window_size, win_type='tukey', center=True).mean()
    
    # Convert back to xarray.DataArray preserving coordinates
    filtered_da = xr.DataArray.from_series(series_filtered)
    filtered_da.attrs = series.attrs.copy()
    filtered_da.attrs['processing'] = f"Tukey low-pass filtered with window size {window_size}"
    
    return filtered_da




def compute_spectral_analysis(raw_series: xr.DataArray, filtered_series: xr.DataArray, nperseg: int = 36) -> Dict[str, Any]:
    """Compute the Power Spectral Density (PSD) for both raw and filtered series.

    Parameters
    ----------
    raw_series : xr.DataArray
        The original cleaned time series.
    filtered_series : xr.DataArray
        The time series after applying the Tukey filter (contains boundary NaNs).
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 36.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing frequencies, raw PSD, filtered PSD, 
        and variance budget metrics.
    """
    # 1. Strictly align with the official dataset documentation (3-monthly sampling)
    fs = 4.0  

    # 2. Process the Raw Series (Demean to focus purely on variance budget)
    raw_vals = raw_series.values
    raw_demeaned = raw_vals - np.mean(raw_vals)
    frequencies, psd_raw = signal.welch(
        raw_demeaned, fs=fs, nperseg=nperseg, detrend='linear', scaling='density'
    )

    # 3. Process the Filtered Series
    # Strip away the boundary NaNs generated by the rolling window before spectral computation
    clean_filt = filtered_series.dropna(dim='TIME')
    filt_vals = clean_filt.values
    filt_demeaned = filt_vals - np.mean(filt_vals)
    _, psd_filt = signal.welch(
        filt_demeaned, fs=fs, nperseg=nperseg, detrend='linear', scaling='density'
    )

    # 4. Integrate Raw PSD over frequency to verify Parseval's conservation theorem
    df = frequencies[1] - frequencies[0]
    integrated_power = np.sum(psd_raw) * df
    series_variance = np.var(raw_demeaned, ddof=0)

    return {
        "frequencies": frequencies,
        "psd_raw": psd_raw,
        "psd_filt": psd_filt,
        "variance": series_variance,
        "integrated_power": integrated_power,
        "fs": fs
    }





def plot_filter_comparison(frequencies: np.ndarray, magnitude_tukey: np.ndarray, magnitude_boxcar: np.ndarray) -> None:
    """Plot frequency responses to compare Tukey and Boxcar low-pass filtering characteristics.

    Parameters
    ----------
    frequencies : np.ndarray
        Normalized physical frequencies.
    magnitude_tukey : np.ndarray
        Tukey window frequency response magnitude in dB.
    magnitude_boxcar : np.ndarray
        Boxcar window frequency response magnitude in dB.
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'stix'

    plt.figure(figsize=(10, 5), dpi=300)

    # 1. Plot responses using correct English character strings
    plt.plot(frequencies, magnitude_boxcar, color='#e74c3c', linestyle='--', linewidth=1.5,
             label='Plain Boxcar (Rectangular) Window')
    plt.plot(frequencies, magnitude_tukey, color='#16a085', linestyle='-', linewidth=2.0,
             label='Tukey (Tapered) Window')

    # 2. Highlight cutoff threshold reference
    plt.axhline(y=-6, color='#7f8c8d', linestyle=':', linewidth=1)
    
    # Labeling and decorations
    plt.title("Part C: Frequency Response Comparison (Tukey vs. Boxcar)", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Frequency (cycles per year)', fontsize=10, labelpad=8)
    plt.ylabel('Magnitude Response (Gain in dB)', fontsize=10, labelpad=8)
    plt.ylim([-50, 5])
    plt.grid(True, which="both", linestyle=':', alpha=0.5, color='#999999')
    plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#e0e0e0', fontsize=9)

    plt.tight_layout()
    plt.show()







def plot_power_spectrum_with_ci(frequencies: np.ndarray, psd_raw: np.ndarray, psd_filt: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray, dof: float) -> None:
    """Plot the professional Log-scale PSD with a shaded Chi-squared 95% confidence band.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array.
    psd_raw : np.ndarray
        Power spectral density values of the raw series.
    psd_filt : np.ndarray
        Power spectral density values of the filtered series.
    lower_bound : np.ndarray
        Lower confidence interval boundary for the raw spectrum.
    upper_bound : np.ndarray
        Upper confidence interval boundary for the raw spectrum.
    dof : float
        Calculated degrees of freedom.
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'stix'

    plt.figure(figsize=(15, 4.5), dpi=300)

    # 1. Draw the 95% Chi-squared shaded confidence band around Raw Spectrum
    plt.fill_between(frequencies, lower_bound, upper_bound, 
                     color='#95a5a6', alpha=0.18, linestyle=':', edgecolor='#7f8c8d',
                     label=rf'95% $\chi^2$ Confidence Band (DoF = {dof:.1f})')

    # 2. Plot Raw and Filtered continuous lines
    plt.plot(frequencies, psd_raw, color='#7f8c8d', alpha=0.8, linestyle='-', linewidth=1.5,
             label='Raw Spectrum')
    plt.plot(frequencies, psd_filt, color='#c0392b', linestyle='-', linewidth=2.2,
             label='Filtered Spectrum')

    # 3. Reference line at annual cycle
    plt.axvline(x=1.0, color='#555555', linestyle=':', linewidth=1.2, 
                label=r'Annual Cycle ($f = 1.0\ \mathrm{yr}^{-1}$)')

    # Log scaling and academic decorations
    plt.yscale('log')
    plt.title(f"MHT Power Spectral Density with 95% Confidence Band (DoF $\\approx$ {dof:.1f})", 
              fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Frequency (cycles per year)', fontsize=10, labelpad=8)
    plt.ylabel(r'Power Spectral Density ($PW^2 / \mathrm{cyc} \cdot \mathrm{yr}^{-1}$)', fontsize=10, labelpad=8)
    plt.grid(True, which="both", linestyle=':', alpha=0.4, color='#999999')
    plt.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#e0e0e0', fontsize=9)

    plt.tight_layout()
    plt.show()




def plot_transport_distribution(
        series: xr.DataArray,
         mean_val: float, 
         var_name: str,
         dpi = 600,
):
    """Plot the probability distribution histogram for the given transport timeseries."""

    fig, ax = plt.subplots(figsize=(9, 5), dpi = dpi)

    # filter NaNs
    clean_series = series.values[~np.isnan(series.values)]

    # Plot histogram
    ax.hist(clean_series, bins = 100, density = True, alpha = 0.6, color = '#acaaa9', edgecolor = '#626160', linewidth = 0.5, label = 'Histogram')

    # KDE linear line
    mu, std = scipy.stats.norm.fit(clean_series)
    xmin, xmax = ax.get_xlim()
    x_vals = np.linspace(xmin, xmax, 500)
    p = scipy.stats.norm.pdf(x_vals, mu, std)
    ax.plot(x_vals, p, color = '#3d4447', linewidth = 1.5, label = f'Gaussian Fit ($\\mu$ = {mu:.2f}, $\\sigma$ = {std:.2f})')
    
    # Mark mean line
    mean_peak_y = scipy.stats.norm.pdf(mean_val, mu, std)
    ax.vlines(x = mean_val, ymin = 0, ymax = mean_peak_y, color = 'red', linestyle = '--', linewidth = 1.8, label = f'Mean: {mean_val:.2f} Sv')


    ax.set_title(f"Frequency Distribution of {var_name} Transport", fontsize = 16)
    ax.set_xlabel("Volume Transport (Sv)", fontsize = 14)
    ax.set_ylabel("Density", fontsize = 14)
    ax.legend(loc = "best", fontsize= 10)
    ax.grid(True, linestyle = ':', alpha = 0.6)
    fig.tight_layout()

    return fig, ax



def plot_welch_psd(
    freqs, 
    psd, 
    var_name: str,
    scale_type: str,
    dpi = 600,
):
    """Plot Welch's Power Spectral Density using a log-log or semi-log scale."""

    fig, ax = plt.subplots(figsize = (9, 5), dpi = dpi)

    # filter freq ≤ 0 data
    mask = freqs > 0
    f_plot = freqs[mask]
    psd_plot = psd[mask]


    # log-log or semilog
    if scale_type == "log-log":
        ax.loglog(f_plot, psd_plot, color='darkblue', linewidth=1.2, label='Welch PSD (Log-Log)')
        title_suffix = " (Log-Log Scale)"
    elif scale_type == "semilog":
        ax.semilogy(f_plot, psd_plot, color='darkred', linewidth=1.2, label='Welch PSD (Semi-Log)')
        title_suffix = " (Semi-Log Scale)"
    else:
        raise ValueError("scale_type must be 'log-log' or 'semilog'")

    
    ax.set_title(f"Welch's Power Spectral Density of {var_name}", fontsize = 16)
    ax.set_xlabel("Frequency (cycles/day)", fontsize = 14)
    ax.set_ylabel(r"PSD [Sv$^2$ / (cycles/day)]", fontsize = 14)
    
    ax.grid(True, which = "both", linestyle = ":", alpha = 0.6)
    ax.legend(fontsize = 10)
    fig.tight_layout()

    return fig, ax



def plot_filtered_comparison(
    original_series, 
    filtered_series, 
    var_name,
    dpi = 600,
):
    """
    Figure 1: Overlay original and filtered time series.
    """

    fig, ax = plt.subplots(figsize = (12, 5), dpi = dpi)

    ax.plot(original_series['TIME'].values, original_series.values, color = 'lightblue', alpha = 0.6, label = 'Original (Hourly)')
    ax.plot(filtered_series['TIME'].values, filtered_series.values, color = 'darkred', linewidth = 1.5, label = 'Tukey Filtered (Low-pass)')
    
    ax.set_title(f"Time Series Comparison of Original and Low-Pass Filtered {var_name}", fontsize = 16)
    ax.set_xlabel("Time", fontsize = 14)
    ax.set_ylabel("Volume Transport (Sv)", fontsize = 14)
    ax.legend(fontsize = 14)
    ax.grid(True, linestyle = ':', alpha = 0.6)

    fig.tight_layout()
    return fig, ax
    


def plot_psd_comparison(
    freqs_orig, 
    psd_orig, 
    freqs_filt, 
    psd_filt, 
    var_name,
    dpi = 600,
):
    """
    Figure 2: Overlay original and filtered power spectra (Log-Log scale).
    """
    fig, ax = plt.subplots(figsize=(9, 5), dpi = dpi)
    
    mask_o = freqs_orig > 0
    mask_f = freqs_filt > 0
    
    ax.loglog(freqs_orig[mask_o], psd_orig[mask_o], color = 'lightblue', alpha = 0.7, label = 'Original PSD')
    ax.loglog(freqs_filt[mask_f], psd_filt[mask_f], color = 'darkred', linewidth = 1.5, label = 'Filtered PSD (Low-pass)')
    
    ax.set_title(f"PSD Comparison of Original and Filtered {var_name}", fontsize = 16)
    ax.set_xlabel("Frequency (cycles/day)", fontsize = 14)
    ax.set_ylabel(r"PSD [Sv$^2$ / (cycles/day)]", fontsize = 14)
    ax.legend(fontsize = 14)
    ax.grid(True, which = "both", linestyle = ':', alpha = 0.6)

    fig.tight_layout()
    return fig, ax



def plot_filter_frequency_responses(
    freqs: np.ndarray, 
    tukey_db: np.ndarray, 
    boxcar_db: np.ndarray,
    dpi = 600,
):
    """
    Plot and compare the frequency responses of Tukey and Boxcar windows.
    """

    fig, ax = plt.subplots(figsize=(9, 5), dpi = dpi)

    ax.plot(freqs, tukey_db, color = 'darkred', linewidth = 2, label = r'Tukey Window ($\alpha=0.5$)')
    ax.plot(freqs, boxcar_db, color = 'gray', linestyle = '--', linewidth = 1.5, label = 'Boxcar Window (Rectangular)')
    
    ax.set_title("Filter Frequency Response Comparison (Magnitude)", fontsize = 16)
    ax.set_xlabel("Frequency [cycles/day]", fontsize = 14)
    ax.set_ylabel("Magnitude [dB]", fontsize = 14)
    ax.legend(fontsize = 14)
    ax.grid(True, which = "both", linestyle = ':', alpha = 0.6)

    fig.tight_layout()
    return fig, ax


def plot_psd_with_confidence_interval(
    freqs: np.ndarray, 
    psd: np.ndarray, 
    lower_bound: np.ndarray, 
    upper_bound: np.ndarray, 
    dof: float,
    dpi = 600,
):
    """
    Plot Welch PSD with Chi-squared 95% confidence interval bands (Log-Log scale).
    """
    fig, ax = plt.subplots(figsize=(9, 5), dpi = dpi)
    
    mask = freqs > 0
    ax.loglog(freqs[mask], psd[mask], color = 'darkblue', linewidth = 1.5, label = 'Welch PSD')
    ax.loglog(freqs[mask], lower_bound[mask], color = 'gray', linestyle = '--', label = '95% CI Lower Bound')
    ax.loglog(freqs[mask], upper_bound[mask], color = 'gray', linestyle = '--', label = '95% CI Upper Bound')
    ax.fill_between(freqs[mask], lower_bound[mask], upper_bound[mask], color = 'gray', alpha = 0.2, label = '95% Confidence Band')
    
    ax.set_title(f"Welch PSD with Chi-Squared Confidence Interval (EDF = {dof:.1f})", fontsize = 16)
    ax.set_xlabel("Frequency (cycles/day)", fontsize = 14)
    ax.set_ylabel(r"PSD [Sv$^2$ / (cycles/day)]", fontsize = 14)
    ax.legend(fontsize = 14)
    ax.grid(True, which = "both", linestyle = ':', alpha = 0.6)

    fig.tight_layout()
    return fig, ax

