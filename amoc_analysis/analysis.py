from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from urllib.parse import urlparse

import requests
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

import scipy.signal as signal
from scipy.signal import welch
import numpy as np


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







def compute_filter_responses(window_size: int = 12) -> Dict[str, Any]:
    """Compute the frequency response of a Tukey window and a plain Boxcar window.

    Parameters
    ----------
    window_size : int, optional
        The width of the moving window. Default is 12.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing normalized frequencies, Tukey magnitude (dB),
        and Boxcar magnitude (dB).
    """
    # 1. Generate window coefficients
    # Tukey window with shape parameter alpha=0.5 (standard default in pandas)
    window_tukey = signal.windows.tukey(window_size, alpha=0.5)
    window_boxcar = signal.windows.boxcar(window_size)

    # Normalize weights so the DC gain is exactly 1 (0 dB)
    window_tukey /= np.sum(window_tukey)
    window_boxcar /= np.sum(window_boxcar)

    # 2. Compute frequency response using scipy.signal.freqz
    # We sample 512 points between 0 and Nyquist frequency
    w_tuk, h_tuk = signal.freqz(window_tukey, worN=512)
    _, h_box = signal.freqz(window_boxcar, worN=512)

    # 3. Convert digital frequency (radians/sample) to physical frequency (cycles/year)
    # Sampling frequency fs = 4.0 yr^-1, Nyquist is fs / 2 = 2.0 yr^-1
    frequencies = (w_tuk / np.pi) * 2.0

    # 4. Convert magnitudes to decibels (dB), setting a lower floor to prevent log(0)
    magnitude_tukey_db = 20 * np.log10(np.maximum(np.abs(h_tuk), 1e-5))
    magnitude_boxcar_db = 20 * np.log10(np.maximum(np.abs(h_box), 1e-5))

    return {
        "frequencies": frequencies,
        "magnitude_tukey": magnitude_tukey_db,
        "magnitude_boxcar": magnitude_boxcar_db
    }






def compute_confidence_interval(psd: np.ndarray, nperseg: int = 36, n_total: int = 68, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute the Chi-squared confidence intervals for Welch's PSD estimate.

    Parameters
    ----------
    psd : np.ndarray
        The calculated power spectral density array.
    nperseg : int
        Length of each segment used in Welch's method (default 36).
    n_total : int
        Total length of the time series (the cleaned MHT has ~68 points).
    alpha : float
        Significance level (default 0.05 for 95% confidence).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        Lower bound array, upper bound array, and the calculated degrees of freedom.
    """
    
    from scipy.stats import chi2

    # 1. Estimate Equivalent Degrees of Freedom (EDF) for Welch method with 50% overlap
    dof = 2.0 * (n_total / nperseg) * (8.0 / 9.0)
    
    # 2. Fetch critical values from Chi-squared distribution table
    c_lower = chi2.ppf(1.0 - alpha / 2.0, df=dof)
    c_upper = chi2.ppf(alpha / 2.0, df=dof)

    # 3. Compute spectral scaling multiplier bounds
    lower_bound = (dof * psd) / c_lower
    upper_bound = (dof * psd) / c_upper

    return lower_bound, upper_bound, dof







##########################################################################
#------------------------------------------------------------------------#
##########################################################################

def test_calculate_annual_mean():
    """Test that annual mean correctly downsamples daily data and renames coordinate."""
    # Create 2 years of dummy daily data (730 days)
    times = pd.date_range(start="2004-01-01", periods=730, freq="D")
    da_daily = xr.DataArray(
        data=np.ones(730) * 15.0,
        dims=["time"],
        coords={"time": times}
    )
    
    # Run annual mean calculation
    da_annual = analysis.calculate_annual_mean(da_daily)
    
    # Assert dimension renaming and correct shape (2 years)
    assert "TIME" in da_annual.dims
    assert len(da_annual) == 2
    # Assert values are mathematically averaged correctly
    np.testing.assert_allclose(da_annual.values, [15.0, 15.0])


def test_calculate_statistics():
    """Test the temporal mean and standard deviation extraction."""
    # Create a simple array with known mean=10 and std=2
    # np.array([8, 12]) has mean=10, variance=4, std=2
    da = xr.DataArray(data=np.array([8.0, 12.0]), dims=["TIME"])
    
    mean_val, std_val = analysis.calculate_statistics(da)
    
    assert pytest.approx(mean_val) == 10.0
    assert pytest.approx(std_val) == 2.0


def test_compute_spectral_analysis_satisfies_parseval():
    """Property-based test: Verify Welch PSD satisfies Parseval's variance conservation."""
    # 1. Generate a reproducible random white noise series (matches PDF page 13 guidance)
    rng = np.random.default_rng(42)
    n_points = 200
    time_coords = pd.date_range(start="2004-01-01", periods=n_points, freq="3ME") # 3-monthly sampling
    
    raw_series = xr.DataArray(
        data=rng.standard_normal(n_points),
        dims=["TIME"],
        coords={"TIME": time_coords}
    )
    
    # Create a dummy filtered series of the same shape
    filtered_series = raw_series.copy()
    
    # 2. Execute spectral analysis
    spec_results = analysis.compute_spectral_analysis(raw_series, filtered_series, nperseg=64)
    
    # 3. Assert Nyquist frequency and sampling alignment (fs = 4.0 yr^-1)
    assert spec_results["fs"] == 4.0
    assert np.max(spec_results["frequencies"]) == 2.0  # Nyquist = fs / 2
    
    # 4. Parseval Budget Verification: Integrated power should approximately equal time-domain variance
    # Setting a loose relative tolerance due to Welch windowing effects, detrending, and discrete integration
    assert spec_results["integrated_power"] == pytest.approx(spec_results["variance"], rel=0.15)


def test_compute_confidence_interval():
    """Test that Chi-squared bounds scale logically with degrees of freedom."""
    dummy_psd = np.array([1.0, 2.0, 3.0])
    
    # Compute bounds with 95% confidence (alpha=0.05)
    lower, upper, dof = analysis.compute_confidence_interval(dummy_psd, nperseg=36, n_total=68, alpha=0.05)
    
    # Assert Degrees of Freedom calculation follows Welch formula
    expected_dof = 2.0 * (68 / 36) * (8.0 / 9.0)
    assert dof == pytest.approx(expected_dof)
    
    # Assert statistical logic: lower bound <= raw PSD <= upper bound
    assert np.all(lower <= dummy_psd)
    assert np.all(upper >= dummy_psd)


def report_dataset_info(data_input, var_name: str = "TRANS_DSO") -> None:
    """Report the record length, sampling interval, and the number/pattern of missing values."""

    if isinstance(data_input, xr.Dataset):
        series = data_input[var_name]
    else:
        series = data_input    # if input format is DataArray


    # record length and timespan
    total_steps = series.sizes.get('TIME', len(series))
    start_time = pd.to_datetime(series['TIME'].values[0]).strftime('%Y-%m-%d')
    end_time = pd.to_datetime(series['TIME'].values[-1]).strftime('%Y-%m-%d')


    # sampling interval
    time_diff = pd.to_datetime(series['TIME'].values[1]) - pd.to_datetime(series['TIME'].values[0])
    total_hours = time_diff.total_seconds() / 3600

    if total_hours >= 24:
        sampling_interval = f"Approximately {total_hours / 24:.1f} days"
    else:
        sampling_interval = f"Approximately {total_hours:.0f} hours"

    # number of missing values
    nan_count = int(series.isnull().sum().item())

    # report
    print("*" * 20)
    print(f"Record length is {total_steps} time steps")
    print(f"Timespan is from {start_time} to {end_time}")
    print(f"Sampling Interval is {sampling_interval}")
    print(f"{nan_count} Missing values")


def compute_transport_stats(series: xr.DataArray, var_name: str = "TRANS_DSO") -> dict:
    """Compute and report the mean, standard deviation and range."""

    mean_val = series.mean().item()
    std_val = series.std().item()
    min_val = series.min().item()
    max_val = series.max().item()
    transport_range = max_val - min_val

    print(f"Mean: {mean_val:.3f} Sv")
    print(f"Standard Deviation: {std_val:.3f} Sv")
    print(f"Minimum: {min_val:.3f} Sv")
    print(f"Maximum: {max_val:.3f} Sv")
    print(f"Range: {transport_range:.3f} Sv")

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "range": transport_range
    }



def calculate_welch_psd(
    series: xr.DataArray, 
    fs: float = 24.0, 
    nperseg: int = None
) -> tuple:
    """
    Compute Welch's Power Spectral Density (PSD) for the time series.
    fs: sampling frequency. If data is hourly, fs=24.0 means frequency unit is cycles/day.
    nperseg: length of each segment. Default is set to 90 days of hourly data (24 * 90).
    """

    # da → numpy array, demean
    data = series.values
    data_demean = data - np.mean(data)

    # set the length of each segment
    if nperseg is None:
        nperseg = 24 * 90 # 24h*90 → 90 days

    # compute
    freqs, psd = welch(data_demean, fs = fs, nperseg = nperseg, scaling = 'density')

    return freqs, psd




def verify_with_parseval(
    series: xr.DataArray, 
    freqs: np.ndarray, 
    psd: np.ndarray,
    tolerance: float
) -> dict:
    """Verify the variance budget with Parseval's Theorem."""

    # Time-domain variance of demeaned series
    data = series.values
    data_demean = data - np.mean(data)
    time_variance = np.var(data_demean, ddof = 1)

    # Frequency-domain integral of PSD
    df = freqs[1] - freqs[0]
    # freq_integral = np.trapz(psd, freqs)
    freq_integral = np.trapezoid(psd, freqs)

    print(f"Time-domain Variance is {time_variance:.6f} Sv²")
    print(f"Frequency-domain Integral (PSD) is {freq_integral:.6f} Sv²")

    # relative error
    diff = abs(time_variance - freq_integral)
    rel_error = (diff / time_variance) * 100

    print(f"Difference is {diff:.3f}")
    print(f"Relative Error is {rel_error:.3f}%")

    # add tolerance test
    assert rel_error < tolerance * 100, \
        f"Parseval verification failed! Relative error {rel_error}% exceeds tolerance."
    print("✅ Test Passed: Parseval's theorem verified successfully!")
    
    return {
        "time_variance": time_variance,
        "freq_integral": freq_integral,
        "relative_error": rel_error
    }

    return {
        "time_variance": time_variance,
        "freq_integral": freq_integral,
        "relative_error": rel_error
    }


def apply_tukey_lowpass_filter(
    series: xr.DataArray,
    window_size: int,
    alpha: float = 0.5
) -> xr.DataArray:
    """
    Apply a low-pass filter using a Tukey window via pandas rolling.
    window_size: length of the rolling window (e.g., 24 * 30 for a 30-day window).
    alpha: shape parameter of the Tukey window (0 = rectangular, 1 = hann). Default is 0.5.
    """

    s_pd = series.to_pandas()

    #
    weights = signal.windows.tukey(window_size, alpha=alpha)
    weights /= weights.sum()

    # smoothing by convolution
    filtered_pd = s_pd.rolling(
        window=window_size,
        center=True
    ).apply(lambda x: np.dot(x, weights), raw=True)

    # filter NaNs at the ends
    filtered_series = xr.DataArray(
        filtered_pd,
        dims=series.dims,
        coords=series.coords
    )

    # drop NaNs due to window convolution
    filtered_series = filtered_series.dropna(dim="TIME")
    
    return filtered_series