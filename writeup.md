# Assignment 1 Report | Characterising an AMOC timeseries

· Author: Hengxi Yang
· Date: Sep 2026
· Course : 63-731 Data Analysis in Physical Oceanography

## 1. Introduction

### **1.1 Aim**

### **1.2 Data and Series**

· **Dataset**: Denmark Strait Overflow, DSO, hourly

· **Series**: Overflow time-series through Denmark Strait

· **Timespa**n: 1996-05-01 to 2021-08-07

· **Record length**: 221,514



## 2. Results and Discussion

### **Part A Characterise the Series**

The series I chose is Denmark Strait Overflow (DSO). I tried Calafat 2025 before, which includes a meridional heat transport timeseries with a sample size of only 66, it was too small for the following processing. I also tried OSNAP west, which also has a small sample size. Admittedly, DSO is the lower limb of the AMOC rather than direct overturning transport, but it is a core component of the AMOC. Futhermore, options are limited, and the DSO transport timeseries has a long timespan with 221,514 data points, making it an optimal choice.

Here is some info on trans_DSO:

| Parameter| Value |
| :--- | :--- |
| **Record Length** | 221,514 time steps |
| **Timespan** | 1996-05-01 to 2021-08-07 |
| **Sampling Interval** | ~ 1 hour |
| **Missing Values** | 18,456 |


Because there're a lot pf NaNs, I can't just drop them. So I used `linear interpolation` to fill the NaN positions. I didn't know if there's any pattern or period in Trans_DSO, so linear interpolation is the safest way. For the remaining NaNs at both ends, I dropped them directly. Here is some info on the new Tran_DSO.

| Parameter | Value |
| :--- | :--- |
| **Record Length** | 220,956 time steps |
| **Timespan** | 1996-05-24 to 2021-08-07 |
| **Sampling Interval** | ~ 1 hour |
| **Missing Values** | 0 |

The timespan shortened, but it was worth it.

![Frequency Distribution of DSO Transport](figures/PartA_frequency_distribution_of_DSO_Transport.png)


| Parameter | Value (Sv) |
| :--- | :--- |
| **Mean** | -3.125 |
| **Standard Deviation** | 1.457 |
| **Minimum** | -9.661 |
| **Maximum** | 3.084 |
| **Range** | 12.745 |


$\text{Bin width} = \frac{12.745\text{ Sv}}{100} = \mathbf{0.12745\text{ Sv}}$

According to the distribution figure, the southward transport concentrates around the mean, which indicates that the DSO might have some specific flow characteristics. The standard deviation $\sigma = 1.46 Sv$ indicates that the transport timeseries has significant fluctuations.



### **Part B The Spectrum**

![Welch's Power Spectral Density of DSO](figures/PartB1_welchs_PSD_log_log.png)

| Parameter | Value |
| :--- | :--- |
| `segment length` |  100-day seasonal window |
| `noverlap` | 50% (default)|

The PSD figure shows that most of power concentrates around the **low frequence area** . There's a steep decrease from $10^{-1}$ to ${10^0} $ cycles/day, which means the power of DSO distributes in seasonal and longer time scales.

Addtionally, there's a significant peak around 2 cycles/day, which can be seen clearly in the semi-log figure:

![Welch's Power Spectral Density of DSO](figures/PartB1_welchs_PSD_semilog.png)

Denmark Strait is situated between Greenland and Iceland, where semidiurnal tides (such as the M2) dominate. The period of M2 is about half of a day, so most of the energy is focused around 2 cycles/day.

![Location of Denmark Strait](figures/PartB2_location_of_Denmark_Strait.png)

The red noise spans from $10^{-2}$ to ${10^0} cycles/day, the low frequency range reveals long-term signal, while in the later area, the higher the frequency, the lower the PSD.

The white noise starts from ~ 4 cycles/day, where the PSD keeps nearly constant across frequencies, indicating that the real oceanographic signals have decayed.







