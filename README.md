# one_point_stats
Simulation and analysis tools for 21 cm one-point statistics

## Subpackages
- `stats`: codes to calculate variance, skewness and kurtosis and thermal noise errors of the statistics based on a mathematical framework from Kittiwisit et al. 2019.
- `obs`: classes and routines to help simulate HERA and MWA observations. Many modern, up-to-date, and regularly maintain packages now exist and should be used.
- `foreground_filter`: tools for generating and applying foreground wedge-cut filters
- `utils`: various utility codes for data modeling - fitting, conversion, smoothing, projecting, interpolating and gridding, radial profile calculation, etc.