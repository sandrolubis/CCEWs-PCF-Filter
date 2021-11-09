# CCEW-PCF-Filter
2D spatial projection of CCEWs using parabolic cylinder functions (PCFs) following Yang et al., (2003). Parabolic Cylinder Functions (PCFs) are the meridional basis of solutions of the rotating, linearized shallow-water equations on the tropical β-plane

1. Data folder includes netcdf file of daily anomalies of zonal wind (u), meridional wind (v), and geopotential height (phi) at 850 hPa over 20<sup>o</sup>N-20<sup>o</sup>S for the period of 2016-2020.
2. project_waves.py includes the codes to project the dynamical fields onto different types of equatorial waves.

<p align="center">
  <img src="https://github.com/sandrolubis/CCEW-PCF-Filter/blob/main/snapshot_waves_20191216.png" width="500">
</p>

**Figure 1.** Horizontal winds (vectors) and geopotential height (color shading) at 850 hPa from NCEP-NCAR daily data projected onto different types of equatorial waves during the period of 16 December 2019. The trapping scale used for the parabolic cylinder functions is <img src="https://render.githubusercontent.com/render/math?math=y_{0}=\left(c/2\beta\right)^{1/2}"> = 6<sup>o</sup>, corresponding to <img src="https://render.githubusercontent.com/render/math?math=c"> = 20 m/s for <img src="https://render.githubusercontent.com/render/math?math=\beta"> = 2.3 <img src="https://render.githubusercontent.com/render/math?math=\times"> 10<sup>-11</sup> m<sup>-1</sup> s<sup>-1</sup>. The data is expanded over 20<sup>o</sup>N-20<sup>o</sup>S and are constrained for <img src="https://render.githubusercontent.com/render/math?math=k"> = 2-20, period = 3-30 days. Units are m/s for winds and m for geopotential height.
