; Authors: (Original code) by Prof. Steven Woolnough and Dr. Gui-Ying Yang
;          (Modified code) by Dr. Sandro W. Lubis (Nov 2021)
;          CCEW Filter via PCFs following Yang et al., (2003)
; Contact: slubis.geomar@gmail.com
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal

ds1 = xr.open_dataset('data/anomaly/u.850.20N-20S.0-360.20160101-20201231.nc')
ds2 = xr.open_dataset('data/anomaly/v.850.20N-20S.0-360.20160101-20201231.nc')
ds3 = xr.open_dataset('data/anomaly/phi.850.20N-20S.0-360.20160101-20201231.nc')

u = ds1.u
v = ds2.v
z = ds3.phi

u.values = signal.detrend(u.values, axis=0)
v.values = signal.detrend(v.values, axis=0)
z.values = signal.detrend(z.values, axis=0)

wgt_taper = signal.tukey(u.shape[0], alpha=0.1)

uT = np.transpose(u.values, (1, 2, 0)) * np.reshape(wgt_taper, (-1, u.shape[0]))
vT = np.transpose(v.values, (1, 2, 0)) * np.reshape(wgt_taper, (-1, u.shape[0]))
zT = np.transpose(z.values, (1, 2, 0)) * np.reshape(wgt_taper, (-1, u.shape[0]))

u.values = np.transpose(uT, (2, 0, 1))
v.values = np.transpose(vT, (2, 0, 1))
z.values = np.transpose(zT, (2, 0, 1))

g     = 9.8
beta  = 2.3e-11
radea = 6.371e+06
spd   = 86400.0
ww    = 2.0 * np.pi / spd

latmax = 20.0

kmin = 2
kmax = 20

pmin = 3.0
pmax = 30.0

# convert trapping scale to meters
y0 = 6.0
y0real = 2.0 * np.pi * radea * y0 / 360.0

ce = 2.0 * y0real**2 * beta

g_on_c = g / ce
c_on_g = ce / g

waves = np.array(['Kelvin', 'WMRG', 'R1', 'R2'])

# transform u,z to q, r using q=z*(g/c) + u; r=z*(g/c) - u 

q = z * g_on_c + u
r = z * g_on_c - u

qf = np.fft.fft2(q, axes=(0, 2))
vf = np.fft.fft2(v, axes=(0, 2))
rf = np.fft.fft2(r, axes=(0, 2))

nf = qf.shape[0]
nlat = qf.shape[1]
nk = qf.shape[2]

# Find frequencies and wavenumbers corresponding to pmin,pmax and kmin,kmax in coeff matrices

f = np.fft.fftfreq(nf)
k = np.fft.fftfreq(nk) * nk

fmin = np.where(f >= 1.0 / pmax)[0][0]
fmax = (np.where(f > 1.0 / pmin)[0][0]) - 1

f1p = fmin
f2p = fmax + 1
f1n = nf - fmax
f2n = nf - fmin + 1

k1p = kmin
k2p = kmax + 1
k1n = nk - kmax
k2n = nk - kmin + 1

# Define the parobolic cylinder functions

spi2 = np.sqrt(2.0 * np.pi)
dsq = np.array([spi2, spi2, 2.0 * spi2, 6.0 * spi2])
d = np.zeros((dsq.size, nlat))
y = u.Y.values / y0
ysq = y**2

d[0, :] = np.exp(-ysq / 4.0)
d[1, :] = y * d[0, :]
d[2, :] = (ysq - 1.0) * d[0, :]
d[3, :] = y * (ysq - 3.0) * d[0, :]

dlat = np.abs(u.Y.values[1] - u.Y.values[0]) * np.pi / 180.0

qf_Kel = np.zeros((nf, nk), dtype='complex')

qf_mode = np.zeros((dsq.size, nf, nk), dtype='complex')
vf_mode = np.zeros((dsq.size, nf, nk), dtype='complex')
rf_mode = np.zeros((dsq.size, nf, nk), dtype='complex')

# reorder the spectral coefficents to make the latitudes the last dimension

qf = np.transpose(qf, (0, 2, 1))
vf = np.transpose(vf, (0, 2, 1))
rf = np.transpose(rf, (0, 2, 1))

for m in np.arange(dsq.size):
    if m == 0:
        qf_Kel[f1n:f2n, k1p:k2p] = np.sum(qf[f1n:f2n, k1p:k2p, :] * d[m, :] * dlat, axis=-1) / (dsq[m] / y0)
        qf_Kel[f1p:f2p, k1n:k2n] = np.sum(qf[f1p:f2p, k1n:k2n, :] * d[m, :] * dlat, axis=-1) / (dsq[m] / y0)
    
    qf_mode[m, f1n:f2n, k1n:k2n] = np.sum(qf[f1n:f2n, k1n:k2n, :] * d[m, :] * dlat, axis=-1) / (dsq[m] / y0)
    qf_mode[m, f1p:f2p, k1p:k2p] = np.sum(qf[f1p:f2p, k1p:k2p, :] * d[m, :] * dlat, axis=-1) / (dsq[m] / y0)
    
    vf_mode[m, f1n:f2n, k1n:k2n] = np.sum(vf[f1n:f2n, k1n:k2n, :] * d[m, :] * dlat, axis=-1) / (dsq[m] / y0)
    vf_mode[m, f1p:f2p, k1p:k2p] = np.sum(vf[f1p:f2p, k1p:k2p, :] * d[m, :] * dlat, axis=-1) / (dsq[m] / y0)
    
    rf_mode[m, f1n:f2n, k1n:k2n] = np.sum(rf[f1n:f2n, k1n:k2n, :] * d[m, :] * dlat, axis=-1) / (dsq[m] / y0)
    rf_mode[m, f1p:f2p, k1p:k2p] = np.sum(rf[f1p:f2p, k1p:k2p, :] * d[m, :] * dlat, axis=-1) / (dsq[m] / y0)

uf_wave = np.zeros((waves.size, nf, nlat, nk), dtype='complex')
vf_wave = np.zeros((waves.size, nf, nlat, nk), dtype='complex')
zf_wave = np.zeros((waves.size, nf, nlat, nk), dtype='complex')

for w in np.arange(waves.size):
    if waves[w] == 'Kelvin':
        for j in np.arange(nlat):
            uf_wave[w, :, j, :] = 0.5 * qf_Kel * d[0, j]
            zf_wave[w, :, j, :] = 0.5 * qf_Kel * d[0, j] * c_on_g
    
    if waves[w] == 'WMRG':
        for j in np.arange(nlat):
            uf_wave[w, :, j, :] = 0.5 * qf_mode[1, :, :] * d[1, j]
            vf_wave[w, :, j, :] = 0.5 * vf_mode[0, :, :] * d[0, j]
            zf_wave[w, :, j, :] = 0.5 * qf_mode[1, :, :] * d[1, j] * c_on_g
    
    if waves[w] == 'R1':
        for j in np.arange(nlat):
            uf_wave[w, :, j, :] = 0.5 * (qf_mode[2, :, :] * d[2, j] - rf_mode[0, :, :] * d[0, j])
            vf_wave[w, :, j, :] = 0.5 * vf_mode[1, :, :] * d[1, j]
            zf_wave[w, :, j, :] = 0.5 * (qf_mode[2, :, :] * d[2, j] + rf_mode[0, :, :] * d[0, j]) * c_on_g
    
    if waves[w] == 'R2':
        for j in np.arange(nlat):
            uf_wave[w, :, j, :] = 0.5 * (qf_mode[3, :, :] * d[3, j] - rf_mode[1, :, :] * d[1, j])
            vf_wave[w, :, j, :] = 0.5 * vf_mode[2, :, :] * d[2, j]
            zf_wave[w, :, j, :] = 0.5 * (qf_mode[3, :, :] * d[3, j] + rf_mode[1, :, :] * d[1, j]) * c_on_g

u_Kelvin = xr.DataArray(np.real(np.fft.ifft2(uf_wave[0, :, :, :], axes=(0, 2))), coords=[ds1.T, ds1.Y, ds1.X], dims=['T', 'Y', 'X'], name='u')
v_Kelvin = xr.DataArray(np.real(np.fft.ifft2(vf_wave[0, :, :, :], axes=(0, 2))), coords=[ds1.T, ds1.Y, ds1.X], dims=['T', 'Y', 'X'], name='v')
z_Kelvin = xr.DataArray(np.real(np.fft.ifft2(zf_wave[0, :, :, :], axes=(0, 2))), coords=[ds1.T, ds1.Y, ds1.X], dims=['T', 'Y', 'X'], name='z')

u_Kelvin.attrs['long_name'] = 'Kelvin Waves in 850 hPa Zonal Wind'
v_Kelvin.attrs['long_name'] = 'Kelvin Waves in 850 hPa Meridional Wind'
z_Kelvin.attrs['long_name'] = 'Kelvin Waves in 850 hPa Geopotential Height'

u_Kelvin.attrs['units'] = 'm/s'
v_Kelvin.attrs['units'] = 'm/s'
z_Kelvin.attrs['units'] = 'm'

u_WMRG = xr.DataArray(np.real(np.fft.ifft2(uf_wave[1, :, :, :], axes=(0, 2))), coords=[ds1.T, ds1.Y, ds1.X], dims=['T', 'Y', 'X'], name='u')
v_WMRG = xr.DataArray(np.real(np.fft.ifft2(vf_wave[1, :, :, :], axes=(0, 2))), coords=[ds1.T, ds1.Y, ds1.X], dims=['T', 'Y', 'X'], name='v')
z_WMRG = xr.DataArray(np.real(np.fft.ifft2(zf_wave[1, :, :, :], axes=(0, 2))), coords=[ds1.T, ds1.Y, ds1.X], dims=['T', 'Y', 'X'], name='z')

u_WMRG.attrs['long_name'] = 'Westward Mixed Rossby-Gravity Waves in 850 hPa Zonal Wind'
v_WMRG.attrs['long_name'] = 'Westward Mixed Rossby-Gravity Waves in 850 hPa Meridional Wind'
z_WMRG.attrs['long_name'] = 'Westward Mixed Rossby-Gravity Waves in 850 hPa Geopotential Height'

u_WMRG.attrs['units'] = 'm/s'
v_WMRG.attrs['units'] = 'm/s'
z_WMRG.attrs['units'] = 'm'

u_R1 = xr.DataArray(np.real(np.fft.ifft2(uf_wave[2, :, :, :], axes=(0, 2))), coords=[ds1.T, ds1.Y, ds1.X], dims=['T', 'Y', 'X'], name='u')
v_R1 = xr.DataArray(np.real(np.fft.ifft2(vf_wave[2, :, :, :], axes=(0, 2))), coords=[ds1.T, ds1.Y, ds1.X], dims=['T', 'Y', 'X'], name='v')
z_R1 = xr.DataArray(np.real(np.fft.ifft2(zf_wave[2, :, :, :], axes=(0, 2))), coords=[ds1.T, ds1.Y, ds1.X], dims=['T', 'Y', 'X'], name='z')

u_R1.attrs['long_name'] = 'n = 1 Equatorial Rossby Waves in 850 hPa Zonal Wind'
v_R1.attrs['long_name'] = 'n = 1 Equatorial Rossby Waves in 850 hPa Meridional Wind'
z_R1.attrs['long_name'] = 'n = 1 Equatorial Rossby Waves in 850 hPa Geopotential Height'

u_R1.attrs['units'] = 'm/s'
v_R1.attrs['units'] = 'm/s'
z_R1.attrs['units'] = 'm'

u_R2 = xr.DataArray(np.real(np.fft.ifft2(uf_wave[3, :, :, :], axes=(0, 2))), coords=[ds1.T, ds1.Y, ds1.X], dims=['T', 'Y', 'X'], name='u')
v_R2 = xr.DataArray(np.real(np.fft.ifft2(vf_wave[3, :, :, :], axes=(0, 2))), coords=[ds1.T, ds1.Y, ds1.X], dims=['T', 'Y', 'X'], name='v')
z_R2 = xr.DataArray(np.real(np.fft.ifft2(zf_wave[3, :, :, :], axes=(0, 2))), coords=[ds1.T, ds1.Y, ds1.X], dims=['T', 'Y', 'X'], name='z')

u_R2.attrs['long_name'] = 'n = 2 Equatorial Rossby Waves in 850 hPa Zonal Wind'
v_R2.attrs['long_name'] = 'n = 2 Equatorial Rossby Waves in 850 hPa Meridional Wind'
z_R2.attrs['long_name'] = 'n = 2 Equatorial Rossby Waves in 850 hPa Geopotential Height'

u_R2.attrs['units'] = 'm/s'
v_R2.attrs['units'] = 'm/s'
z_R2.attrs['units'] = 'm'

u_Kelvin.to_netcdf('data/waves/u.850.kelvin.nc')
v_Kelvin.to_netcdf('data/waves/v.850.kelvin.nc')
z_Kelvin.to_netcdf('data/waves/z.850.kelvin.nc')

u_WMRG.to_netcdf('data/waves/u.850.wmrg.nc')
v_WMRG.to_netcdf('data/waves/v.850.wmrg.nc')
z_WMRG.to_netcdf('data/waves/z.850.wmrg.nc')

u_R1.to_netcdf('data/waves/u.850.r1.nc')
v_R1.to_netcdf('data/waves/v.850.r1.nc')
z_R1.to_netcdf('data/waves/z.850.r1.nc')

u_R2.to_netcdf('data/waves/u.850.r2.nc')
v_R2.to_netcdf('data/waves/v.850.r2.nc')
z_R2.to_netcdf('data/waves/z.850.r2.nc')

ds1.close()
ds2.close()
ds3.close()
