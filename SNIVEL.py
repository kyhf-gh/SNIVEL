#!/usr/bin/env python
import numpy
import math
import georinex as gr
import SNIVEL_orbits
from SNIVEL_filedownloader import getbcorbit
from SNIVEL_tools import (gpsweekdow, dxyz2dneu, ecef2lla_optimized,
                          azi_elev_optimized, klobuchar_optimized,
                          getklobucharvalues_optimized, niell_optimized,
                          niell_wet_optimized, writesac, doy_calc)
import os
from scipy.optimize import lsq_linear


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_and_prepare(site, year, doy, samprate, event='post'):
    ensure_dir('output')
    ensure_dir(f'output/sac')
    ensure_dir(f'output/sacfilt')
    ensure_dir(f'output/fig')
    outfile = f'output/observables_{site}_{doy}_{year}.txt'
    veloutfile = f'output/velocities_{site}_{doy}_{year}.txt'
    navfile = f'nav/brdc{doy}0.{str(year)[-2:]}n'
    pngfile = f'output/fig/{event}_vel_{site}_{year}_{doy}.png'
    sr = "{0:.2f}".format(float(samprate))
    sacnfile = f'output/sacfilt/{event}_{site.upper()}.{sr}.filt.LXN.sac'
    sacefile = f'output/sacfilt/{event}_{site.upper()}.{sr}.filt.LXE.sac'
    saczfile = f'output/sacfilt/{event}_{site.upper()}.{sr}.filt.LXZ.sac'
    getbcorbit(str(year), str(doy))
    return outfile, veloutfile, navfile, pngfile, sacnfile, sacefile, saczfile


def gnss_vel_cal(rinex_file_path):
    obsfile = rinex_file_path
    site = os.path.basename(obsfile)[0:4]
    header = gr.rinexheader(rinex_file_path)
    samfreq = float(header['interval'])
    t0 = header['t0']
    year, month, day = t0.year, t0.month, t0.day
    doy = doy_calc(year, month, day)
    outfile, veloutfile, navfile, pngfile, sacnfile, sacefile, saczfile = download_and_prepare(
        site, year, doy, samfreq)
    [gpsweek, gpsdow] = gpsweekdow(int(year), int(doy))

    c = 299792458.0
    fL1 = 1575.42e6
    fL2 = 1227.60e6
    wL1 = c / fL1
    wL2 = c / fL2
    elevmask = 10
    clockdrift = 1e-7
    maxvel = 1

    try:
        (x0, y0, z0) = header['position']
        sampersec = 1 / samfreq
        [latsta, lonsta, altsta] = ecef2lla_optimized(float(x0), float(y0),
                                                      float(z0))
        nav = gr.load(navfile)
        [alpha, beta] = getklobucharvalues_optimized(navfile)
        obs = gr.load(obsfile)
        L1 = obs['L1'].values
        L2 = obs['L2'].values
        obs_time = obs.time.values
        nt = len(obs_time)
        svs = obs.sv
        ns = len(svs)
        lines_to_write = []
        for i in range(0, nt):
            gps_time = (numpy.datetime64(obs_time[i]) -
                        numpy.datetime64('1980-01-06T00:00:00')
                        ) / numpy.timedelta64(1, 's')
            gps_sow = float(gps_time) - gpsweek * 604800
            for j in range(0, ns):
                sv1 = svs.sv[j]
                total_val = float(float(L2[i, j]) + float(L1[i, j]))
                if "G" not in str(sv1.values) or not math.isfinite(total_val):
                    continue
                [x, y, z, tr, rho,
                 tsv] = SNIVEL_orbits.bcorbit(nav, sv1.values, gps_sow, x0, y0,
                                              z0)
                [azi, elev] = azi_elev_optimized(x0, y0, z0, x, y, z)
                Mdry = niell_optimized(elev, latsta * 180 / math.pi, altsta,
                                       int(doy))
                Mwet = niell_wet_optimized(elev, latsta * 180 / math.pi)
                [dIon1, dIon2] = klobuchar_optimized(
                    float(latsta) * 180 / math.pi,
                    float(lonsta) * 180 / math.pi, float(elev), float(azi),
                    gps_sow, alpha, beta)
                rclock = (tr + tsv) * c
                l1 = "{0:.5f}".format(float(L1[i, j]) * wL1 + rclock)
                l2 = "{0:.5f}".format(float(L2[i, j]) * wL2 + rclock)
                rx0 = "{0:.9f}".format(float((x0 - x) / rho))
                ry0 = "{0:.9f}".format(float((y0 - y) / rho))
                rz0 = "{0:.9f}".format(float((z0 - z) / rho))
                az = "{0:.2f}".format(float(azi))
                el = "{0:.2f}".format(float(elev))
                rhotrue = "{0:.5f}".format(float(rho))
                di1 = "{0:.5f}".format(dIon1)
                di2 = "{0:.5f}".format(dIon2)
                shd = "{0:.5f}".format(Mdry)
                swd = "{0:.5f}".format(Mwet)
                gpst = "{0:.2f}".format(float(gps_time))
                gpsw = "{0:.0f}".format(float(gpsweek))
                gpss = "{0:.2f}".format(float(gps_sow))
                svstrip = str(sv1.values)[1:]
                dxsat = "{0:.6f}".format(float((x0 - x)))
                dysat = "{0:.6f}".format(float((y0 - y)))
                dzsat = "{0:.6f}".format(float((z0 - z)))
                line = f"{i} {gpst} {gpsw} {gpss} {svstrip} {rx0} {ry0} {rz0} {l1} {l2} {az} {el} {di1} {di2} {shd} {dxsat} {dysat} {dzsat} {swd}\n"
                lines_to_write.append(line)
        with open(outfile, 'w') as ffo:
            ffo.writelines(lines_to_write)
        lines_to_write = []
        a = numpy.loadtxt(outfile)
        tind = a[:, 0]
        gtime = a[:, 1]
        svind = a[:, 4]
        i1corr = a[:, 12]
        i2corr = a[:, 13]
        shdcorr = a[:, 14]

        rx = a[:, 5]
        ry = a[:, 6]
        rz = a[:, 7]
        l1 = a[:, 8] + i1corr - shdcorr
        l2 = a[:, 9] + i2corr - shdcorr

        el = a[:, 11]
        dxsat = a[:, 15]
        dysat = a[:, 16]
        dzsat = a[:, 17]
        ub = numpy.zeros([1, 4])
        lb = numpy.zeros([1, 4])
        ub[0, 0] = maxvel
        lb[0, 0] = -maxvel
        ub[0, 1] = maxvel
        lb[0, 1] = -maxvel
        ub[0, 2] = maxvel
        lb[0, 2] = -maxvel
        ub[0, 3] = clockdrift * c
        lb[0, 3] = -clockdrift * c

        tstart = numpy.amin(tind) + 1
        tstop = numpy.amax(tind)
        for i in range(int(tstart), int(tstop) + 1):
            a0 = numpy.where(tind == i - 1)[0]
            a1 = numpy.where(tind == i)[0]
            l10 = l1[a0]
            l11 = l1[a1]
            l20 = l2[a0]
            l21 = l2[a1]
            sv0 = svind[a0]
            sv1 = svind[a1]
            dxsat0 = dxsat[a0]
            dxsat1 = dxsat[a1]
            dysat0 = dysat[a0]
            dysat1 = dysat[a1]
            dzsat0 = dzsat[a0]
            dzsat1 = dzsat[a1]
            rx0 = rx[a0]
            rx1 = rx[a1]
            ry0 = ry[a0]
            ry1 = ry[a1]
            rz0 = rz[a0]
            rz1 = rz[a1]

            el1 = el[a1]

            G = list()
            W = list()
            Vdat = list()
            if ((gtime[a1[0]] - gtime[a0[0]]) < 1 / sampersec + 0.005):
                for j in range(0, len(sv1)):
                    asv = numpy.where(sv0 == sv1[j])[0]
                    if (len(asv) > 0 and el1[j] > elevmask):
                        dran0 = math.sqrt(
                            math.pow(dxsat0[int(asv)], 2) +
                            math.pow(dysat0[int(asv)], 2) +
                            math.pow(dzsat0[int(asv)], 2))
                        dran1 = math.sqrt(
                            math.pow(dxsat1[j], 2) + math.pow(dysat1[j], 2) +
                            math.pow(dzsat1[j], 2))
                        dran = dran1 - dran0
                        l1diff = l11[j] - l10[int(asv)] - dran
                        l2diff = l21[j] - l20[int(asv)] - dran
                        nl = [
                            fL1 / (fL1 + fL2) * l1diff + fL2 /
                            (fL1 + fL2) * l2diff
                        ]
                        varvalL = [nl]
                        Grow = [rx1[j], ry1[j], rz1[j], 1]
                        Wrow = [el1[j]]
                        W.append(Wrow)
                        G.append(Grow)
                        Vdat.append(nl)

                Winv = numpy.asarray(W)
                Winv = numpy.diagflat(Winv)
                Ginv = numpy.asarray(G)
                Vinv = numpy.asarray(Vdat)

                WV = numpy.matmul(numpy.matmul(numpy.transpose(Ginv), Winv),
                                  Vinv)
                GWGT = numpy.matmul(numpy.transpose(Ginv),
                                    numpy.matmul(Winv, Ginv))
                S = lsq_linear(GWGT,
                               WV.flatten(),
                               bounds=(lb.flatten(), ub.flatten()),
                               lsmr_tol='auto')
                [dn, de, du] = dxyz2dneu(S.x[0], S.x[1], S.x[2],
                                         latsta * 180 / math.pi,
                                         lonsta * 180 / math.pi)
                gpst = "{0:.2f}".format(float(gtime[a1[0]]))
                nvel = "{0:.5f}".format(float(dn) * sampersec)
                evel = "{0:.5f}".format(float(de) * sampersec)
                uvel = "{0:.5f}".format(float(du) * sampersec)

                line = f"{i} {gpst} {nvel} {evel} {uvel}\n"
                lines_to_write.append(line)

        with open(veloutfile, 'w') as ffo:
            ffo.writelines(lines_to_write)

        writesac(veloutfile, site, latsta * 180 / math.pi,
                 lonsta * 180 / math.pi, doy, year, 1 / sampersec, "post")
        return "OK", [veloutfile, pngfile, sacnfile, sacefile, saczfile]
    except Exception as e:
        return "Err", e


if __name__ == '__main__':
    print(gnss_vel_cal('rinex_hr/duow3521.23o'))

# pip install georinex
# pip install wget
# pip install obspy
