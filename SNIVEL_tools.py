#!/usr/bin/env python
import datetime
import calendar
import math
import georinex as gr
import obspy
import scipy
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
from obspy.io.sac import SACTrace
import numpy as np
#####################################################################################
#SNIVEL_tools.py
#Written by Brendan Crowell, University of Washington
#Last edited January 11, 2019
#####################################################################################
c = 299792458.0  #speed of light
fL1 = 1575.42e6  #L1 frequency
fL2 = 1227.60e6  #L2 frequency


# Print iterations progress
def printProgressBar(iteration,
                     total,
                     prefix='',
                     suffix='',
                     decimals=1,
                     length=100,
                     fill='â–ˆ',
                     printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def month_converter(month):
    months = [
        'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT',
        'NOV', 'DEC'
    ]
    return months.index(month) + 1


def doy2month(doy, year):
    isleap = calendar.isleap(year)
    if str(isleap) == 'True':
        dom = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        dom = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    dayind = int(dom[0])
    monind = 1
    for i in range(1, 12):
        if (int(doy) < dayind):
            month = monind
        else:
            dayind = dayind + dom[i]
            monind = monind + 1

    return month


def doy_calc(year, month, day):
    isleap = calendar.isleap(year)
    if str(isleap) == 'True':
        dom = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        dom = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    doy = int(np.sum(dom[:(month - 1)]) + day)
    return (doy)


def gpsweekdow(year, doy):
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)
    gpstime = (np.datetime64(date) -
               np.datetime64('1980-01-06T00:00:00')) / np.timedelta64(1, 's')
    gpsweek = int(gpstime / 604800)
    gpsdow = math.floor((gpstime - gpsweek * 604800) / 86400)
    return (gpsweek, gpsdow)


def gpsleapsec(gpssec):
    leaptimes = np.array([
        46828800, 78364801, 109900802, 173059203, 252028804, 315187205,
        346723206, 393984007, 425520008, 457056009, 504489610, 551750411,
        599184012, 820108813, 914803214, 1025136015, 1119744016, 1167264017
    ])
    leapseconds = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    a1 = np.where(gpssec > leaptimes)[0]
    leapsec = len(a1)
    return (leapsec)


def ecef2lla(x, y, z):
    a = 6378137
    e = 8.1819190842622e-2

    b = math.sqrt(math.pow(a, 2) * (1 - math.pow(e, 2)))
    ep = math.sqrt((math.pow(a, 2) - math.pow(b, 2)) / math.pow(b, 2))
    p = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
    th = math.atan2(a * z, b * p)
    lon = math.atan2(y, x)
    lat = math.atan2((z + math.pow(ep, 2) * b * math.pow(np.sin(th), 3)),
                     (p - math.pow(e, 2) * a * math.pow(np.cos(th), 3)))
    N = a / math.sqrt(1 - math.pow(e, 2) * math.pow(np.sin(lat), 2))
    alt = p / np.cos(lat) - N

    return (lat, lon, alt)


def ecef2lla_optimized(x, y, z):
    a = 6378137
    e = 8.1819190842622e-2
    e_squared = e**2  # Avoid repeated calculation

    b = a * np.sqrt(1 - e_squared)
    ep_squared = (a**2 - b**2) / b**2
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep_squared * b * np.sin(th)**3,
                     p - e_squared * a * np.cos(th)**3)
    N = a / np.sqrt(1 - e_squared * np.sin(lat)**2)
    alt = p / np.cos(lat) - N

    return (lat, lon, alt)


def azi_elev(xsta, ysta, zsta, xsat, ysat, zsat):
    [latsta, lonsta, altsta] = ecef2lla(xsta, ysta, zsta)
    [latsat, lonsat, altsat] = ecef2lla(xsat, ysat, zsat)
    re = math.sqrt(math.pow(xsta, 2) + math.pow(ysta, 2) + math.pow(zsta, 2))
    rs = math.sqrt(math.pow(xsat, 2) + math.pow(ysat, 2) + math.pow(zsat, 2))
    gamma = math.acos(
        np.cos(latsta) * np.cos(latsat) * np.cos(lonsat - lonsta) +
        np.sin(latsta) * np.sin(latsat))
    elev = math.acos(
        np.sin(gamma) /
        math.sqrt(1 + math.pow(re / rs, 2) - 2 * re / rs * np.cos(gamma)))

    deltalon = lonsat - lonsta

    azi = math.atan2(
        np.sin(deltalon) * np.cos(latsat),
        np.cos(latsta) * np.sin(latsat) -
        np.sin(latsta) * np.cos(latsat) * np.cos(deltalon))

    azi = azi * 180 / np.pi

    if (azi < 0):
        azi = azi + 360
    elev = elev * 180 / np.pi
    return (azi, elev)


def azi_elev_optimized(xsta, ysta, zsta, xsat, ysat, zsat):
    # 使用优化后的ecef2lla函数
    [latsta, lonsta, _] = ecef2lla_optimized(xsta, ysta, zsta)
    [latsat, lonsat, _] = ecef2lla_optimized(xsat, ysat, zsat)

    re = np.sqrt(xsta**2 + ysta**2 + zsta**2)
    rs = np.sqrt(xsat**2 + ysat**2 + zsat**2)

    gamma = np.arccos(
        np.cos(latsta) * np.cos(latsat) * np.cos(lonsat - lonsta) +
        np.sin(latsta) * np.sin(latsat))

    elev = np.arccos(
        np.sin(gamma) / np.sqrt(1 +
                                (re / rs)**2 - 2 * re / rs * np.cos(gamma)))

    deltalon = lonsat - lonsta

    azi = np.arctan2(
        np.sin(deltalon) * np.cos(latsat),
        np.cos(latsta) * np.sin(latsat) -
        np.sin(latsta) * np.cos(latsat) * np.cos(deltalon))

    # 将角度转换为度数并确保范围在0到360度之间
    azi = np.degrees(azi) % 360
    elev = np.degrees(elev)

    return (azi, elev)


def getklobucharvalues(navfile):
    navhead = gr.rinexheader(
        navfile)  #load navigation header to obtain klobuchar
    alpha = navhead['ION ALPHA']  #klobuchar alphas
    beta = navhead['ION BETA']  #klobuchar betas
    alpha2 = alpha.replace(
        "D", "E")  #nav headers sometimes use D instead of E for power
    beta2 = beta.replace("D", "E")
    alp1 = np.asarray(alpha2.split())
    bet1 = np.asarray(beta2.split())
    alp = alp1.astype(np.float64)  #np array of klobuchar alphas
    bet = bet1.astype(np.float64)  #np array of klobuchar betas
    return (alp, bet)


def getklobucharvalues_optimized(navfile):
    try:
        navhead = gr.rinexheader(navfile)  # Load navigation header
        # Extract Klobuchar alpha and beta values, replace 'D' with 'E' and split
        alpha = navhead['ION ALPHA'].replace("D", "E").split()
        beta = navhead['ION BETA'].replace("D", "E").split()

        # Convert to numpy arrays of type float64
        alp = np.array(alpha, dtype=np.float64)
        bet = np.array(beta, dtype=np.float64)

        return (alp, bet)
    except KeyError as e:
        raise ValueError(f"Missing required data in RINEX header: {e}")


#Klobuchar ionospheric correction


def klobuchar(latsta, lonsta, elev, azimuth, tow, alpha, beta):
    a1 = float(alpha[0])
    a2 = float(alpha[1])
    a3 = float(alpha[2])
    a4 = float(alpha[3])
    b1 = float(beta[0])
    b2 = float(beta[1])
    b3 = float(beta[2])
    b4 = float(beta[3])

    elev = elev / 180  #elevation angle in semicircles
    azimuth = azimuth * np.pi / 180

    psi = 0.0137 / (elev + 0.11) - 0.022  #earth centered angle

    lat_i = latsta / 180 + psi * np.cos(azimuth)  #subionospheric latitude

    if (lat_i > 0.416):
        lat_i = 0.416
    if (lat_i < -0.416):
        lat_i = -0.416

    lon_i = lonsta / 180 + psi * np.sin(azimuth) / np.cos(
        lat_i * np.pi)  #subionospheric longitude

    lat_m = lat_i + 0.064 * np.cos((lon_i - 1.617) * np.pi)  #geomagnetic lat

    t = 4.32e4 * lon_i + tow
    t = t % 86400
    if (t > 86400):
        t = t - 86400
    if (t < 0):
        t = t + 86400

    sF = 1 + 16 * math.pow(0.53 - elev, 3)  #slant factor

    PER = b1 + b2 * lat_m + b3 * math.pow(lat_m, 2) + b4 * math.pow(lat_m, 3)

    if (PER < 72000):
        PER = 72000

    x = 2 * np.pi * (t - 50400) / PER

    AMP = a1 + a2 * lat_m + a3 * math.pow(lat_m, 2) + a4 * math.pow(lat_m, 3)

    if (AMP < 0):
        AMP = 0

    if (abs(x) > 1.57):
        dIon1 = sF * (5e-9)
    else:
        dIon1 = sF * (5e-9 + AMP *
                      (1 - math.pow(x, 2) / 2 + math.pow(x, 4) / 24))

    dIon1 = c * dIon1

    dIon2 = math.pow(fL1 / fL2, 2) * dIon1

    return (dIon1, dIon2)


def klobuchar_optimized(latsta, lonsta, elev, azimuth, tow, alpha, beta):
    a = np.array(alpha, dtype=float)
    b = np.array(beta, dtype=float)

    elev_rad = np.deg2rad(elev)  # Convert elevation to radians
    azimuth_rad = np.deg2rad(azimuth)  # Convert azimuth to radians

    psi = 0.0137 / (elev_rad / np.pi +
                    0.11) - 0.022  # Earth centered angle in radians

    lat_i = latsta + psi * np.cos(azimuth_rad)  # Subionospheric latitude
    lat_i = np.clip(lat_i, -0.416, 0.416)  # Clip to the range [-0.416, 0.416]

    lon_i = lonsta + (psi * np.sin(azimuth_rad) / np.cos(lat_i * np.pi)
                      )  # Subionospheric longitude

    lat_m = lat_i + 0.064 * np.cos(
        (lon_i - 1.617) * np.pi)  # Geomagnetic latitude

    t = (4.32e4 * lon_i + tow) % 86400  # Time adjusted to be within 0 to 86400

    sF = 1 + 16 * (0.53 - elev_rad / np.pi)**3  # Slant factor

    PER = np.polyval(b[::-1], lat_m)  # Evaluate polynomial for PER
    PER = np.maximum(PER, 72000)

    x = 2 * np.pi * (t - 50400) / PER

    AMP = np.polyval(a[::-1], lat_m)  # Evaluate polynomial for AMP
    AMP = np.maximum(AMP, 0)

    if np.abs(x) > 1.57:
        dIon1 = sF * 5e-9
    else:
        dIon1 = sF * (5e-9 + AMP * (1 - x**2 / 2 + x**4 / 24))

    # Assuming c and fL1/fL2 are defined elsewhere in the code
    dIon1 = c * dIon1
    dIon2 = (fL1 / fL2)**2 * dIon1

    return (dIon1, dIon2)


def niell(elev, lat, alt, doy):
    aht = 2.53e-5
    bht = 5.49e-3
    cht = 1.14e-3

    aavg15 = 1.2769934e-3
    bavg15 = 2.9153695e-3
    cavg15 = 62.610505e-3
    aamp15 = 0.0
    bamp15 = 0.0
    camp15 = 0.0

    aavg30 = 1.2683230e-3
    bavg30 = 2.9152299e-3
    cavg30 = 62.837393e-3
    aamp30 = 1.2709626e-5
    bamp30 = 2.1414979e-5
    camp30 = 9.0128400e-5

    aavg45 = 1.2465397e-3
    bavg45 = 2.9288445e-3
    cavg45 = 63.721774e-3
    aamp45 = 2.6523662e-5
    bamp45 = 3.0160779e-5
    camp45 = 4.3497037e-5

    aavg60 = 1.2196049e-3
    bavg60 = 2.9022565e-3
    cavg60 = 63.824265e-3
    aamp60 = 3.4000452e-5
    bamp60 = 7.2562722e-5
    camp60 = 84.795348e-5

    aavg75 = 1.2045996e-3
    bavg75 = 2.9024912e-3
    cavg75 = 64.258455e-3
    aamp75 = 4.1202191e-5
    bamp75 = 11.723375e-5
    camp75 = 170.37206e-5

    if (abs(lat) <= 15):
        aavg = aavg15
        bavg = bavg15
        cavg = cavg15
        aamp = aamp15
        bamp = bamp15
        camp = camp15

    if (abs(lat) > 15 and abs(lat) <= 30):
        amavg = 15.0 / (aavg30 - aavg15)
        aavg = (abs(lat) - 15) / amavg + aavg15
        amamp = 15.0 / (aamp30 - aamp15)
        aamp = (abs(lat) - 15) / amamp + aamp15

        bmavg = 15.0 / (bavg30 - bavg15)
        bavg = (abs(lat) - 15) / bmavg + bavg15
        bmamp = 15.0 / (bamp30 - bamp15)
        bamp = (abs(lat) - 15) / bmamp + bamp15

        cmavg = 15.0 / (cavg30 - cavg15)
        cavg = (abs(lat) - 15) / cmavg + cavg15
        cmamp = 15.0 / (camp30 - camp15)
        camp = (abs(lat) - 15) / cmamp + camp15

    if (abs(lat) > 30 and abs(lat) <= 45):
        amavg = 15.0 / (aavg45 - aavg30)
        aavg = (abs(lat) - 30) / amavg + aavg30
        amamp = 15.0 / (aamp45 - aamp30)
        aamp = (abs(lat) - 30) / amamp + aamp30

        bmavg = 15.0 / (bavg45 - bavg30)
        bavg = (abs(lat) - 30) / bmavg + bavg30
        bmamp = 15.0 / (bamp45 - bamp30)
        bamp = (abs(lat) - 30) / bmamp + bamp30

        cmavg = 15.0 / (cavg45 - cavg30)
        cavg = (abs(lat) - 30) / cmavg + cavg30
        cmamp = 15.0 / (camp45 - camp30)
        camp = (abs(lat) - 30) / cmamp + camp30

    if (abs(lat) > 45 and abs(lat) <= 60):
        amavg = 15.0 / (aavg60 - aavg45)
        aavg = (abs(lat) - 45) / amavg + aavg45
        amamp = 15.0 / (aamp60 - aamp45)
        aamp = (abs(lat) - 45) / amamp + aamp45

        bmavg = 15.0 / (bavg60 - bavg45)
        bavg = (abs(lat) - 45) / bmavg + bavg45
        bmamp = 15.0 / (bamp60 - bamp45)
        bamp = (abs(lat) - 45) / bmamp + bamp45

        cmavg = 15.0 / (cavg60 - cavg45)
        cavg = (abs(lat) - 45) / cmavg + cavg45
        cmamp = 15.0 / (camp60 - camp45)
        camp = (abs(lat) - 45) / cmamp + camp45

    if (abs(lat) > 60 and abs(lat) <= 75):
        amavg = 15.0 / (aavg75 - aavg60)
        aavg = (abs(lat) - 60) / amavg + aavg60
        amamp = 15.0 / (aamp75 - aamp60)
        aamp = (abs(lat) - 60) / amamp + aamp60

        bmavg = 15.0 / (bavg75 - bavg60)
        bavg = (abs(lat) - 60) / bmavg + bavg60
        bmamp = 15.0 / (bamp75 - bamp60)
        bamp = (abs(lat) - 60) / bmamp + bamp60

        cmavg = 15.0 / (cavg75 - cavg60)
        cavg = (abs(lat) - 60) / cmavg + cavg60
        cmamp = 15.0 / (camp75 - camp60)
        camp = (abs(lat) - 60) / cmamp + camp60

    if (abs(lat) > 75):
        aavg = aavg75
        bavg = bavg75
        cavg = cavg75
        aamp = aamp75
        bamp = bamp75
        camp = camp75

    a = aavg - aamp * np.cos(2 * np.pi * (doy - 28) / 365.25)
    b = bavg - bamp * np.cos(2 * np.pi * (doy - 28) / 365.25)
    c = cavg - camp * np.cos(2 * np.pi * (doy - 28) / 365.25)

    el = np.sin(elev * np.pi / 180)

    m = (1 + a / (1 + b / (1 + c))) / (el + a / (el + b / (el + c)))
    mh = (1 + aht / (1 + bht / (1 + cht))) / (el + aht / (el + bht /
                                                          (el + cht)))
    dm = (1 / el - mh) * alt / 1000

    Mdry = m + dm

    Tropdelay = 2.3 * math.exp(-0.116e-3 * alt) * Mdry

    return (Tropdelay)


def niell_optimized(elev, lat, alt, doy):
    # 定义常数
    aht, bht, cht = 2.53e-5, 5.49e-3, 1.14e-3
    coeffs = [
        [1.2769934e-3, 2.9153695e-3, 62.610505e-3, 0.0, 0.0, 0.0],  # 15
        [
            1.2683230e-3, 2.9152299e-3, 62.837393e-3, 1.2709626e-5,
            2.1414979e-5, 9.0128400e-5
        ],  # 30
        [
            1.2465397e-3, 2.9288445e-3, 63.721774e-3, 2.6523662e-5,
            3.0160779e-5, 4.3497037e-5
        ],  # 45
        [
            1.2196049e-3, 2.9022565e-3, 63.824265e-3, 3.4000452e-5,
            7.2562722e-5, 84.795348e-5
        ],  # 60
        [
            1.2045996e-3, 2.9024912e-3, 64.258455e-3, 4.1202191e-5,
            11.723375e-5, 170.37206e-5
        ]  # 75
    ]

    abs_lat = abs(lat)
    lat_index = int(min(max(0, (abs_lat - 15) // 15), 4))
    current_coeffs = coeffs[lat_index]

    # 插值，如果纬度在定义的范围内
    if 15 < abs_lat < 75:
        next_coeffs = coeffs[lat_index + 1]
        factor = (abs_lat - (lat_index * 15 + 15)) / 15
        for i in range(6):
            current_coeffs[i] += factor * (next_coeffs[i] - current_coeffs[i])

    aavg, bavg, cavg, aamp, bamp, camp = current_coeffs

    a = aavg - aamp * np.cos(2 * np.pi * (doy - 28) / 365.25)
    b = bavg - bamp * np.cos(2 * np.pi * (doy - 28) / 365.25)
    c = cavg - camp * np.cos(2 * np.pi * (doy - 28) / 365.25)

    el_rad = np.sin(np.radians(elev))
    m = (1 + a / (1 + b / (1 + c))) / (el_rad + a / (el_rad + b /
                                                     (el_rad + c)))
    mh = (1 + aht / (1 + bht / (1 + cht))) / (el_rad + aht / (el_rad + bht /
                                                              (el_rad + cht)))
    dm = (1 / el_rad - mh) * alt / 1000

    Mdry = m + dm
    Tropdelay = 2.3 * np.exp(-0.116e-3 * alt) * Mdry

    return Tropdelay


#This takes displacements in x, y, z and converts them to north, east up
def dxyz2dneu(dx, dy, dz, lat, lon):
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    dn = -np.sin(lat) * np.cos(lon) * dx - np.sin(lat) * np.sin(
        lon) * dy + np.cos(lat) * dz
    de = -np.sin(lon) * dx + np.cos(lon) * dy
    du = np.cos(lat) * np.cos(lon) * dx + np.cos(lat) * np.sin(
        lon) * dy + np.sin(lat) * dz
    return (dn, de, du)


def niell_wet(elev, lat):

    aavg15 = 5.8021897e-4
    bavg15 = 1.4275268e-3
    cavg15 = 4.3472961e-2

    aavg30 = 5.6794847e-4
    bavg30 = 1.5138625e-3
    cavg30 = 4.6729510e-2

    aavg45 = 5.8118019e-4
    bavg45 = 1.4572752e-3
    cavg45 = 4.3908931e-2

    aavg60 = 5.9727542e-4
    bavg60 = 1.5007428e-3
    cavg60 = 4.4626982e-2

    aavg75 = 6.1641693e-4
    bavg75 = 1.7599082e-3
    cavg75 = 5.4736038e-2

    if (abs(lat) <= 15):
        aavg = aavg15
        bavg = bavg15
        cavg = cavg15

    if (abs(lat) > 15 and abs(lat) <= 30):
        amavg = 15.0 / (aavg30 - aavg15)
        aavg = (abs(lat) - 15) / amavg + aavg15

        bmavg = 15.0 / (bavg30 - bavg15)
        bavg = (abs(lat) - 15) / bmavg + bavg15

        cmavg = 15.0 / (cavg30 - cavg15)
        cavg = (abs(lat) - 15) / cmavg + cavg15

    if (abs(lat) > 30 and abs(lat) <= 45):
        amavg = 15.0 / (aavg45 - aavg30)
        aavg = (abs(lat) - 30) / amavg + aavg30

        bmavg = 15.0 / (bavg45 - bavg30)
        bavg = (abs(lat) - 30) / bmavg + bavg30

        cmavg = 15.0 / (cavg45 - cavg30)
        cavg = (abs(lat) - 30) / cmavg + cavg30

    if (abs(lat) > 45 and abs(lat) <= 60):
        amavg = 15.0 / (aavg60 - aavg45)
        aavg = (abs(lat) - 45) / amavg + aavg45

        bmavg = 15.0 / (bavg60 - bavg45)
        bavg = (abs(lat) - 45) / bmavg + bavg45

        cmavg = 15.0 / (cavg60 - cavg45)
        cavg = (abs(lat) - 45) / cmavg + cavg45

    if (abs(lat) > 60 and abs(lat) <= 75):
        amavg = 15.0 / (aavg75 - aavg60)
        aavg = (abs(lat) - 60) / amavg + aavg60

        bmavg = 15.0 / (bavg75 - bavg60)
        bavg = (abs(lat) - 60) / bmavg + bavg60

        cmavg = 15.0 / (cavg75 - cavg60)
        cavg = (abs(lat) - 60) / cmavg + cavg60

    if (abs(lat) > 75):
        aavg = aavg75
        bavg = bavg75
        cavg = cavg75

    a = aavg
    b = bavg
    c = cavg

    el = np.sin(elev * np.pi / 180)

    m = (1 + a / (1 + b / (1 + c))) / (el + a / (el + b / (el + c)))

    Mwet = m

    return (Mwet)


def niell_wet_optimized(elev, lat):
    # 定义不同纬度下的平均值
    coeffs = [
        (5.8021897e-4, 1.4275268e-3, 4.3472961e-2),  # 15
        (5.6794847e-4, 1.5138625e-3, 4.6729510e-2),  # 30
        (5.8118019e-4, 1.4572752e-3, 4.3908931e-2),  # 45
        (5.9727542e-4, 1.5007428e-3, 4.4626982e-2),  # 60
        (6.1641693e-4, 1.7599082e-3, 5.4736038e-2)  # 75
    ]

    abs_lat = abs(lat)
    lat_index = int(min(max(0, (abs_lat - 15) // 15), 4))
    aavg, bavg, cavg = coeffs[lat_index]

    # Interpolate if latitude is between defined ranges
    if 15 < abs_lat < 75:
        next_coeffs = coeffs[lat_index + 1]
        factor = (abs_lat - (lat_index * 15 + 15)) / 15
        aavg += factor * (next_coeffs[0] - aavg)
        bavg += factor * (next_coeffs[1] - bavg)
        cavg += factor * (next_coeffs[2] - cavg)

    # 计算湿延迟映射函数
    el_rad = np.sin(np.radians(elev))
    m = (1 + aavg / (1 + bavg / (1 + cavg))) / (el_rad + aavg /
                                                (el_rad + bavg /
                                                 (el_rad + cavg)))

    return m


def writesac(velfile, site, stalat, stalon, doy, year, samprate, event):
    a = np.loadtxt(velfile)
    tind = a[:, 0]
    gtime = a[:, 1]
    leapsec = gpsleapsec(gtime[0])

    #Get the start time of the file in UTC
    date = datetime.datetime(int(year), 1,
                             1) + datetime.timedelta(int(doy) - 1)
    gpstime = (np.datetime64(date) -
               np.datetime64('1980-01-06T00:00:00')) / np.timedelta64(1, 's')
    stime = (gtime[0] - leapsec) * np.timedelta64(
        1, 's') + np.datetime64('1980-01-06T00:00:00')
    sitem = stime.item()
    #print(sitem)
    styr = sitem.year
    stdy = sitem.day
    stmon = sitem.month
    sthr = sitem.hour
    stmin = sitem.minute
    stsec = sitem.second

    nunf = a[:, 2] - np.nanmean(a[:, 2])
    eunf = a[:, 3] - np.nanmean(a[:, 3])
    uunf = a[:, 4] - np.nanmean(a[:, 4])
    t = gtime - gtime[0]
    #print(samprate)
    samplerate = float(samprate)
    bf, af = butter(4, 1 / 4 / samplerate / 0.5 * samplerate, btype='low')
    #print(1 / 4 / samplerate / 0.5 * samplerate)
    nv = filtfilt(bf, af, nunf)
    ev = filtfilt(bf, af, eunf)
    uv = filtfilt(bf, af, uunf)
    plotvelocities(event, site, t, nunf * 100, eunf * 100, uunf * 100, doy,
                   year)
    sr = "{0:.2f}".format(float(samprate))
    print('Writing SAC file ' + 'output/' + event + '.' + site + '.' + sr +
          '.LXN.sac')
    headN = {
        'kstnm': site,
        'kcmpnm': 'LXN',
        'stla': float(stalat),
        'stlo': float(stalon),
        'nzyear': int(year),
        'nzjday': int(doy),
        'nzhour': int(sthr),
        'nzmin': int(stmin),
        'nzsec': int(stsec),
        'nzmsec': int(0),
        'delta': float(samprate)
    }

    sacn = SACTrace(data=nv, **headN)
    sacn.write(f'output/sacfilt/{event}_{site.upper()}.{sr}.filt.LXN.sac')
    sacn = SACTrace(data=nunf, **headN)
    sacn.write(f'output/sac/{event}_{site.upper()}.{sr}.LXN.sac')
    print('Writing SAC file ' +
          f'output/sac/{event}_{site.upper()}.{sr}.LXN.sac')

    headE = {
        'kstnm': site,
        'kcmpnm': 'LXE',
        'stla': float(stalat),
        'stlo': float(stalon),
        'nzyear': int(year),
        'nzjday': int(doy),
        'nzhour': int(sthr),
        'nzmin': int(stmin),
        'nzsec': int(stsec),
        'nzmsec': int(0),
        'delta': float(samprate)
    }
    sace = SACTrace(data=ev, **headE)
    sace.write(f'output/sacfilt/{event}_{site.upper()}.{sr}.filt.LXE.sac')
    sace = SACTrace(data=eunf, **headE)
    sace.write(f'output/sac/{event}_{site.upper()}.{sr}.LXE.sac')

    print('Writing SAC file ' +
          f'output/sac/{event}_{site.upper()}.{sr}.LXE.sac')

    headZ = {
        'kstnm': site,
        'kcmpnm': 'LXZ',
        'stla': float(stalat),
        'stlo': float(stalon),
        'nzyear': int(year),
        'nzjday': int(doy),
        'nzhour': int(sthr),
        'nzmin': int(stmin),
        'nzsec': int(stsec),
        'nzmsec': int(0),
        'delta': float(samprate)
    }
    sacu = SACTrace(data=uv, **headZ)
    sacu.write(f'output/sacfilt/{event}_{site.upper()}.{sr}.filt.LXZ.sac')
    sacu = SACTrace(data=uunf, **headZ)
    sacu.write(f'output/sac/{event}_{site.upper()}.{sr}.LXZ.sac')
    print('Writing SAC file ' +
          f'output/sac/{event}_{site.upper()}.{sr}.LXZ.sac')


def plotvelocities(event, site, t, n, e, v, doy, year):
    fname = f'output/fig/{event}_vel_{site}_{year}_{doy}.png'
    ax1 = plt.subplot(311)
    plt.plot(t, n)
    plt.title(f"{event}_vel_{site}_{year}_{doy}")
    plt.ylabel('North (cm/s)')
    ax2 = plt.subplot(312)
    plt.plot(t, e)
    plt.ylabel('East (cm/s)')
    ax3 = plt.subplot(313)
    plt.plot(t, v)
    plt.xlabel('Seconds after OT')
    plt.ylabel('Vertical (cm/s)')
    plt.savefig(fname)
    plt.clf()
    plt.close()
    return


def getpeakvals(velfile, site, samprate, event):
    a = np.loadtxt(velfile)
    tind = a[:, 0]
    gtime = a[:, 1]
    nunf = a[:, 2] - np.nanmean(a[:, 2])
    eunf = a[:, 3] - np.nanmean(a[:, 3])
    uunf = a[:, 4] - np.nanmean(a[:, 4])
    bf, af = butter(4, 1.25 / 0.5 * 0.2, btype='low')
    nv = filtfilt(bf, af, nunf)
    ev = filtfilt(bf, af, eunf)
    uv = filtfilt(bf, af, uunf)
    peakn = np.amax(np.absolute(nv)) * 100
    peake = np.amax(np.absolute(ev)) * 100
    peaku = np.amax(np.absolute(uv)) * 100
    print(peakn, peake, peaku)
    return (peakn, peake, peaku)


#getpeakvals('output/velocities_mom0_111_2022.txt','mom0',0.2,'masachapa')

#writesac('output/velocities_boar_130_2012.txt','boar','35','-120','130','2012',30,'pig')
