from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pandas import to_datetime


def dt64todt(dt64):
    ts = to_datetime(dt64).timestamp()
    return datetime.fromtimestamp(ts)

def geo_plot(lon, lat, data, levels=None):
    fig = plt.figure(figsize=(5, 5))
    proj = ccrs.PlateCarree(central_longitude=0)
    axe:Axes = plt.axes(projection=proj)
    axe.gridlines(
        draw_labels=True, dms=True, 
        x_inline=False, y_inline=False
    )
    axe.coastlines()
    axe.add_feature(cfeature.OCEAN)
    axe.add_feature(cfeature.LAND, edgecolor='b')

    contours = axe.contourf(
        lon, lat, data, 
        levels=levels,
        extend="both",
        transform=proj
    )
    fig.colorbar(contours, shrink=0.6, pad=0.15)
    return fig

def edh_plot(lon, lat, data):
    levels = [0,10,20,30,40,50,60,70,80,90]
    return geo_plot(lon, lat, data, levels)

def u10_plot(lon, lat, data):
    levels = [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]
    return geo_plot(lon, lat, data, levels)

def v10_plot(lon, lat, data):
    levels = [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]
    return geo_plot(lon, lat, data, levels)

def t2m_plot(lon, lat, data):
    levels = [290,291,292,293,294,295,296,297,298,299,300,301,302]
    return geo_plot(lon, lat, data, levels)

def msl_plot(lon, lat, data):
    levels = [100800,101000,101200,101400,101600,101800,102000,102200]
    return geo_plot(lon, lat, data, levels)

def sst_plot(lon, lat, data):
    levels = [290,291,292,293,294,295,296,297,298,299,300,301,302]
    return geo_plot(lon, lat, data, levels)

def q2m_plot(lon, lat, data):
    levels = [0.006,0.008,0.010,0.012,0.014,0.016,0.018,0.020]
    return geo_plot(lon, lat, data, levels)

def edh_subplot(lon, lat, data, row, col):
    fig = plt.figure(
        figsize=(5*col, 5*row)
    )
    proj = ccrs.PlateCarree(central_longitude=0)
    axes = fig.subplots(
        row, col, 
        subplot_kw={'projection':proj}
    ).reshape((-1,))
    for i in range(len(axes)):
        axes[i].gridlines(
            draw_labels=True, dms=True, 
            x_inline=False, y_inline=False
        )
        axes[i].coastlines()
        axes[i].add_feature(cfeature.OCEAN)
        axes[i].add_feature(cfeature.LAND, edgecolor='b')

        levels = [0,10,20,30,40,50,60,70,80,90]
        contours = axes[i].contourf(
            lon, lat, data[i], 
            levels=levels,
            extend="both",
            transform=ccrs.PlateCarree()
        )
        fig.colorbar(
            contours, shrink=0.6,
            ax=axes[i], pad=0.15
        )
    fig.tight_layout()
    return fig

def edh_animate(lon, lat, data, time=None):
    plt.ioff()
    plt.figure(figsize=(5, 5))
    proj = ccrs.PlateCarree(central_longitude=0)
    levels = [0,10,20,30,40,50,60,70,80,90]
    for i in range(data.shape[0]):
        axe:Axes = plt.axes(projection=proj)
        # axe.set_extent([-180, 180, -90, 90])
        axe.gridlines(
            draw_labels=True, dms=True, 
            x_inline=False, y_inline=False
        )
        if time is not None:
            axe.set_title(
                dt64todt(time[i]).strftime(r"%Y-%m-%d %H")
            )
        axe.coastlines()
        axe.add_feature(cfeature.OCEAN)
        axe.add_feature(cfeature.LAND, edgecolor='b')
        axe.contourf(
            lon, lat, data[i], 
            levels=levels,
            extend="both",
            transform=proj
        )
        plt.draw()
        plt.pause(0.05)
        plt.clf()