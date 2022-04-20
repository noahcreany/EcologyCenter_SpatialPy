import json
import os
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import arcpy
from arcgis import GIS


def _get_symbol(geom, **kwargs):
    point_sym = dict(type='esriSMS', style='esriSMSCircle', color=[255, 0, 0, 255], size=12)
    line_sym = dict(type='esriSLS', style='esriSLSSolid', color=[255, 0, 0, 255], width=2)
    poly_sym = dict(type='esriSFS', style='esriSFSSolid', color=[255, 0, 0, 150], outline=line_sym)
    if geom.type in ['point', 'multipoint']:
        symbol = point_sym
    elif geom.type == 'polyline':
        symbol = line_sym
    elif geom.type == 'polygon':
        symbol = poly_sym
    symbol.update(kwargs)
    return symbol

def _get_geom(geom, wkid):
    geom = json.loads(geom.JSON)
    if not geom['spatialReference']['wkid'] and wkid:
        geom['spatialReference']['wkid'] = wkid
    return geom

def make_map(place, geom, zoom, wkid=4326, symbol=None, **kwargs):
    symbol = symbol or _get_symbol(geom, **kwargs)
    geom = _get_geom(geom, wkid)
    m = GIS().map(place, zoomlevel=zoom)
    m.draw(geom, symbol=symbol)
    return m

def add_geom(m, geom, wkid=4326, symbol=None, **kwargs):
    symbol = symbol or _get_symbol(geom, **kwargs)
    geom = _get_geom(geom, wkid)
    m.draw(geom, symbol=symbol)


def expand_extent(extent, percent=0.1):
    extent = get_extent(extent)
    xdelta = (extent.XMax-extent.XMin) * percent
    ydelta = (extent.YMax-extent.YMin) * percent
    return arcpy.Extent(extent.XMin-xdelta, extent.YMin-ydelta, extent.XMax+xdelta, extent.YMax+ydelta)


def get_extent(data):
    try:
        if isinstance(data, arcpy.Extent):
            return data
        elif isinstance(data, str):
            return arcpy.Describe(data).extent
        elif isinstance(data, arcpy.Geometry):
            return data.extent
        else:
            print('ERROR: Cannot get extent for {} objects.'.format(type(data).__name__))
            return None
    except Exception as e:
        print('ERROR: Cannot get extent for {}: {}'.format(data, e))
        return None


def is_path(data):
    return isinstance(data, str) and os.path.exists(data)


def plot(data, symbols=None, extent=None, margin=0.2, **kwargs):
    if not isinstance(data, list):
        data = [data]
    if not symbols and len(data) <= len(Plotter.colors):
        symbols = Plotter.colors[:len(data)]
    elif not symbols or len(symbols) != len(data):
        print('ERROR: You must provide a symbol for each item to be plotted.')
        return
    plotter = Plotter()
    plotter.plot(data, symbols, margin, extent, **kwargs)


def table2pd(table):
    """Puts attribute table data in a Pandas dataframe."""
    with arcpy.da.SearchCursor(table, '*') as rows:
        return pd.DataFrame(data=list(rows), columns=rows.fields)


class Limits(object):
    def __init__(self, data=None):
        self.data = self._extent_to_list(data) if data else None

    def expand(self, percent):
        xdelta = (self.data[1] - self.data[0]) * percent
        ydelta = (self.data[3] - self.data[2]) * percent
        self.data = [self.data[0] - xdelta, self.data[1] + xdelta,
                     self.data[2] - ydelta, self.data[3] + ydelta]

    def union(self, extent):
        if self.data:
            limits = self._extent_to_list(extent)
            self.data = [min(self.data[0], limits[0]), max(self.data[1], limits[1]),
                         min(self.data[2], limits[2]), max(self.data[3], limits[3])]
        else:
            self.data = self._extent_to_list(extent)

    def _extent_to_list(self, extent):
        return [extent.XMin, extent.XMax, extent.YMin, extent.YMax]

class Plotter(object):

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    symbols = ['-', '--', '-.', ':', '.', ',', 'o', 'v', '^', '<', '>',
               '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x',
               'D', 'd', '|', '_']

    def __init__(self):
        self.fig, self.ax = plt.subplots()

    def plot(self, data, symbols, margin, extent=None, **kwargs):
        if extent and get_extent(extent):
            limits = Limits(get_extent(extent))
        else:
            limits = Limits()
        for d, s in zip(data, symbols):
            try:
                if isinstance(d, str):
                    self._plot_feature_class(d, s, **kwargs)
                elif isinstance(d, arcpy.Geometry):
                    self._plot_geom(d, s, **kwargs)
                elif isinstance(d, arcpy.Extent):
                    self._plot_extent(d, s, **kwargs)
                else:
                    print('ERROR: Plotting {} objects is not supported'.format(type(d).__name__))
                    continue
                if not extent:
                    limits.union(get_extent(d))
            except Exception as e:
                print('Could not plot {}: {}'.format(d, e))
        limits.expand(margin)
        self.ax.axis('equal')
        self.ax.set_xlim(*limits.data[:2])
        self.ax.set_ylim(*limits.data[2:])
        plt.show()

    def _get_polygon_rings(self, poly):
        """Returns polygon rings as lists of coordinates."""
        rings = []
        ring = []
        for pt in poly:
            if pt is not None:
                ring.append((pt.X, pt.Y))
            elif ring:
                rings.append(ring)
                ring = []
        rings.append(ring)
        return rings

    def _make_codes(self, n):
        """Makes a list of path codes."""
        codes = [Path.LINETO] * n
        codes[0] = Path.MOVETO
        return codes

    def _order_coords(self, coords, clockwise):
        """Orders coordinates."""
        total = 0
        x1, y1 = coords[0]
        for x, y in coords[1:]:
            total += (x - x1) * (y + y1)
            x1, y1 = x, y
        x, y = coords[0]
        total += (x - x1) * (y + y1)
        is_clockwise = total > 0
        if clockwise != is_clockwise:
            coords.reverse()
        return coords

    def _plot(self, data, symbol, **kwargs):
        if is_path(data):
            self._plot_feature_class(data, symbol, **kwargs)
        elif isinstance(data, arcpy.Geometry):
            self.plot_geom(data, symbol, **kwargs)
        elif isinstance(data, arcpy.Extent):
            self._plot_extent(data, symbol, **kwargs)
        self.limits.expand(get_extent(data))

    def _plot_extent(self, extent, symbol, **kwargs):
        x = [extent.XMin, extent.XMin, extent.XMax, extent.XMax, extent.XMin]
        y = [extent.YMin, extent.YMax, extent.YMax, extent.YMin, extent.YMin]
        self.ax.plot(x, y, symbol, **kwargs)

    def _plot_geom(self, geom, symbol, **kwargs):
        if geom.type == 'point':
            self._plot_point_geom(geom, symbol, **kwargs)
        elif geom.type == 'polyline':
            symbol = symbol or next(self.ax._get_lines.color_cycle) + '-'
            for i in range(geom.partCount):
                self._plot_line(geom.getPart(i), symbol, **kwargs)
        elif geom.type == 'polygon':
            for i in range(geom.partCount):
                self._plot_polygon(geom.getPart(i), symbol, **kwargs)
        elif geom.type == 'multipoint':
            symbol = symbol or next(self.ax._get_lines.color_cycle) + 'o'
            for i in range(geom.partCount):
                self._plot_point(geom.getPart(i), symbol, **kwargs)

    def _plot_feature_class(self, fn, symbol, **kwargs):
        with arcpy.da.SearchCursor(fn, 'SHAPE@') as rows:
            for row in rows:
                self._plot_geom(row[0], symbol, **kwargs)

    def _plot_line(self, line, symbol, **kwargs):
        """Plots a line."""
        x, y = zip(*[(pt.X, pt.Y) for pt in line])
        self.ax.plot(x, y, symbol, **kwargs)

    def _plot_point(self, pt, symbol, **kwargs):
        """Plots a point."""
        if not any([s in symbol for s in self.symbols]):
            symbol += 'o'
        self.ax.plot(pt.X, pt.Y, symbol, **kwargs)

    def _plot_point_geom(self, point, symbol, **kwargs):
        """Plots a point geometry."""
        self._plot_point(point.getPart(0), symbol, **kwargs)

    def _plot_polygon(self, poly, color, **kwargs):
        """Plots a polygon as a patch."""
        rings = self._get_polygon_rings(poly)

        # Outer clockwise path.
        coords = self._order_coords(rings[0], True)
        codes = self._make_codes(len(coords))

        # Inner counter-clockwise paths.
        for ring in rings[1:]:
            inner_coords = self._order_coords(ring, False)
            inner_codes = self._make_codes(len(inner_coords))

            # Concatenate the paths.
            coords = np.concatenate((coords, inner_coords))
            codes = np.concatenate((codes, inner_codes))

        # Add the patch to the plot
        path = Path(coords, codes)
        if color:
            patch = patches.PathPatch(path, facecolor=color, **kwargs)
        else:
            patch = patches.PathPatch(path, **kwargs)
        self.ax.add_patch(patch)
