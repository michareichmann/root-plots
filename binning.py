# --------------------------------------------------------
#       Functions for binning histograms
# created on January 12th 2023 by M. Reichmann
# --------------------------------------------------------
import numpy as np

from .utils import choose, ufloat, is_iter, mean_sigma


def freedman_diaconis(x):
    return 2 * (np.quantile(x, .75) - np.quantile(x, .25)) / x.size ** (1 / 3)


def width(x):
    w = freedman_diaconis(x[np.isfinite(x)])
    return w if w else 3.49 * mean_sigma(x)[1].n / x.size ** (1 / 3)


def n(x):
    return int((x.max() - x.min()) / width(x))


def increase_range(low, high, fl, fh, to_int=False):
    """increases the range [low, high] by the given factors [fl] on the low end and [fh] on the high end."""
    d = abs(high - low)
    l, h = low - d * fl, high + d * fh
    return [int(l), int(np.ceil(h))] if to_int else [l, h]


def entries(h):
    return np.array([h.GetBinEntries(i) for i in range(1, h.GetNbinsX() + 1)], 'i')


def single_entries_2d(h, ix, iy, nx):
    return int((h.GetBinEntries if 'Prof' in h.ClassName() else h.GetBinContent)((nx + 2) * iy + ix))


def entries_2d(h, flat=False):
    nx, ny = h.GetNbinsX(), h.GetNbinsY()
    e = np.array([[single_entries_2d(h, ix, iy, nx) for ix in range(1, nx + 1)] for iy in range(1, ny + 1)], 'i')
    return e.flatten() if flat else e


def from_uvec(x):
    return [x.size, np.append([i.n - i.s for i in x], x[-1].n + x[-1].s).astype('d')]


def from_vec(x, centre=False):
    x = np.array(x, 'd')
    if centre:
        w0 = (x[1] - x[0])
        x = np.append(x, x[-1] + w0)
        x -= np.append(w0 / 2, np.diff(x) / 2)
    return [x.size - 1, x]


def from_p(x):
    d = x[-1] - x[-2]
    x = np.append(x, [x[-1] + d, x[-1] + 2 * d])
    return [x.size - 2, x[:-1] - np.diff(x) / 2]


def make(xmin, xmax=None, w=1, last=False, nb=None, off=0):
    bins = np.array(xmin, 'd')
    if not is_iter(xmin):
        xmax = choose(xmax, xmin + nb * w) if nb is not None else xmax
        xmin, xmax = sorted([xmin, choose(xmax, 0)])
        bins = np.arange(xmin, xmax + (w if last else 0), w, dtype='d') if nb is None else np.linspace(xmin, xmax, int(nb) + 1, endpoint=True, dtype='d')
    return [bins.size - 1, bins + off]


def make2d(x, y, wx=1, wy=1, nx=None, ny=None, last=True):
    x, y = [(v, None) if len(v) > 2 else v for v in [x, y]]
    return make(*x, wx, last, nx) + make(*y, wy, last, ny)


# ----------------------------------------
# region FIND
def find_range(values, lfac=.2, rfac=.2, q=.02, lq=None):
    q = np.quantile(values[np.isfinite(values)], [choose(lq, q), 1 - q])
    return increase_range(*[min(values), max(values)] if q[0] == q[1] else q, lfac, rfac)


def find(values, lfac=.2, rfac=.2, q=.02, nbins=1, lq=None, w=None, x0=None, x1=None, r=None):
    if all([values == values[0]]):
        return [3, np.array([-.15, -.05, .05, 0.15], 'd') * values[0] + values[0]]
    w, (xmin, xmax) = choose(w, width(values) * nbins), find_range(values, lfac, rfac, q, lq) if r is None else np.array(r, 'd')
    bins = np.arange(choose(x0, xmin), choose(x1, xmax) + w, w, dtype='d')
    return [bins.size - 1, bins]


def find_2d(x, y, lfac=.2, rfac=.2, q=.02, nb=1, lq=None, w=None, x0=None):
    return sum([find(i, lfac, rfac, q, nb, lq, w, x0) for i in [x, y]], start=[])
# endregion
# ----------------------------------------


# ----------------------------------------
# region HISTOGRAM
def hn(h, axis='X'):
    return range(1, getattr(h, f'GetNbins{axis}')() + 1)


def from_hist(h, err=True, raw=False, axis='X'):
    ax = getattr(h, f'Get{axis.title()}axis')()
    if raw:
        return np.array([ax.GetBinLowEdge(i) for i in range(1, ax.GetNbins() + 2)], 'd')
    return np.array([ufloat(ax.GetBinCenter(ibin), ax.GetBinWidth(ibin) / 2) if err else ax.GetBinCenter(ibin) for ibin in range(1, ax.GetNbins() + 1)])


def hx(h, err=True):
    return from_hist(h, err, axis='X')


def hy(h, err=True):
    return from_hist(h, err, axis='Y')


def h2d(h, arr=False):
    x, y = [from_hist(h, raw=True, axis=ax) for ax in ['X', 'Y']]
    return [x, y] if arr else make2d(x, y)


def h2dgrid(h):
    x, y = [from_hist(h, err=False, raw=False, axis=ax) for ax in ['X', 'Y']]
    return np.array([[ix, iy] for iy in y for ix in x]).T


def set_2d_values(h, arr):
    [h.SetBinContent(ix + 1, iy + 1, arr[iy, ix]) for ix in range(arr.shape[1]) for iy in range(arr.shape[0])]


def set_2d_entries(h, arr):
    ny, nx = arr.shape
    [h.SetBinEntries((nx + 2) * (iy + 1) + (ix + 1), arr[iy, ix]) for ix in range(nx) for iy in range(ny)]
# endregion HISTOGRAM
# ----------------------------------------
