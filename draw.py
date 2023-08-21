#!/usr/bin/env python
# --------------------------------------------------------
#       Class for all the ROOT drawing stuff
# created on February 15th 2018 by M. Reichmann (remichae@phys.ethz.ch)
# --------------------------------------------------------

from functools import partial
from inspect import signature
from typing import Any
from warnings import catch_warnings, simplefilter

from ROOT import TGraphErrors, TGaxis, TLatex, TGraphAsymmErrors, TCanvas, gStyle, TLegend, TArrow, TPad, TCutG, TLine, TPaveText, TPaveStats, TH1F, TEllipse, TColor, TProfile
from ROOT import TProfile2D, TH2F, TH3F, THStack, TMultiGraph, TPie, gROOT, TF1
from scipy.stats import binned_statistic
from screeninfo import get_monitors, Monitor, common

import numpy as np
import plotting.binning as bins
from .binning import increase_range, quantile
from .info import Info
from .utils import *


class FitRes(np.ndarray):

    def __new__(cls, f):
        return np.ndarray.__new__(cls, f.GetNpar() if 'TF1' in f.ClassName() else f.NPar(), object)

    def __init__(self, f, **kwargs):
        super().__init__(**kwargs)
        is_tf1 = 'TF1' in f.ClassName()
        self.Fit = f
        self.NPar = f.GetNpar() if is_tf1 else f.NPar()
        self.Pars = np.array([f.GetParameter(i) for i in range(self.NPar)] if is_tf1 else list(f.Parameters()))
        self.Errors = [f.GetParError(i) for i in range(self.NPar)] if is_tf1 else list(f.Errors())
        self.Names = [f.GetParName(i) if is_tf1 else f.ParName(i) for i in range(self.NPar)]
        self.vChi2 = f.GetChisquare() if is_tf1 else f.Chi2()
        self.vNdf = f.GetNDF() if is_tf1 else f.Ndf()
        self.put(np.arange(self.NPar), self.get_pars())

    def __get__(self, obj, objtype=None):
        return self.get_pars()

    def __getitem__(self, item):
        return self.get_pars()[item]

    def __repr__(self):
        return f'[{", ".join(f"{par:1.2e}" for par in self.Pars)}]'

    def __reduce__(self):
        pickled_state = super(FitRes, self).__reduce__()
        return pickled_state[0], pickled_state[1], pickled_state[2] + (self.Pars, self.Errors, self.Names, self.vChi2, self.vNdf)

    def __setstate__(self, state, *args, **kwargs):
        self.Pars, self.Errors, self.Names, self.vChi2, self.vNdf = state[-5:]
        super(FitRes, self).__setstate__(state[0:-5])

    def __call__(self, x):
        return self.Fit(x) if self.is_tf1 else warning('not implemented')

    @property
    def is_tf1(self):
        return 'TF1' in self.Fit.ClassName()

    def get_pars(self, err=True):
        return np.array([ufloat(p, e) for p, e in zip(self.Pars, self.Errors)]) if err else self.Pars

    def get_chi2(self):
        return self.vChi2 / self.vNdf

    def get_integral(self, xmin=None, xmax=None):
        xmin, xmax = choose(xmin, self.Fit.GetXmin()), choose(xmax, self.Fit.GetXmax())
        return ufloat(self.Fit.Integral(xmin, xmax), self.Fit.IntegralError(xmin, xmax))

    def Parameter(self, arg):  # noqa
        return self.Pars[arg]

    def ParError(self, arg):  # noqa
        return self.Errors[arg]

    def ParName(self, arg):  # noqa
        return self.Names[arg]

    def Chi2(self):  # noqa
        return self.vChi2

    def Ndf(self):  # noqa
        return self.vNdf


def get_color_gradient():
    stops = np.array([0., .5, 1], 'd')
    green = np.array([0. / 255., 200. / 255., 80. / 255.], 'd')
    blue = np.array([0. / 255., 0. / 255., 0. / 255.], 'd')
    red = np.array([180. / 255., 200. / 255., 0. / 255.], 'd')
    color_gradient = TColor.CreateGradientColorTable(len(stops), stops, red, green, blue, 255)
    return np.array([color_gradient + ij for ij in range(255)])


class Draw(object):

    Dir = Path(__file__).resolve().parent
    Verbose = False
    Config = None
    Monitor = None
    Res = None

    Count = {}
    Colors = get_color_gradient()
    Objects = []

    Show = True
    Title = True
    Info = None
    FillColor = 871
    Font = 42
    Solid = 1001
    Palette = 1

    DefaultStats = {'x2': None, 'y2': None, 'h': None, 'w': .3, 'entries': False, 'm': False, 'rms': False, 'all_stat': True, 'fit': False, 'center_x': False, 'center_y': False, 'form': None}
    Stats = {}

    def __init__(self, config=None, verbose=True):

        if Draw.Config is None:  # only run the setup once
            # Basics
            Draw.Verbose = verbose
            Draw.Config = Config(choose(config, default=join(Draw.Dir, 'main.ini')))

            # Settings
            Draw.Title = Draw.Config.get_value('SAVE', 'activate title', default=True)
            Draw.FillColor = Draw.Config.get_value('PLOTS', 'fill color', default=821)
            Draw.Font = Draw.Config.get_value('PLOTS', 'legend font', default=42)
            Draw.Show = Draw.Config.get_value('SAVE', 'show', default=True)
            Draw.Monitor = Draw.find_monitor()
            Draw.Res = Draw.load_resolution()
            Draw.Palette = Draw.Config.get_value('PLOTS', 'palette', default=1)

            Draw.setup()
            Draw.Info = Info(self)

        self.Info = self.init_info()
        self.IColor = 0  # color index
        self.Dic = {'TH1F': self.distribution, 'TH1I': self.distribution, 'TH1D': self.distribution,
                    'TH1': self.function,
                    'TGraph': self.graph, 'TGraphErrors': self.graph, 'TGraphAsymmErrors': self.graph,
                    'TProfile': self.profile,
                    'TH2I': self.histo_2d, 'TH2D': self.histo_2d, 'TH2F': self.histo_2d,
                    'TProfile2D': self.prof2d,
                    'TMultiGraph': self.multigraph,
                    'TF1': self.function}

    def __call__(self, th, *args, **kwargs):
        if th.ClassName() in self.Dic:
            return self.Dic[th.ClassName()](th, *args, **kwargs)
        return Draw.histo(th, *args, **kwargs)

    def __repr__(self):
        return f'ROOT {self.__class__.__name__} instance: Title = {get_stat(Draw.Title)}, Show = {get_stat(Draw.Show)}, Info = {get_stat(self.Info)}'

    @property
    def activate(self):
        Draw.Show = True
        return True
    a = activate

    @property
    def deactivate(self):
        Draw.Show = False
        return False
    d = deactivate

    # ----------------------------------------
    # region INIT
    @staticmethod
    def find_monitor():
        if Draw.Config.get_value('MONITOR', 'load', default=True):
            try:
                monitors = sorted(get_monitors(), key=lambda mon: mon.x)
                imon = Draw.Config.get_value('MONITOR', 'number', default=0)
                return monitors[imon if imon < len(monitors) else 0]
            except (common.ScreenInfoError, IndexError):
                pass
        return Monitor(0, 0, *Draw.Config.get_list('MONITOR', 'default'))

    @staticmethod
    def load_resolution():
        """ returns: default plot height in pixels."""
        h = Draw.Config.get_value('DRAW', 'plot height ndc', float, default=.7)
        return int(Draw.Monitor.height * h)

    def init_info(self):
        return Info(Draw)
    # endregion INIT
    # ----------------------------------------

    @staticmethod
    def add(*args):
        for obj in args:
            Draw.Objects.append(obj)
        Draw.clean()
        return args[0] if len(args) == 1 else args

    @staticmethod
    def clean():
        for obj in Draw.Objects:
            # if '0x(nil)' in str(obj) or obj is None:
            if obj is None:
                Draw.Objects.remove(obj)

    # ----------------------------------------
    # region SET
    @staticmethod
    def setup():
        gStyle.SetLegendFont(Draw.Font)
        gStyle.SetOptTitle(Draw.Title)
        gStyle.SetPalette(Draw.Palette)
        gStyle.SetNumberContours(Draw.Config.get_value('PLOTS', 'contours', default=20))

    @staticmethod
    def set_margin(c, side, value=None, default=.1, off=0):
        do(getattr(c, f'Set{side}Margin'), None if round(getattr(c, f'Get{side}Margin')(), 2) != .1 and value is None else max(choose(value, default) + off, 0))

    @staticmethod
    def set_pad_margins(c=None, l_=None, r=None, b=None, t=None, fix=False):
        Draw.set_margin(c, 'Left', l_, default=.13)
        Draw.set_margin(c, 'Right', r, default=.02)
        Draw.set_margin(c, 'Bottom', b, default=.116, off=.06 if Draw.Info.ShowLegend and not fix else 0)
        Draw.set_margin(c, 'Top', t, default=.02, off=.08 if Draw.Title and not fix else 0)

    @staticmethod
    def set_show(status=ON):
        set_root_output(status and Draw.Show)
    # endregion SET
    # ----------------------------------------

    # ----------------------------------------
    # region GET
    @staticmethod
    def color(n, i):
        return Draw.get_colors(n)[i]

    def get_color(self, n, i=None):
        color = Draw.get_colors(n)[choose(i, self.IColor)]
        if i is None:
            self.IColor = self.IColor + 1 if self.IColor < n - 1 else 0
        return color

    @staticmethod
    def get_colors(n):
        return Draw.Colors[np.linspace(0, Draw.Colors.size - 1, n).round().astype(int)].tolist()

    @staticmethod
    def get_count(name='a'):
        if name not in Draw.Count:
            Draw.Count[name] = -1
        Draw.Count[name] += 1
        return Draw.Count[name]

    @staticmethod
    def reset_count(name='a'):
        Draw.Count[name] = 0
    
    @staticmethod
    def get_name(string='a'):
        return '{}{}'.format(string, Draw.get_count(string))

    @staticmethod
    def get_margins(c):
        return [getattr(c, f'Get{n}Margin')() for n in ['Left', 'Right', 'Bottom', 'Top']] if c is not None else None
    # endregion GET
    # ----------------------------------------

    # ----------------------------------------
    # region DRAWING
    @staticmethod
    def canvas(title='c', x=None, y=None, w=1., h=1., logx=None, logy=None, logz=None, gridx=None, gridy=None, transp=None, divide=None, show=True):
        Draw.set_show(show)
        c0 = get_last_canvas(warn=False)
        x = x if x is not None else 0 if c0 is None else c0.GetWindowTopX() + 50
        y = y if y is not None else 0 if c0 is None else c0.GetWindowTopY() + 20
        c = TCanvas(Draw.get_name('c'), title, int(x), int(y), int(w * Draw.Res), int(h * Draw.Res))
        do([c.SetLogx, c.SetLogy, c.SetLogz], [logx, logy, logz])
        do([c.SetGridx, c.SetGridy], [gridx, gridy])
        do(make_transparent, c, transp)
        if divide is not None:
            c.Divide(*(divide if type(divide) in [list, tuple] else [divide]))
        return Draw.add(c)

    @staticmethod
    def axis(x1, x2, y1, y2, title, limits=None, name='ax', col=1, width=1, off=.15, tit_size=.035, lab_size=0.035, tick_size=0.03, line=False, opt='+SU', l_off=.01, log=False, center=None):
        limits = ([y1, y2] if x1 == x2 else [x1, x2]) if limits is None else limits
        a = TGaxis(x1, y1, x2, y2, limits[0], limits[1], 510, opt + ('G' if log else ''))
        a.SetName(name)
        a.SetLineColor(col)
        a.SetLineWidth(width)
        a.SetLabelSize(lab_size if not line else 0)
        a.SetTitleSize(tit_size)
        a.SetTitleOffset(off)
        a.SetTitle(title)
        a.SetTitleColor(col)
        a.SetLabelColor(col)
        a.SetLabelFont(Draw.Font)
        a.SetTitleFont(Draw.Font)
        do(a.CenterTitle, center)
        a.SetTickSize(tick_size if not line else 0)
        a.SetTickLength(tick_size if not line else 0)
        a.SetNdivisions(0) if line else do_nothing()
        a.SetLabelOffset(l_off)
        a.Draw()
        return Draw.add(a)

    @staticmethod
    def y_axis(x, ymin, ymax, tit, limits=None, name='ax', col=1, off=1, w=1, opt='+L', tit_size=.035, lab_size=0.035, tick_size=0.03, l_off=.01, line=False, log=False, center=None):
        return Draw.axis(x, x, ymin, ymax, tit, limits, name, col, w, off, tit_size, lab_size, tick_size, line, opt, l_off, log, center)

    @staticmethod
    def x_axis(y, xmin, xmax, tit, limits=None, name='ax', col=1, off=1, w=1, opt='+L', tit_size=.035, lab_size=0.035, tick_size=0.03, l_off=.01, line=False, log=False, center=None):
        return Draw.axis(xmin, xmax, y, y, tit, limits, name, col, w, off, tit_size, lab_size, tick_size, line, opt, l_off, log, center)

    @staticmethod
    def line(x1, x2, y1, y2, color=1, width=1, style=1, show=True):
        line = TCutG(Draw.get_name('l'), 2, np.array([x1, x2], 'd'), np.array([y1, y2], 'd'))
        line.SetLineColor(color)
        line.SetLineWidth(width)
        line.SetLineStyle(style)
        line.Draw('same') if show else do_nothing()
        return Draw.add(line)

    @staticmethod
    def tline(x1, x2, y1, y2, color=1, width=1, style=1, ndc=None):
        line = TLine(x1, y1, x2, y2)
        line.SetLineColor(color)
        line.SetLineWidth(width)
        line.SetLineStyle(style)
        do(line.SetNDC, ndc)
        line.Draw()
        return Draw.add(line)

    @staticmethod
    def vertical_line(x, ymin=-1e9, ymax=1e9, color=1, w=1, style=1, tline=False, show=True):
        return Draw.line(x, x, ymin, ymax, color, w, style, show) if not tline else Draw.tline(x, x, ymin, ymax, color, w, style)

    @staticmethod
    def horizontal_line(y, xmin=-1e9, xmax=1e9, color=1, w=1, style=1, tline=False, ndc=None, show=True):
        return Draw.line(xmin, xmax, y, y, color, w, style, show) if not tline else Draw.tline(xmin, xmax, y, y, color, w, style, ndc)

    @staticmethod
    def polygon(x, y, line_color=1, width=1, style=1, name=None, fillstyle=None, fill_color=None, opacity=None, show=True, closed=True):
        if get_object(name) is not None:  # check if name already exists
            get_object(name).Clear()
        s, x, y = (len(x) + 1, np.append(x, x[0]).astype('d'), np.append(y, y[0]).astype('d')) if closed else (len(x), np.array(x, 'd'), np.array(y, 'd'))
        line = TCutG(choose(name, Draw.get_name('poly')), s, x, y)
        format_histo(line, line_color=line_color, lw=width, line_style=style, fill_color=fill_color, fill_style=fillstyle, opacity=opacity)
        line.SetFillStyle(Draw.Solid if fill_color else 0)
        if show:
            line.Draw('l')
            line.Draw('f') if fill_color is not None or fillstyle is not None and 4000 > fillstyle > 0 else do_nothing()
        return Draw.add(line)

    @staticmethod
    def polyline(x, y, line_color=1, width=1, style=1, name=None, fillstyle=None, fill_color=None, opacity=None, show=True):
        return Draw.polygon(x, y, line_color, width, style, name, fillstyle, fill_color, opacity, show, closed=False)

    @staticmethod
    def box(x1, y1, x2, y2, line_color=1, width=1, style=1, name=None, fillstyle=None, fillcolor=None, opacity=None, show=True):
        x, y = make_box_args(x1, y1, x2, y2)
        return Draw.polygon(x, y, line_color, width, style, name, fillstyle, fillcolor, opacity, show)

    @staticmethod
    def fypolygon(f, x1, x2, y, name=None, n=100, **kwargs):
        x, y = np.array([(x, f(x)) for x in np.linspace(x1, x2, n)] + [(x2, y), (x1, y)]).T
        Draw.polygon(x, y, name=name, **kwargs)

    def segment(self, x, y, w, ox=0, oy=0, color=2, **dkw):
        x, y = x - ox, y - oy
        x, y = np.append(x, x[0]), np.append(y, y[0])
        xi, yi = [[j + (w if j == min(i) else -w) for j in i] for i in [x, y]]
        self.graph(x, y, draw_opt='samel', **prep_kw(dkw, color=color))
        self.graph(xi, yi, draw_opt='samel', **prep_kw(dkw, color=color))
        return self.graph(np.concatenate([x, xi]), np.concatenate([y, yi]), draw_opt='samef', **prep_kw(dkw, fill_style=Draw.Solid, fill_color=color, color=color))

    @staticmethod
    def tlatex(x, y, text, name=None, align=20, color=1, size=.05, angle=None, ndc=None, font=42, show=True):
        tlatex = TLatex(x, y, text)
        format_text(tlatex, choose(name, Draw.get_name('t')), align, color, size, angle, ndc, font)
        tlatex.Draw() if show else do_nothing()
        return Draw.add(tlatex)

    @staticmethod
    def date(x, y, align=20, color=1, size=.05, angle=None, font=42, **kwargs):
        return Draw.tlatex(x, y, datetime.now().strftime('%Y-%m-%d %H:%M'), None, align, color, size, angle, ndc=True, font=font, **kwargs)

    @staticmethod
    def arrow(x1, x2, y1, y2, col=1, width=1, opt='<|', size=.005):
        ar = TArrow(x1, y1, x2, y2, size, opt)
        ar.SetLineWidth(width)
        ar.SetLineColor(col)
        ar.SetFillColor(col)
        ar.Draw()
        return Draw.add(ar)

    @staticmethod
    def tpad(tit='', pos=None, fill_col=0, gridx=None, gridy=None, margins=None, transparent=False, logy=None, logx=None, logz=None, lm=None, rm=None, bm=None, tm=None, c=None, fix=False):
        c.cd() if c is not None else do_nothing()
        pos = [0, 0, 1, 1] if pos is None else pos
        p = TPad(Draw.get_name('pd'), tit, *pos)
        p.SetFillColor(fill_col)
        margins = margins if all([m is None for m in [lm, rm, bm, tm]]) else [lm, rm, bm, tm]
        Draw.set_pad_margins(p, *margins if margins is not None else full(4, .1) if c is None else Draw.get_margins(c), fix=fix)
        do([p.SetLogx, p.SetLogy, p.SetLogz], [logx, logy, logz])
        do([p.SetGridx, p.SetGridy], [gridx, gridy])
        make_transparent(p) if transparent else do_nothing()
        p.Draw()
        p.cd()
        return Draw.add(p)

    @staticmethod
    def tpavetext(text, x1, x2, y1, y2, font=42, align=0, size=0, angle=0, margin=.05, color=1, opt='ndc', show=True):
        p = TPaveText(x1, y1, x2, y2, opt)
        p.SetFillColor(0)
        p.SetFillStyle(0)
        p.SetBorderSize(0)
        p.SetMargin(margin)
        t = p.AddText(text)
        format_text(t, 'pave', align, color, size, angle, ndc=True, font=font)
        p.Draw() if show else do_nothing()
        return Draw.add(p)

    @staticmethod
    def textbox(text, x=.5, y=.5, s=.05, font=42, color=1, fill_color=0, fill_style=1000, margin=.05, opacity=1, ndc=True, rounded=True, border=False, show=True):
        t = Draw.tpavetext(text, x - s, x + s, y - s, y + s, font, color=color, margin=margin, opt=['', 'NDC'][ndc] + ['', 'ARC'][rounded], show=show)
        t.SetLineColorAlpha(1 if border else fill_color, 1 if border else 0)
        t.SetFillStyle(fill_style)
        t.SetFillColorAlpha(fill_color, opacity)
        return t

    @staticmethod
    def stats(fit, x2=None, y2=None, w=.3, prec='1.1f', names=None, rm_entries=None, left=False, **_kw):
        if fit is not None:
            names = fit.Names if names is None else names
            c = get_last_canvas()
            p = TPaveStats(*get_stat_pos(c, fit.NPar - make_list(rm_entries).size + 1, x2, y2, w=w, left=left), 'ndc')
            p.SetBorderSize(1)
            p.SetFillColor(0)
            p.SetFillStyle(Draw.Solid)
            leg = p.AddText('Fit Result')
            leg.SetTextFont(42)
            ls = p.GetListOfLines()
            ls.Add(Draw.tlatex(0, 0, '#chi^{{2}} / ndf  = {chi2:{p}} / {ndf}'.format(ndf=fit.Ndf(), chi2=fit.Chi2(), p=prec), size=0, align=0, font=42))
            for i in range(fit.NPar):
                if i not in make_list(rm_entries):
                    ls.Add(Draw.tlatex(0, 0, '{n}  = {v:{p}} #pm {e:{p}}'.format(n=names[i], v=fit.Parameter(i), e=fit.ParError(i), p=prec), size=0, align=0, font=42))
            p.Draw()
            return Draw.add(p)

    @staticmethod
    def add_stats_entry(h, key, value, form='.2f', line=None):
        s = h.GetListOfFunctions()[0]
        s.SetName(Draw.get_name('st'))
        value = f'{value.n:{form}} #pm {value.s:{form}}' if is_ufloat(value) else f'{value:{form}}'
        text = f'{key} = {value}'
        h.SetStats(0)
        y2, hl = s.GetY2NDC(), (s.GetY2NDC() - s.GetY1NDC()) / s.GetSize()
        s.AddText(text)
        s.SetY1NDC(s.GetY1NDC() - hl)
        [Draw.horizontal_line(y2 - i * hl, s.GetX1NDC(), s.GetX2NDC(), tline=True, ndc=True) for i in make_list(line)] if line is not None else do_nothing()

    @staticmethod
    def frame(pad, xmin, xmax, ymin, ymax, tit, div=None, y_cent=None):
        pad.cd()
        fr = pad.DrawFrame(xmin, ymin, xmax, ymax)
        pad.Modified()
        fr.GetYaxis().SetTitle(tit)
        do(fr.GetYaxis().CenterTitle, y_cent)
        fr.GetYaxis().SetNdivisions(div) if div is not None else do_nothing()
        format_frame(fr)
        Draw.add(fr)

    @staticmethod
    def grid(x_vals, y_vals, **dkw):
        lx = [Draw.line(x, x, min(y_vals), max(y_vals), **prep_kw(dkw, width=1, color=1)) for x in x_vals]
        return lx + [Draw.line(min(x_vals), max(x_vals), y, y, **prep_kw(dkw, width=1, color=1)) for y in y_vals]

    @staticmethod
    def ellipse(a=1, b=1, x_off=0, y_off=0, color=2, w=2, fill=False, fill_color=None, show=True):
        e = TEllipse(x_off, y_off, a, b)
        do(e.SetLineColor, color)
        do(e.SetFillColor, choose(fill_color, color))
        do(e.SetLineWidth, w)
        e.SetFillStyle(Draw.Solid if fill else 0)
        e.Draw() if show else do_nothing()
        return Draw.add(e)

    @staticmethod
    def circle(r, x_off=0, y_off=0, color=None, w=None, fill=False, fill_color=None, show=True):
        return Draw.ellipse(r, r, x_off, y_off, color, w, fill, fill_color, show)

    @staticmethod
    def preliminary(canvas=None, height=.06):
        c = get_last_canvas() if canvas is None else canvas
        c.cd()
        return Draw.tpavetext('#font[62]{RD42} Preliminary', c.GetLeftMargin(), .5, 1 - height - c.GetTopMargin(), 1 - c.GetTopMargin(), font=72, align=12, margin=0.04)

    @staticmethod
    def irradiation(irr, canvas=None, height=.06, left=True):
        c = get_last_canvas() if canvas is None else canvas
        c.cd()
        x1, x2 = (c.GetLeftMargin(), .5) if left else (.5, 1 - c.GetRightMargin())
        return Draw.tpavetext('Irradiation: {}'.format(irr), x1, x2, 1 - height - c.GetTopMargin(), 1 - c.GetTopMargin(), font=42, align=12, margin=0.04)

    @staticmethod
    def legend(histos, titles, styles=None, x2=None, y2=None, show=True, *args, **kwargs):
        leg = Draw.make_legend(x2, y2, *args, **prep_kw(kwargs, nentries=len(histos)))
        for i in range(len(histos)):
            leg.AddEntry(histos[i], titles[i], 'lpf' if styles is None else styles[i] if is_iter(styles) and not type(styles) is str else styles)
        leg.Draw('same') if show else do_nothing()
        return leg

    @staticmethod
    def histo(th, show=True, lm=None, rm=None, bm=None, tm=None, m=None, draw_opt=None, wx=1, hy=1, logx=None, logy=None, logz=None, grid=None, gridy=None, gridx=None, phi=None, theta=None,
              leg=None, ldraw=None, canvas=None, sumw2=None, stats=False, all_pads=False, info_leg=True, **kwargs):
        wx += .16 if not Draw.Title and wx == 1 else 0  # rectify if there is no title
        th.Sumw2(sumw2) if hasattr(th, 'Sumw2') and sumw2 is not None else do_nothing()
        Draw.set_show(show)
        c = get_last_canvas() if canvas is None and 'same' in str(draw_opt) else canvas
        c = Draw.canvas(th.GetTitle().split(';')[0], None, None, wx, hy, logx, logy, logz, gridx or grid, gridy or grid, show=show) if c is None else c
        Draw.set_pad_margins(c, *[lm, rm, bm, tm] if m is None else m)
        do([c.SetLogx, c.SetLogy, c.SetLogz], [logx, logy, logz])
        do([c.SetGridx, c.SetGridy], [gridx or grid, gridy or grid])
        do([c.SetPhi, c.SetTheta], [phi, theta])
        c.cd()
        th.Draw(draw_opt if draw_opt is not None else 'ap' if is_graph(th) else 'hist' if 'TH' in th.ClassName() else '')
        if leg is not None:
            update_canvas()
            for i_leg in make_list(leg):
                if 'TF1' not in i_leg.ClassName() and callable(i_leg):
                    i_leg = i_leg()
                i_leg.Draw(choose(ldraw, 'same'))
                th.GetListOfFunctions().Add(i_leg) if hasattr(th, 'GetListOfFunctions') else do_nothing()
        if info_leg:
            Draw.Info.draw(c, all_pads)
        Draw.set_show(True)
        if stats or stats is None:
            for i in (th.GetListOfGraphs() if 'Multi' in th.ClassName() else [th]):
                format_statbox(i, **Draw.Stats if stats else Draw.DefaultStats, c=canvas)
        return Draw.add(c, th)[0]

    @staticmethod
    def mode(m=1, **kwargs):
        d = {1: {'tit_size': .05, 'lab_size': .045, 'y_off': 1.35},
             2: {'wx': 1.5, 'hy': .75, 'tit_size': .06, 'lab_size': .05, 'y_off': .7, 'lm': .08, 'bm': .15},
             3: {'wx': 1.5, 'hy': .5, 'tit_size': .09, 'lab_size': .08, 'y_off': .45, 'lm': .08, 'bm': .18, 'rm': .03, 'x_tit': 'Time [ns]', 'y_tit': 'Signal [mV]', 'markersize': .5},
             4: {'tit_size': .05, 'lab_size': .045, 'tick_size': 0, 'l_off_y': 10, 'l_off_x': 10, 'center_x': True, 'center_y': True, 'y_off': .5, 'x_off': .5, 'lm': .066, 'bm': .066},
             5: {'tit_size': .08, 'lab_size': .07, 'y_off': 1.02, 'lm': .17, 'bm': .15}
             }[m]
        return prep_kw(kwargs, **d)

    def distribution(self, x, binning=None, title='', q=.02, lf=.2, rf=.2, n=1, r=None, w=None, x0=None, x1=None, **kwargs):
        if is_root_object(x):
            th = x
        else:
            th = TH1F(Draw.get_name('h'), title, *choose(binning, bins.find, values=x, q=q, nbins=n, lfac=lf, rfac=rf, r=r, w=w, x0=x0, x1=x1))
            fill_hist(th, x)
        format_histo(th, **prep_kw(kwargs, **Draw.mode(), fill_color=Draw.FillColor, y_tit='Number of Entries' if not th.GetYaxis().GetTitle() else None))
        self.histo(th, **prep_kw(kwargs, stats=None))
        return th

    def function(self, f, title='', c=None, graph=False, **dkw):
        x = np.linspace(f.GetXmin(), f.GetXmax(), 100)
        f = Draw.make_tgraph(x, [f(i) for i in x]) if graph else f
        format_histo(f, title=title, **prep_kw(dkw, **Draw.mode()))
        self.histo(f, **prep_kw(dkw, canvas=c, line_color=2, draw_opt='al' if graph else None))
        return f

    def functions(self, f, leg_titles=None, wl=.2, **dkw):
        y_range = [min([i.GetMinimum() for i in f]), 1.1 * max([i.GetMaximum() for i in f])]
        self(f[0], **prep_kw(dkw, y_range=y_range, color=self.get_color(len(f))))
        self.legend(f, leg_titles, 'l', w=wl, **rm_key(dkw, 'w'))
        [self(i, draw_opt='same', **prep_kw(dkw, color=self.get_color(len(f)))) for i in f[1:]]
        return get_last_canvas()

    def graph(self, x, y=None, title='', bin_labels=None, **dkw):
        g = x if y is None else Draw.make_tgraph(x, y)
        format_histo(g, title=title, **prep_kw(dkw, **Draw.mode(), fill_color=Draw.FillColor))
        set_bin_labels(g, bin_labels)
        self.histo(g, **prep_kw(dkw, bm=.24 if bin_labels else None))
        return g

    def trend(self, x, y, title='', bw=None, n=20, **dkw):
        x, y = [np.array_split(i, n if bw is None else np.arange(bw, x.size, bw)) for i in [x, y]]
        x = abs(array([mean(i) + np.array([0, -i[0], -i[-1]]) for i in x]))
        y = [mean_sigma(i)[0] for i in y]
        return self.graph(x, y, title, **dkw)

    def profile(self, x, y=None, binning=None, title='', q=.02, lf=.2, rf=.2, w=None, x0=None, graph=False, **dkw):
        if y is None:
            p = x
        else:
            x, y = np.array(x, dtype='d'), np.array(y, dtype='d')
            p = TProfile(Draw.get_name('p'), title, *choose(binning, bins.find, lfac=lf, rfac=rf, values=x, q=q, w=w, x0=x0))
            fill_hist(p, x, y)
        p = self.make_graph_from_profile(p) if graph else p
        format_histo(p, **prep_kw(dkw, **Draw.mode(), fill_color=Draw.FillColor))
        self.histo(p, **prep_kw(dkw, stats=choose(get_kw('stats', dkw), set_statbox, entries=True, w=.25)))
        return p

    def sum_hist(self, x, y=None, binning=None, title='', q=.02, lf=.2, rf=.2, w=None, x0=None, **dkw):
        """ creates a histogram summing up all values in a bin """
        if y is None:
            h = x
        else:
            x, y = np.array(x, dtype='d'), np.array(y, dtype='d')
            binning = choose(binning, bins.find, lfac=lf, rfac=rf, values=x, q=q, w=w, x0=x0)
            e = np.histogram(x, binning[1])[0]  # number of entries in each bin
            x = [np.sum(v) for v in np.split(y, np.unique(np.cumsum(e)))]
            h = fill_hist(TH1F(Draw.get_name('sh'), title, *binning), x, set_bins=True)
        format_histo(h, **prep_kw(dkw, **Draw.mode(), fill_color=Draw.FillColor, stats=False, y_range=[0, 1.1 * h.GetMaximum()]))
        self.histo(h, **prep_kw(dkw, stats=choose(get_kw('stats', dkw), set_statbox, entries=True, w=.25)))
        return h

    def prof2d(self, x, y=None, zz=None, binning=None, title='', qz=None, z0=None, rot=None, mirror=None, centre=None, **dkw):
        if is_root_object(x):
            p = x
        else:
            x, y, zz = arr2coods(x) if y is None else (x, y, zz)
            dflt_bins = bins.find(x) + bins.find(y) if binning is None else None
            p = TProfile2D(Draw.get_name('p2'), title, *choose(binning, dflt_bins))
            fill_hist(p, x, y, uarr2n(zz))
        p = self.rotate_2d(p, rot)
        p = self.flip_2d(p, mirror)
        (rx, ry), rz = get_2d_centre_ranges(p, centre), find_z_range(p, qz, z0)
        format_histo(p, **prep_kw(dkw, **Draw.mode(), z_off=1.2, pal=55, x_range=rx, y_range=ry, z_range=rz))
        draw_opt = choose(get_kw('draw_opt', dkw), 'colz')
        self.histo(p, **prep_kw(dkw,  rm=.17 if 'z' in draw_opt else None, stats=choose(get_kw('stats', dkw), set_statbox, entries=True, w=.25), draw_opt=draw_opt))
        return p

    def histo_2d(self, x, y=None, binning=None, title='', q=.02, n=1, lf=.2, rf=.2, w=None, x0=None, x1=None, y0=None, y1=None, qz=None, z0=None, canvas=None, rot=None,
                 mirror=None, centre=None, **dkw):
        if y is None:
            th = x
        else:
            x, y = np.array(x, dtype='d'), np.array(y, dtype='d')
            b = partial(bins.find, q=q, nbins=n, rfac=rf, lfac=lf, w=w)
            th = TH2F(Draw.get_name('h2'), title, *(b(x, x0=x0, x1=x1) + b(y, x0=y0, x1=y1)) if binning is None else binning)
            fill_hist(th, x, y)
        th = self.rotate_2d(th, rot)
        th = self.flip_2d(th, mirror)
        (rx, ry), rz = get_2d_centre_ranges(th, centre), find_z_range(th, qz, z0)
        format_histo(th, **prep_kw(dkw, **Draw.mode(), z_off=1.2, z_tit='Number of Entries', pal=55, x_range=rx, y_range=ry, z_range=rz))
        draw_opt = choose(get_kw('draw_opt', dkw), 'colz')
        self.histo(th, canvas=canvas, **prep_kw(dkw, rm=.17 if 'z' in draw_opt else None, stats=choose(get_kw('stats', dkw), set_statbox, entries=True, w=.25), draw_opt=draw_opt))
        return th

    def histo_3d(self, x, y, zz, binning=None, title='', q=.02, **dkw):
        th = TH3F(Draw.get_name('h3'), title, *bins.find(x, q=q) + bins.find(y, q=q) + bins.find(zz, q=q) if binning is None else binning)
        fill_hist(th, x, y, zz)
        format_histo(th, **prep_kw(dkw))
        self.histo(th, **prep_kw(dkw, draw_opt='colz', show=False))
        return th

    def efficiency(self, x, e, binning=None, q=.02, w=None, x0=None, **kwargs):
        p = self.profile(x, e, binning, q=q, w=w, x0=x0, show=False)
        x, y = bins.hx(p), np.array([calc_eff(p0 * n, n) if n else [-1, 0, 0] for p0, n in [[p.GetBinContent(ibin), p.GetBinEntries(ibin)] for ibin in range(1, p.GetNbinsX() + 1)]])
        return self.graph(x[y[:, 0] != -1], y[y[:, 0] != -1], **prep_kw(kwargs, title='Efficiency', y_tit='Efficiency [%]'))

    def pull(self, h, binning=None, ret_h=False, **dkw):
        x = h if type(h) in [list, np.ndarray] else h_y(h)
        m, s = mean_sigma(x)
        x = uarr2n((x - m) / s)
        th = self.distribution(x, binning, **prep_kw(dkw, rf=.5, lf=.5, n=2, x_tit=f'Normalised {h.GetYaxis().GetTitle()}'.split('[')[0] if hasattr(h, 'Class') else None))
        return th if ret_h else mean_sigma(x[x != 0])

    def stack(self, histos, title='', leg_titles=None, leg_head=None, scale=False, fill=None, ldraw='l', lw=.2, **dkw):
        s = THStack(Draw.get_name('s'), title)
        for h in histos:
            s.Add(h, 'hist')
            color = self.get_color(len(histos))
            format_histo(h, color=color, fill_color=choose(color, 0, fill), fill_style=choose(Draw.Solid, 4000, fill), stats=0, **dkw)
            if scale:
                h.Scale(1 / h.GetMaximum())
        h0 = histos[0]
        self.histo(s, show=False, save=False, draw_opt='nostack')
        format_histo(s, **prep_kw(dkw, x_tit=h0.GetXaxis().GetTitle(), y_tit=h0.GetYaxis().GetTitle(), **Draw.mode(1, y_off=h0.GetYaxis().GetTitleOffset())))
        leg = self.legend(histos, leg_titles, ldraw, header=leg_head, w=lw) if leg_titles is not None else None
        self.histo(s, **prep_kw(dkw, draw_opt='nostack', leg=leg, lm=get_last_canvas().GetLeftMargin()))
        return s

    def multigraph(self, graphs, title='', leg_titles=None, bin_labels=None, draw_opt='p', wleg=.2, **dkw):
        if hasattr(graphs, 'GetName'):
            m, g0 = graphs, graphs.GetListOfGraphs()[0]
        else:
            g0 = graphs[0]
            m = TMultiGraph(Draw.get_name('mg'), ';'.join([title, g0.GetXaxis().GetTitle(), g0.GetYaxis().GetTitle()]))
            for i, g in enumerate(graphs):
                m.Add(g, draw_opt)
                color = None if get_kw('color', dkw, 2) is None else self.get_color(len(graphs))
                marker = get_kw('marker', dkw) if 'marker' in dkw else markers(i)
                format_histo(g, **prep_kw(dkw, marker=marker, color=color, stats=False))
        y_range = ax_range(graph_y(graphs, err=False), 0, .3, .6)
        self.histo(m, show=False, save=False, canvas=get_kw('canvas', dkw, None))
        format_histo(m, **prep_kw(dkw, **Draw.mode(1, y_off=g0.GetYaxis().GetTitleOffset()), y_tit=g0.GetYaxis().GetTitle(), y_range=y_range, x_tit=choose('', None, bin_labels)))
        set_bin_labels(m, bin_labels)
        leg = self.legend(graphs, leg_titles, draw_opt, w=wleg) if leg_titles else None
        self.histo(m, **prep_kw(dkw, leg=leg, bm=choose(.26, None, bin_labels), draw_opt='ap'))
        return m

    def pie(self, labels, values=None, colors=None, title='', offset=0, show=True, flat=False, draw_opt=None, **kwargs):
        labels, (values, colors) = (labels.keys(), np.array(list(labels.values())).T) if values is None else (labels, (values, choose(colors, Draw.get_colors(len(labels)))))
        pie = TPie(self.get_name('pie'), title, len(labels), np.array(values, 'f'), np.array(colors, 'i'))
        for i, label in enumerate(labels):
            pie.SetEntryRadiusOffset(i, offset)
            pie.SetEntryLabel(i, label)
        format_pie(pie, **kwargs)
        draw_opt = choose(draw_opt, f'{"" if flat else "3d"}rsc')
        self.histo(pie, draw_opt=draw_opt, show=show)
        return pie

    def prof2hist(self, p):
        b = bins.h2d(p)
        h, nx, ny = self.histo_2d([], [], b, show=False), b[0], b[2]
        xax, yax = p.GetXaxis(), p.GetYaxis()
        [h.Fill(xax.GetBinCenter(ix), yax.GetBinCenter(iy)) for ix in range(1, nx + 2) for iy in range(1, ny + 2) for _ in range(bins.single_entries_2d(p, ix, iy, nx))]
        return h

    @staticmethod
    def info(txt, c=None, size=.04):
        c = (get_last_canvas() if c is None else c).cd()
        Draw.tlatex(c.GetLeftMargin() + .02, 1 - (c.GetTopMargin() + .02), txt, align=13, ndc=True, size=size)

    @staticmethod
    def bin_numbers(h, show=True):
        if show:
            x, y = bins.h2d(h, arr=True)
            dx, dy = np.diff(x)[0] / 2, np.diff(y)[0] / 2
            [Draw.tlatex(x[m] + dx, y[n] + dy, str((x.size - 1) * n + m)) for n in range(y.size - 1) for m in range(x.size - 1)]

    def maps_correlation(self, m1, m2, sx=0, sy=0, thresh=.1, **dkw):
        x, y = get_correlation_arrays(m1, m2, sx, sy, thresh, flat=True)
        return self.histo_2d(x[(x != 0) & (y != 0)], y[(x != 0) & (y != 0)], **prep_kw(dkw, x_tit=m1.GetZaxis().GetTitle(), y_tit=m2.GetZaxis().GetTitle()))

    def maps_profile(self, m1, m2, sx=0, sy=0, thresh=.1, **dkw):
        x, y = get_correlation_arrays(m1, m2, sx, sy, thresh, flat=True)
        return self.profile(x[(x != 0) & (y != 0)], y[(x != 0) & (y != 0)], **prep_kw(dkw, graph=True, x_tit=m1.GetZaxis().GetTitle(), y_tit=m2.GetZaxis().GetTitle()))
    # endregion DRAW
    # ----------------------------------------

    # ----------------------------------------
    # region OPERATIONS
    def operate(self, h, f, *args, **kwargs):
        h0, h = h, self.prof2d([], [], [], bins.h2d(h), show=False)
        prof = 'Profile' in h0.ClassName()
        x, n = hist_values_2d(h0, err=False, flat=False), bins.entries_2d(h0) if prof else 1
        bins.set_2d_values(h, f(x * n, *args, **kwargs))
        bins.set_2d_entries(h, f(n, *args, **kwargs)) if prof else do_nothing()
        h.SetEntries(int(h0.GetEntries()))
        format_histo(h, z_range=[h0.GetMinimum(), h0.GetMaximum()], **{f'{i}_tit': getattr(h0, f'Get{i.title()}axis')().GetTitle() for i in ['x', 'y', 'z']}, ncont=h0.GetContour())
        return h

    def rotate_2d(self, h, n=2):
        return self.operate(h, np.rot90, n) if n is not None else h

    def flip_2d(self, h, axis=0):
        return self.operate(h, np.flip, axis=axis) if axis is not None else h
    # endregion OPERATIONS
    # ----------------------------------------

    # ----------------------------------------
    # region CREATE
    @staticmethod
    def make_histo(title, binning):
        h = TH1F(Draw.get_name('h'), title, *binning)
        return Draw.add(h)

    @staticmethod
    def make_f(name, function, xmin=0, xmax=1, pars=None, limits=None, fix=None, npx=None, parnames=None, **kwargs):
        f = TF1(choose(name, Draw.get_name('f')), function, xmin, xmax)
        f.SetParameters(*pars) if pars is not None else do_nothing()
        [f.SetParLimits(i, *lim) for i, lim in enumerate(limits)] if limits else do_nothing()
        [f.FixParameter(i, value) for i, value in enumerate(make_list(fix))] if fix is not None else do_nothing()
        f.SetParNames(*make_list(parnames)) if parnames is not None else do_nothing()
        do(f.SetNpx, npx)
        format_histo(f, **kwargs)
        return Draw.add(f)

    @staticmethod
    def make_tf1(name, f, xmin=0, xmax=1, pars0: Any = 0, color=None, w=None, style=None, title=None, npx=None, *args, **kwargs):
        def tmp(x, pars):
            return f(x[0], pars, *args, **kwargs) if 'pars' in signature(f).parameters else f(x[0], *args, **kwargs)

        Draw.add(tmp)
        f0 = TF1(choose(name, Draw.get_name('f')), tmp, xmin, xmax, len(pars0) if is_iter(pars0) else pars0)
        [f0.SetParameter(i, p) for i, p in enumerate(pars0)] if is_iter(pars0) else do_nothing()
        do(f0.SetNpx, npx)
        format_histo(f0, choose(title, name), line_color=color, line_style=style, lw=w)
        return Draw.add(f0)

    @staticmethod
    def make_tgraph(x=None, y=None, **kwargs):
        if len(list(x)) != len(list(y)) or not len(x):
            return warning('Arrays have different size!')
        d = [[make_list(i) for i in lst] for lst in [x, y]]  # make all entries np.arrays
        if any([i.size == 3 for lst in d for i in lst]):
            x, y = np.array(array([[[v[0].n, v[0].s, v[0].s] if is_ufloat(v[0]) else append(v, zeros(3 - v.size)) for v in lst] for lst in d]).tolist())  # noqa
            x, ex1, ex2, y, ey1, ey2 = [a.astype('d') for a in np.concatenate([x.T, y.T])]
            g = TGraphAsymmErrors(len(x), x, y, ex1, ex2, ey1, ey2)
        else:
            x, y = np.array([[[v[0].n, v[0].s] if is_ufloat(v[0]) else np.append(v, zeros(2 - v.size)) for v in lst] for lst in d])
            x, ex, y, ey = [a.astype('d') for a in np.concatenate([x.T, y.T])]
            g = TGraphErrors(len(x), x, y, ex, ey)
        format_histo(g, Draw.get_name('g'), **prep_kw(kwargs, marker=20, markersize=1.2))
        return Draw.add(g)

    @staticmethod
    def make_graph_from_profile(p, nmin=2):
        (x, y), n = hist_xy(p), bins.entries(p)
        cut = (y != 0) & (n >= nmin)
        return Draw.make_tgraph(x[cut], y[cut], title=p.GetTitle(), x_tit=p.GetXaxis().GetTitle(), y_tit=p.GetYaxis().GetTitle())

    @staticmethod
    def make_legend(x2=None, y2=None, w=.25, nentries=2, scale=1, ts=None, d=.01, y1=None, x1=None, header=None, clean=False, margin=.25, cols=None, fix=False, bottom=False, left=False, c=None,
                    **kwargs):
        _ = kwargs
        use_margins = y2 is None
        h = nentries // choose(cols, 1) * .06 * scale
        x2, y2 = get_stat_margins(c, x2, y2, d, bottom, left, h, w)
        x1 = choose(x1, x2 - w)
        y1 = choose(y1, y2 - h)
        if not use_margins:
            y1 += .07 if not Draw.Title and y1 + h > .8 and not fix else 0
            y1 -= .07 if not Draw.Info.ShowLegend and y1 < .3 and not fix else 0
        leg = TLegend(x1, max(y1, 0), x1 + w, min(y1 + h, 1))
        leg.SetName(Draw.get_name('l'))
        do(leg.SetTextSize, ts)
        leg.SetHeader(header, 'c') if header is not None else do_nothing()
        leg.SetTextFont(Draw.Font)
        leg.SetMargin(margin)
        do(leg.SetNColumns, cols)
        if clean:
            leg.SetLineWidth(2)
            leg.SetBorderSize(0)
            leg.SetFillColor(0)
            leg.SetFillStyle(0)
            leg.SetTextAlign(12)
        return Draw.add(leg)
    # endregion CREATE
    # ----------------------------------------

# END OF CLASS ---------------------------


# ----------------------------------------
# region FORMATTING TODO: make separate class
def format_histo(histo, name=None, title=None, x_tit=None, y_tit=None, z_tit=None, marker=None, color=None, line_color=None, line_style=None, markersize=None, x_off=None, y_off=None, z_off=None,
                 lw=None, fill_color=None, fill_style=None, stats=None, tit_size=None, lab_size=None, xls=None, yls=None, l_off_y=None, l_off_x=None, draw_first=False,
                 x_range=None, xr=None, y_range=None, yr=None, z_range=None, zr=None,
                 sumw2=None, style=None, ndivx=None, ndivy=None, ncont=None, tick_size=None, t_ax_off=None, tform='%H:%M', center_y=False, center_x=False, yax_col=None,
                 normalise=None, pal=None, rebin=None, y_ticks=None, x_ticks=None, z_ticks=None, opacity=None, center_tit=None, bar_w=None, bar_off=None, **kwargs):
    _ = kwargs
    h = histo
    if draw_first:
        Draw.set_show(False)
        h.Draw('nostack' if h.ClassName() == 'THStack' else 'a')
        Draw.set_show(True)
    do(h.SetTitle, title)
    do(h.SetName, name)
    do(set_palette, pal)
    if normalise is not None:
        old_tit = choose(y_tit, h.GetYaxis().GetTitle())
        y_tit = old_tit.replace('Number', 'Frequency') if 'Number' in old_tit else old_tit
        normalise_histo(h)
    try:
        do(h.SetStats, stats)
        do(h.SetContour, ncont)
    except AttributeError or ReferenceError:
        pass
    do(h.Rebin, rebin) if hasattr(h, 'Rebin') else do_nothing()
    format_markers(h, marker, color, markersize)
    format_bar(h, bar_w, bar_off)
    format_lines(h, choose(line_color, color), lw, line_style)
    format_fill(h, fill_color, opacity, choose(style, fill_style))
    # axes
    try:
        x_args = [x_tit, x_off, tit_size, choose(center_tit, center_x), choose(xls,  lab_size), l_off_x, choose(xr, x_range), ndivx, choose(x_ticks, tick_size), ]
        y_args = [y_tit, y_off, tit_size, choose(center_tit, center_y), choose(yls, lab_size), l_off_y, choose(yr, y_range), ndivy, choose(y_ticks, tick_size), yax_col]
        z_args = [z_tit, z_off, tit_size, False, lab_size, None, choose(zr, z_range), None, choose(z_ticks, tick_size)]
        for i, name in enumerate(['X', 'Y', 'Z']):
            format_axis(getattr(h, f'Get{name}axis')(), h, *[x_args, y_args, z_args][i])
    except AttributeError or ReferenceError:
        pass
    set_time_axis(h, tform, t_ax_off) if t_ax_off is not None else do_nothing()
    do(h.Sumw2, sumw2) if hasattr(h, 'Sumw2') else do_nothing()
    update_canvas()
    return h


def format_bar(h, w, off):
    try:
        do(h.SetBarWidth, w)
        do(h.SetBarOffset, off)
    except AttributeError or ReferenceError:
        pass


def format_markers(h, style, color, size):
    if style or size:
        try:
            do(h.SetMarkerStyle, style)
            do(h.SetMarkerColor, color)
            do(h.SetMarkerSize, size)
        except AttributeError or ReferenceError:
            pass


def format_lines(h, color, w, style):
    if hasattr(h, 'SetLineColor'):
        do(h.SetLineColor, color)
        do(h.SetLineWidth, w)
        do(h.SetLineStyle, style)


def format_fill(h, color, opacity, style):
    if hasattr(h, 'SetFillColor'):
        if color is not None:
            h.SetFillColor(color) if opacity is None else h.SetFillColorAlpha(color, opacity)
        do(h.SetFillStyle, style)


def set_statbox(x2=None, y2=None, h=None, w=.3, entries=False, m=False, rms=False, all_stat=False, fit=False, center_x=False, center_y=False, form=None, stats=True, **kw):
    Draw.Stats = {'x2': x2, 'y2': y2, 'h': h, 'w': w, 'entries': entries, 'm': m, 'rms': rms, 'all_stat': all_stat, 'fit': fit, 'center_x': center_x, 'center_y': center_y, 'form': form, **kw}
    return stats


def set_entries():
    Draw.Stats = {'entries': True, 'w': .2}
    return True


def get_window_ratio(c=None):
    c = choose(c, get_last_canvas(warn=False))
    return c.GetWindowHeight() / c.GetWindowWidth() if c.ClassName() == 'TCanvas' else c.GetAbsHNDC() / c.GetAbsWNDC()


def get_stat_margins(c=None, x2=None, y2=None, d=.01, bottom=False, left=False, h=0., w=0.):
    c = choose(c, get_last_canvas(warn=False))
    r = get_window_ratio(c)
    x2 = choose(x2, c.GetLeftMargin() + w + 4 * d * r if left else 1 - c.GetRightMargin() - d * r)
    y2 = choose(y2, c.GetBottomMargin() + h + 4 * d if bottom else 1 - c.GetTopMargin() - d)
    return x2, y2


def get_stat_pos(c, nentries, x2=None, y2=None, d=.01, h=None, w=.3, center_x=False, center_y=False, bottom=False, left=False):
    r = get_window_ratio(c)
    h = choose(h, .05 / r * nentries)
    x2, y2 = get_stat_margins(c, x2, y2, d, bottom, left, h, w)
    cx, cy = mean([1 - c.GetRightMargin(), c.GetLeftMargin()]), mean([1 - c.GetTopMargin(), c.GetBottomMargin()])
    x1, x2 = (cx - w / 2, cx + w / 2) if center_x else (x2 - w, x2)
    y1, y2 = (cy - h / 2, cy + h / 2) if center_y else (y2 - h, y2)
    return x1, y1, x2, y2


def format_statbox(th, x2=None, y2=None, d=.01, h=None, w=.3, entries=False, m=False, rms=False, all_stat=False, fit=False, fit_opt=None, stat_opt=None, center_x=False,
                   center_y=False, bottom=False, left=False, form=None, c=None):
    c = choose(c, get_last_canvas(warn=False))
    update_canvas(c)
    f = None if 'TF1' in th.ClassName() else next((o for o in th.GetListOfFunctions() if 'TF1' in o.ClassName()), None)
    if 'TGraph' in th.ClassName() and fit and f:
        gStyle.SetOptFit(True)
    p = None if 'TF1' in th.ClassName() else next((o for o in th.GetListOfFunctions() if 'Pave' in o.ClassName()), None)
    if p is not None:
        stats = np.ones(3, 'i') if all_stat else np.array([rms, m, entries], 'i')
        nentries = stats.nonzero()[0].size + (f.GetNpar() + 1 if fit and f is not None else 0)
        [getattr(p, f'Set{n}NDC')(i) for i, n in zip(get_stat_pos(c, nentries, x2, y2, d, h, w, center_x, center_y, bottom, left), ['X1', 'Y1', 'X2', 'Y2'])]
        p.SetOptStat(choose(stat_opt, int('100000{}{}{}0'.format(*stats * 2))))
        p.SetOptFit(choose(fit_opt, int(fit)))
        p.SetStatFormat(choose(form, '1.1f'))
        p.SetFitFormat(form) if form is not None else do_nothing()
        p.Draw()
        update_canvas(c)


def format_axis(axis, h, title, tit_offset, tit_size, centre_title, lab_size, label_offset, limits, ndiv, tick_size, color=None):
    do(axis.SetTitle, title)
    do(axis.SetTitleOffset, tit_offset)
    do(axis.SetTitleSize, tit_size)
    axis.CenterTitle(centre_title)
    do(axis.SetLabelSize, lab_size)
    do(axis.SetLabelOffset, label_offset)
    if limits is not None:
        axis.SetLimits(*limits) if is_graph(h) and 'xaxis' in axis.GetName() else axis.SetRangeUser(*limits)
    do(axis.SetNdivisions, ndiv)
    do(axis.SetTickSize, tick_size)
    do(axis.SetTitleColor, color)
    do(axis.SetLabelColor, color)
    do(axis.SetAxisColor, color)


def format_pie(pie, h=None, r=None, text_size=None, angle3d=None, angle_off=None, label_format=None):
    do([pie.SetHeight, pie.SetRadius], [h, r])
    do(pie.SetTextSize, text_size)
    do(pie.SetAngle3D, angle3d)
    do(pie.SetLabelFormat, label_format)
    do(pie.SetAngularOffset, angle_off)


def format_text(t, name='text', align=20, color=1, size=.05, angle=None, ndc=None, font=None):
    t.SetName(name)
    t.SetTextAlign(align)
    t.SetTextColor(color)
    t.SetTextSize(size)
    do(t.SetTextAngle, angle)
    do(t.SetTextFont, font)
    do(t.SetNDC, ndc)
    return t


def format_frame(frame):
    fr = frame
    fr.GetYaxis().SetTitleSize(.06)
    fr.GetYaxis().SetTitleOffset(.6)
    fr.GetYaxis().SetLabelSize(.06)
    fr.SetTitleSize(.05)
    fr.GetXaxis().SetTickLength(0)
    fr.GetXaxis().SetLabelOffset(99)
    fr.SetLineColor(0)
    fr.GetXaxis().SetTimeDisplay(1)
# endregion FORMATTING
# ----------------------------------------


def fill_hist(h, x, y=None, zz=None, set_bins=False):
    if set_bins:
        for i, v in enumerate(x, 1):
            h.SetBinContent(i, np.array(v, dtype='d'))
        return h
    if len(x) and is_ufloat(x[0]):
        for i, v in enumerate(x, 1):
            h.SetBinContent(i, v.n)
            h.SetBinError(i, v.s)
        return
    x, y, zz = np.array(x).astype('d'), np.array(y).astype('d'), np.array(zz).astype('d')
    if len(x.shape) > 1:
        y = np.array(x[:, 1])
        x = np.array(x[:, 0])
    if not x.size:  # return if there are no entries
        return
    if h.ClassName() == 'TProfile2D':
        for i in range(x.size):
            h.Fill(x[i], y[i], zz[i])
    elif 'TH1' in h.ClassName():
        h.FillN(x.size, x, np.ones(x.size))
    elif any(name in h.ClassName() for name in ['TH2', 'TProfile']):
        h.FillN(x.size, x, y, np.ones(x.size))
    elif h.ClassName() == 'TProfile2D':
        h.FillN(x.size, x, y, zz, np.ones(x.size))
    else:
        for i in range(x.size):
            h.Fill(x[i], y[i], zz[i])
    return h


def set_2d_ranges(h, dx, dy):
    # find centers in x and y
    xmid, ymid = [(p.GetBinCenter(p.FindFirstBinAbove(0)) + p.GetBinCenter(p.FindLastBinAbove(0))) / 2 for p in [h.ProjectionX(), h.ProjectionY()]]
    format_histo(h, x_range=[xmid - dx, xmid + dx], y_range=[ymid - dy, ymid + dx])


def arr2coods(a):
    i = np.indices(a.shape)
    cut = ~np.isnan(a).flatten()
    return [v.flatten()[cut] for v in [i[1], i[0], a]]


def fix_chi2(g, prec=.01, show=True):
    it = 0
    error = 2
    chi2 = 0
    fit = None
    while abs(chi2 - 1) > prec and it < 20:
        for i in range(g.GetN()):
            g.SetPointError(i, g.GetErrorX(i), error)
        fit = g.Fit('pol0', 'qs{}'.format('' if show else 0))
        chi2 = fit.Chi2() / fit.Ndf()
        error += .5 ** it * np.sign(chi2 - 1)
        it += 1
    return None if fit is None else FitRes(fit)


def make_darray(values):
    return np.array([v.n for v in values] if is_ufloat(values[0]) else values, dtype='d')


# ----------------------------------------
# region GRAPH VALUES
def graph_values(g, m, err=False, as_u=True):
    if is_iter(g):
        return np.array([v for ig in g for v in graph_values(ig, m, err)])
    v = np.frombuffer(getattr(g, f'Get{m}')())
    if 'Asym' in g.ClassName() and err:
        e = np.array([np.frombuffer(getattr(g, f'GetE{m}{att}')(), count=g.GetN()) for att in ['low', 'high']]).T
        return make_ufloat(v, mean(e, axis=1)) if as_u else np.column_stack([v, e])
    elif 'Error' in g.ClassName() and err:
        e = np.frombuffer(getattr(g, f'GetE{m}')())
        return make_ufloat(v, e) if as_u else np.array([v, e]).T
    return v


def graph_xy(g, err=True, as_u=True):
    return graph_x(g, err, as_u), graph_y(g, err, as_u)


def graph_x(g, err=True, as_u=True):
    return graph_values(g, 'X', err, as_u)


def graph_y(g, err=True, as_u=True):
    return graph_values(g, 'Y', err, as_u)
# endregion GRAPH VALUES
# ----------------------------------------


# ----------------------------------------
# region HISTOGRAM VALUES
def hist_values(h, err=True):
    return np.array([ufloat(h.GetBinContent(i), h.GetBinError(i)) if err else h.GetBinContent(i) for i in bins.hn(h)])


def hist_xy(h, err=True, raw=False):
    if type(h) in [np.ndarray, list]:
        return np.array([tup for ip in h for tup in np.array(hist_xy(ip, err, raw)).T]).T
    return bins.from_hist(h, err, raw), hist_values(h, err)


def hist_values_2d(h, err=True, flat=True, z_sup=True):
    xbins, ybins = bins.hn(h, axis='X'), bins.hn(h, axis='Y')
    values = np.array([ufloat(h.GetBinContent(xbin, ybin), h.GetBinError(xbin, ybin)) for ybin in ybins for xbin in xbins])
    values = values if err else np.array([v.n for v in values])
    return (values[values != 0] if z_sup else values) if flat else values.reshape(len(ybins), len(xbins))


def hist_xyz(h, err=True, flat=False, z_sup=True, grid=False):
    z_ = hist_values_2d(h, err, flat=False, z_sup=False)
    if grid:
        return *bins.h2dgrid(h), z_.flatten()
    x, y = bins.h2d(h, arr=True)
    cut = np.where(z_ != 0) if z_sup else [..., ...]
    return x[cut[1]], y[cut[0]], z_[cut].flatten() if flat else z_[cut] if z_sup else z_
# endregion HISTOGRAM VALUES
# ----------------------------------------


def h_y(h):
    return graph_y(h) if 'Graph' in h.ClassName() else hist_values(h)


def h_x(h):
    return graph_x(h) if 'Graph' in h.ClassName() else bins.from_hist(h)


def h_xy(h):
    return h_x(h), h_y(h)


def set_bin_labels(g, labels):
    if labels is not None:
        for i, label in enumerate(labels):
            g.GetXaxis().SetBinLabel(g.GetXaxis().FindBin(i), label)
        update_canvas()


def make_box_args(x1, y1, x2, y2):
    return np.array([[x1, x1, x2, x2], [y1, y2, y2, y1]])


def make_poly_args(x, y, last_x=None):
    return np.append(x.repeat(2)[1:], choose(last_x, x[-1] + x[-1] - x[-2])), y.repeat(2)


def make_star(cx=0, cy=0, r=1, n=5):
    coods = pol2cart(np.tile([r, r / (2 * cos(np.pi / n) + 1)], n), np.linspace(0, 2 * np.pi, 2 * n, endpoint=False) - np.pi / 2)
    return (coods.T + np.array([cx, cy])).T


def set_titles(status=True):
    gStyle.SetOptTitle(status)


def shift_graph(g, ox=0, oy=0):
    if is_iter(g):
        return [shift_graph(ig, ox, oy) for ig in g]
    if 'Asym' in g.ClassName():
        x, y = graph_xy(g, as_u=False)
        for i in range(x.shape[0]):
            g.SetPoint(i, x[i][0] + ox, y[i][0] + oy)
            g.SetPointError(i, *x[i][1:], *y[i][1:])
    else:
        for i, (x, y) in enumerate(array(graph_xy(g)).T + [ox, oy]):
            g.SetPoint(i, x.n, y.n)
            if 'Error' in g.ClassName():
                g.SetPointError(i, x.s, y.s)
    return g


def get_3d_profiles(h, opt, err=True):
    px, py = [], []
    for ibin in range(1, h.GetNbinsX() + 1):
        h.GetXaxis().SetRange(ibin, ibin + 1)
        p = h.Project3D(opt)
        px.append(deepcopy(p.ProfileX()))
        py.append(deepcopy(p.ProfileY()))
    return bins.hx(h, err), px, py


def get_3d_correlations(h, opt='yz', thresh=.25, err=True, z_supp=True):
    corr = []
    for ibin in range(1, h.GetNbinsX() + 1):
        h.GetXaxis().SetRange(ibin, ibin + 1)
        corr.append(remove_low_stat_bins(h.Project3D(opt), thresh=thresh).GetCorrelationFactor())
    c = np.array(corr)
    return (bins.hx(h, err)[c != 0], c[c != 0]) if z_supp else (bins.hx(h, err), c)


def scale_graph(gr, scale=None, val=1, to_low_flux=False):
    x, y = graph_xy(gr)
    if scale is None:
        m, s = mean_sigma(y, err=False)
        scale = val / (y[np.where(x == min(x))[0]] if to_low_flux else m)
    for i in range(x.size):
        gr.SetPoint(i, gr.GetX()[i], gr.GetY()[i] * scale)
        gr.SetPointError(i, gr.GetErrorX(i), gr.GetErrorY(i) * scale) if 'Error' in gr.ClassName() else do_nothing()
    return scale


def get_quantile(h, q):
    quantiles = make_list(q)
    v = zeros(quantiles.size)
    h.GetQuantiles(v.size, v, quantiles)
    return v[0] if v.size == 1 else v


def markers(i, duo=False):
    return duo_markers(i) if duo else ((list(range(20, 24)) + [29, 33, 34]) * 2)[i]


def duo_markers(i):
    """ :returns full and open marker"""
    full_, open_ = [20, 21, 22, 23, 29, 33, 34], [24, 25, 26, 32, 30, 27, 28]
    return list(concatenate(array([full_, open_]).T).tolist())[i]  # noqa


def set_palette(pal):
    Draw.Palette = pal
    gStyle.SetPalette(*[len(pal), np.array(pal).astype('i')] if is_iter(pal) and type(pal) is not str else [pal])


def n_pal(n):
    return np.append(0, Draw.get_colors(n)).astype('i')


def set_n_palette(n):
    set_palette(n_pal(n))


def is_graph(h):
    return 'Graph' in h.ClassName()


def update_canvas(c=None):
    c = choose(c, get_last_canvas(warn=False))
    if c is not None:
        c.Modified()
        c.Update()
    return c


def show_colors(colors):
    n = len(colors)
    c = Draw.canvas(divide=(int(np.ceil(sqrt(n))), int(np.ceil(sqrt(n)))))
    for i, col in enumerate(colors, 1):
        c.cd(i)
        Draw.box(0, 0, 1, 1, fillstyle=Draw.Solid, fillcolor=col)
        Draw.tlatex(.5, .5, str(i - 1), align=22, size=.2)


def show_wheel():
    from ROOT import TColorWheel
    t = TColorWheel()
    t.Draw()
    Draw.add(t)


def show_line_styles():
    Draw.canvas(w=1.5, title='Line Styles')
    for i in range(1, 11):
        Draw.horizontal_line(1 / 11 * i, 0.1, .95, w=2, style=i)
        Draw.tlatex(.07, 1 / 11 * i, str(i), align=32)


def ax_range(low: Any = None, high=None, fl=0., fh=0., h=None, to_int=False, thresh=None):
    if type(low) in [list, np.ndarray]:
        utypes = [Variable, AffineScalarFunc]
        if len(low) == 2 and not is_ufloat(low[0]):
            return ax_range(low[0], low[1], fl, fh)
        m, s = mean_sigma(low, err=0)
        v = low[np.absolute(low - m) < thresh * s] if thresh is not None else low
        return ax_range(min(v).n if type(v[0]) in utypes else min(v), max(v).n if type(v[0]) in utypes else max(v), fl, fh, to_int=to_int)
    if h is not None:
        lo, hi = choose(thresh, low), choose(thresh, high)
        if 'TH2' in h.ClassName() or '2D' in h.ClassName():
            axes = enumerate([h.GetXaxis(), h.GetYaxis()], 1)
            return [ax_range(ax.GetBinCenter(h.FindFirstBinAbove(lo, i)), ax.GetBinCenter(h.FindLastBinAbove(hi, i)), fl, fh, to_int=to_int) for i, ax in axes]
        return ax_range(h.GetBinCenter(h.FindFirstBinAbove(lo)), h.GetBinCenter(h.FindLastBinAbove(hi)), fl, fh, to_int=to_int)
    return increase_range(low, high, fl, fh, to_int)


def find_z_range(h, q=None, z0=None):
    if q is not None:
        x = hist_values_2d(h, err=False, flat=True, z_sup=False)
        zmin, zmax = choose(z0, quantile, a=x, q=1 - q), quantile(x, q)
        return [zmin, zmax]


def set_drawing_range(h, legend=True, lfac=None, rfac=None, thresh=None):
    for i in range(1, 4):
        h.SetBinContent(i, 0)
    thresh = choose(thresh, .05 * h.GetMaximum())
    range_ = [h.GetBinCenter(i) for i in [h.FindFirstBinAbove(thresh), h.FindLastBinAbove(thresh)]]
    lfac = lfac if lfac is not None else .2
    rfac = rfac if rfac is not None else .55 if legend else .1
    h.GetXaxis().SetRangeUser(*ax_range(range_, lfac, rfac))


def normalise_histo(histo, x_range=None, from_min=False):
    h = histo
    x_axis = h.GetXaxis()
    x_axis.SetRangeUser(*x_range) if x_range is not None else do_nothing()
    min_bin = h.GetMinimumBin() if from_min else 1
    integral = h.Integral(min_bin, h.GetNbinsX() - 2)
    return scale_histo(h, integral)


def normalise_bins(h):
    px = h.ProjectionX()
    for xbin in range(h.GetNbinsX()):
        for ybin in range(h.GetNbinsY()):
            h.SetBinContent(xbin, ybin, h.GetBinContent(xbin, ybin) / (px.GetBinContent(xbin) if px.GetBinContent(xbin) else 1))
    update_canvas()


def set_z_range(zmin, zmax):
    c = get_last_canvas()
    h = c.GetListOfPrimitives()[1]
    h.GetZaxis().SetRangeUser(zmin, zmax)


def set_axes_range(xmin, xmax, ymin, ymax, c=None):
    set_x_range(xmin, xmax, c)
    set_y_range(ymin, ymax, c)
    update_canvas()


def get_ax_range(h, d='x'):
    ax = getattr(h, f'Get{d.title()}axis')()
    return [ax.GetXmin(), ax.GetXmax()]


def get_dax(h, d='x'):
    return np.diff(get_ax_range(h, d))[0]


def set_x_range(xmin, xmax, c=None):
    c = choose(c, get_last_canvas())
    h = c.GetListOfPrimitives()[1]
    h.GetXaxis().SetRangeUser(xmin, xmax)


def set_y_range(ymin, ymax, c=None):
    c = choose(c, get_last_canvas())
    h = c.GetListOfPrimitives()[1]
    h.GetYaxis().SetRangeUser(ymin, ymax)


def get_last_canvas(warn=True):
    try:
        return gROOT.GetListOfCanvases()[-1]
    except IndexError:
        warning('There is no canvas is in the list...', prnt=warn)


def close_last_canvas():
    get_last_canvas().Close()


def get_object(name):
    if name is not None:
        o = gROOT.FindObject(name)
        return None if o.__class__.Class_Name() == 'TObject' else o


def set_time_axis(histo, form='%H:%M', off=0, axis='X'):
    ax = getattr(histo, f'Get{axis.upper()}axis')()
    ax.SetTimeFormat(form)
    ax.SetTimeOffset(-off - 3600 if off else 0)
    ax.SetTimeDisplay(1)
    update_canvas()


def find_mpv_fwhm(histo, nbins=15):
    max_bin = histo.GetMaximumBin()
    fit = TF1('fit', 'gaus', 0, 500)
    histo.Fit('fit', 'qs0', '', histo.GetBinCenter(max_bin - nbins), histo.GetBinCenter(max_bin + nbins))
    mpv = ufloat(fit.GetParameter(1), fit.GetParError(1))
    fwhm = histo.FindLastBinAbove(fit(mpv.n) / 2) - histo.FindFirstBinAbove(fit(mpv.n) / 2)
    return mpv, fwhm, mpv / fwhm


def get_fw_center(h):
    (low, high), bw = get_fwhm(h, ret_edges=True), h.GetBinWidth(1)
    return mean([add_err(low, bw), add_err(high, bw)])  # center of FWHM as MPV


def find_mpv(h, r=.8, show_fit=False):
    b, y = [f(hist_values(h, err=False)) for f in [np.argsort, sorted]]
    bmax, ymax = (b[-1] + 1, y[-1]) if y[-1] < 2 * y[-2] else (b[-2] + 1, y[-2])
    fit_range = [f(ymax * r) for f in [h.FindFirstBinAbove, h.FindLastBinAbove]]
    fit_range = fit_range if np.diff(fit_range)[0] > 5 else (bmax + np.array([-5, 5])).tolist()
    yfit, xfit = FitRes(h.Fit('gaus', f'qs{"" if show_fit else"0"}', '', *[h.GetBinCenter(i) for i in fit_range]))[:2]  # fit the top with a gaussian to get better maxvalue
    return (xfit, yfit) if abs(yfit - ymax) < .2 * ymax else (h.GetBinCenter(int(bmax)) + ufloat(0, h.GetBinWidth(1) / 2), ymax * ufloat(1, .02))  # check if fit value is reasonable ...


def get_fwhm(h, fit_range=.8, ret_edges=False, err=True):
    half_max = find_mpv(h, fit_range)[1] * .5
    low, high = [ufloat(v.n, v.s + abs(v.n - i.n)) for v, i in zip(_get_fwhm(h, half_max), _get_fwhm(h, half_max - half_max.s))]
    return ((low, high) if err else (low.n, high.n)) if ret_edges else high - low


def _get_fwhm(h, half_max):
    blow, bhigh, w = h.FindFirstBinAbove(half_max.n), h.FindLastBinAbove(half_max.n), h.GetBinWidth(1)
    low = get_x(h.GetBinCenter(blow - 1), h.GetBinCenter(blow), h.GetBinContent(blow - 1), h.GetBinContent(blow), half_max)
    high = get_x(h.GetBinCenter(bhigh), h.GetBinCenter(bhigh + 1), h.GetBinContent(bhigh), h.GetBinContent(bhigh + 1), half_max)
    return low, high


def fit_fwhm(h, fitfunc='gaus', show=False, fit_range=.8):
    low, high = get_fwhm(h, fit_range, ret_edges=True)
    return FitRes(h.Fit(fitfunc, 'qs{}'.format('' if show else 0), '', low.n, high.n))


def get_f_fwhm(f: TF1):
    half_max = f.GetMaximum() / 2
    return f.GetX(half_max, f.GetMaximumX(), 1e9) - f.GetX(half_max)


def scale_histo(histo, value=None, to_max=False, x_range=None):
    h = histo
    maximum = h.GetBinContent(h.GetMaximumBin())
    if x_range is not None:
        h.GetXaxis().SetRangeUser(*x_range) if x_range is not None else do_nothing()
        maximum = h.GetBinContent(h.GetMaximumBin())
        h.GetXaxis().UnZoom()
    value = maximum if to_max else value
    if value:
        h.Scale(1 / value)
    return h


def find_2d_centre(h, thresh=.5):
    px, py = h.ProjectionX(), h.ProjectionY()
    return np.array([mean([p.GetBinCenter(b) for b in [p.FindFirstBinAbove(p.GetMaximum() * thresh), p.FindLastBinAbove(p.GetMaximum() * thresh)]]) for p in [px, py]])


def get_2d_centre_ranges(h, dx, dy=None, thresh=.5):
    if dx is None:
        return None, None
    cx, cy = find_2d_centre(h, thresh)
    dx, dy = dx / 2, choose(dy, dx) / 2
    return [cx - dx, cx + dx], [cy - dy, cy + dy]


def centre_2d(h, dx, dy=None, thresh=.5):
    c = get_last_canvas()
    set_axes_range(*np.concatenate(get_2d_centre_ranges(h, dx, dy, thresh)), c)


def make_transparent(pad):
    pad.SetFillStyle(4000)
    pad.SetFillColor(0)
    pad.SetFrameFillStyle(4000)


def hide_axis(axis):
    axis.SetTickLength(0)
    axis.SetLabelOffset(99)
    axis.SetTitleOffset(99)


def remove_low_stat_bins(h, q=.9, thresh=None):
    if h.GetEntries() > 0:
        e = bins.entries_2d(h) if 'Profile' in h.ClassName() else hist_values_2d(h, err=False, z_sup=False, flat=False)
        e0 = e.flatten()
        t = quantile(e0[e0 > 0], q) if thresh is None else thresh * h.GetMaximum()
        e0[e0 < t] = 0
        (bins.set_2d_entries if 'Profile' in h.ClassName() else bins.set_2d_values)(h, e0.reshape(e.shape))
        update_canvas()
    return h


def get_correlation_arrays(m1, m2, sx=0, sy=0, thresh=.1, flat=False):
    a1, a2 = [hist_values_2d(sm, err=False, flat=False) for sm in [m1, m2]]
    n1, n2 = [bins.entries_2d(sm) for sm in [m1, m2]]
    a1[n1 < thresh * n1.max()] = 0  # set bins with low stats to 0
    a2[n2 < thresh * n2.max()] = 0
    a2 = np.roll(a2, [sx, sy], axis=[0, 1])  # shift through second np.array
    return (a1.flatten(), a2.flatten()) if flat else (a1, a2)


def correlate_maps(m1, m2, sx=0, sy=0, thresh=.1):
    a1, a2 = (m1, m2) if type(m1) is np.ndarray else get_correlation_arrays(m1, m2, thresh=thresh)
    return correlate(a1, np.roll(a2, [sx, sy], axis=[0, 1]))


def correlate_all_maps(m1, m2, thresh=.1):
    a1, a2 = get_correlation_arrays(m1, m2, thresh=thresh)
    return np.array([[correlate_maps(a1, a2, x, y) for x in range(a1.shape[0])] for y in range(a1.shape[1])])


def set_root_warnings(status, fatal=False):
    gROOT.ProcessLine('gErrorIgnoreLevel = {e};'.format(e='0' if status else 'kFatal' if fatal else 'kError'))


def set_root_output(status=True):
    gROOT.SetBatch(not status)
    set_root_warnings(status)


def is_root_object(o):
    return hasattr(o, 'GetName')


def np_profile(x, y, u=False):
    with catch_warnings():
        simplefilter("ignore")
        m, s, n = [binned_statistic(x, y.astype('d'), bins=bins.n(x), statistic=stat) for stat in ['mean', 'std', 'count']]
        c = n[0] > 1
        b, m, s, n = m[1], m[0][c], s[0][c], n[0][c]
        return ((b[:-1] + np.diff(b) / 2)[c], ) + ((arr2u(m, s / sqrt(n)), ) if u else (m, s / sqrt(n)))


if __name__ == '__main__':
    z = Draw()
