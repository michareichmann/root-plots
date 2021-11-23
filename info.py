#!/usr/bin/env python
# --------------------------------------------------------
#       Class to draw the info legend for an analysis class
# created on Jan 30th 2018 by M. Reichmann (remichae@phys.ethz.ch)
# --------------------------------------------------------

from os import chdir
from subprocess import check_output
from .utils import warning


class Info(object):

    ShowLegend = False

    def __init__(self, draw):
        self.Draw = draw
        self.ShowDate = draw.Config.get_value('SAVE', 'date', default=False)
        self.ShowGit = draw.Config.get_value('SAVE', 'git hash', default=False) and not self.ShowDate

        self.Objects = []

    def __str__(self):
        return '' if not any([self.ShowDate, self.ShowGit, self.ShowLegend]) else 'ON'

    def __repr__(self):
        on = ['OFF', 'ON']
        return f'Plotting {self.__class__.__name__}: legend {on[self.ShowLegend]}, git hash {on[self.ShowGit]}, date {on[self.ShowDate]}'

    def is_active(self):  # noqa
        return True

    def get(self, canvas=None):
        return self.draw(canvas)

    def draw_legend(self):  # noqa
        return False

    def draw(self, canvas=None, all_pads=True):
        """ draws the active information on the canvas """
        if not self.is_active():
            return
        if canvas is not None:
            canvas.cd()
            if canvas.GetBottomMargin() < .105 and self.ShowLegend:
                canvas.SetBottomMargin(0.15)
        else:
            canvas = gROOT.GetSelectedPad()
            if not canvas:
                return warning('Cannot access an active Pad')

        pads = [i for i in canvas.GetListOfPrimitives() if i.IsA().GetName() == 'TPad'] if all_pads else [canvas]
        leg, git = None, None
        for pad in [canvas] if not pads else pads:
            pad.cd()
            git = self.draw_git()
            leg = self.draw_legend()
            self.draw_date()
            pad.Modified()
        canvas.Modified()
        canvas.Update()
        return leg, git

    def draw_git(self):
        chdir(self.Draw.Dir)
        txt = 'git hash: {ver}'.format(ver=check_output(['git', 'describe', '--always']).decode('utf-8').strip('\n'))
        return self.Draw.tlatex(.9 if self.ShowLegend else 0.02, .02, txt, show=self.ShowGit, ndc=True, size=.02)

    def draw_date(self):
        x, y, align = (.995, .005, 31) if self.ShowLegend else (.005, .05, 12)
        self.Draw.date(x, y, align, size=.02, show=self.ShowDate)
