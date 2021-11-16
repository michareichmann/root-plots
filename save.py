#!/usr/bin/env python
# --------------------------------------------------------
#       Class for saving the root plots
# created on September 25th 2020 by M. Reichmann (remichae@phys.ethz.ch)
# --------------------------------------------------------

from os.path import expanduser, basename, isdir
from pathlib import Path
from ROOT import TFile

from . import html
from .draw import *
from .utils import BaseDir


class SaveDraw(Draw):

    Save = True

    ServerMountDir = None
    MountExists = None
    File = None
    Dummy = TFile('dummy.root', 'RECREATE')

    def __init__(self, analysis=None, results_dir=None, sub_dir=''):
        self.Analysis = analysis
        super(SaveDraw, self).__init__(None if analysis is None else analysis.MainConfig.FileName)

        # INFO
        SaveDraw.Save = Draw.Config.get_value('SAVE', 'save', default=False)
        self.Legends = self.init_legends()  # TODO make standard legend

        # Results
        self.ResultsDir = join(BaseDir, 'Results', choose(results_dir, default='' if analysis is None else analysis.TCString))
        self.SubDir = str(sub_dir)

        # Server
        SaveDraw.ServerMountDir = expanduser(Draw.Config.get_value('SAVE', 'server mount directory', default=None))
        SaveDraw.server_is_mounted(analysis)
        self.ServerDir = self.load_server_save_dir()

    def __del__(self):
        if isfile('dummy.root'):
            remove_file('dummy.root', prnt=False)

    # ----------------------------------------
    # region INIT
    def init_legends(self,):
        try:
            from helpers.info_legend import InfoLegend
            return InfoLegend(self.Analysis)
        except (ImportError, AttributeError):
            return

    def load_server_save_dir(self):
        if self.Analysis is not None and SaveDraw.MountExists and SaveDraw.ServerMountDir is not None:
            if hasattr(self.Analysis, 'load_selections'):
                ensure_dir(join(SaveDraw.ServerMountDir, 'content', 'selections', str(self.Analysis)))
                return join(SaveDraw.ServerMountDir, 'content', 'selections', str(self.Analysis))
            if not hasattr(self.Analysis, 'DUT'):
                return
            run_string = f'RP-{self.Analysis.Ensemble.Name.lstrip("0").replace(".", "-")}' if hasattr(self.Analysis, 'RunPlan') else str(self.Analysis.Run.Number)
            return join(SaveDraw.ServerMountDir, 'content', 'diamonds', self.Analysis.DUT.Name, self.Analysis.TCString, run_string)
    # endregion INIT
    # ----------------------------------------

    # ----------------------------------------
    # region SET
    def open_file(self, *exclude, prnt=False):
        if SaveDraw.File is None or exclude:
            info('opening ROOT file on server ...', prnt=prnt)
            f = TFile(join(self.ServerDir, 'plots.root'), 'UPDATE')
            data = {key.GetName(): f.Get(key.GetName()) for key in f.GetListOfKeys()}
            f = TFile(join(self.ServerDir, 'plots.root'), 'RECREATE')
            for key, c in data.items():
                if c and key not in exclude:
                    c.Write(key)
            f.Write()
            SaveDraw.File = f

    def remove_plots(self, *exclude):
        self.open_file(*exclude, prnt=False)

    def create_overview(self, x=4, y=3, redo=True):
        if self.ServerDir is not None:
            p = Path(self.ServerDir, 'plots.root')
            html.create_tree(p.with_name('index.html'))
            if not p.with_suffix('.html').exists() or redo:
                html.create_root_overview(p, x, y)

    def set_sub_dir(self, name):
        self.SubDir = name

    def set_results_dir(self, name):
        self.ResultsDir = join(Draw.Dir, 'Results', name)
    # endregion SET
    # ----------------------------------------

    @staticmethod
    def server_is_mounted(ana):
        if ana is None:
            return False
        if SaveDraw.MountExists is not None:
            return SaveDraw.MountExists
        SaveDraw.MountExists = isdir(join(SaveDraw.ServerMountDir, 'data'))
        if not SaveDraw.MountExists:
            warning('Diamond server is not mounted in {}'.format(SaveDraw.ServerMountDir))

    # ----------------------------------------
    # region SAVE
    def save_full(self, h, filename, cname='c', **kwargs):
        self(h, **prep_kw(kwargs, show=False, save=False))
        self.save_plots(None, full_path=join(self.Dir, filename), show=False, cname=cname, **kwargs)

    def histo(self, histo, file_name=None, show=True, all_pads=False, prnt=True, save=True, info_leg=True, *args, **kwargs):
        c = super(SaveDraw, self).histo(histo, show, *args, **kwargs)
        histo.SetTitle('') if not Draw.Title else do_nothing()
        if info_leg:
            self.Legends.draw(c, all_pads, show and Draw.Legend)
        self.save_plots(file_name, prnt=prnt, show=show, save=save)
        return c

    def save_plots(self, savename, sub_dir=None, canvas=None, full_path=None, prnt=True, ftype=None, show=True, save=True, cname=None, **kwargs):
        """ Saves the canvas at the desired location. If no canvas is passed as argument, the active canvas will be saved. However for applications without graphical interface,
         such as in SSl terminals, it is recommended to pass the canvas to the method. """
        kwargs = prep_kw(kwargs, save=save, prnt=prnt, show=show)
        if not kwargs['save'] or not SaveDraw.Save or (savename is None and full_path is None):
            return
        canvas = get_last_canvas() if canvas is None else canvas
        if cname is not None:
            canvas.SetName(cname)
        update_canvas(canvas)
        try:
            self.__save_canvas(canvas, sub_dir=sub_dir, file_name=savename, ftype=ftype, full_path=full_path, **kwargs)
            return Draw.add(canvas)
        except Exception as inst:
            warning('Error saving plots ...:\n  {}'.format(inst))

    def __save_canvas(self, canvas, file_name, res_dir=None, sub_dir=None, full_path=None, ftype=None, prnt=True, show=True, **kwargs):
        """should not be used in analysis methods..."""
        _ = kwargs
        file_path = join(choose(res_dir, self.ResultsDir), choose(sub_dir, self.SubDir), file_name) if full_path is None else full_path
        file_name = basename(file_path)
        ensure_dir(dirname(file_path))
        info(f'saving plot: {file_name}', prnt=prnt and self.Verbose)
        canvas.Update()
        Draw.set_show(show)  # needs to be in the same batch so that the pictures are created, takes forever...
        set_root_warnings(False)
        for f in choose(make_list(ftype), default=['pdf'], decider=ftype):
            canvas.SaveAs('{}.{}'.format(file_path, f.strip('.')))
        self.save_on_server(canvas, file_name, save=full_path is None, prnt=prnt)
        Draw.set_show(True)

    def print_http(self, file_name, prnt=True, force_print=False):
        info(join('https://diamond.ethz.ch', 'psi2', Path(self.ServerDir, file_name).relative_to(self.ServerMountDir)), prnt=force_print or prnt and self.Verbose and not Draw.Show)

    def save_on_server(self, canvas, file_name, save=True, prnt=True):
        if self.ServerDir is not None and save:
            self.open_file()
            p = Path(self.ServerDir, f'{basename(file_name)}.html')
            if file_name in SaveDraw.File.GetListOfKeys():
                SaveDraw.File.Delete(f'{file_name};1')
            else:
                html.create_root(p, title=p.parent.name, pal=53 if 'SignalMap' in file_name else 55)
            SaveDraw.File.cd()
            canvas.Write(file_name)
            SaveDraw.File.Write()
            SaveDraw.Dummy.cd()
            self.print_http(p.name, prnt)
            self.create_overview(redo=False)

    @staticmethod
    def save_last(canvas=None, ext='pdf'):
        filename = input(f'Enter the name of the {ext}-file: ')
        choose(canvas, get_last_canvas()).SaveAs(join(BaseDir, f'{filename.split(".")[0]}.{ext}'))
    # endregion SAVE
    # ----------------------------------------


if __name__ == '__main__':
    z = SaveDraw()
