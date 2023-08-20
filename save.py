#!/usr/bin/env python
# --------------------------------------------------------
#       Class for saving the root plots
# created on September 25th 2020 by M. Reichmann (remichae@phys.ethz.ch)
# --------------------------------------------------------

from ROOT import TFile

from . import html
from .draw import *
from .utils import BaseDir


class SaveDraw(Draw):

    Save = True
    SaveOnServer = True

    ServerMountDir: Path = None
    Dummy = TFile(str(Draw.Dir.joinpath('dummy.root')), 'RECREATE')

    def __init__(self, analysis=None, results_dir='', sub_dir=''):
        self.Analysis = analysis
        super(SaveDraw, self).__init__(self.find_config())

        self.File = None

        # INFO
        SaveDraw.Save = Draw.Config.get_value('SAVE', 'save', default=False)

        # Results
        self.ResultsDir = BaseDir.joinpath('results', results_dir)
        self.SubDir = str(sub_dir)

        # Server
        SaveDraw.ServerMountDir = Path(Draw.Config.get_value('SAVE', 'server mount directory', default=None)).expanduser()

    def __del__(self):
        remove_file(join(self.Dir, 'dummy.root'), warn=False)

    # ----------------------------------------
    # region INIT
    def find_config(self):
        if hasattr(self.Analysis, 'MainConfig'):
            return self.Analysis.MainConfig.FilePath
        if hasattr(self.Analysis, 'Config'):
            return self.Analysis.Config.FilePath

    @property
    def mount_exists(self):
        x = False if SaveDraw.ServerMountDir is None else SaveDraw.ServerMountDir.joinpath('data').exists()
        if not x:
            warning(f'Diamond server is not mounted in {SaveDraw.ServerMountDir}')
        return x

    @property
    def server_dir(self):
        if self.Analysis is not None:
            return SaveDraw.ServerMountDir.joinpath('content', self.Analysis.server_save_dir)

    def init_info(self):
        return super().init_info() if self.Analysis is None or not hasattr(self.Analysis, 'InfoLegend') else self.Analysis.InfoLegend(self)

    @property
    def file_name(self):
        d = self.server_dir
        return None if d is None else d.joinpath('plots.root')
    # endregion INIT
    # ----------------------------------------

    # ----------------------------------------
    # region SET
    def open_file(self, *exclude, prnt=False):
        if self.File is None or exclude:
            pal = Draw.Palette
            info('opening ROOT file on server ...', prnt=prnt)
            data = {}
            if self.file_name.exists():
                if self.file_name.stat().st_size < 1000:   # file must be corrupted or empty
                    self.rm_plots()
                f0 = TFile(str(self.file_name), 'UPDATE')
                data = {key.GetName(): f0.Get(key.GetName()) for key in f0.GetListOfKeys()}
            f = TFile(str(self.file_name), 'RECREATE')
            set_palette(pal)
            for key, c in data.items():
                if c and key not in exclude:
                    c.Write(key)
            f.Write()
            self.File = f

    def close_file(self):
        if self.File is not None:
            self.File.Close()
            self.File = None

    def rm_plots(self):
        remove_file(self.file_name)

    def remove_plots(self, *exclude):
        self.open_file(*exclude, prnt=False)

    def create_overview(self, x=4, y=3, redo=True):
        if self.server_dir is not None:
            html.create_tree(self.file_name.with_name('tree.html'))
            if not self.file_name.with_suffix('.html').exists() or redo:
                html.create_root_overview(self.file_name, x, y, verbose=self.Verbose)

    def set_sub_dir(self, name):
        self.SubDir = name

    def set_results_dir(self, name):
        self.ResultsDir = join(Draw.Dir, 'Results', name)
    # endregion SET
    # ----------------------------------------

    # ----------------------------------------
    # region SAVE
    def save_full(self, h, filename, cname='c', **kwargs):
        self(h, **prep_kw(kwargs, show=False, save=False))
        self.save_plots(None, full_path=join(self.Dir, filename), show=False, cname=cname, **kwargs)

    def histo(self, histo, file_name=None, show=True, prnt=True, save=True, info_leg=True, all_pads=False, fn=None, *args, **kwargs):
        c = super(SaveDraw, self).histo(histo, show, info_leg=False, *args, **kwargs)
        if info_leg:
            self.Info.draw(c, all_pads)
        histo.SetTitle('') if not Draw.Title else do_nothing()
        self.save_plots(choose(fn, file_name), prnt=prnt, show=show, save=save)
        return c

    def save_plots(self, savename, sub_dir=None, canvas=None, full_path=None, prnt=True, ftype=None, show=True, save=True, cname=None, **kwargs):
        """ Saves the canvas at the desired location. If no canvas is passed as argument, the active canvas will be saved. However, for applications without graphical interface,
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
        file_path = Path(join(choose(res_dir, self.ResultsDir), choose(sub_dir, self.SubDir), file_name) if full_path is None else full_path)
        ensure_dir(file_path.parent)
        info(f'saving plot: {file_path.name}', prnt=prnt and self.Verbose)
        canvas.Update()
        Draw.set_show(show)  # needs to be in the same batch so that the pictures are created, takes forever...
        set_root_warnings(False)
        for f in choose(make_list(ftype), default=['pdf'], decider=ftype):
            canvas.SaveAs(f'{file_path}.{f.strip(".")}')
        self.save_on_server(canvas, file_path.name, save=full_path is None, prnt=prnt)
        Draw.set_show(True)

    def print_http(self, file_name, prnt=True, force_print=False):
        prnt = force_print or prnt and Draw.Verbose and not Draw.Show
        info(join('https://diamond.ethz.ch', self.ServerMountDir.name, Path(self.server_dir, file_name).relative_to(self.ServerMountDir)), prnt=prnt)

    def save_on_server(self, canvas, file_name, save=True, prnt=True):
        d = self.server_dir
        if d is not None and save and SaveDraw.SaveOnServer and self.mount_exists:
            d.mkdir(parents=True, exist_ok=True)
            p = d.joinpath(f'{Path(file_name).stem}.html')
            self.open_file(file_name)
            html.create_root(p, title=p.parent.name, pal=55 if is_iter(Draw.Palette) else Draw.Palette, verbose=self.Verbose)
            self.File.cd()
            canvas.Write(file_name)
            self.File.Write()
            SaveDraw.Dummy.cd()
            self.print_http(p.name, prnt)
            self.create_overview(redo=False)
            self.close_file()

    @staticmethod
    def save_last(canvas=None, ext='pdf', prnt=None):
        filename = BaseDir.joinpath('tmp', f'{input(f"Enter the name of the {ext}-file: ").split(".")[0]}.{ext}')
        ensure_dir(filename.parent)
        choose(canvas, get_last_canvas()).SaveAs(str(filename))
        info(f'saved to: {filename}', prnt=choose(prnt, Draw.Verbose))
        return filename

    @property
    def sl(self):
        return self.save_last()
    # endregion SAVE
    # ----------------------------------------


if __name__ == '__main__':
    z = SaveDraw()
