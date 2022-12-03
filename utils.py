# --------------------------------------------------------
#       PLOTTING UTILITY FUNCTIONS
# created on October 27th 2021 by M. Reichmann
# --------------------------------------------------------
from configparser import ConfigParser, NoOptionError, NoSectionError
from copy import deepcopy
from datetime import datetime
from json import loads, load
from os import _exit, makedirs, remove
from os.path import exists, isfile, join, sep
from time import time
from pathlib import Path
from subprocess import check_call, check_output

from numpy import array, zeros, count_nonzero, sqrt, average, full, all, quantile, arctan2, cos, sin, corrcoef, isfinite, mean
from uncertainties import ufloat_fromstr, ufloat
from uncertainties.core import Variable, AffineScalarFunc
from inspect import getframeinfo, stack


BaseDir = Path(__file__).resolve().parent.parent


ON = True
OFF = False

GREEN = '\033[92m'
WHITE = '\033[98m'
ENDC = '\033[0m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RED = '\033[91m'
UP1 = '\033[1A'
ERASE = '\033[K'

COUNT = 0


def get_t_str():
    return datetime.now().strftime('%H:%M:%S')


def colored(txt, color=None):
    return f'{color}{txt}{ENDC}' if color else txt


def prnt_msg(txt, head='', color=None, blank_lines=0, endl=True, prnt=True):
    if prnt:
        print('\n' * blank_lines + f'\r{colored(f"{head}:", color):<18} {get_t_str()} --> {txt}', end='\n' if endl else ' ')


def info(txt, blank_lines=0, endl=True, prnt=True):
    prnt_msg(txt, 'INFO', GREEN, blank_lines, endl, prnt)
    return time()


def add_to_info(t, msg='Done', color=None, prnt=True):
    print(colored(f'{msg} ({time() - t:2.2f} s)', color)) if prnt else do_nothing()


def warning(txt, blank_lines=0, prnt=True):
    prnt_msg(txt, 'WARNING', YELLOW, blank_lines, prnt=prnt)


def critical(txt):
    i = getframeinfo(stack()[1][0])
    print(f'{sep.join(Path(i.filename).parts[-2:])}: {i.lineno}')
    prnt_msg(txt, 'CRITICAL', RED)
    _exit(2)


def get_stat(status):
    return 'ON' if status else 'OFF'


def choose(v, default, decider='None', *args, **kwargs):
    use_default = decider is None if decider != 'None' else v is None  # noqa
    if callable(default) and use_default:
        default = default(*args, **kwargs)
    return default if use_default else v(*args, **kwargs) if callable(v) else v


def round_up_to(num, val=1):
    return int(num) // val * val + val


def do(fs, pars, exe=-1):
    fs, pars = ([fs], [pars]) if type(fs) is not list else (fs, pars)  # noqa
    exe = pars if exe == -1 else [exe]
    for f, p, e in zip(fs, pars, exe):
        f(p) if e is not None else do_nothing()


def do_nothing():
    pass


def is_iter(v):
    try:
        iter(v)
        return True
    except TypeError:
        return False


def is_ufloat(value):
    return type(value) in [Variable, AffineScalarFunc, AsymVar]


def uarr2n(x):
    return array([i.n for i in x]) if len(x) and is_ufloat(x[0]) else x


def uarr2s(arr):
    return array([i.s for i in arr]) if len(arr) and is_ufloat(arr[0]) else arr


def arr2u(x, ex):
    return array([ufloat(i, e) for i, e in zip(x, ex)])


def add_err(u, e):
    return u + ufloat(0, e)


def add_perr(u, e):
    return u * ufloat(1, e)


def eff2u(eff):
    return ufloat(eff[0], mean(eff[1:])) if eff.shape == (3,) else array([eff2u(e) for e in eff])


def make_ufloat(n, s=0):
    return (eff2u(n) if len(n) == 3 and s == 0 else array([ufloat(*v) for v in array([n, s]).T])) if is_iter(n) else n if is_ufloat(n) else ufloat(n, s)


def make_list(value):
    return array([value], dtype=object).flatten()


def prep_kw(dic, **default):
    d = deepcopy(dic)
    for kw, value in default.items():
        if kw not in d:
            d[kw] = value
    return d


def get_kw(kw, kwargs, default=None):
    return kwargs[kw] if kw in kwargs else default


def rm_key(d, *key):
    d = deepcopy(d)
    for k in key:
        if k in d:
            del d[k]
    return d


def mean_sigma(values, weights=None, err=True):
    """ Return the weighted average and standard deviation. values, weights -- Numpy ndarrays with the same shape. """
    if len(values) == 1:
        value = make_ufloat(values[0])
        return (value, ufloat(value.s, 0)) if err else (value.n, value.s)
    weights = full(len(values), 1) if weights is None else weights
    # variance defined weights: https://en.wikipedia.org/wiki/Inverse-variance_weighting
    use_variance = is_ufloat(values[0])
    if use_variance:
        errors = uarr2s(values)
        weights = full(errors.size, 1) if all(errors == errors[0]) else [1 / e ** 2 if e else 0 for e in errors]
        values = uarr2n(values)
    if all(weights == 0):
        return [0, 0]
    n, avrg = values.size, average(values, weights=weights)
    sigma = sqrt(n / (n - 1) * average((values - avrg) ** 2, weights=weights))  # Fast and numerically precise
    m = ufloat(avrg, sqrt(1 / sum(weights))) if use_variance else ufloat(avrg, sigma / sqrt(len(values) - 1))
    s = ufloat(sigma, sigma / sqrt(2 * len(values)))
    return (m, s) if err else (m.n, s.n)


def calc_eff(k=0, n=0, values=None):
    values = array(values) if values is not None else None
    if n == 0 and (values is None or not values.size):
        return zeros(3)
    k = float(k if values is None else count_nonzero(values))
    n = float(n if values is None else values.size)
    m = (k + 1) / (n + 2)
    mode = k / n
    s = sqrt(((k + 1) / (n + 2) * (k + 2) / (n + 3) - ((k + 1) ** 2) / ((n + 2) ** 2)))
    return array([mode, max(s + (mode - m), 0), max(s - (mode - m), 0)]) * 100


def freedman_diaconis(x):
    return 2 * (quantile(x, .75) - quantile(x, .25)) / x.size ** (1 / 3)


def bin_width(x):
    w = freedman_diaconis(x[isfinite(x)])
    return w if w else 3.49 * mean_sigma(x)[1].n / x.size ** (1 / 3)


def cart2pol(x, y):
    return array([sqrt(x ** 2 + y ** 2), arctan2(y, x)])


def pol2cart(rho, phi):
    return array([rho * cos(phi), rho * sin(phi)])


def get_x(x1, x2, y1, y2, y):
    return (x2 - x1) / (y2 - y1) * (y - y1) + x1


def get_y(x1, x2, y1, y2, x):
    return get_x(y1, y2, x1, x2, x)


def ensure_dir(path):
    if not exists(path):
        info('Creating directory: {d}'.format(d=path))
        makedirs(path)
    return path


def remove_file(*file_path, string=None, warn=True):
    for f in file_path:
        if Path(f).exists():
            warning(f'removing {choose(string, f)}', prnt=warn)
            remove(f)


def correlate(l1, l2):
    if len(l1.shape) == 2:
        x, y = l1.flatten(), l2.flatten()
        cut, s = (x > 0) & (y > 0), count_nonzero(x)
        return correlate(x[cut], y[cut]) if count_nonzero(cut) > .6 * s else 0
    return corrcoef(l1, l2)[0][1]


def add_spaces(s):
    return ''.join(f' {s[i]}' if i and (s[i].isupper() or s[i].isdigit()) and not s[i - 1].isdigit() and not s[i - 1].isupper() else s[i] for i in range(len(s)))


def print_check(reset=False):
    global COUNT
    COUNT = 0 if reset else COUNT
    print('======={}========'.format(COUNT))
    COUNT += 1


def sum_times(t, fmt='%H:%M:%S'):
    return sum(array([datetime.strptime(i, fmt) for i in t]) - datetime.strptime('0', '%H'))


def load_json(filename):
    if not isfile(filename):
        warning(f'json file does not exist: {filename}')
        return {}
    with open(filename) as f:
        return load(f)


class Config(ConfigParser):

    def __init__(self, file_name, section=None, from_json=False, required=False, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.FilePath = Path(file_name)
        if required and not self.FilePath.exists():
            critical(f'{self!r} does not exist!')
        self.read_dict(load_json(file_name)) if from_json else self.read(file_name) if type(file_name) is not list else self.read_file(file_name)
        self.Section = self.check_section(section)

    def __call__(self, section):
        return Config(self.FilePath, section)

    def __repr__(self):
        return f'{self.__class__.__name__}: {join(*self.FilePath.parts[-2:])}' + (f' (section = {self.Section})' if hasattr(self, 'Section') and self.Section else '')

    def options(self, section=None):
        return super().options(choose(section, self.Section))

    def check_section(self, section):
        return section if section is None or section in self else critical(f'No section {section} in {self}')

    def set_section(self, sec):
        self.Section = self.check_section(sec)

    def get_value(self, section, option=None, dtype: type = str, default=None):
        dtype = type(default) if default is not None else dtype
        s, o = (self.Section, section) if option is None else (section, option)
        try:
            if dtype is bool:
                return self.getboolean(s, o)
            v = self.get(s, o)
            return loads(v.replace('\'', '\"')) if '[' in v or '{' in v and dtype is not str else dtype(v)
        except (NoOptionError, NoSectionError, ValueError):
            return default

    def get_values(self, section=None):
        return [*self[choose(section, self.Section)].values()]

    def get_list(self, section, option=None, default=None):
        return self.get_value(section, option, list, choose(default, []))

    def get_float(self, section: str, option: str = None) -> float:
        return self.get_value(section, option, float)

    def get_ufloat(self, section, option=None, default=None):
        return ufloat_fromstr(self.get_value(section, option, default=default))

    def show(self):
        for key, section in self.items():
            print(colored(f'[{key}]', YELLOW))
            for option in section:
                print(f'{option} = {self.get(key, option)}')
            print()

    def write(self, file_name=None, space_around_delimiters=True):
        with open(choose(file_name, self.FilePath), 'w') as f:
            super(Config, self).write(f, space_around_delimiters)


class AsymVar:

    def __init__(self, value, err_down, err_up, fmt='.2f'):
        self.NominalValue = float(value)
        self.ErrDown = float(err_down)
        self.ErrUp = float(err_up)
        self.Format = fmt

    def __len__(self):
        return 3

    def __str__(self):
        return self.format()

    def __repr__(self):
        return self.format()

    def __gt__(self, other):
        return self.n > (other.n if type(other) is AsymVar else other)

    def __lt__(self, other):
        return self.n < (other.n if type(other) is AsymVar else other)

    def __neg__(self):
        return AsymVar(-self.n, self.s0, self.s1)

    def __add__(self, other):
        if other == self:
            self.NominalValue *= 2
            self.ErrDown *= 2
            self.ErrUp *= 2
        elif type(other) is AsymVar:
            self.NominalValue += other.n
            self.ErrDown = (ufloat(0, self.s0) + ufloat(0, other.s1)).s
            self.ErrUp = (ufloat(0, self.s1) + ufloat(0, other.s0)).s
        elif is_ufloat(other):
            self.NominalValue += other.n
            self.ErrDown = (ufloat(0, self.s0) + other).s
            self.ErrUp = (ufloat(0, self.s1) + other).s
        else:
            self.NominalValue += float(other)
        return self

    def __sub__(self, other):
        return self.__add__(-other)

    def __truediv__(self, other):
        if is_ufloat(other):
            return warning('not implemented')
        return AsymVar(*self.a / other)

    def __mul__(self, other):
        if is_ufloat(other):
            return warning('not implemented')
        return AsymVar(*self.a * other)

    def __iter__(self):
        return iter([self.n, self.s0, self.s1])

    def __getitem__(self, item):
        return [self.NominalValue, self.ErrUp, self.ErrDown][item]

    def __format__(self, f):
        return f'{self.n:{f}}+{self.s0:{f}}-{self.s1:{f}}'

    def format(self, fmt=None, unit=''):
        return f'({self.__format__(choose(fmt, self.Format))}){unit}'

    @property
    def to_ufloat(self):
        return ufloat(self.n, mean(self.s))
    u = to_ufloat

    @property
    def to_array(self):
        return array(self)
    a = to_array

    @property
    def nominal_value(self):
        return self.NominalValue
    n = nominal_value  # abbreviation

    @property
    def lower_error(self):
        return self.ErrDown
    s0 = lower_error

    @property
    def upper_error(self):
        return self.ErrUp
    s1 = upper_error

    @property
    def error(self):
        return array([self.s0, self.s1])
    s = error

    @property
    def flip_errors(self):
        return AsymVar(self.n, self.s1, self.s0)
    f = flip_errors


def aufloat(n, s0=0, s1=0):
    return AsymVar(n, s0, s1)


def add_asym_error(v, s0=0, s1=0):
    return array([add_asym_error(i, s0, s1) for i in v], dtype=AsymVar) if is_iter(v) else AsymVar(0, s0, s1) + v


def download_file(server, loc, target, out=True):
    cmd = f'rsync -aPvL {server}:{loc} {target}'
    return (check_call if out else check_output)(cmd, shell=True)
