from numpy import array


def f(name, *args):
    return f'\\{name}' + ''.join(f'{{{i}}}' for i in args)


def multirow(txt, n, pos='*'):
    return f('multirow', n, pos, txt)


def makecell(*txt):
    return f('makecell', '\\\\'.join(txt))


def bold(*txt):
    return [f('textbf', i) for i in txt]


def unit(txt, custom=False):
    return f('unit', '' if custom else "\\" + txt)


def si(*v, fmt='.1f', unt=None):
    return num(*v, fmt=fmt) if unt is None else [f('SI', f'{float(i):{fmt}}', f'\\{unt}').replace('/', '') for i in v]


def num(*v, fmt='.1f'):
    return [f('num', f'{float(i):{fmt}}').replace('/', '') for i in v]


def si_range(v0, v1, fmt='.0f', unt=None):
    return num_range(v0, v1, fmt) if unt is None else f('SIrange', f'{float(v0):{fmt}}', f'{float(v1):{fmt}}', f'\\{unt}')


def num_range(v0, v1, fmt='.0f'):
    return f('numrange', f'{float(v0):{fmt}}', f'{float(v1):{fmt}}')


def hline(word):
    return word + ' \\\\' + f('hline') if 'hline' not in word else word


def table_row(*words, endl=False):
    row = f'  { " & ".join(words)}'
    return hline(row) if endl or 'hline' in row else f'{row} \\\\'


def table(header, rows, endl=False):
    cols = array(rows, str).T
    max_width = [len(max(col, key=len).replace(' \\\\\\hline', '')) for col in cols]  # noqa
    rows = array([[f'{word:<{w}}' for word in col] for col, w in zip(cols, max_width)]).T
    rows = '\n'.join(table_row(*row, endl=endl) for row in rows)
    return f'{table_row(*header, endl=True)}\n{rows}' if header else rows
