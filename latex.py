from numpy import array, column_stack


def f(name, *args):
    return f'\\{name}' + ''.join(f'{{{i}}}' for i in args)


def multirow(txt, n, pos='*'):
    return f('multirow', n, pos, txt)


def makecell(*txt):
    return f('makecell', '\\\\'.join(txt))


def bold(*txt):
    return [f('textbf', i) for i in txt]


def math(txt):
    return f'${txt}$'


def unit(*txt, custom=False):
    return f('unit', ''.join(f' {t}' if custom else f(t) for t in txt))


def si(*v, fmt='.1f', unt=None):
    return qty(*v, fmt=fmt, unt=unt)


def qty(*v, fmt='.1f', unt=None):
    return num(*v, fmt=fmt) if unt is None else [f('qty', f'{i:{fmt}}', f'\\{unt}').replace('/', '') for i in v]


def si_2err(*v, fmt='.1f', unt=None):
    return f('SIserr', *[f'{i:{fmt}}' for i in v], f(unt))


def num(*v, fmt='.1f', rm='@'):
    return [num_2err(i) if hasattr(i, '__len__') else f('num', f'{i:{fmt}}').replace('/', '').replace(rm, '') for i in v]


def num_2err(v, fmt='.1f'):
    return f('numerr', *[f'{i:{fmt}}' for i in v])


def si_range(v0, v1, fmt='.0f', unt=None):
    return qty_range(v0, v1, fmt, unt)


def qty_range(v0, v1, fmt='.0f', unt=None):
    return num_range(v0, v1, fmt) if unt is None else f('qtyrange', f'{v0:{fmt}}', f'{float(v1):{fmt}}', f'\\{unt}')


def num_range(v0, v1, fmt='.0f'):
    return f('numrange', f'{v0:{fmt}}', f'{v1:{fmt}}')


def hline(word):
    return word + ' \\\\' + f('hline') if 'hline' not in word else word


def table_row(*words, endl=False):
    row = f'  { " & ".join(words)}'
    return hline(row) if endl or 'hline' in row else f'{row} \\\\'


def table(header, rows, endl=False, align_header=False):
    cols = array(rows, str).T
    max_width = [len(max(col, key=len).replace(' \\\\\\hline', '')) for col in (column_stack([cols, header]) if align_header else cols)]  # noqa
    rows = array([[f'{word:<{w}}' for word in col] for col, w in zip(cols, max_width)]).T
    rows = '\n'.join(table_row(*row, endl=endl) for row in rows)
    return f'{table_row(*header, endl=True)}\n{rows}' if header else rows
