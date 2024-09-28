"""
Provides a simple table class.  A SimpleTable is essentially
a list of lists plus some formatting functionality.

Dependencies: the Python 2.5+ standard library.

Installation: just copy this module into your working directory (or
   anywhere in your pythonpath).

Basic use::

   mydata = [[11,12],[21,22]]  # data MUST be 2-dimensional
   myheaders = [ "Column 1", "Column 2" ]
   mystubs = [ "Row 1", "Row 2" ]
   tbl = SimpleTable(mydata, myheaders, mystubs, title="Title")
   print( tbl )
   print( tbl.as_csv() )

A SimpleTable is inherently (but not rigidly) rectangular.
You should create it from a *rectangular* (2d!) iterable of data.
Each item in your rectangular iterable will become the data
of a single Cell.  In principle, items can be any object,
not just numbers and strings.  However, default conversion
during table production is by simple string interpolation.
(So you cannot have a tuple as a data item *and* rely on
the default conversion.)

A SimpleTable allows only one column (the first) of stubs at
initilization, concatenation of tables allows you to produce tables
with interior stubs.  (You can also assign the datatype 'stub' to the
cells in any column, or use ``insert_stubs``.) A SimpleTable can be
concatenated with another SimpleTable or extended by another
SimpleTable. ::

    table1.extend_right(table2)
    table1.extend(table2)


A SimpleTable can be initialized with `datatypes`: a list of ints that
provide indexes into `data_fmts` and `data_aligns`.  Each data cell is
assigned a datatype, which will control formatting.  If you do not
specify the `datatypes` list, it will be set to ``range(ncols)`` where
`ncols` is the number of columns in the data.  (I.e., cells in a
column have their own datatype.) This means that you can just specify
`data_fmts` without bothering to provide a `datatypes` list.  If
``len(datatypes)<ncols`` then datatype assignment will cycle across a
row.  E.g., if you provide 10 columns of data with ``datatypes=[0,1]``
then you will have 5 columns of datatype 0 and 5 columns of datatype
1, alternating.  Corresponding to this specification, you should provide
a list of two ``data_fmts`` and a list of two ``data_aligns``.

Cells can be assigned labels as their `datatype` attribute.
You can then provide a format for that lable.
Us the SimpleTable's `label_cells` method to do this.  ::

    def mylabeller(cell):
        if cell.data is np.nan:
            return 'missing'

    mytable.label_cells(mylabeller)
    print(mytable.as_text(missing='-'))


Potential problems for Python 3
-------------------------------

- Calls ``next`` instead of ``__next__``.
  The 2to3 tool should handle that no problem.
  (We will switch to the `next` function if 2.5 support is ever dropped.)
- Let me know if you find other problems.

:contact: alan dot isaac at gmail dot com
:requires: Python 2.5.1+
:note: current version
:note: HTML data format currently specifies tags
:todo: support a bit more of http://www.oasis-open.org/specs/tr9503.html
:todo: add labels2formatters method, that associates a cell formatter with a
       datatype
:todo: add colspan support to Cell
:since: 2008-12-21
:change: 2010-05-02 eliminate newlines that came before and after table
:change: 2010-05-06 add `label_cells` to `SimpleTable`
"""
from statsmodels.compat.python import lmap, lrange
from itertools import cycle, zip_longest
import csv

def csv2st(csvfile, headers=False, stubs=False, title=None):
    """Return SimpleTable instance,
    created from the data in `csvfile`,
    which is in comma separated values format.
    The first row may contain headers: set headers=True.
    The first column may contain stubs: set stubs=True.
    Can also supply headers and stubs as tuples of strings.
    """
    pass

class SimpleTable(list):
    """Produce a simple ASCII, CSV, HTML, or LaTeX table from a
    *rectangular* (2d!) array of data, not necessarily numerical.
    Directly supports at most one header row,
    which should be the length of data[0].
    Directly supports at most one stubs column,
    which must be the length of data.
    (But see `insert_stubs` method.)
    See globals `default_txt_fmt`, `default_csv_fmt`, `default_html_fmt`,
    and `default_latex_fmt` for formatting options.

    Sample uses::

        mydata = [[11,12],[21,22]]  # data MUST be 2-dimensional
        myheaders = [ "Column 1", "Column 2" ]
        mystubs = [ "Row 1", "Row 2" ]
        tbl = text.SimpleTable(mydata, myheaders, mystubs, title="Title")
        print( tbl )
        print( tbl.as_html() )
        # set column specific data formatting
        tbl = text.SimpleTable(mydata, myheaders, mystubs,
            data_fmts=["%3.2f","%d"])
        print( tbl.as_csv() )
        with open('c:/temp/temp.tex','w') as fh:
            fh.write( tbl.as_latex_tabular() )
    """

    def __init__(self, data, headers=None, stubs=None, title='', datatypes=None, csv_fmt=None, txt_fmt=None, ltx_fmt=None, html_fmt=None, celltype=None, rowtype=None, **fmt_dict):
        """
        Parameters
        ----------
        data : list of lists or 2d array (not matrix!)
            R rows by K columns of table elements
        headers : list (or tuple) of str
            sequence of K strings, one per header
        stubs : list (or tuple) of str
            sequence of R strings, one per stub
        title : str
            title of the table
        datatypes : list of int
            indexes to `data_fmts`
        txt_fmt : dict
            text formatting options
        ltx_fmt : dict
            latex formatting options
        csv_fmt : dict
            csv formatting options
        hmtl_fmt : dict
            hmtl formatting options
        celltype : class
            the cell class for the table (default: Cell)
        rowtype : class
            the row class for the table (default: Row)
        fmt_dict : dict
            general formatting options
        """
        self.title = title
        self._datatypes = datatypes
        if self._datatypes is None:
            self._datatypes = [] if len(data) == 0 else lrange(len(data[0]))
        self._txt_fmt = default_txt_fmt.copy()
        self._latex_fmt = default_latex_fmt.copy()
        self._csv_fmt = default_csv_fmt.copy()
        self._html_fmt = default_html_fmt.copy()
        self._csv_fmt.update(fmt_dict)
        self._txt_fmt.update(fmt_dict)
        self._latex_fmt.update(fmt_dict)
        self._html_fmt.update(fmt_dict)
        self._csv_fmt.update(csv_fmt or dict())
        self._txt_fmt.update(txt_fmt or dict())
        self._latex_fmt.update(ltx_fmt or dict())
        self._html_fmt.update(html_fmt or dict())
        self.output_formats = dict(txt=self._txt_fmt, csv=self._csv_fmt, html=self._html_fmt, latex=self._latex_fmt)
        self._Cell = celltype or Cell
        self._Row = rowtype or Row
        rows = self._data2rows(data)
        list.__init__(self, rows)
        self._add_headers_stubs(headers, stubs)
        self._colwidths = dict()

    def __str__(self):
        return self.as_text()

    def __repr__(self):
        return str(type(self))

    def _add_headers_stubs(self, headers, stubs):
        """Return None.  Adds headers and stubs to table,
        if these were provided at initialization.
        Parameters
        ----------
        headers : list[str]
            K strings, where K is number of columns
        stubs : list[str]
            R strings, where R is number of non-header rows

        :note: a header row does not receive a stub!
        """
        pass

    def insert(self, idx, row, datatype=None):
        """Return None.  Insert a row into a table.
        """
        pass

    def insert_header_row(self, rownum, headers, dec_below='header_dec_below'):
        """Return None.  Insert a row of headers,
        where ``headers`` is a sequence of strings.
        (The strings may contain newlines, to indicated multiline headers.)
        """
        pass

    def insert_stubs(self, loc, stubs):
        """Return None.  Insert column of stubs at column `loc`.
        If there is a header row, it gets an empty cell.
        So ``len(stubs)`` should equal the number of non-header rows.
        """
        pass

    def _data2rows(self, raw_data):
        """Return list of Row,
        the raw data as rows of cells.
        """
        pass

    def pad(self, s, width, align):
        """DEPRECATED: just use the pad function"""
        pass

    def _get_colwidths(self, output_format, **fmt_dict):
        """Return list, the calculated widths of each column."""
        pass

    def get_colwidths(self, output_format, **fmt_dict):
        """Return list, the widths of each column."""
        pass

    def _get_fmt(self, output_format, **fmt_dict):
        """Return dict, the formatting options.
        """
        pass

    def as_csv(self, **fmt_dict):
        """Return string, the table in CSV format.
        Currently only supports comma separator."""
        pass

    def as_text(self, **fmt_dict):
        """Return string, the table as text."""
        pass

    def as_html(self, **fmt_dict):
        """Return string.
        This is the default formatter for HTML tables.
        An HTML table formatter must accept as arguments
        a table and a format dictionary.
        """
        pass

    def as_latex_tabular(self, center=True, **fmt_dict):
        """Return string, the table as a LaTeX tabular environment.
        Note: will require the booktabs package."""
        pass

    def extend_right(self, table):
        """Return None.
        Extend each row of `self` with corresponding row of `table`.
        Does **not** import formatting from ``table``.
        This generally makes sense only if the two tables have
        the same number of rows, but that is not enforced.
        :note: To extend append a table below, just use `extend`,
        which is the ordinary list method.  This generally makes sense
        only if the two tables have the same number of columns,
        but that is not enforced.
        """
        pass

    def label_cells(self, func):
        """Return None.  Labels cells based on `func`.
        If ``func(cell) is None`` then its datatype is
        not changed; otherwise it is set to ``func(cell)``.
        """
        pass

def pad(s, width, align):
    """Return string padded with spaces,
    based on alignment parameter."""
    pass

class Row(list):
    """Provides a table row as a list of cells.
    A row can belong to a SimpleTable, but does not have to.
    """

    def __init__(self, seq, datatype='data', table=None, celltype=None, dec_below='row_dec_below', **fmt_dict):
        """
        Parameters
        ----------
        seq : sequence of data or cells
        table : SimpleTable
        datatype : str ('data' or 'header')
        dec_below : str
          (e.g., 'header_dec_below' or 'row_dec_below')
          decoration tag, identifies the decoration to go below the row.
          (Decoration is repeated as needed for text formats.)
        """
        self.datatype = datatype
        self.table = table
        if celltype is None:
            if table is None:
                celltype = Cell
            else:
                celltype = table._Cell
        self._Cell = celltype
        self._fmt = fmt_dict
        self.special_fmts = dict()
        self.dec_below = dec_below
        list.__init__(self, (celltype(cell, row=self) for cell in seq))

    def add_format(self, output_format, **fmt_dict):
        """
        Return None. Adds row-instance specific formatting
        for the specified output format.
        Example: myrow.add_format('txt', row_dec_below='+-')
        """
        pass

    def insert_stub(self, loc, stub):
        """Return None.  Inserts a stub cell
        in the row at `loc`.
        """
        pass

    def _get_fmt(self, output_format, **fmt_dict):
        """Return dict, the formatting options.
        """
        pass

    def get_aligns(self, output_format, **fmt_dict):
        """Return string, sequence of column alignments.
        Ensure comformable data_aligns in `fmt_dict`."""
        pass

    def as_string(self, output_format='txt', **fmt_dict):
        """Return string: the formatted row.
        This is the default formatter for rows.
        Override this to get different formatting.
        A row formatter must accept as arguments
        a row (self) and an output format,
        one of ('html', 'txt', 'csv', 'latex').
        """
        pass

    def _decorate_below(self, row_as_string, output_format, **fmt_dict):
        """This really only makes sense for the text and latex output formats.
        """
        pass

class Cell:
    """Provides a table cell.
    A cell can belong to a Row, but does not have to.
    """

    def __init__(self, data='', datatype=None, row=None, **fmt_dict):
        if isinstance(data, Cell):
            self.data = data.data
            self._datatype = data.datatype
            self._fmt = data._fmt
        else:
            self.data = data
            self._datatype = datatype
            self._fmt = dict()
        self._fmt.update(fmt_dict)
        self.row = row

    def __str__(self):
        return '%s' % self.data

    def _get_fmt(self, output_format, **fmt_dict):
        """Return dict, the formatting options.
        """
        pass

    def format(self, width, output_format='txt', **fmt_dict):
        """Return string.
        This is the default formatter for cells.
        Override this to get different formating.
        A cell formatter must accept as arguments
        a cell (self) and an output format,
        one of ('html', 'txt', 'csv', 'latex').
        It will generally respond to the datatype,
        one of (int, 'header', 'stub').
        """
        pass
    datatype = property(get_datatype, set_datatype)
' Some formatting suggestions:\n\n- if you want rows to have no extra spacing,\n  set colwidths=0 and colsep=\'\'.\n  (Naturally the columns will not align.)\n- if you want rows to have minimal extra spacing,\n  set colwidths=1.  The columns will align.\n- to get consistent formatting, you should leave\n  all field width handling to SimpleTable:\n  use 0 as the field width in data_fmts.  E.g., ::\n\n        data_fmts = ["%#0.6g","%#0.6g","%#0.4g","%#0.4g"],\n        colwidths = 14,\n        data_aligns = "r",\n'
default_txt_fmt = dict(fmt='txt', table_dec_above='=', table_dec_below='-', title_align='c', row_pre='', row_post='', header_dec_below='-', row_dec_below=None, colwidths=None, colsep=' ', data_aligns='r', data_fmts=['%s'], stub_align='l', header_align='c', header_fmt='%s', stub_fmt='%s', header='%s', stub='%s', empty_cell='', empty='', missing='--')
default_csv_fmt = dict(fmt='csv', table_dec_above=None, table_dec_below=None, row_pre='', row_post='', header_dec_below=None, row_dec_below=None, title_align='', data_aligns='l', colwidths=None, colsep=',', data_fmt='%s', data_fmts=['%s'], stub_align='l', header_align='c', header_fmt='"%s"', stub_fmt='"%s"', empty_cell='', header='%s', stub='%s', empty='', missing='--')
default_html_fmt = dict(table_dec_above=None, table_dec_below=None, header_dec_below=None, row_dec_below=None, title_align='c', colwidths=None, colsep=' ', row_pre='<tr>\n  ', row_post='\n</tr>', data_aligns='c', data_fmts=['<td>%s</td>'], data_fmt='<td>%s</td>', stub_align='l', header_align='c', header_fmt='<th>%s</th>', stub_fmt='<th>%s</th>', empty_cell='<td></td>', header='<th>%s</th>', stub='<th>%s</th>', empty='<td></td>', missing='<td>--</td>')
default_latex_fmt = dict(fmt='ltx', table_dec_above='\\toprule', table_dec_below='\\bottomrule', header_dec_below='\\midrule', row_dec_below=None, strip_backslash=True, row_post='  \\\\', data_aligns='c', colwidths=None, colsep=' & ', data_fmts=['%s'], data_fmt='%s', stub_align='l', header_align='c', empty_align='l', header_fmt='\\textbf{%s}', stub_fmt='\\textbf{%s}', empty_cell='', header='\\textbf{%s}', stub='\\textbf{%s}', empty='', missing='--', replacements={'#': '\\#', '$': '\\$', '%': '\\%', '&': '\\&', '>': '$>$', '_': '\\_', '|': '$|$'})
default_fmts = dict(html=default_html_fmt, txt=default_txt_fmt, latex=default_latex_fmt, csv=default_csv_fmt)
output_format_translations = dict(htm='html', text='txt', ltx='latex')