"""
Substantially copied from NumpyDoc 1.0pre.
"""
from collections import namedtuple
from collections.abc import Mapping
import copy
import inspect
import re
import textwrap
from statsmodels.tools.sm_exceptions import ParseError

def dedent_lines(lines):
    """Deindent a list of lines maximally"""
    pass

def strip_blank_lines(line):
    """Remove leading and trailing blank lines from a list of lines"""
    pass

class Reader:
    """
    A line-based string reader.
    """

    def __init__(self, data):
        """
        Parameters
        ----------
        data : str
           String with lines separated by '
'.
        """
        if isinstance(data, list):
            self._str = data
        else:
            self._str = data.split('\n')
        self.reset()

    def __getitem__(self, n):
        return self._str[n]
Parameter = namedtuple('Parameter', ['name', 'type', 'desc'])

class NumpyDocString(Mapping):
    """Parses a numpydoc string to an abstract representation

    Instances define a mapping from section title to structured data.
    """
    sections = {'Signature': '', 'Summary': [''], 'Extended Summary': [], 'Parameters': [], 'Returns': [], 'Yields': [], 'Receives': [], 'Raises': [], 'Warns': [], 'Other Parameters': [], 'Attributes': [], 'Methods': [], 'See Also': [], 'Notes': [], 'Warnings': [], 'References': '', 'Examples': '', 'index': {}}

    def __init__(self, docstring):
        orig_docstring = docstring
        docstring = textwrap.dedent(docstring).split('\n')
        self._doc = Reader(docstring)
        self._parsed_data = copy.deepcopy(self.sections)
        try:
            self._parse()
        except ParseError as e:
            e.docstring = orig_docstring
            raise

    def __getitem__(self, key):
        return self._parsed_data[key]

    def __setitem__(self, key, val):
        if key not in self._parsed_data:
            self._error_location('Unknown section %s' % key)
        else:
            self._parsed_data[key] = val

    def __iter__(self):
        return iter(self._parsed_data)

    def __len__(self):
        return len(self._parsed_data)
    _role = ':(?P<role>\\w+):'
    _funcbacktick = '`(?P<name>(?:~\\w+\\.)?[a-zA-Z0-9_\\.-]+)`'
    _funcplain = '(?P<name2>[a-zA-Z0-9_\\.-]+)'
    _funcname = '(' + _role + _funcbacktick + '|' + _funcplain + ')'
    _funcnamenext = _funcname.replace('role', 'rolenext')
    _funcnamenext = _funcnamenext.replace('name', 'namenext')
    _description = '(?P<description>\\s*:(\\s+(?P<desc>\\S+.*))?)?\\s*$'
    _func_rgx = re.compile('^\\s*' + _funcname + '\\s*')
    _line_rgx = re.compile('^\\s*' + '(?P<allfuncs>' + _funcname + '(?P<morefuncs>([,]\\s+' + _funcnamenext + ')*)' + ')' + '(?P<trailing>[,\\.])?' + _description)
    empty_description = '..'

    def _parse_see_also(self, content):
        """
        func_name : Descriptive text
            continued text
        another_func_name : Descriptive text
        func_name1, func_name2, :meth:`func_name`, func_name3
        """
        pass

    def _parse_index(self, section, content):
        """
        .. index: default
           :refguide: something, else, and more
        """
        pass

    def _parse_summary(self):
        """Grab signature (if given) and summary"""
        pass

    def __str__(self, func_role=''):
        out = []
        out += self._str_signature()
        out += self._str_summary()
        out += self._str_extended_summary()
        for param_list in ('Parameters', 'Returns', 'Yields', 'Receives', 'Other Parameters', 'Raises', 'Warns'):
            out += self._str_param_list(param_list)
        out += self._str_section('Warnings')
        out += self._str_see_also(func_role)
        for s in ('Notes', 'References', 'Examples'):
            out += self._str_section(s)
        for param_list in ('Attributes', 'Methods'):
            out += self._str_param_list(param_list)
        out += self._str_index()
        return '\n'.join(out)

class Docstring:
    """
    Docstring modification.

    Parameters
    ----------
    docstring : str
        The docstring to modify.
    """

    def __init__(self, docstring):
        self._ds = None
        self._docstring = docstring
        if docstring is None:
            return
        self._ds = NumpyDocString(docstring)

    def remove_parameters(self, parameters):
        """
        Parameters
        ----------
        parameters : str, list[str]
            The names of the parameters to remove.
        """
        pass

    def insert_parameters(self, after, parameters):
        """
        Parameters
        ----------
        after : {None, str}
            If None, inset the parameters before the first parameter in the
            docstring.
        parameters : Parameter, list[Parameter]
            A Parameter of a list of Parameters.
        """
        pass

    def replace_block(self, block_name, block):
        """
        Parameters
        ----------
        block_name : str
            Name of the block to replace, e.g., 'Summary'.
        block : object
            The replacement block. The structure of the replacement block must
            match how the block is stored by NumpyDocString.
        """
        pass

    def __str__(self):
        return str(self._ds)

def remove_parameters(docstring, parameters):
    """
    Parameters
    ----------
    docstring : str
        The docstring to modify.
    parameters : str, list[str]
        The names of the parameters to remove.

    Returns
    -------
    str
        The modified docstring.
    """
    pass

def indent(text, prefix, predicate=None):
    """
    Non-protected indent

    Parameters
    ----------
    text : {None, str}
        If None, function always returns ""
    prefix : str
        Prefix to add to the start of each line
    predicate : callable, optional
        If provided, 'prefix' will only be added to the lines
        where 'predicate(line)' is True. If 'predicate' is not provided,
        it will default to adding 'prefix' to all non-empty lines that do not
        consist solely of whitespace characters.

    Returns
    -------

    """
    pass