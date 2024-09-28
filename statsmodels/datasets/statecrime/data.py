"""Statewide Crime Data"""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'Public domain.'
TITLE = 'Statewide Crime Data 2009'
SOURCE = '\nAll data is for 2009 and was obtained from the American Statistical Abstracts except as indicated below.\n'
DESCRSHORT = 'State crime data 2009'
DESCRLONG = DESCRSHORT
NOTE = '::\n\n    Number of observations: 51\n    Number of variables: 8\n    Variable name definitions:\n\n    state\n        All 50 states plus DC.\n    violent\n        Rate of violent crimes / 100,000 population. Includes murder, forcible\n        rape, robbery, and aggravated assault. Numbers for Illinois and\n        Minnesota do not include forcible rapes. Footnote included with the\n        American Statistical Abstract table reads:\n        "The data collection methodology for the offense of forcible\n        rape used by the Illinois and the Minnesota state Uniform Crime\n        Reporting (UCR) Programs (with the exception of Rockford, Illinois,\n        and Minneapolis and St. Paul, Minnesota) does not comply with\n        national UCR guidelines. Consequently, their state figures for\n        forcible rape and violent crime (of which forcible rape is a part)\n        are not published in this table."\n    murder\n        Rate of murders / 100,000 population.\n    hs_grad\n        Percent of population having graduated from high school or higher.\n    poverty\n        % of individuals below the poverty line\n    white\n        Percent of population that is one race - white only. From 2009 American\n        Community Survey\n    single\n        Calculated from 2009 1-year American Community Survey obtained obtained\n        from Census. Variable is Male householder, no wife present, family\n        household combined with Female householder, no husband present, family\n        household, divided by the total number of Family households.\n    urban\n        % of population in Urbanized Areas as of 2010 Census. Urbanized\n        Areas are area of 50,000 or more people.'

def load():
    """
    Load the statecrime data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass