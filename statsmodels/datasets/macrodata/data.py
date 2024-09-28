"""United States Macroeconomic data"""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'This is public domain.'
TITLE = __doc__
SOURCE = '\nCompiled by Skipper Seabold. All data are from the Federal Reserve Bank of St.\nLouis [1] except the unemployment rate which was taken from the National\nBureau of Labor Statistics [2]. ::\n\n    [1] Data Source: FRED, Federal Reserve Economic Data, Federal Reserve Bank of\n        St. Louis; http://research.stlouisfed.org/fred2/; accessed December 15,\n        2009.\n\n    [2] Data Source: Bureau of Labor Statistics, U.S. Department of Labor;\n        http://www.bls.gov/data/; accessed December 15, 2009.\n'
DESCRSHORT = 'US Macroeconomic Data for 1959Q1 - 2009Q3'
DESCRLONG = DESCRSHORT
NOTE = '::\n    Number of Observations - 203\n\n    Number of Variables - 14\n\n    Variable name definitions::\n\n        year      - 1959q1 - 2009q3\n        quarter   - 1-4\n        realgdp   - Real gross domestic product (Bil. of chained 2005 US$,\n                    seasonally adjusted annual rate)\n        realcons  - Real personal consumption expenditures (Bil. of chained\n                    2005 US$, seasonally adjusted annual rate)\n        realinv   - Real gross private domestic investment (Bil. of chained\n                    2005 US$, seasonally adjusted annual rate)\n        realgovt  - Real federal consumption expenditures & gross investment\n                    (Bil. of chained 2005 US$, seasonally adjusted annual rate)\n        realdpi   - Real private disposable income (Bil. of chained 2005\n                    US$, seasonally adjusted annual rate)\n        cpi       - End of the quarter consumer price index for all urban\n                    consumers: all items (1982-84 = 100, seasonally adjusted).\n        m1        - End of the quarter M1 nominal money stock (Seasonally\n                    adjusted)\n        tbilrate  - Quarterly monthly average of the monthly 3-month\n                    treasury bill: secondary market rate\n        unemp     - Seasonally adjusted unemployment rate (%)\n        pop       - End of the quarter total population: all ages incl. armed\n                    forces over seas\n        infl      - Inflation rate (ln(cpi_{t}/cpi_{t-1}) * 400)\n        realint   - Real interest rate (tbilrate - infl)\n'

def load():
    """
    Load the US macro data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    The macrodata Dataset instance does not contain endog and exog attributes.
    """
    pass
variable_names = ['realcons', 'realgdp', 'realinv']

def __str__():
    return 'macrodata'