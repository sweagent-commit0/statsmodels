"""World Bank Fertility Data."""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'This data is distributed according to the World Bank terms of use. See SOURCE.'
TITLE = 'World Bank Fertility Data'
SOURCE = '\nThis data has been acquired from\n\nThe World Bank: Fertility rate, total (births per woman): World Development Indicators\n\nAt the following URL: http://data.worldbank.org/indicator/SP.DYN.TFRT.IN\n\nThe sources for these statistics are listed as\n\n(1) United Nations Population Division. World Population Prospects\n(2) United Nations Statistical Division. Population and Vital Statistics Repot (various years)\n(3) Census reports and other statistical publications from national statistical offices\n(4) Eurostat: Demographic Statistics\n(5) Secretariat of the Pacific Community: Statistics and Demography Programme\n(6) U.S. Census Bureau: International Database\n\nThe World Bank Terms of Use can be found at the following URL\n\nhttp://go.worldbank.org/OJC02YMLA0\n'
DESCRSHORT = 'Total fertility rate represents the number of children that would be born to a woman if she were to live to the end of her childbearing years and bear children in accordance with current age-specific fertility rates.'
DESCRLONG = DESCRSHORT
NOTE = '\n::\n\n    This is panel data in wide-format\n\n    Number of observations: 219\n    Number of variables: 58\n    Variable name definitions:\n        Country Name\n        Country Code\n        Indicator Name - The World Bank Series indicator\n        Indicator Code - The World Bank Series code\n        1960 - 2013 - The fertility rate for the given year\n'

def load():
    """
    Load the data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass