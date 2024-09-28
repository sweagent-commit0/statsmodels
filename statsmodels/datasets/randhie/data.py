"""RAND Health Insurance Experiment Data"""
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'This is in the public domain.'
TITLE = __doc__
SOURCE = '\nThe data was collected by the RAND corporation as part of the Health\nInsurance Experiment (HIE).\n\nhttp://www.rand.org/health/projects/hie.html\n\nThis data was used in::\n\n    Cameron, A.C. amd Trivedi, P.K. 2005.  `Microeconometrics: Methods\n        and Applications,` Cambridge: New York.\n\nAnd was obtained from: <http://cameron.econ.ucdavis.edu/mmabook/mmadata.html>\n\nSee randhie/src for the original data and description.  The data included\nhere contains only a subset of the original data.  The data varies slightly\ncompared to that reported in Cameron and Trivedi.\n'
DESCRSHORT = 'The RAND Co. Health Insurance Experiment Data'
DESCRLONG = ''
NOTE = '::\n\n    Number of observations - 20,190\n    Number of variables - 10\n    Variable name definitions::\n\n        mdvis   - Number of outpatient visits to an MD\n        lncoins - ln(coinsurance + 1), 0 <= coninsurance <= 100\n        idp     - 1 if individual deductible plan, 0 otherwise\n        lpi     - ln(max(1, annual participation incentive payment))\n        fmde    - 0 if idp = 1; ln(max(1, MDE/(0.01 coinsurance))) otherwise\n        physlm  - 1 if the person has a physical limitation\n        disea   - number of chronic diseases\n        hlthg   - 1 if self-rated health is good\n        hlthf   - 1 if self-rated health is fair\n        hlthp   - 1 if self-rated health is poor\n        (Omitted category is excellent self-rated health)\n'

def load():
    """
    Loads the RAND HIE data and returns a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    endog - response variable, mdvis
    exog - design
    """
    pass

def load_pandas():
    """
    Loads the RAND HIE data and returns a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    endog - response variable, mdvis
    exog - design
    """
    pass