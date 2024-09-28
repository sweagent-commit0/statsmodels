"""American National Election Survey 1996"""
from numpy import log
from statsmodels.datasets import utils as du
__docformat__ = 'restructuredtext'
COPYRIGHT = 'This is public domain.'
TITLE = __doc__
SOURCE = '\nhttp://www.electionstudies.org/\n\nThe American National Election Studies.\n'
DESCRSHORT = 'This data is a subset of the American National Election Studies of 1996.'
DESCRLONG = DESCRSHORT
NOTE = '::\n\n    Number of observations - 944\n    Number of variables - 10\n\n    Variables name definitions::\n\n            popul - Census place population in 1000s\n            TVnews - Number of times per week that respondent watches TV news.\n            PID - Party identification of respondent.\n                0 - Strong Democrat\n                1 - Weak Democrat\n                2 - Independent-Democrat\n                3 - Independent-Indpendent\n                4 - Independent-Republican\n                5 - Weak Republican\n                6 - Strong Republican\n            age : Age of respondent.\n            educ - Education level of respondent\n                1 - 1-8 grades\n                2 - Some high school\n                3 - High school graduate\n                4 - Some college\n                5 - College degree\n                6 - Master\'s degree\n                7 - PhD\n            income - Income of household\n                1  - None or less than $2,999\n                2  - $3,000-$4,999\n                3  - $5,000-$6,999\n                4  - $7,000-$8,999\n                5  - $9,000-$9,999\n                6  - $10,000-$10,999\n                7  - $11,000-$11,999\n                8  - $12,000-$12,999\n                9  - $13,000-$13,999\n                10 - $14,000-$14.999\n                11 - $15,000-$16,999\n                12 - $17,000-$19,999\n                13 - $20,000-$21,999\n                14 - $22,000-$24,999\n                15 - $25,000-$29,999\n                16 - $30,000-$34,999\n                17 - $35,000-$39,999\n                18 - $40,000-$44,999\n                19 - $45,000-$49,999\n                20 - $50,000-$59,999\n                21 - $60,000-$74,999\n                22 - $75,000-89,999\n                23 - $90,000-$104,999\n                24 - $105,000 and over\n            vote - Expected vote\n                0 - Clinton\n                1 - Dole\n            The following 3 variables all take the values:\n                1 - Extremely liberal\n                2 - Liberal\n                3 - Slightly liberal\n                4 - Moderate\n                5 - Slightly conservative\n                6 - Conservative\n                7 - Extremely Conservative\n            selfLR - Respondent\'s self-reported political leanings from "Left"\n                to "Right".\n            ClinLR - Respondents impression of Bill Clinton\'s political\n                leanings from "Left" to "Right".\n            DoleLR  - Respondents impression of Bob Dole\'s political leanings\n                from "Left" to "Right".\n            logpopul - log(popul + .1)\n'

def load_pandas():
    """Load the anes96 data and returns a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass

def load():
    """Load the anes96 data and returns a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    pass