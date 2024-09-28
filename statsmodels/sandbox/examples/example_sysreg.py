"""Example: statsmodels.sandbox.sysreg
"""
from statsmodels.compat.python import asbytes, lmap
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from statsmodels.sandbox.sysreg import SUR, Sem2SLS
grun_data = sm.datasets.grunfeld.load()
firms = ['General Motors', 'Chrysler', 'General Electric', 'Westinghouse', 'US Steel']
firms = lmap(asbytes, firms)
grun_exog = grun_data.exog
grun_endog = grun_data.endog
grun_sys = []
for i in firms:
    index = grun_exog['firm'] == i
    grun_sys.append(grun_endog[index])
    exog = grun_exog[index][['value', 'capital']].view(float).reshape(-1, 2)
    exog = sm.add_constant(exog, prepend=True)
    grun_sys.append(exog)
grun_sys[-2][5] = 261.6
grun_sys[-2][-3] = 645.2
grun_sys[-1][11, 2] = 232.6
grun_mod = SUR(grun_sys)
grun_res = grun_mod.fit()
print('Results for the 2-step GLS')
print('Compare to Greene Table 14.1, 5th edition')
print(grun_res.params)
print('Results for iterative GLS (equivalent to MLE)')
print('Compare to Greene Table 14.3')
grun_imod = SUR(grun_sys)
grun_ires = grun_imod.fit(igls=True)
print(grun_ires.params)
macrodata = sm.datasets.macrodata.load().data
macrodata = np.sort(macrodata, order=['year', 'quarter'])
y = macrodata['realcons'] + macrodata['realinv'] + macrodata['realgovt']
macro_sys = []
macro_sys.append(macrodata['realcons'][1:])
exog1 = np.column_stack((y[1:], macrodata['realcons'][:-1]))
exog1 = sm.add_constant(exog1, prepend=True)
macro_sys.append(exog1)
macro_sys.append(macrodata['realinv'][1:])
exog2 = np.column_stack((macrodata['tbilrate'][1:], np.diff(y)))
exog2 = sm.add_constant(exog2, prepend=True)
macro_sys.append(exog2)
indep_endog = {0: [1]}
instruments = np.column_stack((macrodata[['realgovt', 'tbilrate']][1:].view(float).reshape(-1, 2), macrodata['realcons'][:-1], y[:-1]))
instruments = sm.add_constant(instruments, prepend=True)
macro_mod = Sem2SLS(macro_sys, indep_endog=indep_endog, instruments=instruments)
macro_params = macro_mod.fit()
print('The parameters for the first equation are correct.')
print('The parameters for the second equation are not.')
print(macro_params)
y_instrumented = macro_mod.wexog[0][:, 1]
whitened_ydiff = y_instrumented - y[:-1]
wexog = np.column_stack((macrodata['tbilrate'][1:], whitened_ydiff))
wexog = sm.add_constant(wexog, prepend=True)
correct_params = sm.GLS(macrodata['realinv'][1:], wexog).fit().params
print('If we correctly instrument everything, then these are the parameters')
print('for the second equation')
print(correct_params)
print('Compare to output of R script statsmodels/sandbox/tests/macrodata.s')
print('\nUsing IV2SLS')
miv = IV2SLS(macro_sys[0], macro_sys[1], instruments)
resiv = miv.fit()
print('equation 1')
print(resiv.params)
miv2 = IV2SLS(macro_sys[2], macro_sys[3], instruments)
resiv2 = miv2.fit()
print('equation 2')
print(resiv2.params)
run_greene = 0
if run_greene:
    try:
        data3 = np.genfromtxt('/home/skipper/school/MetricsII/Greene TableF5-1.txt', names=True)
    except:
        raise ValueError('Based on Greene TableF5-1.  You should download it from his web site and edit this script accordingly.')
    sys3 = []
    sys3.append(data3['realcons'][1:])
    y = data3['realcons'] + data3['realinvs'] + data3['realgovt']
    exog1 = np.column_stack((y[1:], data3['realcons'][:-1]))
    exog1 = sm.add_constant(exog1, prepend=False)
    sys3.append(exog1)
    sys3.append(data3['realinvs'][1:])
    exog2 = np.column_stack((data3['tbilrate'][1:], np.diff(y)))
    exog2 = sm.add_constant(exog2, prepend=False)
    sys3.append(exog2)
    indep_endog = {0: [0]}
    instruments = np.column_stack((data3[['realgovt', 'tbilrate']][1:].view(float).reshape(-1, 2), data3['realcons'][:-1], y[:-1]))
    instruments = sm.add_constant(instruments, prepend=False)
    sem_mod = Sem2SLS(sys3, indep_endog=indep_endog, instruments=instruments)
    sem_params = sem_mod.fit()
    y_instr = sem_mod.wexog[0][:, 0]
    wyd = y_instr - y[:-1]
    wexog = np.column_stack((data3['tbilrate'][1:], wyd))
    wexog = sm.add_constant(wexog, prepend=False)
    params = sm.GLS(data3['realinvs'][1:], wexog).fit().params
    print("These are the simultaneous equation estimates for Greene's example 13-1 (Also application 13-1 in 6th edition.")
    print(sem_params)
    print('The first set of parameters is correct.  The second set is not.')
    print('Compare to the solution manual at http://pages.stern.nyu.edu/~wgreene/Text/econometricanalysis.htm')
    print('The reason is the restriction on (y_t - y_1)')
    print('Compare to R script GreeneEx15_1.s')
    print('Somehow R carries y.1 in yd to know that it needs to be instrumented')
    print('If we replace our estimate with the instrumented one')
    print(params)
    print('We get the right estimate')
    print('Without a formula framework we have to be able to do restrictions.')