_plot_added_variable_doc = '    Create an added variable plot for a fitted regression model.\n\n    Parameters\n    ----------\n    %(extra_params_doc)sfocus_exog : int or string\n        The column index of exog, or a variable name, indicating the\n        variable whose role in the regression is to be assessed.\n    resid_type : str\n        The type of residuals to use for the dependent variable.  If\n        None, uses `resid_deviance` for GLM/GEE and `resid` otherwise.\n    use_glm_weights : bool\n        Only used if the model is a GLM or GEE.  If True, the\n        residuals for the focus predictor are computed using WLS, with\n        the weights obtained from the IRLS calculations for fitting\n        the GLM. If False, unweighted regression is used.\n    fit_kwargs : dict, optional\n        Keyword arguments to be passed to fit when refitting the\n        model.\n    ax: Axes\n        Matplotlib Axes instance\n\n    Returns\n    -------\n    Figure\n        A matplotlib figure instance.\n'
_plot_partial_residuals_doc = "    Create a partial residual, or 'component plus residual' plot for a\n    fitted regression model.\n\n    Parameters\n    ----------\n    %(extra_params_doc)sfocus_exog : int or string\n        The column index of exog, or variable name, indicating the\n        variable whose role in the regression is to be assessed.\n    ax: Axes\n        Matplotlib Axes instance\n\n    Returns\n    -------\n    Figure\n        A matplotlib figure instance.\n"
_plot_ceres_residuals_doc = "    Conditional Expectation Partial Residuals (CERES) plot.\n\n    Produce a CERES plot for a fitted regression model.\n\n    Parameters\n    ----------\n    %(extra_params_doc)s\n    focus_exog : {int, str}\n        The column index of results.model.exog, or the variable name,\n        indicating the variable whose role in the regression is to be\n        assessed.\n    frac : float\n        Lowess tuning parameter for the adjusted model used in the\n        CERES analysis.  Not used if `cond_means` is provided.\n    cond_means : array_like, optional\n        If provided, the columns of this array span the space of the\n        conditional means E[exog | focus exog], where exog ranges over\n        some or all of the columns of exog (other than the focus exog).\n    ax : matplotlib.Axes instance, optional\n        The axes on which to draw the plot. If not provided, a new\n        axes instance is created.\n\n    Returns\n    -------\n    Figure\n        The figure on which the partial residual plot is drawn.\n\n    Notes\n    -----\n    `cond_means` is intended to capture the behavior of E[x1 |\n    x2], where x2 is the focus exog and x1 are all the other exog\n    variables.  If all the conditional mean relationships are\n    linear, it is sufficient to set cond_means equal to the focus\n    exog.  Alternatively, cond_means may consist of one or more\n    columns containing functional transformations of the focus\n    exog (e.g. x2^2) that are thought to capture E[x1 | x2].\n\n    If nothing is known or suspected about the form of E[x1 | x2],\n    set `cond_means` to None, and it will be estimated by\n    smoothing each non-focus exog against the focus exog.  The\n    values of `frac` control these lowess smooths.\n\n    If cond_means contains only the focus exog, the results are\n    equivalent to a partial residual plot.\n\n    If the focus variable is believed to be independent of the\n    other exog variables, `cond_means` can be set to an (empty)\n    nx0 array.\n\n    References\n    ----------\n    .. [1] RD Cook and R Croos-Dabrera (1998).  Partial residual plots\n       in generalized linear models.  Journal of the American\n       Statistical Association, 93:442.\n\n    .. [2] RD Cook (1993). Partial residual plots.  Technometrics 35:4.\n\n    Examples\n    --------\n    Using a model built from the the state crime dataset, make a CERES plot with\n    the rate of Poverty as the focus variable.\n\n    >>> import statsmodels.api as sm\n    >>> import matplotlib.pyplot as plt\n    >>> import statsmodels.formula.api as smf\n    >>> from statsmodels.graphics.regressionplots import plot_ceres_residuals\n\n    >>> crime_data = sm.datasets.statecrime.load_pandas()\n    >>> results = smf.ols('murder ~ hs_grad + urban + poverty + single',\n    ...                   data=crime_data.data).fit()\n    >>> plot_ceres_residuals(results, 'poverty')\n    >>> plt.show()\n\n    .. plot:: plots/graphics_regression_ceres_residuals.py\n"
_plot_influence_doc = "    Plot of influence in regression. Plots studentized resids vs. leverage.\n\n    Parameters\n    ----------\n    {extra_params_doc}\n    external : bool\n        Whether to use externally or internally studentized residuals. It is\n        recommended to leave external as True.\n    alpha : float\n        The alpha value to identify large studentized residuals. Large means\n        abs(resid_studentized) > t.ppf(1-alpha/2, dof=results.df_resid)\n    criterion : str {{'DFFITS', 'Cooks'}}\n        Which criterion to base the size of the points on. Options are\n        DFFITS or Cook's D.\n    size : float\n        The range of `criterion` is mapped to 10**2 - size**2 in points.\n    plot_alpha : float\n        The `alpha` of the plotted points.\n    ax : AxesSubplot\n        An instance of a matplotlib Axes.\n    **kwargs\n        Additional parameters passed through to `plot`.\n\n    Returns\n    -------\n    Figure\n        The matplotlib figure that contains the Axes.\n\n    Notes\n    -----\n    Row labels for the observations in which the leverage, measured by the\n    diagonal of the hat matrix, is high or the residuals are large, as the\n    combination of large residuals and a high influence value indicates an\n    influence point. The value of large residuals can be controlled using the\n    `alpha` parameter. Large leverage points are identified as\n    hat_i > 2 * (df_model + 1)/nobs.\n\n    Examples\n    --------\n    Using a model built from the the state crime dataset, plot the influence in\n    regression.  Observations with high leverage, or large residuals will be\n    labeled in the plot to show potential influence points.\n\n    >>> import statsmodels.api as sm\n    >>> import matplotlib.pyplot as plt\n    >>> import statsmodels.formula.api as smf\n\n    >>> crime_data = sm.datasets.statecrime.load_pandas()\n    >>> results = smf.ols('murder ~ hs_grad + urban + poverty + single',\n    ...                   data=crime_data.data).fit()\n    >>> sm.graphics.influence_plot(results)\n    >>> plt.show()\n\n    .. plot:: plots/graphics_regression_influence.py\n    "
_plot_leverage_resid2_doc = "    Plot leverage statistics vs. normalized residuals squared\n\n    Parameters\n    ----------\n    results : results instance\n        A regression results instance\n    alpha : float\n        Specifies the cut-off for large-standardized residuals. Residuals\n        are assumed to be distributed N(0, 1) with alpha=alpha.\n    ax : Axes\n        Matplotlib Axes instance\n    **kwargs\n        Additional parameters passed the plot command.\n\n    Returns\n    -------\n    Figure\n        A matplotlib figure instance.\n\n    Examples\n    --------\n    Using a model built from the the state crime dataset, plot the leverage\n    statistics vs. normalized residuals squared.  Observations with\n    Large-standardized Residuals will be labeled in the plot.\n\n    >>> import statsmodels.api as sm\n    >>> import matplotlib.pyplot as plt\n    >>> import statsmodels.formula.api as smf\n\n    >>> crime_data = sm.datasets.statecrime.load_pandas()\n    >>> results = smf.ols('murder ~ hs_grad + urban + poverty + single',\n    ...                   data=crime_data.data).fit()\n    >>> sm.graphics.plot_leverage_resid2(results)\n    >>> plt.show()\n\n    .. plot:: plots/graphics_regression_leverage_resid2.py\n    "