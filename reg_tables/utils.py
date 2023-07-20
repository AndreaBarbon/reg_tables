from __future__ import annotations
from linearmodels.panel.results import PanelModelComparison,PanelModelResults
from linearmodels.panel.model import PanelOLS
from typing import Dict, List, Union, cast
from linearmodels.compat.statsmodels import Summary
from pandas import  Series, concat
from linearmodels.shared.io import add_star, pval_format
from linearmodels.iv.results import default_txt_fmt, stub_concat, table_concat
from linearmodels.panel.covariance import (
    setup_covariance_estimator,
)
from linearmodels.panel.data import PanelData
from linearmodels.panel.results import (
    PanelEffectsResults,
)
from linearmodels.panel.utility import (
    AbsorbingEffectWarning,
    absorbing_warn_msg,
    check_absorbed,
    not_absorbed,
)

from linearmodels.shared.hypotheses import (
    WaldTestStatistic,
)

from linearmodels.typing import (
    Float64Array,
    IntArray,
)
from scipy.linalg import lstsq as sp_lstsq
from statsmodels.iolib.summary import SimpleTable
import numpy as np
from pandas import  DataFrame, Series
import numpy as np
import re

def _lstsq(
    x: Float64Array, y: Float64Array, rcond: float | None = None
) -> tuple[Float64Array, Float64Array, int, Float64Array]:
    if rcond is None:
        eps = np.finfo(np.float64).eps
        cond = float(max(x.shape) * eps)
    else:
        cond = rcond
    return sp_lstsq(x, y, cond=cond, lapack_driver="gelsy")

def align_latex_table(table):
    lines = table.split('\n')
    start_idx = lines.index(list(filter(lambda x: '&' in x, lines))[0])
    try:
        end_idx = lines.index(list(filter(lambda x: 'Entity FEs ' in x, lines))[0])+1
    except IndexError:
        end_idx = lines.index(list(filter(lambda x: 'R-squared ' in x, lines))[0])+1
    columns = lines[start_idx].count('&')+1
    max_len = [0] * columns
    f_lines = []
    #changhing lines formatting  
    for idx, line in enumerate(lines):
        if (idx < start_idx) | (idx > end_idx):
            f_lines.append(line)
            continue

        if '&' not in line:
            f_lines.append(line)
            continue
        
        items = line.split('&')
        new_line = []
        for item_idx, item in enumerate(items):
            if '\\\\' in item:
                item = item.replace('\\\\','')
            try:
                item_0 = re.search(r"\S", item).span()[0]
            except Exception:
                new_line.append(' ')
                continue
            item_1 = re.search(r"\S\s*$", item).span()[0]+1
            new_item = item[item_0:item_1]
            new_line.append(' '+new_item+' ')
        f_lines.append('&'.join(new_line))


    for idx, line in enumerate(f_lines):
        if (idx < start_idx) | (idx > end_idx):
            continue

        if '&' not in line:
            continue
        
        items = line.split('&')
        for item_idx,item in enumerate(items):
            length = len(item)
            if item_idx == len(max_len)-1:
                length += 4
            if length> max_len[item_idx]:
                max_len[item_idx] = length

    new_lines = []    
    for idx, line in enumerate(f_lines):
        if (idx < start_idx) | (idx > end_idx):
            new_lines.append(line)
            continue

        if '&' not in line:
            new_lines.append(line)
            continue
        
        items = line.split('&')
        new_line = []
        for item_idx, item in enumerate(items):


            if len(item) == max_len[item_idx]:
                if item_idx == len(max_len)-1:
                    item = item[:-4] + '\\\\'
                new_line.append(item)
            else:
                if item_idx == len(max_len)-1:
                    item = item + ' '*(max_len[item_idx] - len(item) -4) + '\\\\'
                    new_line.append(item)
                else:
                    item = item + ' '*(max_len[item_idx] - len(item))
                    
                    new_line.append(item)
        
        new_lines.append('&'.join(new_line))
    new_table = "\n".join(new_lines)
    return new_table



def _str(v: float) -> str:
            if np.isnan(v):
                return " "

            av = abs(v)
            digits = 0

            if av != 0:
                digits = int(np.ceil(np.log10(av)))

            if digits > 4 or digits <= -4:
                return "{0:8.4g}".format(v)
            else:
                return "{:.20f}".format(v)


class InvalidDataset(Exception):
    "Raised when there are issues with the dataset"
    pass

class PanelOLSNew(PanelOLS):
    def fit(
        self,
        *,
        use_lsdv: bool = False,
        use_lsmr: bool = False,
        low_memory: bool | None = None,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        auto_df: bool = True,
        count_effects: bool = True,
        **cov_config: bool | float | str | IntArray | DataFrame | PanelData,
    ) -> PanelEffectsResults:
        """
        Estimate model parameters

        Parameters
        ----------
        use_lsdv : bool
            Flag indicating to use the Least Squares Dummy Variable estimator
            to eliminate effects.  The default value uses only means and does
            note require constructing dummy variables for each effect.
        use_lsmr : bool
            Flag indicating to use LSDV with the Sparse Equations and Least
            Squares estimator to eliminate the fixed effects.
        low_memory : {bool, None}
            Flag indicating whether to use a low-memory algorithm when a model
            contains two-way fixed effects. If `None`, the choice is taken
            automatically, and the low memory algorithm is used if the
            required dummy variable array is both larger than then array of
            regressors in the model and requires more than 1 GiB .
        cov_type : str
            Name of covariance estimator. See Notes.
        debiased : bool
            Flag indicating whether to debiased the covariance estimator using
            a degree of freedom adjustment.
        auto_df : bool
            Flag indicating that the treatment of estimated effects in degree
            of freedom adjustment is automatically handled. This is useful
            since clustered standard errors that are clustered using the same
            variable as an effect do not require degree of freedom correction
            while other estimators such as the unadjusted covariance do.
        count_effects : bool
            Flag indicating that the covariance estimator should be adjusted
            to account for the estimation of effects in the model. Only used
            if ``auto_df=False``.
        **cov_config
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        PanelEffectsResults
            Estimation results

        Examples
        --------
        >>> from linearmodels import PanelOLS
        >>> mod = PanelOLS(y, x, entity_effects=True)
        >>> res = mod.fit(cov_type="clustered", cluster_entity=True)

        Notes
        -----
        Three covariance estimators are supported:

        * "unadjusted", "homoskedastic" - Assume residual are homoskedastic
        * "robust", "heteroskedastic" - Control for heteroskedasticity using
          White's estimator
        * "clustered` - One- or two-way clustering.  Configuration options are:

          * ``clusters`` - Input containing 1 or 2 variables.
            Clusters should be integer valued, although other types will
            be coerced to integer values by treating as categorical variables
          * ``cluster_entity`` - Boolean flag indicating to use entity
            clusters
          * ``cluster_time`` - Boolean indicating to use time clusters

        * "kernel" - Driscoll-Kraay HAC estimator. Configurations options are:

          * ``kernel`` - One of the supported kernels (bartlett, parzen, qs).
            Default is Bartlett's kernel, which is produces a covariance
            estimator similar to the Newey-West covariance estimator.
          * ``bandwidth`` - Bandwidth to use when computing the kernel.  If
            not provided, a naive default is used.
        """

        weighted = np.any(self.weights.values2d != 1.0)

        if use_lsmr:
            y, x, ybar, y_effects, x_effects = self._lsmr_path()
        elif use_lsdv:
            y, x, ybar, y_effects, x_effects = self._slow_path()
        else:
            low_memory = (
                self._choose_twoway_algo() if low_memory is None else low_memory
            )
            if not weighted:
                y, x, ybar = self._fast_path(low_memory=low_memory)
                y_effects = np.array([0.0])
                x_effects = np.zeros(x.shape[1])
            else:
                y, x, ybar, y_effects, x_effects = self._weighted_fast_path(
                    low_memory=low_memory
                )

        neffects = 0
        drop_first = self.has_constant
        if self.entity_effects:
            neffects += self.dependent.nentity - drop_first
            drop_first = True
        if self.time_effects:
            neffects += self.dependent.nobs - drop_first
            drop_first = True
        if self.other_effects:
            assert self._other_effect_cats is not None
            oe = self._other_effect_cats.dataframe
            for c in oe:
                neffects += oe[c].nunique() - drop_first
                drop_first = True

        if self.entity_effects or self.time_effects or self.other_effects:
            if not self._drop_absorbed:
                check_absorbed(x, [str(var) for var in self.exog.vars])
            else:
                # TODO: Need to special case the constant here when determining which
                #  to retain since we always want to retain the constant if present
                retain = not_absorbed(x, self._constant, self._constant_index)
                if not retain:
                    raise ValueError(
                        "All columns in exog have been fully absorbed by the included"
                        " effects. This model cannot be estimated."
                    )
                if len(retain) != x.shape[1]:
                    drop = set(range(x.shape[1])).difference(retain)
                    dropped = ", ".join([str(self.exog.vars[i]) for i in drop])
                    import warnings

                    warnings.warn(
                        absorbing_warn_msg.format(absorbed_variables=dropped),
                        AbsorbingEffectWarning,
                        stacklevel=2,
                    )
                    x = x[:, retain]
                    # Update constant index loc
                    if self._constant:
                        assert isinstance(self._constant_index, int)
                        self._constant_index = int(
                            np.argwhere(np.array(retain) == self._constant_index)
                        )

                    # Adjust exog
                    self.exog = PanelData(self.exog.dataframe.iloc[:, retain])
                    x_effects = x_effects[retain]

        params = _lstsq(x, y, rcond=None)[0]
        nobs = self.dependent.dataframe.shape[0]
        df_model = x.shape[1] + neffects
        df_resid = nobs - df_model
        # Check clusters if singletons were removed
        cov_config = self._setup_clusters(cov_config)
        if auto_df:
            count_effects = self._determine_df_adjustment(cov_type, **cov_config)
        extra_df = neffects if count_effects else 0
        cov = setup_covariance_estimator(
            self._cov_estimators,
            cov_type,
            y,
            x,
            params,
            self.dependent.entity_ids,
            self.dependent.time_ids,
            debiased=debiased,
            extra_df=extra_df,
            **cov_config,
        )

        weps = y - x @ params
        eps = weps
        _y = self.dependent.values2d
        _x = self.exog.values2d
        if weighted:
            eps = (_y - y_effects) - (_x - x_effects) @ params
            if self.has_constant:
                # Correction since y_effects and x_effects @ params add mean
                w = self.weights.values2d
                eps -= (w * eps).sum() / w.sum()
        index = self.dependent.index
        fitted = DataFrame(_x @ params, index, ["fitted_values"])
        idiosyncratic = DataFrame(eps, index, ["idiosyncratic"])
        eps_effects = _y - fitted.values

        sigma2_tot = float(np.squeeze(eps_effects.T @ eps_effects) / nobs)
        sigma2_eps = float(np.squeeze(eps.T @ eps) / nobs)
        sigma2_effects = sigma2_tot - sigma2_eps
        rho = sigma2_effects / sigma2_tot if sigma2_tot > 0.0 else 0.0

        resid_ss = float(np.squeeze(weps.T @ weps))
        if self.has_constant:
            mu = ybar
        else:
            mu = np.array([0.0])
        total_ss = float(np.squeeze((y - mu).T @ (y - mu)))
        r2 = 1 - resid_ss / total_ss if total_ss > 0.0 else 0.0

        root_w = cast(Float64Array, np.sqrt(self.weights.values2d))
        y_ex = root_w * self.dependent.values2d
        mu_ex = 0
        if (
            self.has_constant
            or self.entity_effects
            or self.time_effects
            or self.other_effects
        ):
            mu_ex = root_w * ((root_w.T @ y_ex) / (root_w.T @ root_w))
        total_ss_ex_effect = float(np.squeeze((y_ex - mu_ex).T @ (y_ex - mu_ex)))
        r2_ex_effects = (
            1 - resid_ss / total_ss_ex_effect if total_ss_ex_effect > 0.0 else 0.0
        )

        res = self._postestimation(params, cov, debiased, df_resid, weps, y, x, root_w)
        ######################################
        # Pooled f-stat
        ######################################
        if self.entity_effects or self.time_effects or self.other_effects:
            wy, wx = root_w * self.dependent.values2d, root_w * self.exog.values2d
            df_num, df_denom = (df_model - wx.shape[1]), df_resid
            if not self.has_constant:
                # Correction for when models does not have explicit constant
                wy -= root_w * _lstsq(root_w, wy, rcond=None)[0]
                wx -= root_w * _lstsq(root_w, wx, rcond=None)[0]
                df_num -= 1
            weps_pooled = wy - wx @ _lstsq(wx, wy, rcond=None)[0]
            resid_ss_pooled = float(np.squeeze(weps_pooled.T @ weps_pooled))
            num = (resid_ss_pooled - resid_ss) / df_num

            denom = resid_ss / df_denom
            if denom == 0:
                raise InvalidDataset('There is a problem with your dataset. This error may indicate that the dependent variable in your data is constant.') 
            stat = num / denom
            f_pooled = WaldTestStatistic(
                stat,
                "Effects are zero",
                df_num,
                df_denom=df_denom,
                name="Pooled F-statistic",
            )
            res.update(f_pooled=f_pooled)
            effects = DataFrame(
                eps_effects - eps,
                columns=["estimated_effects"],
                index=self.dependent.index,
            )
        else:
            effects = DataFrame(
                np.zeros_like(eps),
                columns=["estimated_effects"],
                index=self.dependent.index,
            )

        res.update(
            dict(
                df_resid=df_resid,
                df_model=df_model,
                nobs=y.shape[0],
                residual_ss=resid_ss,
                total_ss=total_ss,
                wresids=weps,
                resids=eps,
                r2=r2,
                entity_effects=self.entity_effects,
                time_effects=self.time_effects,
                other_effects=self.other_effects,
                sigma2_eps=sigma2_eps,
                sigma2_effects=sigma2_effects,
                rho=rho,
                r2_ex_effects=r2_ex_effects,
                effects=effects,
                fitted=fitted,
                idiosyncratic=idiosyncratic,
            )
        )

        return PanelEffectsResults(res)


class PanelModelComparisonNew(PanelModelComparison):

    @property
    def summary(self) -> Summary:
        """
        Model estimation summary.

        Returns
        -------
        Summary
            Summary table of model estimation results

        Notes
        -----
        Supports export to csv, html and latex  using the methods ``summary.as_csv()``,
        ``summary.as_html()`` and ``summary.as_latex()``.
        """


        
        smry = Summary()
        models = list(self._results.keys())
        title = "Model Comparison"
        stubs = [
            "Dep. Variable",
            "Estimator",
            "No. Observations",
            "Cov. Est.",
            "R-squared",
            "R-Squared (Within)",
            "R-Squared (Between)",
            "R-Squared (Overall)",
            "F-statistic",
            "P-value (F-stat)",
        ]
        dep_name = {}
        for key in self._results:
            dep_name[key] = self._results[key].model.dependent.vars[0]
        dep_name = Series(dep_name)

        vals = concat(
            [
                dep_name,
                self.estimator_method,
                self.nobs,
                self.cov_estimator,
                self.rsquared,
                self.rsquared_within,
                self.rsquared_between,
                self.rsquared_overall,
                self.f_statistic,
            ],
            axis=1,
        )
        vals = [[i for i in v] for v in vals.T.values]
        vals[2] = [str(v) for v in vals[2]]
        for i in range(4, len(vals)):
            f = _str
            if i == 9:
                f = pval_format
            vals[i] = [f(v) for v in vals[i]]

        params = self.params
        precision = getattr(self, self._precision)
        pvalues = np.asarray(self.pvalues)
        params_fmt = []
        params_stub = []
        for i in range(len(params)):
            formatted_and_starred = []
            for v, pv in zip(params.values[i], pvalues[i]):
                formatted_and_starred.append(add_star(_str(v), pv, self._stars))
            params_fmt.append(formatted_and_starred)

            precision_fmt = []
            for v in precision.values[i]:
                v_str = _str(v)
                v_str = "({0})".format(v_str) if v_str.strip() else v_str
                precision_fmt.append(v_str)
            params_fmt.append(precision_fmt)
            params_stub.append(params.index[i])
            params_stub.append(" ")

        vals = table_concat((vals, params_fmt))
        stubs = stub_concat((stubs, params_stub))

        all_effects = []
        for key in self._results:
            res = self._results[key]
            effects = getattr(res, "included_effects", [])
            all_effects.append(effects)

        neffect = max(map(len, all_effects))
        effects = []
        effects_stub = ["Effects"]
        for i in range(neffect):
            if i > 0:
                effects_stub.append("")
            row = []
            for j in range(len(self._results)):
                effect = all_effects[j]
                if len(effect) > i:
                    row.append(effect[i])
                else:
                    row.append("")
            effects.append(row)
        if effects:
            vals = table_concat((vals, effects))
            stubs = stub_concat((stubs, effects_stub))

        txt_fmt = default_txt_fmt.copy()
        txt_fmt["data_aligns"] = "r"
        txt_fmt["header_align"] = "r"
        table = SimpleTable(
            vals, headers=models, title=title, stubs=stubs, txt_fmt=txt_fmt
        )
        smry.tables.append(table)
        prec_type = self._PRECISION_TYPES[self._precision]
        smry.add_extra_txt(["{0} reported in parentheses".format(prec_type)])
        return smry


def compare(
    results: Union[List[PanelModelResults], Dict[str, PanelModelResults]],
    *,
    precision: str = "tstats",
    stars: bool = False,
) -> PanelModelComparisonNew:
    """
    Compare the results of multiple models

    Parameters
    ----------
    results : {list, dict}
        Set of results to compare.  If a dict, the keys will be used as model
        names.
    precision : {"tstats","std_errors", "std-errors", "pvalues"}
        Estimator precision estimator to include in the comparison output.
        Default is "tstats".
    stars : bool
        Add stars based on the p-value of the coefficient where 1, 2 and
        3-stars correspond to p-values of 10%, 5% and 1%, respectively.

    Returns
    -------
    PanelModelComparison
        The model comparison object.
    """
    return PanelModelComparisonNew(results, precision=precision, stars=stars)
