from linearmodels.panel.results import PanelModelComparison,PanelModelResults
from typing import Dict, List, Union
from linearmodels.compat.statsmodels import Summary
from pandas import  Series, concat
from linearmodels.shared.io import add_star, pval_format
from linearmodels.iv.results import default_txt_fmt, stub_concat, table_concat
from statsmodels.iolib.summary import SimpleTable
import numpy as np
import re

def align_latex_table(table):
    lines = table.split('\n')
    start_idx = lines.index(list(filter(lambda x: '&' in x, lines))[0])
    end_idx = lines.index(list(filter(lambda x: 'Entity FEs ' in x, lines))[0])+1
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
