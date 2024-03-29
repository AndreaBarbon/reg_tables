import warnings
from reg_tables.utils import compare, align_latex_table, PanelOLSNew
from statsmodels.tools.tools import add_constant
import copy
import io
import pandas as pd
import numpy as np
import re
from varname import argname
from varname.utils import ImproperUseError, UsingExecWarning
from ast import Subscript
from tqdm import tqdm as tq

class Spec():
    """
    Contains specification of regression

    Parameters
    ----------
    data : {np.ndarray, pd.DataFrame}
        Dataset from which 'x' and 'y' variables are going to be sourced from
    y : str
        Name of the column with 'y' variable
    x_vars : {str,list, dict, set, tuple, np.ndarray, pd.core.series.Series}
        Name of the columns with 'x' variables
    entity_effects : bool
        Peform regression with entity effects
    time_effects : bool
        Peform regression with time effects
    other_effects : {str,list, dict, set, tuple, np.ndarray, pd.core.series.Series}
        Category codes to use for any effects that are not entity or time effects. Each variable is treated as an effect.
    cluster_entity : bool
        Cluster standard errors by entity
    cluster_time : bool
        Cluster standard errors by time
    double_cluster : bool
        Cluster standard errors bith by entity and time
    intercept : bool
        Include intercept in the regression
    check_rank : bool
        Check rank during regression
    """

    def __init__(self, 
            data, y, x_vars, 
            entity_effects=False, time_effects=False,
            cluster_entity=False, other_effects=None, cluster_time=False, 
            double_cluster=False, intercept=True,check_rank=True, data_name=None
        ):
        if ((
            (time_effects==True) and
            (entity_effects==True)
            )and\
          (other_effects!=None)):
            raise ValueError('At most two fixed effects are supported.')
        self.data = data
        if data_name != None:
            self.data_name = data_name
        else:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UsingExecWarning)
                    self.data_name = argname('data')
            except ImproperUseError:
                print("Can't retrieve name of dataset. Using default value. Rename the dataset or provide a 'data_name' argument.")
                self.data_name = 'data'


        
        self.y = y
        if isinstance(x_vars, (list, dict, set, tuple,
                               np.ndarray, pd.core.series.Series)) != True: x_vars = [x_vars]
        self.x_vars = x_vars
        self.entity_effects = entity_effects
        self.time_effects = time_effects
        if other_effects == []:
            self.other_effects = None
        else:
            self.other_effects = other_effects
        self.cluster_entity = cluster_entity
        self.cluster_time = cluster_time
        self.double_cluster = double_cluster
        self.intercept=intercept
        self.check_rank=check_rank
        if (time_effects or entity_effects): intercept=False
        

    def __repr__(self):
        return (f'dataset : {self.data_name}\
                x-vars: {self.x_vars}, y: {self.y},\
                Entity Effects: {self.entity_effects},\
                Time Effects: {self.time_effects},\
                Other Effects: {self.other_effects},\
                Cluster Entity: {self.cluster_entity},\
                Cluster Time: {self.cluster_time},\
                Double Cluster: {self.double_cluster},\
                Intercept: {self.intercept},\
                Check rank: {self.check_rank}')  

    def to_model(self, model):
        """
        Adds the current spec to specified model
        """
        if isinstance(model, Model):
            model.add_spec(self)
        else:
            raise TypeError('Please provide a Model object as argument')

    def run(self):
        """
        Run this regression
        
        Returns
        -------
        PanelEffectsResults
            The panel effects results object.
        
        """
        if self.other_effects != None:             
            reg = PanelOLSNew(
                    self.data[[self.y]],
                    add_constant(self.data[self.x_vars])if self.intercept == True else self.data[self.x_vars], 
                    entity_effects = self.entity_effects, 
                    time_effects = self.time_effects,
                    other_effects = self.data[self.other_effects]
                ).fit(
                cov_type='clustered', 
                cluster_entity=(self.cluster_entity | self.double_cluster),
                cluster_time=(self.cluster_time   | self.double_cluster)
            )
        else:
            reg = PanelOLSNew(
                    self.data[[self.y]],
                    add_constant(self.data[self.x_vars])if self.intercept == True else self.data[self.x_vars], 
                    entity_effects = self.entity_effects, 
                    time_effects = self.time_effects
                ).fit(
                cov_type='clustered', 
                cluster_entity=(self.cluster_entity | self.double_cluster),
                cluster_time=(self.cluster_time   | self.double_cluster)
            )
        return reg
    
                
class Model():
    """
    Contains multiple Spec objects

    Parameters
    ----------
    baseline : Spec
        First regression of the model
    rename_dict : dict
        Rename columns with the variables
    time_entity_effects : bool
        Peform all regressions both with entity and time effects
    """
    def __init__(self, baseline, rename_dict={}, time_entity_effects=False):
        self._rename_dict = rename_dict#add check
        baseline.intercept = True
        self.baseline = baseline
        self.specs = []
        if time_entity_effects:
            for comb in [(False, False), (True, False), (False, True), (True, True)]:
                new_spec = copy.deepcopy(self.baseline)
                new_spec.entity_effects = comb[0]
                new_spec.time_effects = comb[1]
                self.specs.append(new_spec)
        else:
            new_spec = copy.deepcopy(self.baseline)
            self.specs.append(new_spec)
    
    def __repr__(self):
        strr = ''
        for idx, spec in enumerate(self.specs):
            strr = strr+(f'Spec {idx+1}: '+ spec.__repr__()+'\n')
        return strr
    
    def remove_spec (self, idx1, idx2=None):
        """
        Remove regression from the model

        Parameters
        ----------
        idx1 : {float, int}
            Index of the model that needs to be removed 
            (numeration starts from 1)
        idx2 : {float, int}
            If passed a slice of [idx1:idx2] will be removed 
            (numeration starts from 1)
        """
        if idx2 != None:del self.specs[idx1-1:idx2-1] 
        else:
            del self.specs[idx1-1] 
    
    def add_spec(self, **kwargs):
        """
        Add specs to the model
        Parameters
        ----------
        **kwargs:
            kwargs describing the models. Possible arguments :
                data : {np.ndarray, pd.DataFrame}
                    Dataset from which 'x' and 'y' variables are going to be sourced from
                y : str
                    Name of the column with 'y' variable
                x_vars : {str,list, dict, set, tuple, np.ndarray, pd.core.series.Series}
                    Name of the columns with 'x' variables
                entity_effects : bool
                    Peform regression with entity effects
                time_effects : bool
                    Peform regression with time effects
                time_entity_effects : bool
                    Peform regression both with entity and time effects
                other_effects : {str,list, dict, set, tuple, np.ndarray, pd.core.series.Series}
                    Category codes to use for any effects that are not entity or time effects. Each variable is treated as an effect.
                cluster_entity : bool
                    Cluster standard errors by entity
                cluster_time : bool
                    Cluster standard errors by time
                double_cluster : bool
                    Cluster standard errors bith by entity and time
                intercept : bool
                    Include intercept in the regression
                check_rank : bool
                    Check rank during regression
        
        Examples
        --------
        >>> model.add_spec(y='y2', entity_effects=True)
        >>> model.add_spec(y='y2', time_effects=True) 
        """
        new_spec = copy.deepcopy(self.baseline)
        try:
            if isinstance(kwargs[x_vars],(list,dict,set,tuple,np.ndarray,pd.core.series.Series)) != True:
                kwargs[x_vars] = [x_vars] 
        except:pass

        if 'data' in kwargs.keys():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UsingExecWarning)
                new_spec.data_name = argname('kwargs[data]')
            if isinstance(new_spec.data_name, Subscript):
                if 'data_name' not in kwargs.keys():
                    new_spec.data_name = 'data'
                    print("Can't retrieve name of dataset. Using default value. Rename the dataset or provide a 'data_name' argument.")

        if (('time_entity_effects' in kwargs) and ('other_effects' in kwargs)) or\
        ((
            ('time_effects' in kwargs) and
            ('entity_effects' in kwargs)
            )\
         and ('other_effects' in kwargs)):
            raise ValueError('At most two fixed effects are supported.')
            
        if 'other_effects' in kwargs:
            if kwargs['other_effects'] == []:
                kwargs['other_effects'] = None


        for key in kwargs: setattr(new_spec, key, kwargs[key])

        if (new_spec.time_effects or new_spec.entity_effects): new_spec.intercept = False

        
        if 'time_entity_effects' in kwargs:
            for comb in [(False, False), (True, False), (False, True), (True, True)]:
                variation = copy.deepcopy(new_spec)
                variation.entity_effects = comb[0]
                variation.time_effects = comb[1]
                variation.intercept = False
                self.specs.append(variation)


        else:self.specs.append(new_spec)


        
    def rename(self, rename_dict):
        """
        Rename the variables in the output table

        Parameters
        ----------
        rename_dict : dict
            Rename columns with the variables
        """
        for key in rename_dict.keys(): self._rename_dict[key] = rename_dict[key]
        
    def run(self,coeff_decimals=2,latex_path=None,
            time_fe_name='Time FEs', entity_fe_name='Entity FEs',
            other_fe_name = 'Other FEs',
            custom_row=None, display_datasets=False,
            rsquared='Inclusive'):
        """
        Run all regressions in the models

        Parameters
        ----------
        coeff_decimals : int
            Display the numbers in the results table 
            with certain number of fraction digits
        latex_path : str
            Write the table in LaTeX format to specified path
        time_fe_name : str
            Name for time fixed effects column
        entity_fe_name : str
            Name for entity fixed effects column
        custom_row : str
            Adds a custom row to the end of the table
        display_datasets : {bool,list}
            Display the names of databases in the results table. 
            If the value is 'True' then use original names of variables.
            Pass a list of strings to define custom names for databases.
        rsquared : str
            Type of R-squared. Default is 'Inclusive'. Other supported types - 'Standard', 'Between',
            'Overall', 'Within'.


        Returns
        -------
        pd.DataFrame
            Table with the results of the regressions
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
            "ignore",
            message="In a future version of pandas all arguments of concat\
            except for the argument 'objs' will be keyword-only.")
            if custom_row !=None:
                if isinstance (custom_row,list)!=True:
                    print('Custom row is not a list')
            regs = []
            for spec in tq(self.specs, 'Running regressions'):
                regs.append(spec.run())
            

            if rsquared == "Inclusive":
                rsquared_func = lambda reg: reg.rsquared_inclusive
            elif rsquared == "Standard":
                rsquared_func = lambda reg: reg.rsquared
            elif rsquared == "Between":
                rsquared_func = lambda reg: reg.rsquared_between
            elif rsquared == "Overall":
                rsquared_func = lambda reg: reg.rsquared_overall
            elif rsquared == "Within":
                rsquared_func = lambda reg: reg.rsquared_within    

            R2s  = list(map(rsquared_func, regs))
            regs = compare(regs, stars=True, precision='tstats')
            csv = regs.summary.as_csv()
            tab = pd.read_csv(io.StringIO(csv), skiprows=1)

            other_eff_dict = {}
            for idx, spec in enumerate(self.specs):
                if spec.other_effects != None:
                    other_eff_dict[idx] = spec.other_effects
            if other_eff_dict != {}:
                rows = tab.shape[0]
                cols = tab.shape[1]
                last_row = tab[rows-1:rows].copy()
                tab = tab[:rows-1]
                tab.loc[tab.shape[0]] = ['Other Effects'] + ['']*(cols-1)
                tab = pd.concat([tab,last_row])
                for key, value in other_eff_dict.items():
                    if isinstance (value, list):
                        renamed_items = []
                        for item in value:
                            if item in self._rename_dict.keys():
                                item = self._rename_dict[item]
                            renamed_items.append(item)
                        tab.iat[tab.shape[0]-2,int(key)+1] = ', '.join(renamed_items)
                    else:  
                        if value in self._rename_dict.keys():
                                    value = self._rename_dict[value]
                        tab.iat[tab.shape[0]-2,int(key)+1] = value
            tab = tab.set_index([tab.columns[0]])
            col_dict = dict(zip(tab.columns.to_list(), 
                              list(map(lambda x:'('+str(int(x.replace(' ','').replace('Model',''))+1)+')',
                                   tab.columns.to_list())))
            )
            coeff_borders = []
            observ = int()
            r2 = int()
            const = int()
            for idx, x in enumerate(tab.index):
                if 'No. Observations' in x:observ = idx
                if 'const' in x:const = idx
                if re.match('R-squared    ',x) != None:
                    r2 = idx
                if '===' in x:coeff_borders.append(idx)
            
            tab.rename(index={tab.index[observ]:'Observations', tab.index[const]:'Intercept'}, columns=col_dict, inplace=True)
            tab.loc['Observations'] = ["{0:0,.0f}".format(float(x)) for x in tab.loc['Observations']]
            try:coeffs = tab[coeff_borders[0]+1:coeff_borders[1]].copy()
            except:coeffs = tab[coeff_borders[0]+1:-1].copy()
            if coeff_decimals != None:
                def change_decimals(cell):
                    try:
                        if '(' in cell:
                            sub_str = float(re.search('\(-?[0-9]*\.[0-9]*' ,cell)[0][1:])
                            return '('+re.sub('\(-?[0-9]*\.[0-9]*', f'{sub_str:.{coeff_decimals}f}', cell)
                        else:
                            sub_str = float(re.search('^-?[0-9]*\.[0-9]*', cell)[0])
                            return re.sub('^-?[0-9]*\.[0-9]*', f'{sub_str:.{coeff_decimals}f}', cell)
                    except TypeError:
                        return cell
                coeffs = coeffs.applymap(change_decimals)
                s = "{0:0."+str(coeff_decimals)+"f}"
                R2s  = [ s.format(x) for x in R2s ]
            else:
                R2s  = [ "{0:0.4f}".format(x) for x in R2s ]

            if const != 0:
                coeffs = pd.concat([coeffs[2:], coeffs[0:2]])
            coeffs_dict = {}
            
            for idx,name in enumerate(coeffs.index):
                if re.sub('[ \t]+$','',name) in self._rename_dict.keys():
                    coeffs_dict[name] = self._rename_dict[re.sub('[ \t]+$', '', name)]
            coeffs.rename(index=coeffs_dict, inplace=True)
            final = pd.concat([tab.head(1), coeffs])
            for idx,name in enumerate(final.iloc[0]):
                if re.sub('[ \t]+$','', name) in self._rename_dict.keys():
                   final.iat[0,idx] = self._rename_dict[re.sub('[ \t]+$','', name)]

            # Add spacing
            final = pd.concat([final.iloc[:1], pd.DataFrame(index=[' ']), final.iloc[1:]])
            final = pd.concat([final, pd.DataFrame(index=[' '])])

            
            for line in [observ,r2]:
                final = pd.concat([final, tab[line:].head(1)])

            # Inclusive R2s (including fixed effects)
            final.iloc[-1] = R2s
            
            effects = pd.DataFrame(index=[time_fe_name, entity_fe_name, other_fe_name])
            some_effects = False
            for column in tab.columns:
                for x in tab[column]:
                    if re.search('Time', str(x))!=None: effects.loc[time_fe_name,column]='Yes'; some_effects = True
                    if re.search('Entity', str(x))!=None: effects.loc[entity_fe_name,column]='Yes'; some_effects = True
            if other_eff_dict != {}:
                for x in tab.iloc[tab.shape[0]-2,:].index:
                    effects.at[other_fe_name, x] = tab.iloc[tab.shape[0]-2,:][x]
                some_effects = True
            else:
                effects = effects[:2]
            if some_effects: final=pd.concat([final,effects.dropna(how='all')])
            
            clustering = pd.DataFrame(index=['SEs clustering:'], columns=final.columns)
            for idx, spec in enumerate(self.specs):
                if spec.cluster_entity == True:
                    clustering.iat[0, idx] = 'Entity'
                if spec.cluster_time == True:
                    clustering.iat[0, idx] = 'Time'
                if ((spec.cluster_time == True)\
                      & (spec.cluster_entity == True))\
                      | spec.double_cluster == True:
                    clustering.iat[0, idx] = 'Entity and Time'
            
            clustering.dropna(how='all',inplace=True)
            if clustering.shape[0] == 1:
                final = pd.concat([final,clustering])



            if display_datasets != False:
                if display_datasets == True:
                    data_info = pd.DataFrame(data=[[spec.data_name for spec in self.specs]]
                                            , columns=final.columns, index=['Dataset'])
                else:
                    data_info = pd.DataFrame(data=[display_datasets]
                        , columns=final.columns, index=['Dataset'])
                final = pd.concat([final, data_info])
            
            
            if custom_row != None:
                custom = pd.DataFrame(index=[custom_row[0]])
                for idx,item in enumerate(custom_row[1:]):
                    custom.at[custom_row[0],final.columns[idx]]=item
                final = pd.concat([final,custom])
            final.fillna('', inplace=True)
            final.drop(index=' ', inplace=True)
            if latex_path != None:
                latex_string = final.style.to_latex(column_format = 'l'+ ((final.shape[1])*'c'), hrules=True)
                # latex_string = re.sub('(?<=\{tabular\}\{l)(.*?)(?=\})',
                #                     'c'*len(re.search('(?<=\{tabular\}\{l)(.*?)(?=\})',
                #                     final.style.to_latex())[0]),final.style.to_latex())
                # latex_string = re.sub('{lcccc}\n','{lcccc}\n\\\\toprule\n{}', latex_string)
                # latex_string = re.sub('\nD','\n\\\midrule\nD', latex_string)
                # latex_string = re.sub('\n\\\end{tabular}\n','\n\\\\bottomrule\n\\\end{tabular}\n', latex_string)
                latex_string = align_latex_table(latex_string).replace('\\\\ \n',' \\\\\n')
                with open(latex_path, 'w') as f:
                    f.write(latex_string)   
            return final