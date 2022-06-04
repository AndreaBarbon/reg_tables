from linearmodels import PanelOLS
from linearmodels.panel import compare
import copy
import io
import pandas as pd
import numpy as np
import re

class Spec():
    
    def __init__(self, data, y, x_vars, entity_effects=False,time_effects=False):
        self.data = data
        self.y = y
        if isinstance(x_vars,(list,dict,set,tuple,np.ndarray,pd.core.series.Series))!=True:
            x_vars=[x_vars]
        self.x_vars = x_vars
        self.entity_effects = entity_effects
        self.time_effects = time_effects
        
    def run(self):
        reg = PanelOLS(
                self.data[[self.y]], 
                self.data[self.x_vars], 
                entity_effects=self.entity_effects, 
                time_effects=self.time_effects
            ).fit(
                cov_type = 'clustered', cluster_entity=True, cluster_time=False
        )
        return reg
    
    def rename(self, rename_dict):
        self.data.rename(rename_dict, inplace=True, axis=1)
        if self.y in rename_dict.keys():
            self.y = rename_dict[self.y]
        
        new_x_vars = []
        for x in self.x_vars:
            if x in rename_dict.keys():
                new_x_vars.append(rename_dict[x])
            else:
                new_x_vars.append(x)
                
        self.x_vars = new_x_vars
                

    
class Model():
    
    def __init__(self, baseline, rename_dict={}):
        self.rename_dict = rename_dict
        self.baseline = baseline
        baseline.rename(self.rename_dict)
        self.specs = [self.baseline]
        for comb in [(True,False),(False,True),(True,True)]:
            new_spec = copy.deepcopy(self.baseline)
            new_spec.entity_effects = comb[0]
            new_spec.time_effects = comb[1]
            self.specs.append(new_spec)
        
    def add_spec(self,data=None, y=None, x_vars=None):
        new_spec = copy.deepcopy(self.baseline)
        if data is not None: new_spec.data = data
        if y is not None: new_spec.y = y
        if x_vars is not None: new_spec.x_vars = x_vars
        new_spec.rename(self.rename_dict)
        self.specs.append(new_spec)
        for comb in [(True,False),(False,True),(True,True)]:
            variation = copy.deepcopy(new_spec)
            variation.entity_effects = comb[0]
            variation.time_effects = comb[1]
            self.specs.append(variation)
        
    def rename(self, rename_dict):
        for spec in self.specs: spec.rename(rename_dict)
        
    def run(self):
        regs = [ spec.run() for spec in self.specs ]
        regs= compare(regs, stars=True, precision='tstats')
        csv = regs.summary.as_csv()
        tab = pd.read_csv(io.StringIO(csv), skiprows=1)
        tab = tab.set_index([tab.columns[0]])
        col_dict=dict(zip(tab.columns.to_list(), list(map(lambda x:'('+str(int(x.replace(' ','').replace('Model',''))+1)+')',tab.columns.to_list()))))
        coeff_borders=[]
        observ=int()
        r2=int()
        for idx,x in enumerate(tab.index):
            if 'No. Observations' in x:observ=idx
            if re.match('R-squared    ',x)!=None:
                r2=idx
            if '===' in x:coeff_borders.append(idx)
        tab.rename(index={tab.index[observ]:'Observations'},columns=col_dict, inplace=True)
        final=pd.concat([tab.head(1),tab[coeff_borders[0]+1:coeff_borders[1]]])
        for line in [observ,r2]:
            final=pd.concat([final,tab[line:].head(1)])
        effects=pd.DataFrame(index=['Time FEs', 'Entity FEs'])
        for column in tab.columns:
            for x in tab[column]:
                if re.search('Time', str(x))!=None: effects.loc['Time FEs',column]='Yes'
                if re.search('Entity', str(x))!=None: effects.loc['Entity FEs',column]='Yes'
        return pd.concat([final,effects]).fillna('')
