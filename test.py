from linearmodels import PanelOLS
from linearmodels.panel import compare
import copy
import io
import pandas as pd
import numpy as np
import re 
class Spec():
    
    def __init__(self, 
            data, y, x_vars, 
            entity_effects=False, time_effects=False, all_effects=False,
            cluster_entity=False, cluster_time=False,
        ):
        self.data = data
        self.y = y
        if isinstance(x_vars,(list,dict,set,tuple,np.ndarray,pd.core.series.Series))!=True:
            x_vars=[x_vars]
            
        self.x_vars = x_vars
        self.entity_effects = entity_effects
        self.time_effects = time_effects
        self.all_effects = all_effects
        self.cluster_entity = cluster_entity
        self.cluster_time = cluster_time

    def __repr__(self):
        return (f'x-vars: {self.x_vars}, y-var: {self.y}')  

    def run(self):
        reg = PanelOLS(
                self.data[[self.y]], 
                self.data[self.x_vars], 
                entity_effects=self.entity_effects, 
                time_effects=self.time_effects,
                
            ).fit(
                cov_type = 'clustered', 
            cluster_entity=self.cluster_entity, 
            cluster_time=self.cluster_time
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
    
    def __init__(self, baseline, rename_dict={}, all_effects=False):
        self.rename_dict = rename_dict
        self.baseline = baseline
        baseline.rename(self.rename_dict)
        self.specs = [self.baseline]
        if all_effects:
            for comb in [(True,False),(False,True),(True,True)]:
                new_spec = copy.deepcopy(self.baseline)
                new_spec.entity_effects = comb[0]
                new_spec.time_effects = comb[1]
                self.specs.append(new_spec)
    
    def __repr__(self):
        strr=''
        for idx,basespec in enumerate(self.specs[0:len(self.specs):4]):
            strr=strr+(f'Spec {idx}: x-vars: {basespec.x_vars}, y-var: {basespec.y}\n')
        return strr
    
    def remove_spec (self,base_id):
        del self.specs[base_id*4:(base_id+1)*4]

    def add_spec( self, **kwargs):
        
        new_spec = copy.deepcopy(self.baseline)
        
        for key in kwargs: 
            setattr(new_spec, key, kwargs[key])
            
        print(new_spec.cluster_time)
            
        new_spec.rename(self.rename_dict)
        self.specs.append(new_spec)
        
        if 'all_effects' in kwargs:
            for comb in [(True,False),(False,True),(True,True)]:
                variation = copy.deepcopy(new_spec)
                variation.entity_effects = comb[0]
                variation.time_effects = comb[1]
                self.specs.append(variation)
        
    def rename(self, rename_dict):
        for spec in self.specs: spec.rename(rename_dict)
        
    def run(self,coeff_decimals=None,latex_path=None):
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
        tab.loc['Observations'] = ["{0:0,.0f}".format(float(x)) for x in tab.loc['Observations']]
        try:coeffs=tab[coeff_borders[0]+1:coeff_borders[1]]
        except:coeffs=tab[coeff_borders[0]+1:-1]
        if coeff_decimals!=None:
            def change_decimals(cell,decimals=4):
                if '*' in cell:
                    return  re.sub('^-?[0-9].*?(?=\*)',str(round(float(re.search('^-?[0-9].*?(?=\*)' ,cell)[0]),decimals)),cell)
                elif '(' in cell:
                    return  re.sub('(?<=\()(.*)(?=\))',str(round(float(re.search('(?<=\()(.*)(?=\))' ,cell)[0]),decimals)),cell)
                else:return ''
            coeffs=coeffs.applymap(change_decimals,decimals=coeff_decimals)
        final=pd.concat([tab.head(1),coeffs])
        
        for line in [observ,r2]:
            final=pd.concat([final,tab[line:].head(1)])
        effects=pd.DataFrame(index=['Time FEs', 'Entity FEs'])
        some_effects = False
        for column in tab.columns:
            for x in tab[column]:
                if re.search('Time', str(x))!=None: effects.loc['Time FEs',column]='Yes'; some_effects = True
                if re.search('Entity', str(x))!=None: effects.loc['Entity FEs',column]='Yes'; some_effects = True
        if some_effects: final=pd.concat([final,effects]).fillna('')
        if latex_path!=None:
            f=open(latex_path,'w')
            f.write(final.style.to_latex())  
        return final

