import warnings
from linearmodels import PanelOLS
from linearmodels.panel import compare
from statsmodels.tools.tools import add_constant
import copy
import io
import pandas as pd
import numpy as np
import re 
class Spec():
    """
    Contains specification of regression
    """

    def __init__(self, 
            data, y, x_vars, 
            entity_effects=False, time_effects=False, all_effects=False,
            cluster_entity=False, cluster_time=False, double_cluster=False,
            intercept=True,check_rank=True
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
        self.double_cluster = double_cluster
        self.intercept=intercept
        self.check_rank=check_rank
        if (time_effects or entity_effects): intercept=False

    def __repr__(self):
        return (f'x-vars: {self.x_vars}, y: {self.y}, Entity Effects: {self.entity_effects} , Time Effects: {self.time_effects}, All Effects: {self.all_effects}, Cluster Entity: {self.cluster_entity}, Cluster Time: {self.cluster_time}, Double Cluster: {self.double_cluster}, Intercept: {self.intercept},Check rank: {self.check_rank}')  

    def run(self):
            
        reg = PanelOLS(
                self.data[[self.y]],
                add_constant(self.data[self.x_vars])if self.intercept==True else self.data[self.x_vars], 
                entity_effects=self.entity_effects, 
                time_effects  =self.time_effects,
                #check_rank=self.check_rank
                
            ).fit(
                cov_type   = 'clustered', 
            cluster_entity = (self.cluster_entity | self.double_cluster),
            cluster_time   = (self.cluster_time   | self.double_cluster)
        )
        return reg
    
                
class Model():
    """
    Contains multiple Spec objects
    """
    def __init__(self, baseline, rename_dict={}, all_effects=False):
        self._rename_dict = rename_dict#add check
        baseline.intercept=True
        self.baseline = baseline
        self.specs = []
        if all_effects:
            for comb in [(False,False),(True,False),(False,True),(True,True)]:
                new_spec = copy.deepcopy(self.baseline)
                new_spec.entity_effects = comb[0]
                new_spec.time_effects = comb[1]
                self.specs.append(new_spec)
        else:
            new_spec = copy.deepcopy(self.baseline)
            self.specs.append(new_spec)
    
    def __repr__(self):
        strr=''
        for idx,spec in enumerate(self.specs):
            strr=strr+(f'Spec {idx+1}: '+ spec.__repr__()+'\n')
        return strr
    
    def remove_spec (self,idx1,idx2=None):
        if idx2!=None:del self.specs[idx1-1:idx2-1] 
        else:del self.specs[idx1-1] 

    def add_spec( self, **kwargs):
        
        new_spec = copy.deepcopy(self.baseline)
        try:
            if isinstance(kwargs[x_vars],(list,dict,set,tuple,np.ndarray,pd.core.series.Series))!=True:
                kwargs[x_vars]=[x_vars] 
        except:pass
        for key in kwargs: setattr(new_spec, key, kwargs[key])

        if (new_spec.time_effects or new_spec.entity_effects): new_spec.intercept=False

        
        if 'all_effects' in kwargs:
            for comb in [(False,False),(True,False),(False,True),(True,True)]:
                variation = copy.deepcopy(new_spec)
                variation.entity_effects = comb[0]
                variation.time_effects = comb[1]
                variation.intercept=False
                self.specs.append(variation)


        else:self.specs.append(new_spec)
        
    def rename(self, rename_dict):
        for key in rename_dict.keys(): self._rename_dict[key]=rename_dict[key]
        
    def run(self,coeff_decimals=None,latex_path=None,time_fe_name='Time FEs', entity_fe_name='Entity FEs', custom_row=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="In a future version of pandas all arguments of concat except for the argument 'objs' will be keyword-only.")
            if custom_row !=None:
                if isinstance (custom_row,list)!=True:
                    print('Custom row is not a list')
            regs = [ spec.run() for spec in self.specs ]
            R2s  = [ reg.rsquared_inclusive for reg in regs ]
            regs= compare(regs, stars=True, precision='tstats')
            csv = regs.summary.as_csv()
            tab = pd.read_csv(io.StringIO(csv), skiprows=1)
            tab = tab.set_index([tab.columns[0]])
            col_dict=dict(zip(tab.columns.to_list(), list(map(lambda x:'('+str(int(x.replace(' ','').replace('Model',''))+1)+')',tab.columns.to_list()))))
            coeff_borders=[]
            observ=int()
            r2=int()
            const=int()
            for idx,x in enumerate(tab.index):
                if 'No. Observations' in x:observ=idx
                if 'const' in x:const=idx
                if re.match('R-squared    ',x)!=None:
                    r2=idx
                if '===' in x:coeff_borders.append(idx)
            
            tab.rename(index={tab.index[observ]:'Observations',tab.index[const]:'Intercept'},columns=col_dict, inplace=True)
            tab.loc['Observations'] = ["{0:0,.0f}".format(float(x)) for x in tab.loc['Observations']]
            try:coeffs=tab[coeff_borders[0]+1:coeff_borders[1]].copy()
            except:coeffs=tab[coeff_borders[0]+1:-1].copy()
            if coeff_decimals!=None:
                def change_decimals(cell):
                    try:
                        return re.sub('^-?[0-9]*\.[0-9]*',str(round(float(re.search('^-?[0-9]*\.[0-9]*' ,cell)[0]),coeff_decimals)),cell)
                    except:
                        return cell
                coeffs=coeffs.applymap(change_decimals)
                s = "{0:0."+str(coeff_decimals)+"f}"
                R2s  = [ s.format(x) for x in R2s ]
            else:
                R2s  = [ "{0:0.4f}".format(x) for x in R2s ]

            if const!=0:coeffs=pd.concat([coeffs[2:],coeffs[0:2]])
            coeffs_dict={}
            for idx,name in enumerate(coeffs.index):
                if re.sub('[ \t]+$','',name) in self._rename_dict.keys():
                    coeffs_dict[name]= self._rename_dict[re.sub('[ \t]+$','',name)]
            coeffs.rename(index=coeffs_dict,inplace=True)
            final=pd.concat([tab.head(1),coeffs])
            for idx,name in enumerate(final.iloc[0]):
                if re.sub('[ \t]+$','',name) in self._rename_dict.keys():
                   final.iloc[0][idx]= self._rename_dict[re.sub('[ \t]+$','',name)]

            # Add spacing
            final=pd.concat([final.iloc[:1],pd.DataFrame(index=[' ']), final.iloc[1:]])
            final=pd.concat([final,pd.DataFrame(index=[' '])])

            
            for line in [observ,r2]:
                final=pd.concat([final,tab[line:].head(1)])

            # Inclusive R2s (including fixed effects)
            final.iloc[-1] = R2s

            effects=pd.DataFrame(index=[time_fe_name, entity_fe_name])
            some_effects = False
            for column in tab.columns:
                for x in tab[column]:
                    if re.search('Time', str(x))!=None: effects.loc[time_fe_name,column]='Yes'; some_effects = True
                    if re.search('Entity', str(x))!=None: effects.loc[entity_fe_name,column]='Yes'; some_effects = True
            if some_effects: final=pd.concat([final,effects])
            if custom_row!=None:
                custom=pd.DataFrame(index=[custom_row[0]])
                for idx,item in enumerate(custom_row[1:]):
                    custom.at[custom_row[0],final.columns[idx]]=item
                final=pd.concat([final,custom])
            final.fillna('',inplace=True)
            if latex_path!=None:
                f=open(latex_path,'w')
                f.write(re.sub('(?<=\{tabular\}\{l)(.*?)(?=\})','c'*len(re.search('(?<=\{tabular\}\{l)(.*?)(?=\})',\
                    final.style.to_latex())[0]),final.style.to_latex()))  
            return final