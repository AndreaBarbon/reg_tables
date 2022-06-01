from linearmodels import PanelOLS
from linearmodels.panel import compare
import copy
import io

te = dz.set_index(['artist', 'last_day']).copy()
te['F' ] = te['Followers_fitted_v2_L1']
te['F2'] = te['time2sale_L1']
te['F3'] = te['time2sale']
te = te[['F','F2', 'F3','win_prc']]
te = te.dropna()


class Spec():
    
    def __init__(self, data, y, x_vars, entity_effects=False):
        self.data = data
        self.y = y
        self.x_vars = x_vars
        self.entity_effects = entity_effects
        
    def run(self):
        reg = PanelOLS(
                self.data[[self.y]], 
                self.data[self.x_vars], 
                entity_effects=self.entity_effects, 
                time_effects=True
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
        
    def add_spec(self,data=None, y=None, x_vars=None, entity_effects=None):
        new_spec = copy.deepcopy(self.baseline)
        if data is not None: new_spec.data = data
        if y is not None: new_spec.y = y
        if x_vars is not None: new_spec.x_vars = x_vars
        if entity_effects is not None: new_spec.entity_effects = entity_effects
        new_spec.rename(self.rename_dict)
        self.specs.append(new_spec)
        
    def rename(self, rename_dict):
        for spec in self.specs: spec.rename(rename_dict)
        
    def run(self):
        regs = [ spec.run() for spec in self.specs ]
        return compare(regs, stars=True, precision='tstats')
        

baseline = Spec( te, 'win_prc', ['F', 'F2'] )
rename   = {
    'win_prc': '$P_t$',
    'F' : '$F_1$',
}

model = Model(baseline, rename_dict=rename)
model.add_spec(
    y = 'F',
    x_vars = ['F2', 'F3'],
    entity_effects = True,
)
regs = model.run()
csv = regs.summary.as_csv()
tab = pd.read_csv(io.StringIO(csv), skiprows=1)
tab = tab.set_index([tab.columns[0]])
tab


#spec2.x = ['F2', 'F3']

#compare([spec1.run(), spec2.run()])
