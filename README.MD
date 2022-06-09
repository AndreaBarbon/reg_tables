## Description

## Installation
```
pip install reg-tables
```


## Usage

This package consists of two classes: Spec and Model. Spec defines the specifications of regression and requires to provide data from which the variables are going to be taken, y variable, and x variables. Optional arguments for this class include specifying whether the regressions should be performed with entity effects, time effects or both (entity_effects, time_effects or all_effects arguments respectively). Methods for Spec class include ‘.run’, which runs the regression and ‘.rename’ – a method to rename variable according to the dictionary passed.

Other class of the package is Model. Model is the wrapper for ‘compare’ function of linearmodels. When creating Model, one has to specify the baseline Spec object on which the regression has to be performed. Optional arguments include passing a ‘rename_dict’, according to which the variables are going to be renamed, as well as setting an ‘all_effects’ Boolean variable, which will add the four versions of baseline Spec object with all possible combinations of entity and time effects to the model. Model class has ‘.rename’, ‘.add_spec’ and ‘.remove_spec’ methods. The latter has a mandatory index argument and second optional index argument, which, if passed would work as a end point for slice.  The ‘.run’ method executes all Spec objects within Model and outputs them to a table. Optional argument ‘coeff_decimals’ allows to specify the number of decimals for coefficient estimates and t-values, while ‘latex_path’ allows to save the output table to a disk.
