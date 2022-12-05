import logging
import statsmodels.api as sm
from statsmodels.formula.api import ols

logger = logging.getLogger(__name__)

class StepwiseRegressionSelector():

    def __init__(self, data, target, initial_list=None, threshold_in=0.05, threshold_out=0.10, verbose=True):
        '''
        data: pandas dataframe with all possible predictors and response
        target: string, name of response column in data
        initial_list: initial list of predictors to start with (column names of data)
        threshold_in: include a predictor if its p-value < threshold_in
        threshold_out: exclude a predictor if its p-value > threshold_out
        verbose: whether to logging.info the sequence of inclusions and exclusions
        '''

        self.data = data
        self.target = target
        self.threshold_in = threshold_in
        self.threshold_out = threshold_out
        self.included = initial_list if initial_list else []
        self.verbose = verbose

        # assert that the target variable is in the data
        assert self.target in self.data.columns, \
            f"Target variable {self.target} not in the data"
        # assert whether threshold_in < threshold_out
        assert self.threshold_in < self.threshold_out, \
            f"threshold_in {self.threshold_in} should be less than threshold_out {self.threshold_out}"
        # assert whether the initial_list is in the data
        if self.included:
            assert set(self.included).issubset(
                set(self.data.columns)), f"Initial list {self.included} not in the data"

        # rename all variables in a equal-length manner
        # to avoid the problem of variable name with special characters
        # as well as a certain variable name being the substring of another variable
        self.decode_dict = self.encode()

    def encode(self):
        # rename all variables in a equal-length manner
        # to avoid the problem of variable name with special characters
        # as well as a certain variable name being the substring of another variable

        # if we have 16 variables, they should be renamed as f01, f02, ..., f15, f16
        # if we have 123 variables, they should be renamed as f001, f002, ..., f012, ..., f123
        self.encode_dict = dict()
        for i, var in enumerate(self.data.columns):
            self.encode_dict[var] = 'f' + \
                str(i+1).zfill(len(str(len(self.data.columns))))
        self.data = self.data.rename(columns=self.encode_dict)

        return {v: k for k, v in self.encode_dict.items()}

    def forward_backward(self):

        excluded = self.data.columns.tolist()
        excluded.remove(self.encode_dict[self.target])
        # set a constant model as initial reduced_model
        formula = f'{self.encode_dict[self.target]}~1'
        best_r2_dif = .0

        while True:  # end loop when no variable gets in/out

            changed = False

            full_model = ols(formula=formula, data=self.data).fit()
            last_adj_r2 = full_model.rsquared_adj

            # forward step
            for new_feature in excluded:

                # Note here the test_model has more variable than full_model
                test_model = ols(
                    formula=formula+f'+{new_feature}', data=self.data).fit()

                # find feature whose contribution to adj_r2 largest
                if test_model.rsquared_adj - last_adj_r2 > best_r2_dif:
                    best_r2_dif = test_model.rsquared_adj - last_adj_r2
                    last_adj_r2 = test_model.rsquared_adj
                    best_feature = new_feature

            # Partial F-test
            # Note that in anova_lm models with few variables are put forward
            full_model_pro = ols(
                formula=formula+f'+{best_feature}', data=self.data).fit()
            anova_tbl = sm.stats.anova_lm(full_model, full_model_pro)
            criterion = anova_tbl['Pr(>F)'][1]

            if criterion <= self.threshold_in:
                self.included.append(best_feature)
                excluded.remove(best_feature)
                formula += f'+{best_feature}'
                full_model = full_model_pro
                changed = True
                best_r2_dif = .0
                if self.verbose:
                    logging.info('Add  {:40} with f_pvalue {:.6}'.format(
                        self.decode_dict[best_feature], criterion))

            # backward step
            for old_feature in self.included:
                test_model = ols(formula=formula.replace(
                    f'+{old_feature}', ''), data=self.data).fit()

                # Note here the test_model has less variable than full_model
                anova_tbl = sm.stats.anova_lm(test_model, full_model)
                criterion = anova_tbl['Pr(>F)'][1]
                if criterion >= self.threshold_out:
                    self.included.remove(old_feature)
                    excluded.append(old_feature)
                    formula = formula.replace(f'+{old_feature}', '')
                    changed = True
                    best_r2_dif = .0
                    if self.verbose:
                        logging.info('Drop {:40} with f_pvalue {:.6}'.format(
                            self.decode_dict[old_feature], criterion))

            if not changed:
                break

        # compute r^2 of linear model constructed by selected variables
        full_model = ols(formula=formula, data=self.data).fit()
        r2 = full_model.rsquared_adj
        final_variables = [self.decode_dict[var] for var in self.included]
        logging.info(f'Final variables: {final_variables}')
        logging.info(f'Final r^2      : {r2}')

        # # show formula of linear model constructed by selected variables
        # # with the original variable names according to self.decode_dict
        # for key, value in self.decode_dict.items():
        #     formula = formula.replace(key, value)

        # logging.info(f'Final formula  : {formula}')

        return final_variables