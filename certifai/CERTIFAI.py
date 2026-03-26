# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:37:55 2020

@author: Iacopo
Modified by alanpar97
"""
import importlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import manhattan_distances as L1
from sklearn.metrics.pairwise import euclidean_distances as L2


class CERTIFAI:
    def __init__(self, Pm=.2, Pc=.5, dataset_path=None,
                 numpy_dataset=None, pandas_dataset=None, verbose=False):
        """The class instance is initialised with the probabilities needed
        for the counterfactual generation process and an optional path leading
        to a .csv file containing the training set. If the path is provided,
        the class will assume in some of its method that the training set is tabular
        in nature and pandas built-in functions will be used in several places, instead
        of the numpy or self defined alternatives."""

        self.column_names = None
        self.Pm = Pm
        self.Pc = Pc
        self.Population = None
        self.distance = None
        self.constraints = None
        self.predictions = None
        self.results = None
        self.verbose = verbose

        if dataset_path is not None:
            self.tab_dataset = pd.read_csv(dataset_path)
        elif pandas_dataset is not None:
            if not isinstance(pandas_dataset, pd.DataFrame):
                raise ValueError("pandas_dataset must be a pandas DataFrame")
            self.tab_dataset = pandas_dataset.copy()
        elif numpy_dataset is not None:
            self.tab_dataset = numpy_dataset
        else:
            self.tab_dataset = None

    @classmethod
    def from_csv(cls, path):
        return cls(dataset_path=path)

    def get_con_cat_columns(self, x):

        assert isinstance(x, pd.DataFrame), 'This method can be used only if input\
            is an instance of pandas dataframe at the moment.'

        con = []
        cat = []

        for column in x:
            if x[column].dtype == 'O':
                cat.append(column)
            else:
                con.append(column)

        return con, cat

    def tabular_distance(self, x, y, continuous_distance='L1', con=None,
                     cat=None):
        """Distance function for tabular data, as described in the original
        paper. This function is the default one for tabular data in the paper and
        in the set_distance function below as well. For this function to be used,
        the training set must consist of a .csv file as specified in the class
        instatiation above. This way, pandas can be used to infer whether a
        feature is categorical or not based on its pandas datatype and, as such, it is important that all columns
        in the dataframe have the correct datatype.

        Arguments:
            x (pandas.dataframe): the input sample from the training set. This needs to be
            a row of a pandas dataframe at the moment, but the functionality of this
            function will be extended to accept also numpy.ndarray.

            y (pandas.dataframe or numpy.ndarray): the comparison samples (i.e. here, the counterfactual)
            which distance from x needs to be calculated.

            continuous_distance (bool): the distance function to be applied
            to the continuous features. Default is L1 function.

            con (list): list of the continuous features (i.e. columns) names

            cat (list): list of the categorical features (i.e. columns) names
        """

        assert isinstance(x, pd.DataFrame), 'This distance can be used only if input\
            is a row of a pandas dataframe at the moment.'

        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y, columns=x.columns.tolist())
        else:
            y.columns = x.columns.tolist()

        if con is None or cat is None:
            con, cat = self.get_con_cat_columns(x)

        if 0 < len(cat):
            cat_distance = len(cat) - (x[cat].values == y[cat].values).sum(axis=1)
        else:
            cat_distance = 1

        if len(con) > 0:
            if continuous_distance == 'L1':
                con_distance = L1(x[con], y[con])
            else:
                con_distance = L2(x[con], y[con])
        else:
            con_distance = 1

        return len(con) / x.shape[-1] * con_distance + len(cat) / x.shape[-1] * cat_distance

    def set_distance(self, kind='automatic', x=None):
        """Set the distance function to be used in counterfactual generation.
        The distance function can either be manually chosen by passing the
        relative value to the kind argument or it can be inferred by passing the
        'automatic' value.

        Arguments:
            Inputs:
                kind (string): possible values, representing the different
                distance functions are: 'automatic', 'L1', 'L2' and 'euclidean' (same as L2)

                x (numpy.ndarray or pandas.DataFrame): training set or a sample from it on the basis
                of which the function will decide what distance function to use if kind=='automatic'.
                If the training set is a .csv file then
                the function more suitable for tabular data will be used, otherwise the function
                will backoff to using L1 norm distance.

        Outputs:
            None, set the distance attribute as described above."""

        if kind == 'automatic':
            assert x is not None or self.tab_dataset is not None, 'For using automatic distance assignment,\
                the input data needs to be provided or the class needs to be initialised with a csv file!'

            if x is None:
                x = self.tab_dataset
            else:
                con, cat = self.get_con_cat_columns(x)
                if len(cat) > 0:
                    self.distance = self.tabular_distance
                else:
                    self.distance = L1

        elif kind == 'tab_distance':
            self.distance = self.tabular_distance
        elif kind == 'L1':
            self.distance = L1
        elif kind == 'L2':
            self.distance = L2
        elif kind == 'euclidean':
            self.distance = L2
        else:
            raise ValueError('Distance function specified not recognised:\
                             use one of automatic, L1, L2 or euclidean.')

    def set_population(self, x=None):
        """Set the population limit (i.e. number of counterfactuals created at each generation).
        following the original paper, we define the maximum population as the minum between the squared number of features
        to be generated and 30000.

        Arguments:
            Inputs:
                x (numpy.ndarray or pandas.DataFrame): the training set or a sample from it, so that the number of features can be obtained.

            Outputs:
                None, the Population attribute is set as described above
        """

        if x is None:
            assert self.tab_dataset is not None, 'If input is not provided, the class needs to be instatiated\
                with an associated csv file, otherwise there is no input data for inferring population size.'

            x = self.tab_dataset

        if len(x.shape) > 2:
            self.Population = min(sum(x.shape[1:]) ** 2, 30000)
        else:
            self.Population = min(x.shape[-1] ** 2, 30000)

    def set_constraints(self, x=None, fixed=None):
        '''Set the list of constraints for each input feature, whereas
        each constraint consist in the minimum and maximum value for
        the given continuous feature. If a categorical feature is encountered,
        then the number of unique categories is appended to the list instead.

        Arguments:
            Inputs:
            x (numpy.ndarray): if the training set is not a pandas dataframe (see above),
            this function will expect a numpy array with the entire training set in the
            form of a numpy array.

            fixed (list): a list of features to be kept fixed in counterfactual generation
            (i.e. all counterfactual will have the same value as the given sample for that
             feature). If no list is provided, then no feature will be kept fixed

            Outputs:
                None, an attribute 'constraints' is created for the class, where
                the constraints are stored.
            '''

        fixed_feats = set() if fixed is None else set(fixed)

        self.constraints = []

        if x is None:
            x = self.tab_dataset

        if len(x.shape) > 2:
            x = self.tab_dataset if self.tab_dataset is not None else x.copy()

            x = pd.DataFrame(x.reshape(x.shape[0], -1))

        if isinstance(x, pd.DataFrame):
            for i in x:
                if i in fixed_feats:
                    # Placeholder if the feature needs to be kept fixed in generating counterfactuals
                    self.constraints.append(i)
                # Via a dataframe is also possible to constran categorical fatures (not supported for numpy array)
                elif x.loc[:, i].dtype == 'O':
                    self.constraints.append((0, len(pd.unique(x.loc[:, i]))))
                else:
                    self.constraints.append((min(x.loc[:, i]), max(x.loc[:, i])))
        else:
            assert x is not None, 'A numpy array should be provided to get min-max values of each column,\
                or, alternatively, a .csv file needs to be supplied when instatiating the CERTIFAI class'

            for i in range(x.shape[1]):
                if i in fixed_feats:
                    # Placeholder if the feature needs to be kept fixed in generating counterfactuals
                    self.constraints.append(i)
                else:
                    self.constraints.append((min(x[:, i]), max(x[:, i])))
        if self.verbose:
            print(f'Constraints have been set for the input data.{self.constraints}')

    def generate_prediction(self, model, model_input, model_type="torch", classification=False):
        '''Function to output prediction from a deep learning or machine learning model.

        Arguments:
            Inputs:
                model (torch.nn.Module, tf.Keras.Model, or sklearn model): the trained model.

                model_input (torch.tensor, numpy.ndarray, or pandas.DataFrame): the input to the model.

                model_type (str): the type of model, which can be "torch", "tf", or "sklearn".

                classification (bool): whether a classification or regression task is performed.

            Output:
                prediction (numpy.ndarray): the array containing the predicted values.
        '''
        if classification:
            if model_type == "torch":
                torch = importlib.import_module("torch")
                with torch.no_grad():
                    prediction = np.argmax(model(model_input).numpy(), axis=-1)
            elif model_type == "tf":
                prediction = np.argmax(model.predict(model_input), axis=-1)
            elif model_type == "sklearn":
                prediction = model.predict(model_input)
        else:
            if model_type == "torch":
                torch = importlib.import_module("torch")
                with torch.no_grad():
                    prediction = model(model_input).numpy()
            elif model_type == "tf":
                prediction = model.predict(model_input)
            elif model_type == "sklearn":
                prediction = model.predict(model_input)

        return prediction

    def generate_counterfacts_list_dictionary(self, counterfacts_list,
                                              distances, fitness_dict,
                                              retain_k, start=0):
        '''Function to generate and trim at the same time the list containing
        the counterfactuals and a dictionary having fitness score
        for each counterfactual index in the list.

        Arguments:
            Inputs:
                counterfacts_list (list): list containing each initial counterfactual

                distances (numpy.ndarray): array containing distance from sample
                to each counterfactual

                fitness_dict (dict): dictionary containing the fitness score
                (i.e. distance) for each counterfactual index in the list
                as key. If an empty dictionary is passed to the function, then
                the index of the counterfactual starts from 0, else it starts
                counting from the value of the start argument.

                start (int): index from which to start assigning keys for
                the dictionary

            Outputs:
                selected_counterfacts (list): list of top counterfacts from
                the input list, selected on the basis of their fitness score
                and having length=retain_k

                fitness_dict (dict): dictionary of fitness scores stored
                by the relative counterfactual index.'''

        gen_dict = {i: distance for i, distance in enumerate(distances)}
        gen_dict = {k: v for k, v in sorted(gen_dict.items(), key=lambda item: item[1])}
        selected_counterfacts = []

        k = 0
        for key, value in gen_dict.items():
            if k == retain_k:
                break
            selected_counterfacts.append(counterfacts_list[key])
            fitness_dict[start + k] = value
            k += 1
        return selected_counterfacts, fitness_dict

    def generate_cats_ids(self, dataset=None, cat=None):
        '''Generate the unique categorical values of the relative features
        in the dataset.

        Arguments:
            Inputs:

            dataset (pandas.dataframe): the reference dataset from which to extract
            the categorical values. If not provided, the function will assume that
            a dataframe has been saved in the class instance when created, via the
            option for initialising it from a csv file.

            cat (list): list of categorical features in the reference dataset. If
            not provided, the list will be obtained via the relative function of
            this class.

            Output:

            cat_ids (list): a list of tuples containing the unique categorical values
            for each categorical feature in the dataset, their number and the relative
            column index.
        '''
        if dataset is None:
            assert self.tab_dataset is not None, 'If the dataset is not provided\
            to the function, a csv needs to have been provided when instatiating the class'

            dataset = self.tab_dataset

        if cat is None:
            con, cat = self.get_con_cat_columns(dataset)

        cat_ids = []
        for index, key in enumerate(dataset):
            if key in set(cat):
                cat_ids.append((index,
                                len(pd.unique(dataset[key])),
                                pd.unique(dataset[key])))
        return cat_ids

    def generate_initial_candidates_tab(self,
                                sample,
                                normalisation=None,
                                constrained=True,
                                has_cat=False,
                                cat_ids=None,
                                img=False):
        '''Function to generate the random (constrained or unconstrained)
        candidates for counterfactual generation if the input is a pandas
        dataframe (i.e. tabular data).

        Arguments:
            Inputs:
                sample (pandas.dataframe): the input sample for which counterfactuals
                need to be generated.

                normalisation (str) ["standard", "max_scaler"]: the
                normalisation technique used for the data at hand. Default
                is None, while "standard" and "max_scaler" techniques are
                the other possible options. According to the chosen value
                different random number generators are used.

                constrained (bool): whether the generation of each feature
                will be constrained by its minimum and maximum value in the
                training set (it applies just for the not normalised scenario
                              as there is no need otherwise)

                has_cat (bool): whether the input sample includes categorical
                variables or not (they are treated differently in the distance
                                  function used in the original paper).

                cat_ids (list): list of the names of columns containing categorical
                values.

            Outputs:
                generation (list[list]): a list the random candidates generated.

                distances (numpy.ndarray): an array of distances of the candidates
                from the current input sample.
                '''

        nfeats = sample.shape[-1]

        if normalisation is None:
            if constrained:
                generation = []
                temp = []
                for constraint in self.constraints:
                    if not isinstance(constraint, tuple):
                        # For fixed feature, repeat its value to match the population size
                        temp.append(np.full((self.Population, 1), sample.loc[:, constraint].values))
                    else:
                        #Generate random candidates
                        temp.append(np.random.randint(constraint[0] * 100, (constraint[1] + 1) * 100,
                                                      size=(self.Population, 1)) / 100)
                generation = np.concatenate(temp, axis=-1)
            else:
                # If not constrained, we still don't want to generate values that are not totally unrealistic
                low = min(sample)
                high = max(sample)
                #Generate random candidates
                generation = np.random.randint(low, high + 1, size=(self.Population, nfeats))
        elif normalisation == 'standard':
            generation = np.random.randn(self.Population, nfeats)
        elif normalisation == 'max_scaler':
            generation = np.random.rand(self.Population, nfeats)
        else:
            raise ValueError('Normalisation option not recognised:\
                             choose one of "None", "standard" or\
                                 "max_scaler".')

        if has_cat:
            assert cat_ids is not None, 'If categorical features are included in the dataset,\
                the relative cat_ids (to be generated with the generate_cats_ids method) needs\
                    to be provided to the function.'
            generation = pd.DataFrame(generation, columns=sample.columns.tolist())

            for idx, ncat, cat_value in cat_ids:
                random_indeces = np.random.randint(0, ncat, size=self.Population)
                random_cats = [cat_value[feat] for feat in random_indeces]
                generation.iloc[:, idx] = random_cats

            distances = self.distance(sample, generation)[0]
            generation = generation
        else:
            distances = self.distance(sample, generation)[0]
            generation = generation.tolist()
            generation = pd.DataFrame(generation, columns=sample.columns.tolist())

        for i in sample:
            generation[i] = generation[i].astype(sample[i].dtype)

        return generation.values.tolist(), distances

    def mutate(self, counterfacts_list):
        '''Function to perform the mutation step from the original paper

        Arguments:
            Input:
                counterfacts_list (list): the candidate counterfactuals
                from the selection step.

            Output:
                mutated_counterfacts (numpy.ndarray): the mutated candidate
                counterfactuals.'''

        nfeats = len(counterfacts_list[0])

        dtypes = [type(feat) for feat in counterfacts_list[0]]

        counterfacts_df = pd.DataFrame(counterfacts_list)

        random_indeces = np.random.binomial(1, self.Pm, len(counterfacts_list))

        mutation_indeces = [index for index, i in enumerate(random_indeces) if i]

        for index in mutation_indeces:
            mutation_features = np.random.randint(0, nfeats,
                                                  size=np.random.randint(1, nfeats))

            for feat_ind in mutation_features:
                if isinstance(counterfacts_df.iloc[0, feat_ind], str):
                    counterfacts_df.iloc[index, feat_ind] = np.random.choice(
                        np.unique(counterfacts_df.iloc[:, feat_ind]))

                else:
                    counterfacts_df.iloc[index, feat_ind] = (
                            0.5 * (
                            np.random.choice(counterfacts_df.iloc[:, feat_ind]) +
                            np.random.choice(counterfacts_df.iloc[:, feat_ind])
                    )
                    ).astype(counterfacts_df.dtypes[feat_ind])

        for index, key in enumerate(counterfacts_df):
            counterfacts_df[key] = counterfacts_df[key].astype(dtypes[index])

        return counterfacts_df.values.tolist()

    def crossover(self, counterfacts_list, return_df=False):
        '''Function to perform the crossover step from the original paper

        Arguments:
            Input:
                counterfacts_list (list): the candidate counterfactuals
                from the mutation step.

            Output:
                crossed_counterfacts (numpy.ndarray): the changed candidate
                counterfactuals.'''

        nfeats = len(counterfacts_list[0])

        random_indeces = np.random.binomial(1, self.Pc, len(counterfacts_list))

        mutation_indeces = [index for index, i in enumerate(random_indeces) if i]

        counterfacts_df = pd.DataFrame(counterfacts_list)

        while mutation_indeces:

            individual1 = mutation_indeces.pop(np.random.randint(0, len(mutation_indeces)))

            if len(mutation_indeces) > 0:
                individual2 = mutation_indeces.pop(np.random.randint(0, len(mutation_indeces)))

                mutation_features = np.random.randint(0, nfeats,
                                                      size=np.random.randint(1, nfeats))

                features1 = counterfacts_df.iloc[individual1, mutation_features]

                features2 = counterfacts_df.iloc[individual2, mutation_features]

                counterfacts_df.iloc[individual1, mutation_features] = features2

                counterfacts_df.iloc[individual2, mutation_features] = features1

        if return_df:
            return counterfacts_df

        return counterfacts_df.values.tolist()

    def fit(self,
            model,
            x=None,
            model_input=None,
            trained_with_columns=False,
            target_name = "target",
            model_type="torch",
            classification=False,
            target_lower=None,
            target_upper=None,
            generations=3,
            distance='automatic',
            constrained=True,
            class_specific=None,
            select_retain=1000,
            gen_retain=500,
            final_k=1,
            normalisation=None,
            fixed=None,
            verbose=False):
        '''Generate the counterfactuals for the defined dataset under the
        trained model.

        Arguments:
            Inputs:
                model (torch.nn.module, keras.model or sklearn.model): the trained model
                that will be used to check that original samples and their
                counterfactuals yield different predictions.

                x (pandas.DataFrame or numpy.ndarray): the referenced
                dataset, i.e. the samples for which creating counterfactuals.
                If no dataset is provided, the function assumes that the
                dataset has been previously provided to the class during
                instantiation (see above) and that it is therefore contained
                in the attribute 'tab_dataset'.

                model_input (torch.tensor, numpy.ndarray or pd.Dataframe): the dataset
                for which counterfactuals are generated, but having the form
                required by the trained model to generate predictions based
                on it. If nothing is provided, the model input will be automatically
                generated for each dataset's observation (following the torch
                argument in order to create the correct input).

                trained_with_columns (bool): whether the model was trained with
                data having column names (i.e. pandas dataframe) or not. Default
                is False.

                target_name (str): the name of the target variable in case

                model_type (str): the type of model used. Options are "torch" for PyTorch,
                          "tf" for TensorFlow/Keras, and "sklearn" for Scikit-learn.

                classification (bool): whether the task of the model is to
                classify (classification = True) or to perform regression.

                generations (int): the number of generations, i.e. how many
                times the algorithm will run over each data sample in order to
                generate the relative counterfactuals. In the original paper, the
                value of this parameter was found for each separate example via
                grid search. Computationally, increasing the number of generations
                linearly increases the time of execution.

                distance (str): the type of distance function to be used in
                comparing original samples and counterfactuals. The default
                is "automatic", meaning that the distance function will be guessed,
                based on the form of the input data. Other options are "L1" or "L2".

                constrained (bool): whether to constrain the values of each
                generated feature to be in the range of the observed values
                for that feature in the original dataset.

                class_specific (int): if classification is True, this option
                can be used to further specify that we want to generate
                counterfactuals just for samples yielding a particular prediction
                (whereas the relative integer is the index of the predicted class
                 according to the trained model). This is useful, e.g., if the
                analysis needs to be restricted on data yielding a specific
                prediction. Default is None, meaning that all data will be used
                no matter what prediction they yield.

                select_retain (int): hyperparameter specifying the (max) amount
                of counterfactuals to be retained after the selection step
                of the algorithm.

                gen_retain (int): hyperparameter specifying the (max) amount
                of counterfactuals to be retained at each generation.

                final_k (int): hyperparameter specifying the (max) number
                of counterfactuals to be kept for each data sample.

                normalisation (str) ["standard", "max_scaler"]: the
                normalisation technique used for the data at hand. Default
                is None, while "standard" and "max_scaler" techniques are
                the other possible options. According to the chosen value
                different random number generators are used. This option is
                useful to speed up counterfactuals' generation if the original
                data underwent some normalisation process or are in some
                specific range.

                fixed (list): a list of features to be kept fixed in counterfactual generation
                (i.e. all counterfactual will have the same value as the given sample for that
                feature). If no list is provided, then no feature will be kept fixed.

                verbose (bool): whether to print the generation status via
                a progression bar.

            Outputs
                None: the function does not output any value but it populates
                the result attribute of the instance with a list of tuples each
                containing the original data sample, the generated  counterfactual(s)
                for that data sample and their distance(s).
        '''

        # Reset per-run state so repeated fit() calls don't use stale values
        self.predictions = None
        self.constraints = None
        self.Population = None
        self.distance = None
        self.column_names = None

        if x is None:
            assert self.tab_dataset is not None, 'Either an input is passed into the function or the class needs to be instantiated with a dataset.'
            x = self.tab_dataset.copy()
        else:
            x = x.copy()

        if not classification:
            if (target_lower is not None) ^ (target_upper is not None):
                raise ValueError("Provide BOTH target_lower and target_upper or neither.")
            if target_lower is not None:
                if not isinstance(target_lower, (pd.Series, np.ndarray)) \
                        or not isinstance(target_upper, (pd.Series, np.ndarray)):
                    raise TypeError("target_lower/upper must be pd.Series or 1-D numpy arrays.")
                if len(target_lower) != len(x) or len(target_upper) != len(x):
                    raise ValueError("target_lower/upper must have the same length as x.")
                # make them quick to index later
                target_lower = np.asarray(target_lower)
                target_upper = np.asarray(target_upper)

        if self.constraints is None:
            self.set_constraints(x, fixed)
        if self.Population is None:
            self.set_population(x)
        if self.distance is None:
            self.set_distance(distance, x)
        if model_input is None:
            model_input = self.result_to_input(x, model_type=model_type)
        else:
            if trained_with_columns:
                self.column_names = model_input.columns

        if model_type == "torch" and hasattr(model, 'eval'):
            model.eval()

        if self.predictions is None:
            self.predictions = self.generate_prediction(model, model_input, model_type=model_type, classification=classification)
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)


        self.results = []
        if isinstance(x, pd.DataFrame):
            con, cat = self.get_con_cat_columns(x)
            has_cat = len(cat) > 0
            cat_ids = self.generate_cats_ids(x) if has_cat else None
        else:
            x = pd.DataFrame(x)

        if classification and class_specific is not None:
            x = x.iloc[self.predictions == class_specific]
            self.class_specific = class_specific

        tot_samples = range(x.shape[0])

        for i in tot_samples:
            #Get Instance to explain
            sample = x.iloc[i:i + 1, :].copy()
            counterfacts = []
            counterfacts_fit = {}

            for g in range(generations):
                generation, distances = self.generate_initial_candidates_tab(sample,
                                                                     normalisation,
                                                                     constrained,
                                                                     has_cat,
                                                                     cat_ids)

                selected_generation, _ = self.generate_counterfacts_list_dictionary(
                    counterfacts_list=generation,
                    distances=distances,
                    fitness_dict={},
                    retain_k=select_retain,
                    start=0)

                #Generate Counterfactual Candidates
                selected_generation = np.array(selected_generation) #Convert list of CEs to np.array
                mutated_generation = self.mutate(selected_generation) #Mutate CEs
                crossed_generation = self.crossover(mutated_generation, return_df=True) #Crossover CEs.
                #Convert CEs to the Model's input format
                gen_input = self.result_to_input(crossed_generation, model_type=model_type)
                # Get Counterfactual Predictions
                counterfactual_predictions = self.generate_prediction(model, gen_input, model_type, classification)
                if classification or target_lower is None:  # keep legacy behaviour
                    valid_mask = counterfactual_predictions != self.predictions[i]
                else:
                    lb, ub = target_lower[i], target_upper[i]  # row-specific bounds
                    # counterfactual_predictions is (n_candidates, 1) or (n_candidates,)
                    valid_mask = np.squeeze(counterfactual_predictions) >= lb
                    valid_mask &= np.squeeze(counterfactual_predictions) <= ub


                #final_generation = crossed_generation.loc[different_prediction]
                final_generation = crossed_generation.loc[valid_mask]

                if not final_generation.empty:  # Check if final_generation is not empty
                    final_distances = self.distance(sample, final_generation)[0]
                    final_generation = final_generation.copy()
                    final_generation['prediction_target'] = np.array(counterfactual_predictions)[valid_mask]

                    final_generation, counterfacts_fit = self.generate_counterfacts_list_dictionary(
                        counterfacts_list=final_generation.values.tolist(),
                        distances=final_distances,
                        fitness_dict=counterfacts_fit,
                        retain_k=gen_retain,
                        start=len(counterfacts_fit))

                    counterfacts.extend(final_generation)

            if counterfacts:
                counterfacts, fitness_dict = self.generate_counterfacts_list_dictionary(
                    counterfacts_list=counterfacts,
                    distances=list(counterfacts_fit.values()),
                    fitness_dict={},
                    retain_k=final_k,
                    start=0)
                sample[target_name] = self.predictions[i]
                # Split prediction_target (last element) from feature values
                prediction_targets = [cf[-1] for cf in counterfacts]
                counterfacts = [cf[:-1] for cf in counterfacts]
                self.results.append((sample, counterfacts, prediction_targets))

    def result_to_input(self, x, model_type="torch"):
        '''Function to transform the raw input to the required format for the ML model.

        Arguments:
            x (pandas.DataFrame or numpy.ndarray): The "raw" input to be transformed.
            model_type (str): The model type used. Options: "sklearn", "torch", "tf".

        Outputs:
            Transformed input as torch.tensor, numpy.ndarray, or pandas.DataFrame.
        '''

        # Ensure the model_type is valid
        if model_type not in ["sklearn", "torch", "tf"]:
            raise ValueError("model_type must be one of ['sklearn', 'torch', 'tf']")

        if isinstance(x, pd.DataFrame):
            x = x.copy()
            model_input = x

            # Apply transformations only if needed
            if model_type in ["torch", "tf"]:
                con, cat = self.get_con_cat_columns(x)
                if len(cat) > 0:
                    for feature in cat:
                        enc = LabelEncoder()
                        x[feature] = enc.fit_transform(x[feature])
                model_input = x.to_numpy()  # Convert DataFrame to NumPy array

        elif isinstance(x, np.ndarray):
            model_input = x  # NumPy array remains unchanged
        else:
            raise ValueError("The input x must be a pandas DataFrame or a numpy array")

        # Convert based on model type
        if model_type == "sklearn":
            if self.column_names is None:
                if isinstance(model_input, pd.DataFrame):
                    result = model_input.to_numpy()
                    return result
                else:
                    return model_input
            else:
                if isinstance(model_input, pd.DataFrame):
                    model_input = model_input.copy()
                    model_input.columns = self.column_names
                    return model_input
                else:
                    return pd.DataFrame(data=model_input, columns=self.column_names)
        elif model_type == "torch":
            torch = importlib.import_module("torch")
            return torch.tensor(model_input, dtype=torch.float32)  # Convert to PyTorch tensor
        elif model_type == "tf":
            return model_input  # For TensorFlow, keep as NumPy array