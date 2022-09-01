# Import statements
# Declare Imports
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import operator
from textwrap import wrap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Set a variable for current notebook's path for various loading/saving mechanisms
nb_path = os.getcwd()

'''Data import functions'''

def load_census_csv(table_list, statistical_area_code='SA3', year=2021):
    '''
    Navigates the file structure to import the relevant files for specified data tables at a defined statistical area level
    
    INPUTS
    table_list: LIST of STRING objects - the ABS Census Datapack table to draw information from (G01-G59)
    statistical_area_code: STRING - the ABS statistical area level of detail required (SA1-SA3)
    year: INT - the reference year for the data you want to bring in, currently only works for 2016 and 2021
    
    OUTPUTS
    A pandas dataframe
    '''
    statistical_area_code = statistical_area_code.upper()
    year_str = str(year)
    if year == 2021:
        geog_level = 'AUST'
    else:
        geog_level = 'AUS'

    df_csv_load = pd.DataFrame()
    for index, table in enumerate(table_list):
        
        if index==0:
            df_csv_load = pd.read_csv('{}\Data\{}\AUST\\{}Census_{}_{}_{}.csv'.format(nb_path,
                                                                                statistical_area_code,
                                                                                year_str,
                                                                                table,
                                                                                geog_level,
                                                                                statistical_area_code
                                                                               ),
                                       engine='python')
        else:
            temp_df = pd.read_csv('{}\Data\{}\AUST\\{}Census_{}_{}_{}.csv'.format(nb_path,
                                                                                statistical_area_code,
                                                                                year_str,
                                                                                table,
                                                                                geog_level,
                                                                                statistical_area_code
                                                                               ),
                                       engine='python')
            merge_col = df_csv_load.columns[0]
            df_csv_load = pd.merge(df_csv_load, temp_df, on=merge_col)
    
    return df_csv_load



def refine_measure_name(table_namer, string_item, category_item, category_list):
    '''Simple function for generating measure names based on custom metadata information on ABS measures'''
    position_list = []
    for i, j in enumerate(category_item.split("|")):
        if j in category_list:
            position_list.append(i)
    return table_namer + '|' + '_'.join([string_item.split("|")[i] for i in position_list])


def load_table_refined(table_ref, category_list, statistical_area_code='SA3', drop_zero_area=True):
    '''
    Function for loading ABS census data tables, and refining/aggregating by a set of defined categories
    (e.g. age, sex, occupation, English proficiency, etc.) where available.
    
    INPUTS
    table_ref: STRING - the ABS Census Datapack table to draw information from (G01-G59)
    category_list: LIST of STRING objects - Cetegorical informatio to slice/aggregate information from (e.g. Age)
    statistical_area_code: STRING - the ABS statistical area level of detail required (SA1-SA3)
    drop_zero_area: BOOLEAN - an option to remove "non-geographical" area data points such as "no fixed address" or "migratory"
    '''
    df_meta = pd.read_csv('{}\Data\Metadata\Metadata_2016_refined.csv'.format(os.getcwd()))
    index_reference = 'Area_index'
    
    # slice meta based on table
    meta_df_select = df_meta[df_meta['Profile table'].str.contains(table_ref)].copy()
    
    # for category in filter_cats, slice based on category >0
    for cat in category_list:
        # First, check if there *are* any instances of the given category
        try:
            if meta_df_select[cat].sum() > 0:
                # If so, apply the filter
                meta_df_select = meta_df_select[meta_df_select[cat]>0]
            else:
                pass # If not, don't apply (otherwise you will end up with no selections)
        except:
            pass
        
    # select rows with lowest value in "Number of Classes Excl Total" field
    min_fields = meta_df_select['Number of Classes Excl Total'].min()
    meta_df_select = meta_df_select[meta_df_select['Number of Classes Excl Total'] == min_fields]
    
    # Select the table file(s) to import
    import_table_list = meta_df_select['DataPack file'].unique()
    
    # Import the SA data tables
    df_data = load_census_csv(import_table_list, statistical_area_code.upper())
    
    # Select only columns included in the meta-sliced table above
    df_data.set_index(df_data.columns[0], inplace=True)
    refined_columns = meta_df_select.Short.tolist()
    df_data = df_data[refined_columns]
    
    # aggregate data by:
    # transposing the dataframe
    df_data_t = df_data.T.reset_index()
    df_data_t.rename(columns={ df_data_t.columns[0]: 'Short' }, inplace = True)
    # merging with the refined meta_df to give table name, "Measures" and "Categories" fields
    meta_merge_ref = meta_df_select[['Short','Table name','Measures','Categories']]
    df_data_t = df_data_t.merge(meta_merge_ref, on='Short')
    
    # from the "Categories" field, you should be able to split an individual entry by the "|" character
    # to give the index of the measure you are interested in grouping by
    # create a new column based on splitting the "Measure" field and selecting the value of this index/indices
    # Merge above with the table name to form "[Table_Name]|[groupby_value]" to have a good naming convention
    # eg "Method_of_Travel_to_Work_by_Sex|Three_methods_Females"
    df_data_t[index_reference] = df_data_t.apply(lambda x: refine_measure_name(x['Table name'], 
                                                                               x['Measures'], 
                                                                               x['Categories'], 
                                                                               category_list), axis=1)
    
    # then groupby this new column 
    # then transpose again and either create the base data_df for future merges or merge with the already existing data_df
    df_data_t = df_data_t.drop(['Short','Table name','Measures','Categories'], axis=1)
    df_data_t = df_data_t.groupby([index_reference]).sum()
    df_data_t = df_data_t.T
    
    if drop_zero_area:
        df_zero_area = pd.read_csv('{}\Data\Metadata\Zero_Area_Territories.csv'.format(os.getcwd()))
        zero_indicies = set(df_zero_area['AGSS_Code_2016'].tolist())
        zero_indicies_drop = set(df_data_t.index.values).intersection(zero_indicies)
        df_data_t = df_data_t.drop(zero_indicies_drop, axis=0)
    
    return df_data_t


def load_tables_specify_cats(table_list, category_list, statistical_area_code='SA3'):
    '''
    Function for loading ABS census data tables, and refining/aggregating by a set of defined categories
    (e.g. age, sex, occupation, English proficiency, etc.) where available.
    
    INPUTS
    table_list: LIST of STRING objects - list of the ABS Census Datapack tables to draw information from (G01-G59)
    category_list: LIST of STRING objects - Cetegorical information to slice/aggregate information from (e.g. Age)
    statistical_area_code: STRING - the ABS statistical area level of detail required (SA1-SA3)
    
    OUTPUTS
    A pandas dataframe
    '''
    for index, table in enumerate(table_list):
        if index==0:
            df = load_table_refined(table, category_list, statistical_area_code)
            df.reset_index(inplace=True)
        else:
            temp_df = load_table_refined(table, category_list, statistical_area_code)
            temp_df.reset_index(inplace=True)
            merge_col = df.columns[0]
            df = pd.merge(df, temp_df, on=merge_col)
    
    df.set_index(df.columns[0], inplace=True)
    
    return df



def build_model(verbosity = 3):
    ''' 
    Builds a Gridsearch object for use in supervised learning modelling.
    Imputes for missing values and build a model to complete a quick Gridsearch over RandomForestRegressor key parameters.
    
    INPUTS
    None
    
    OUTPUTS
    cv - An SKLearn Gridsearch object for a pipeline that includes Median imputation and RandomForestRegressor model.
    '''
    pipeline_model = Pipeline([
        ('impute', SimpleImputer(missing_values=np.nan, strategy='median')),
        ('clf', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=100))
    ])
    # specify parameters for grid search
    parameters = {'clf__n_estimators':[20,40], # this used to start at 10 and go to 80 but was a huge timesuck and not improving performance
              'clf__max_depth':[16,32,64], # this used to go to 128 but had no impact on performance
              #'clf__min_samples_leaf':[1,2,4] This wasn't really having an impact on performance
             }

    # create grid search object
    scorer = make_scorer(r2_score)
    cv = GridSearchCV(pipeline_model, param_grid=parameters, scoring=scorer, verbose = verbosity, cv=3)

    return cv

def WFH_create_Xy(stat_a_level, load_tables, load_features):
    '''
    A function which compiles a set of background information from defined ABS census tables and 
    creates input and output vectors to allow model training for the  "Work from home participation 
    rate" in a given region. Cleans the data for outliers (defined as >3 standard deviations from the mean) in the
    WFH participation rate, and scales all data by dividing by the "Total Population" feature for each region.
    
    INPUTS
    stat_a_level - String. The statistical area level of information the data should be drawn from (SA1-3)
    load_tables - List of Strings. A list of ABS census datapack tables to draw data from (G01-59)
    load_features - List of Strings. A list of population characteristics to use in analysis (Age, Sex, labor force status, etc.)
    
    OUTPUTS
    X - pandas DataFrame - a dataframe of features from the census datapacks tables, normalised by dividing each feature by 
                            the population attributable to the region
    y - pandas series - the Work from Home Participation Rate by region
    
    '''
    # Load table 59 (the one with commute mechanism) and have a quick look at the distribution of WFH by sex
    response_vector = 'WFH_Participation'
    df_travel = load_table_refined('G59', ['Number of Commuting Methods'], statistical_area_code=stat_a_level)
    cols_to_delete = [x for x in df_travel.columns if 'Worked_at_home' not in x]
    df_travel.drop(cols_to_delete,axis=1, inplace=True)

    df_pop = load_census_csv(['G01'], statistical_area_code=stat_a_level)
    df_pop.set_index(df_pop.columns[0], inplace=True)
    df_pop = df_pop.drop([x for x in df_pop.columns if 'Tot_P_P' not in x], axis=1)
    df_travel = df_travel.merge(df_pop, left_index=True, right_index=True)
    
    # Create new "Work From Home Participation Rate" vector to ensure consistency across regions
    # Base this off population who worked from home divided by total population in the region
    df_travel[response_vector] = (df_travel['Method of Travel to Work by Sex|Worked_at_home']/
                                  df_travel['Tot_P_P'])
    # Drop the original absolute values column
    df_travel = df_travel.drop(['Method of Travel to Work by Sex|Worked_at_home'], axis=1)
    
    # load input vectors
    input_vectors = load_tables_specify_cats(load_tables, load_features, statistical_area_code=stat_a_level)
    
    # Remove duplicate column values
    input_vectors = input_vectors.T.drop_duplicates().T

    # Bring in total population field and scale all the values by this item
    input_vectors = input_vectors.merge(df_pop, left_index=True, right_index=True)

    # convert input features to numeric
    cols = input_vectors.columns
    input_vectors[cols] = input_vectors[cols].apply(pd.to_numeric, errors='coerce')

    # Drop rows with zero population
    input_vectors = input_vectors.dropna(subset=['Tot_P_P'])
    input_vectors = input_vectors[input_vectors['Tot_P_P'] > 0]

    # Scale all factors by total region population
    for cols in input_vectors.columns:
        if 'Tot_P_P' not in cols:
            input_vectors[cols] = input_vectors[cols]/input_vectors['Tot_P_P']

    # merge and drop na values from the response vector
    df_travel = df_travel.merge(input_vectors, how='left', left_index=True, right_index=True)
    df_travel = df_travel.dropna(subset=[response_vector])

    df_travel = df_travel.drop([x for x in df_travel.columns if 'Tot_P_P' in x], axis=1)

    # drop outliers based on the WFHPR column
    # only use an upper bound for outlier detection in this case, based on 3-sigma variation 
    # had previously chosen to remove columns based on IQR formula, but given the skew in the data this was not effective
    #drop_cutoff = (((df_travel[response_vector].quantile(0.75)-df_travel[response_vector].quantile(0.25))*1.5)
    #               +df_travel[response_vector].quantile(0.75))
    drop_cutoff = df_travel[response_vector].mean() + (3* df_travel[response_vector].std())
    df_travel = df_travel[df_travel[response_vector] <= drop_cutoff]
    
    # Remove duplicate column values
    df_travel = df_travel.T.drop_duplicates().T

    # Create X & y
    X = df_travel.drop(response_vector, axis=1)
    y = df_travel[response_vector]

    # Get the estimator
    return X, y


def model_WFH(stat_a_level, load_tables, load_features):
    '''
    A function which compiles a set of background information from defined ABS census tables and trains a 
    Random Forest Regression model (including cleaning and gridsearch functions) to predict the "Work from home
    participation rate" in a given region.
    
    INPUTS
    stat_a_level - String. The statistical area level of information the data should be drawn from (SA1-3)
    load_tables - List of Strings. A list of ABS census datapack tables to draw data from (G01-59)
    load_features - List of Strings. A list of population characteristics to use in analysis (Age, Sex, labor force status, etc.)
    
    OUTPUTS
    grid_fit.best_estimator_ - SKLearn Pipeline object. The best grid-fit model in training the data.
    X_train - Pandas dataframe. The training dataset used in fitting the model.
    X_test - Pandas dataframe. A testing dataset for use in analysing model performance.
    y_train - Pandas dataframe. The training dataset for the response vector (WFH participation)
                used in fitting the model.
    y_test - Pandas dataframe. A testing dataset for the response vector (WFH participation)
                for use in analysing model performance.
    
    '''
    
    # Create X & y
    X, y = WFH_create_Xy(stat_a_level, load_tables, load_features)

    # Split the 'features' and 'response' vectors into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # build a model using all the above inputs
    grid_obj = build_model()

    # TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
    grid_fit = grid_obj.fit(X_train, y_train)

    # Get the estimator
    return grid_fit.best_estimator_, X_train, X_test, y_train, y_test


def sort_series_abs(S):
    '''Takes a pandas Series object and returns the series sorted by absolute value'''
    temp_df = pd.DataFrame(S)
    temp_df['abs'] = temp_df.iloc[:,0].abs()
    temp_df.sort_values('abs', ascending = False, inplace = True)
    return temp_df.iloc[:,0]


'''Plotting functions'''

def feature_plot_h(model, X_train, n_features):
    '''
    Takes a trained model and outputs a horizontal bar chart showing the "importance" of the
    most impactful n features.
    
    INPUTS
    model = Trained model in sklearn with  variable ".feature_importances_". Trained supervised learning model.
    X_train = Pandas Dataframe object. Feature set the training was completed using.
    n_features = Int. Top n features you would like to plot.
    '''
    importances = model.feature_importances_
    # Identify the n most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:n_features]]
    values = importances[indices][:n_features]
    
    columns = [ '\n'.join(wrap(c, 30)).replace("_", " ") for c in columns ]
    
    # Create the plot
    fig = plt.figure(figsize = (9,n_features))
    plt.title("Normalized Weights for {} Most Predictive Features".format(n_features), fontsize = 16)
    plt.barh(np.arange(n_features), values, height = 0.4, align="center", color = '#00A000', 
          label = "Feature Weight")
    plt.barh(np.arange(n_features) - 0.3, np.cumsum(values), height = 0.2, align = "center", color = '#00A0A0', 
          label = "Cumulative Feature Weight")
    plt.yticks(np.arange(n_features), columns)
    plt.xlabel("Weight", fontsize = 12)
    
    plt.legend(loc = 'upper right')
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()  

    
def feature_impact_plot(model, X_train, n_features, y_label, pipeline=None, consistent_X=False, share_y = True):
    '''
    Takes a trained model and training dataset and synthesises the impacts of the top n features
    to show their relationship to the response vector (i.e. how a change in the feature changes
    the prediction). Returns n plots showing the variance for min, max, median, 1Q and 3Q.
    
    INPUTS
    model = Trained model in sklearn with  variable ".feature_importances_". Trained supervised learning model.
    X_train = Pandas Dataframe object. Feature set the training was completed using.
    n_features = Int. Top n features you would like to plot.
    y_label = String. Description of response variable for axis labelling.
    pipeline = Optional, sklearn pipeline object. If the sklearn model was compiled using a pipeline, 
                this object needs to be specified separately.
    consistent_X = Optional Boolean. Input True to specify if the range of simulated feature ranges should be consistent.
                        this makes the impact charts easier to compare between features where they have consistent 
                        units of meaure (e.g. share of population).
    share_y = Optional Boolean. Have a shared y-axis for all sub plots. Very good for comparing impacts of changes to 
                individual features, but can make distinguishing the impacts of a feature difficult if there is 
                significant variance in other plots.
    
    OUTPUT
    Plot with n subplots showing the variance for min, max, median, 1Q and 3Q as a result of simulated outcomes.
    '''
    # Display the n most important features
    indices = np.argsort(model.feature_importances_)[::-1]
    columns = X_train.columns.values[indices[:n_features]]
    
    sim_var = [[]]
    
    if pipeline == None:
        pipeline=model
    
    # get statistical descriptors
    X_descriptor = X_train[columns].describe()
    
    # Shorten the simulated outcomes for efficiency
    sample_length = min(X_train.shape[0], 1000)
    
    X_train = X_train.sample(sample_length, random_state=42)
    
    if consistent_X:
        value_dispersion = X_descriptor.loc['std',:].max()
    
    for col in columns:
        base_pred = pipeline.predict(X_train)
        # Add percentiles of base predictions to a df for use in reporting
        base_percentiles = [np.percentile(base_pred, pc) for pc in range(0,101,25)]

        # Create new predictions based on tweaking the parameter
        # copy X, resetting values to align to the base information through different iterations
        df_copy = X_train.copy() 
        
        if consistent_X != True:
            value_dispersion = X_descriptor.loc['std',col] * 1.5

        for val in np.arange(-value_dispersion, value_dispersion, value_dispersion/50):
            df_copy[col] = X_train[col] + val
            # Add new predictions based on changed database
            predictions = pipeline.predict(df_copy)
            
            # Add percentiles of these predictions to a df for use in reporting
            percentiles = [np.percentile(predictions, pc) for pc in range(0,101,25)]
            
            # Add variances between percentiles of these predictions and the base prediction to a df for use in reporting
            percentiles = list(map(operator.sub, percentiles, base_percentiles))
            percentiles = list(map(operator.truediv, percentiles, base_percentiles))
            sim_var.append([val, col] + percentiles)

    # Create a dataframe based off the arrays created above
    df_predictions = pd.DataFrame(sim_var,columns = ['Value','Feature']+[0,25,50,75,100])
    
    # Create a subplot object based on the number of features
    num_cols = 2
    subplot_rows = int(n_features/num_cols) + int(n_features%num_cols)
    fig, axs = plt.subplots(nrows = subplot_rows, ncols = num_cols, sharey = share_y, figsize=(15,5*subplot_rows))

    nlines = 1

    # Plot the feature variance impacts
    for i in range(axs.shape[0]*axs.shape[1]):
        if i < len(columns):
            # Cycle through each plot object in the axs array and plot the appropriate lines
            ax_row = int(i/num_cols)
            ax_column = int(i%num_cols)
            
            axs[ax_row, ax_column].plot(df_predictions[df_predictions['Feature'] == columns[i]]['Value'],
                                        df_predictions[df_predictions['Feature'] == columns[i]][25],
                                        label = '25th perc')
            axs[ax_row, ax_column].plot(df_predictions[df_predictions['Feature'] == columns[i]]['Value'],
                                        df_predictions[df_predictions['Feature'] == columns[i]][50],
                                        label = 'Median')
            axs[ax_row, ax_column].plot(df_predictions[df_predictions['Feature'] == columns[i]]['Value'],
                                        df_predictions[df_predictions['Feature'] == columns[i]][75],
                                        label = '75th perc')
            
            axs[ax_row, ax_column].set_title("\n".join(wrap(columns[i], int(100/num_cols))))
            axs[ax_row, ax_column].legend()
            # Create spacing between charts if chart titles happen to be really long.
            nlines = max(nlines, axs[ax_row, ax_column].get_title().count('\n'))

            axs[ax_row, ax_column].set_xlabel('Simulated +/- change to feature'.format(y_label))
            
            # Format the y-axis as %
            if ax_column == 0 or share_y == False:
                vals = axs[ax_row, ax_column].get_yticks()
                axs[ax_row, ax_column].set_yticklabels(['{:,.2%}'.format(x) for x in vals])
                axs[ax_row, ax_column].set_ylabel('% change to {}'.format(y_label))
        
        # If there is a "spare" plot, hide the axis so it simply shows as an empty space
        else:
            axs[int(i/num_cols),int(i%num_cols)].axis('off')
    
    # Apply spacing between subplots in case of very big headers
    fig.subplots_adjust(hspace=0.5*nlines)
    
    # Return the plot
    plt.tight_layout()    
    plt.show()
    
def model_analyse_pred(X_test, y_test, model):
    ''' 
    A function for outputting a fitting chart, showing the pairing of prediction vs actual in a modelled test set.
    
    INPUTS
    X_test - Pandas dataframe. The test set of characteristics to feed into a prediction model.
    y_test - Pandas Series. The test set of responses to compare to predictions made in the model.
    model - SKLearn fitted supervised learning model.
    
    OUTPUTS
    A plot showing the relationship between prediction and actual sets.
    '''
    preds = model.predict(X_test)
    model_analyse(y_test, preds)

def model_analyse(y_test, y_pred):
    ''' 
    A function for outputting a fitting chart, showing the pairing of prediction vs actual in a modelled test set.
    
    INPUTS
    X_test - Pandas dataframe. The test set of characteristics to feed into a prediction model.
    y_test - Pandas Series. The test set of responses to compare to predictions made in the model.
    
    OUTPUTS
    A plot showing the relationship between prediction and actual sets.
    '''
    lineStart = min(y_pred.min(), y_test.min())  
    lineEnd = max(y_pred.max()*1.2, y_test.max()*1.2)

    plt.figure()
    plt.scatter(y_pred, y_test, color = 'k', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'r')
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.xlabel('Predictions')
    plt.ylabel('Actuals')
    plt.text(y_test.max()/5, y_test.max()*1.1, 'R2 Score: {:.3f}'.format(r2_score(y_test, y_pred)))
    plt.show()
    
def top_n_features(model, X_train, n_features):
    '''
    Takes a trained model and training dataset and returns the top n features by feature importance
    
    INPUTS
    model = Trained model in sklearn with  variable ".feature_importances_". Trained supervised learning model.
    X_train = Pandas Dataframe object. Feature set the training was completed using.
    n_features = Int. Top n features you would like to plot.
    '''
    # Display the n most important features
    indices = np.argsort(model.feature_importances_)[::-1]
    columns = X_train.columns.values[indices[:n_features]].tolist()
    
    return columns