# only one class to deal with all methods
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
class WrongModelName(Exception):pass
class SingleModel:
    def __init__(self,Q,T):
        self.Q =Q
        self.T =T
        self.q_max = max(Q)
        self.T_max = max(T)
        self.models_functions ={
            "ex":self.exposential,
            "hr":self.harmonic,
            "hp":self.hyperbolic
        }
    def normalize(self):
        """ Normalize the the input [ production and time ] for faster and better fitting"""
        self.T = self.T / max(self.T)
        self.Q = self.Q / max(self.Q)
    def RMSE(self,y , y_fit ):
        """Get the root mean squared error between the fit line and the data"""
        N = len(y)
        return np.sqrt(np.sum(y - y_fit) ** 2 / N)
        # De-normalize qi and di
    def hyperbolic(self,t, qi, di, b):
            return qi / (np.abs((1 + b * di * t)) ** (1 / b))

    def exposential(self,t, qi, di):
        return qi * np.exp(-di * t)

    def harmonic(self,t, qi, di):
        return qi / (1 + di * t)

    def fit(self,model):
        """ Takes the name of the model and return the paramters related
        model : str  [ ex -> exponential , hr -> harmonic , hp -> hyperbolic ]
        return:
            parameters [ qi , di , b , RMSE , Model_name ]
        """
        if model not in ("ex","hr","hp"):
            raise WrongModelName("The model name is worng choose between [ ex , hr , hp ]")
        self.normalize()
        params, pcov = curve_fit(self.models_functions[model], self.T, self.Q)
        if model =="ex":
            qi, di = params
            b = 0
        elif model == "hr":
            qi, di = params
            b = 1
        elif model =="hp":
            qi, di , b = params
        qfit_normalized = self.models_functions[model](self.T,*params)
        q_fitted = qfit_normalized * self.q_max
        RMSE = self.RMSE(self.Q, qfit_normalized)
        # De-normalize qi and di
        qi = qi * self.q_max
        di = di / self.T_max
        parameters = [ qi, di, b, RMSE, model ]
        return parameters, q_fitted

# ------------------------- start the whole arps model ------------------------------
class ARPS:
    """
    Takes the productions and date columns and provide ARP's paramters for the 3 models
    """
    def __init__(self, dataframe : pd.DataFrame , production_col : str , date_column :str):
        """
        :param dataframe: the pandas.DataFrame object the contain the data
        :param production_col: name of the production column
        :param date_column: name of the data column
        """
        self.q = production_col
        self.date = date_column
        self.df = dataframe[[date_column,production_col]]
        self.Q = 0
        self.T =0
        self.ex_params = None
        self.hp_params = None
        self.Hr_params = None

    def smooth(self, window_size : int , stds : int , trim : bool ):
        """Smooting the data using moving average method with removing the outliers.
        window_size : int -> the size of the window for the moving average
        stds : int -> number of standard deviations to remove data after [ for the outliers ]
        trim : bool -> whether to trim the data in which the production increase or not
        return :
          df : pd.DataFrame -> the data frame after smoothing
        """
        # smoothing using moving average
        self.df[self.q + '_rol_Av'] = self.df[self.q].rolling(window=window_size, center=True).mean()
        # identify the outliers and remove them
        self.df[self.q + '_rol_Std'] = self.df[self.q].rolling(window=window_size, center=True).std()
        self.df[self.q + '_is_Outlier'] = (abs(self.df[self.q] - self.df[self.q + '_rol_Av']) > ( stds * self.df[self.q+'_rol_Std']))
        result = self.df.drop(self.df[self.df[self.q + '_is_Outlier'] == True].index).reset_index(drop=True)
        # Remove rows where "_rol_Av" has NaNs
        result = result[result[self.q + '_rol_Av'].notna()]
        # remove the increasing part of the curve we only concern about the decline part
        if trim == True:
            # Trim initial buildup
            maxi = result[self.q + '_rol_Av'].max()
            maxi_index = (result[result[self.q + '_rol_Av'] == maxi].index.values)[0]
            result = result.iloc[maxi_index:, :].reset_index(drop=True)
        self.df = result
        return self.df

    def prepocess_date_col(self,frequency = "Daily"):
        """
        Convert the date column into number so that it can be used for fitting
        frequency : the frequency of the production data taken [ Daily , Monthly , Yearly ]
        """
        self.df[self.date] = pd.to_datetime(self.df[self.date])
        self.df["Time [{frequency}]"] = (self.df[self.date] - self.df[self.date].iloc[0])
        if frequency == "Daily":
            self.df["Time [{frequency}]"] = self.df["Time [{frequency}]"] / np.timedelta64(1, 'D')
            self.df["Time [{frequency}]"] = self.df["Time [{frequency}]"].astype(int)
        elif frequency == "Monthly":
            self.df["Time [{frequency}]"] = self.df["Time [{frequency}]"] / np.timedelta64(1, 'M')
            self.df["Time [{frequency}]"] = self.df["Time [{frequency}]"].astype(int)
        elif frequency == "Yearly":
            self.df["Time [{frequency}]"] = self.df["Time [{frequency}]"] / np.timedelta64(1, 'Y')
            self.df["Time [{frequency}]"] = self.df["Time [{frequency}]"].astype(int)
        self.T = self.df["Time [{frequency}]"]
        self.Q = self.df[self.q + '_rol_Av']

    def fit_exponential(self):
         """
         Fit the data only for the Exponential model
         return : parameters , values of the fitted line to draw
         """
         parameters , q_fitted = SingleModel(self.Q, self.T).fit("ex")
         self.ex_params = parameters
         return parameters , q_fitted

    def fit_hyperbolic(self):
        """
        Fit the data only for the Hyperbolic model
        return : parameters , values of the fitted line to draw
               """
        parameters, q_fitted = SingleModel(self.Q, self.T).fit("hp")
        self.hp_params = parameters
        return parameters , q_fitted

    def fit_harmonic(self):
        """
        Fit the data only for the Harmonic model
        return : parameters , values of the fitted line to draw
        """
        parameters, q_fitted = SingleModel(self.Q, self.T).fit("hr")
        self.Hr_params = parameters
        return parameters , q_fitted

    def fit_all_models(self):
        """
        Fit the data for the three model once [ exponential , harmonic , hyperbolic ]
        return :
         data_parameters : Dict -> a dictionary of all the information and parameters of each model [ qi, di , b , rmse ]
         Qs : pandas.DataFrame -> dataframe for these columns [ Time, originalSmoothed_Q , Exponential_fitted_Q , Hyperbolic_fiited_Q , Harmonic_fitted_Q ]
        """
        ex ,qex = self.fit_exponential()
        hp , qhp= self.fit_hyperbolic()
        hr , qhr = self.fit_harmonic()
        all_params = [ex, hp, hr]
        data_info = {
            "Model": [i[-1] for i in all_params],
            "Qi": [i[0] for i in all_params],
            "Di": [i[1] for i in all_params],
            "b": [i[2] for i in all_params],
            "Normalized RMSE": [i[3] for i in all_params],
        }
        Qs = pd.DataFrame({
            "Time": self.T,
            "Original_Smoothed": self.Q,
            "Exponential": qex,
            "Hyperbolic": qhp,
            "Harmonic": qhr
        })
        return data_info , Qs
