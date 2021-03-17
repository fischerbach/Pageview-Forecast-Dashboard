import pandas as pd
import numpy as np
class FES:
    @staticmethod
    def me(f,y):
        f = f.reset_index(drop=True).values.flatten()
        y = y.reset_index(drop=True).values.flatten()
        df = pd.DataFrame({'f_i':f, 'y_i': y})
        df['e'] = df['y_i'] - df['f_i']
        return np.mean(df['e'])
    
    @staticmethod
    def mse(f, y):
        f = f.reset_index(drop=True).values.flatten()
        y = y.reset_index(drop=True).values.flatten()
        df = pd.DataFrame({'f_i':f, 'y_i': y})
        df['e'] = np.square(df['y_i'] - df['f_i'])
        return np.mean(df['e'])

    @staticmethod
    def rmse(f, y):
        return np.sqrt(FES.mse(f,y))

    @staticmethod
    def mae(f, y):
        f = f.reset_index(drop=True).values.flatten()
        y = y.reset_index(drop=True).values.flatten()
        df = pd.DataFrame({'f_i':f, 'y_i': y})
        df['e'] = np.abs(df['y_i'] - df['f_i'])
        return np.mean(df['e'])

    @staticmethod
    def mpe(f, y):
        f = f.reset_index(drop=True).values.flatten()
        y = y.reset_index(drop=True).values.flatten()
        df = pd.DataFrame({'f_i':f, 'y_i': y})
        df['e'] = df['y_i'] - df['f_i']
        df['pe'] = 100*(df['e']/df['y_i'])
        return np.mean(df['pe'])

    @staticmethod
    def mape(f, y):
        f = f.reset_index(drop=True).values.flatten()
        y = y.reset_index(drop=True).values.flatten()
        df = pd.DataFrame({'f_i':f, 'y_i': y})
        df['e'] = df['y_i'] - df['f_i']
        df['ape'] = 100*np.abs(df['e']/df['y_i'])
        return np.mean(df['ape'])
    
    @staticmethod
    def all(f, y):
        return (
            FES.me(f,y),
            FES.mse(f,y),
            FES.rmse(f,y),
            FES.mae(f,y),
            FES.mpe(f,y),
            FES.mape(f,y),
            FES.u1(f,y),
            FES.u2(f,y)
        )

    @staticmethod
    def u1(f,y):
        y = y.reset_index(drop=True).values.flatten()
        f = f.reset_index(drop=True).values.flatten()
        df = pd.DataFrame({'f_i':f, 'y_i': y})
        df['(f_i - y_i)^2'] = np.square(df['f_i'] - df['y_i'])
        df['y_i^2'] = np.square(df['y_i'])
        df['f_i^2'] = np.square(df['f_i'])
        return (np.sqrt(np.mean(df['(f_i - y_i)^2'])))/(np.sqrt(np.mean(df['y_i^2']))+np.sqrt(np.mean(df['f_i^2'])))

    @staticmethod
    def u2(f,y):
        y = y.reset_index(drop=True).values.flatten()
        f = f.reset_index(drop=True).values.flatten()
        df = pd.DataFrame({'f_i+1':f, 'y_i+1': y})
        df['y_i'] = df['y_i+1'].shift(periods=1)
        df['numerator'] = np.square((df['f_i+1'] - df['y_i+1']) / df['y_i'])
        df['denominator'] = np.square((df['y_i+1'] - df['y_i']) / df['y_i'])
        df.dropna(inplace=True)
        return np.sqrt(np.sum(df['numerator'])/np.sum(df['denominator']))