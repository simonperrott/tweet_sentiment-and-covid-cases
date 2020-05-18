import pandas as pd
import dateutil.parser


def load_covid_cases():
    covid_cases = pd.read_csv('data/covid/time_series_covid19_confirmed_global.csv')
    return prepare_df(covid_cases)


def load_covid_deaths():
    covid_deaths = pd.read_csv('data/covid/time_series_covid19_deaths_global.csv')
    return prepare_df(covid_deaths)


def prepare_df(df: pd.DataFrame):
    df.rename(columns={'Country/Region': 'Country'}, inplace=True)
    df = df[df['Country'].isin(['Ireland', 'US', 'United Kingdom']) & df['Province/State'].isnull()]
    df = df.drop(['Province/State', 'Lat', 'Long'], 1)
    df.set_index('Country', inplace=True)
    df = df.transpose()
    df_diff = df.diff(axis=0)
    df_diff['day'] = [dateutil.parser.parse(d, dayfirst=False).strftime("%Y-%m-%d") for d in df_diff.index]
    df_diff.fillna(0, inplace=True)
    df_diff.set_index('day', inplace=True)
    return df_diff
