import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

sandp_df = pd.read_csv('data/^GSPC.csv', sep=',')
sandp_df['Date'] = pd.to_datetime(sandp_df['Date'])
sandp_df = sandp_df.set_index('Date')
# sandp_df = sandp_df.loc['2005-1-1':]

# [(start, end)]
# https://www.thebalance.com/the-history-of-recessions-in-the-united-states-3306011
# https://en.wikipedia.org/wiki/List_of_economic_crises
recessions = [
    ('1953-07-01', '1953-05-01'),  # Post Korean War 6.1% unemployment
    ('1957-08-01', '1958-04-01'),  # 7.5% unemployment
    ('1960-04-01', '1961-02-01'),  # Richard Nixon 7.1% unemployment
    ('1969-12-01', '1970-11-01'),  # 6.1% unemployment (mild)
    ('1973-11-01', '1975-03-01'),  # OPEC - 9% unemployment
    ('1980-01-01', '1980-07-01'),  #
    ('1981-07-01', '1982-11-01'),  # 10% unemployment
    ('1990-07-01', '1991-03-01'),  # 7.8% unemployment
    ('2001-03-01', '2001-11-01'),  # Y2K 6.3% unemployment
    ('2007-12-01', '2009-05-01'),  # 2008
]


def plot_recessions():
    # plt.axvline(x=pd.to_datetime(f'20080915', format='%Y%m%d'), linestyle='dashed', color='grey')
    plt.axvspan(
        pd.to_datetime('1987-10-18'),
        pd.to_datetime('1987-10-20'),
        alpha=0.1,
        color='red')
    for r in recessions:
        plt.axvspan(
            pd.to_datetime(r[0]), pd.to_datetime(r[1]), alpha=0.1, color='gray')


with PdfPages('graphs.pdf') as pdf:

    plt.figure(figsize=(20, 10))
    plt.title('Close')
    plt.plot(sandp_df['Close'])
    plot_recessions()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(sandp_df['Close'])
    plt.yscale('log')
    plt.title('Close Log')
    plot_recessions()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(sandp_df['Close'].pct_change())
    plt.title('close_pct_change')
    plot_recessions()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(sandp_df['Volume'])
    plt.title('Volume')
    plot_recessions()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(sandp_df['Volume'].pct_change())
    plt.title('volume_pct_change')
    plot_recessions()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(sandp_df['Close'].rolling(window=7).var())
    plt.title('close_var_7')
    plot_recessions()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(sandp_df['Volume'].pct_change().rolling(window=7).var())
    plt.title('volume_pct_change_var7')
    plot_recessions()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(sandp_df['Close'].pct_change().rolling(window=7).var())
    plt.title('close_pct_change_var7')
    plot_recessions()
    pdf.savefig()
    plt.close()

    # interest rate
    interest_df = pd.read_csv('data/FEDFUNDS.csv', sep=',')
    interest_df['DATE'] = pd.to_datetime(interest_df['DATE'])
    interest_df = interest_df.set_index('DATE')
    interest_rates = interest_df['FEDFUNDS']

    plt.figure(figsize=(20, 10))
    plt.plot(interest_rates)
    plt.title('interest_rates')
    plot_recessions()
    pdf.savefig()
    plt.close()

    # interest rate derivative
    interest_rates_derivative = interest_rates.diff()

    plt.figure(figsize=(20, 10))
    plt.plot(interest_rates_derivative)
    plt.title('interest_rates_derivative')
    plot_recessions()
    pdf.savefig()
    plt.close()

    # consumer spending
    spending_df = pd.read_csv('data/personal_expenditure_no_food_energy.csv')
    spending_df['DATE'] = pd.to_datetime(spending_df['DATE'])
    spending_df = spending_df.set_index('DATE')
    spending = spending_df['DPCCRC1M027SBEA']

    plt.figure(figsize=(20, 10))
    plt.plot(spending)
    plt.title('consumer_spending_no_food_no_energy')
    plot_recessions()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(spending)
    plt.yscale('log')
    plt.title('consumer_spending_no_food_no_energy log')
    plot_recessions()
    pdf.savefig()
    plt.close()

    # inflation
    df = pd.read_csv('data/inflation.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.set_index('DATE')
    inflation = df['FPCPITOTLZGUSA']

    plt.figure(figsize=(20, 10))
    plt.plot(inflation)
    plt.title('inflation')
    plot_recessions()
    pdf.savefig()
    plt.close()

    # unemployment
    df = pd.read_csv('data/unemployment_rate.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.set_index('DATE')
    unemployment = df['UNRATE']

    plt.figure(figsize=(20, 10))
    plt.plot(unemployment)
    plt.title('unemployment_rate')
    plot_recessions()
    pdf.savefig()
    plt.close()

    # US government debt
    df = pd.read_csv('data/government_debt.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.set_index('DATE')
    federal_debt = df['FYGFD']

    plt.figure(figsize=(20, 10))
    plt.plot(federal_debt)
    plt.title('federal_debt')
    plot_recessions()
    pdf.savefig()
    plt.close()

    # GDP
    df = pd.read_csv('data/GDP.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.set_index('DATE')

    plt.figure(figsize=(20, 10))
    plt.plot(df['GDP'])
    plt.title('GDP')
    plot_recessions()
    pdf.savefig()
    plt.close()
