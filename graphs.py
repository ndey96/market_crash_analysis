import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.rc('axes', labelsize=25)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

sandp_df = pd.read_csv('data/^GSPC.csv', sep=',')
sandp_df['Date'] = pd.to_datetime(sandp_df['Date'])
sandp_df = sandp_df.set_index('Date')
# sandp_df = sandp_df.loc['2005-1-1':]

# [(start, end)]
# https://www.thebalance.com/the-history-of-recessions-in-the-united-states-3306011
# https://en.wikipedia.org/wiki/List_of_economic_crises
recessions = [
    ('1953-07-01', '1954-05-01'),  # Post Korean War 6.1% unemployment
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
pred_style = {
    'alpha': 0.1,
    # 'fill': False,
    'facecolor': 'orange',
    'edgecolor': 'black',
    # 'hatch': '\\\\',
}


def plot_recessions():
    # plt.axvline(x=pd.to_datetime(f'20080915', format='%Y%m%d'), linestyle='dashed', color='grey')
    # plt.axvspan(
    #     pd.to_datetime('1987-10-18'),
    #     pd.to_datetime('1987-10-20'),
    #     alpha=0.1,
    #     color='red')
    # plt.axvline(
    #     pd.to_datetime('1987-10-18'),
    #     linestyle='dashed',
    #     color='gray',
    #     label='1987 Flash Crash')
    label = 'Recession'
    for r in recessions:
        plt.axvspan(
            pd.to_datetime(r[0]),
            pd.to_datetime(r[1]),
            alpha=0.1,
            facecolor='blue',
            edgecolor='black',
            hatch='//',
            label=label)
        label = None


def plot_low_high_thresh_preds(begin_thresh, end_thresh, signal):
    plt.axhline(
        begin_thresh,
        color='orange',
        label='Prediction Thresholds',
        linestyle='dashed')
    plt.axhline(end_thresh, color='orange', linestyle='dashed')
    predicted_recessions = []
    predicted_start = None
    for date, val in signal.iteritems():
        if val < begin_thresh:
            predicted_start = date
        elif (val > end_thresh) and (predicted_start != None):
            if (date - predicted_start).days >= 90:
                predicted_recessions.append((predicted_start, date))
            predicted_start = None
    label = 'Predicted Recession'
    for pred_r in predicted_recessions:
        plt.axvspan(pred_r[0], pred_r[1], label=label, **pred_style)
        label = None


def plot_low_high_cole_thresh_preds(begin_thresh, end_thresh, signal):
    plt.axhline(
        begin_thresh,
        color='orange',
        label='Prediction Thresholds',
        linestyle='dashed')
    plt.axhline(end_thresh, color='orange', linestyle='dashed')
    predicted_recessions = []
    predicted_start = None
    predicted_end = None
    crossed_high_thresh = False
    for date, val in signal.iteritems():
        if (val < begin_thresh) and (predicted_start !=
                                     None) and crossed_high_thresh:
            predicted_recessions.append((predicted_start, date))
            crossed_high_thresh = False
            predicted_start = None
        elif (val < begin_thresh):
            predicted_start = date
        elif (val >= end_thresh) and (predicted_start != None):
            crossed_high_thresh = True

    label = 'Predicted Recession'
    for pred_r in predicted_recessions:
        plt.axvspan(pred_r[0], pred_r[1], label=label, **pred_style)
        label = None


def plot_low_high_cole_thresh_preds2(begin_thresh, end_thresh, signal):
    plt.axhline(
        begin_thresh,
        color='orange',
        label='Prediction Thresholds',
        linestyle='dashed')
    plt.axhline(end_thresh, color='orange', linestyle='dashed')
    predicted_recessions = []
    predicted_start = None
    predicted_end = None
    crossed_high_thresh = False
    for date, val in signal.iteritems():
        if (val > begin_thresh) and (predicted_start !=
                                     None) and crossed_high_thresh:
            predicted_recessions.append((predicted_start, date))
            crossed_high_thresh = False
            predicted_start = None
        elif (val > begin_thresh):
            predicted_start = date
        elif (val <= end_thresh) and (predicted_start != None):
            crossed_high_thresh = True

    label = 'Predicted Recession'
    for pred_r in predicted_recessions:
        plt.axvspan(pred_r[0], pred_r[1], label=label, **pred_style)
        label = None


with PdfPages('graphs.pdf') as pdf:

    plt.figure(figsize=(20, 10))
    # plt.title('Close')
    plt.plot(sandp_df['Close'], label='')
    plot_recessions()
    plt.xlabel('Year')
    plt.ylabel('S&P 500 Close')
    plt.legend()
    pdf.savefig()
    plt.close()

    # plt.figure(figsize=(20, 10))
    # plt.plot(sandp_df['Close'])
    # plt.yscale('log')
    # plt.title('Close Log')
    # plot_recessions()
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(sandp_df['Close'].pct_change(), label='')
    # plt.title('close_pct_change')
    plot_recessions()
    plt.xlabel('Year')
    plt.ylabel('% Change of S&P 500 Close')
    plt.legend()
    pdf.savefig()
    plt.close()

    # plt.figure(figsize=(20, 10))
    # plt.plot(sandp_df['Close'].pct_change().rolling(
    #     closed='right', window=30).mean().pct_change())
    # plt.title('close_pct_change')
    # plot_recessions()
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    # plt.figure(figsize=(20, 10))
    # plt.plot(sandp_df['Volume'])
    # plt.title('Volume')
    # plot_recessions()
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    # plt.figure(figsize=(20, 10))
    # plt.plot(sandp_df['Volume'].pct_change())
    # plt.title('volume_pct_change')
    # plot_recessions()
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    # plt.figure(figsize=(20, 10))
    # plt.plot(sandp_df['Close'].rolling(closed='right', window=7).var())
    # plt.title('close_var_7')
    # plot_recessions()
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    # plt.figure(figsize=(20, 10))
    # plt.plot(sandp_df['Volume'].pct_change().rolling(closed='right', window=7).var())
    # plt.title('volume_pct_change_var7')
    # plot_recessions()
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    # plt.figure(figsize=(20, 10))
    # plt.plot(sandp_df['Close'].pct_change().rolling(closed='right',
    #                                                 window=7).var())
    # plt.title('close_pct_change_var7')
    # plot_recessions()
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    # plt.figure(figsize=(20, 10))
    # plt.plot(sandp_df['Close'].pct_change().rolling(closed='right',
    #                                                 window=30).var())
    # plt.title('close_pct_change_var30')
    # plot_recessions()
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    close_pct_change_var60 = sandp_df['Close'].pct_change().rolling(
        closed='right', window=60).var()
    mins = []
    maxs = []
    for r in recessions[:7]:
        recession_rows = close_pct_change_var60.loc[r[0]:r[1]]
        mins.append(recession_rows.min())
        maxs.append(recession_rows.max())
    maxs = np.array(maxs)
    plt.figure(figsize=(20, 10))
    plt.plot(close_pct_change_var60, label='')
    # plt.title('close_pct_change_var60')
    plot_recessions()
    plot_low_high_cole_thresh_preds(
        np.max(mins), np.min(maxs[maxs > 1e-4]), close_pct_change_var60)
    plt.xlabel('Year')
    plt.ylabel('Variance of % Change of S&P 500 Close')
    plt.legend()
    pdf.savefig()
    plt.close()

    # plt.figure(figsize=(20, 10))
    # plt.plot(sandp_df['Close'].pct_change().rolling(closed='right',
    #                                                 window=180).var())
    # plt.title('close_pct_change_var180')
    # plot_recessions()
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    # plt.figure(figsize=(20, 10))
    # plt.plot(sandp_df['Close'].pct_change().rolling(closed='right',
    #                                                 window=7).mean())
    # plt.title('close_pct_change_mean7')
    # plt.axhline(0)
    # plot_recessions()
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    # plt.figure(figsize=(20, 10))
    # plt.plot(sandp_df['Close'].pct_change().rolling(closed='right',
    #                                                 window=30).mean())
    # plt.title('close_pct_change_mean30')
    # plt.axhline(0)
    # plot_recessions()
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    close_pct_change_mean60 = sandp_df['Close'].pct_change().rolling(
        closed='right', window=60).mean().dropna()
    min_close_pct_change_mean60 = []
    max_close_pct_change_mean60 = []
    for r in recessions[:7]:
        recession_rows = close_pct_change_mean60.loc[r[0]:r[1]]
        min_close_pct_change_mean60.append(recession_rows.min())
        max_close_pct_change_mean60.append(recession_rows.max())

    plt.figure(figsize=(20, 10))
    plt.plot(close_pct_change_mean60, label='')
    plot_low_high_thresh_preds(
        np.max(min_close_pct_change_mean60),
        np.min(max_close_pct_change_mean60), close_pct_change_mean60)
    plt.axhline(0, color='black')
    # plt.title('close_pct_change_mean60')
    plot_recessions()
    plt.xlabel('Year')
    plt.ylabel('Mean of % Change of S&P 500 Close')
    plt.legend()
    pdf.savefig()
    plt.close()

    # plt.figure(figsize=(20, 10))
    # plt.plot(sandp_df['Close'].pct_change().rolling(closed='right',
    #                                                 window=90).mean())
    # plt.axhline(0)
    # plt.title('close_pct_change_mean90')
    # plot_recessions()
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    # plt.figure(figsize=(20, 10))
    # plt.plot(sandp_df['Close'].pct_change().rolling(closed='right',
    #                                                 window=120).mean())
    # plt.axhline(0)
    # plt.title('close_pct_change_mean120')
    # plot_recessions()
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    # interest rate
    interest_df = pd.read_csv('data/FEDFUNDS.csv', sep=',')
    interest_df['DATE'] = pd.to_datetime(interest_df['DATE'])
    interest_df = interest_df.set_index('DATE')
    interest_rates = interest_df['FEDFUNDS']

    plt.figure(figsize=(20, 10))
    plt.plot(interest_rates, label='')
    # plt.title('interest_rates')
    plot_recessions()
    plt.xlabel('Year')
    plt.ylabel('US Interest Rate')
    plt.legend()
    pdf.savefig()
    plt.close()

    # plt.figure(figsize=(20, 10))
    # plt.plot(interest_rates.diff())
    # plt.title('interest_rates diff')
    # plot_recessions()
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    interest_rates_diff = interest_rates.diff().dropna()

    plt.figure(figsize=(20, 10))
    plt.plot(interest_rates_diff, label='')
    # plt.title('interest_rates diff 2a')
    plot_recessions()
    max_interest_rates_diff = []
    for r in recessions[1:7]:
        recession_rows = interest_rates_diff.loc[r[0]:r[1]]
        max_interest_rates_diff.append(recession_rows.min())

    plot_low_high_cole_thresh_preds2(
        begin_thresh=0,
        end_thresh=np.max(max_interest_rates_diff),
        signal=interest_rates_diff)
    plt.xlabel('Year')
    plt.ylabel('US Interest Rate Derivative')
    plt.legend()
    pdf.savefig()
    plt.close()

    # consumer spending
    spending_df = pd.read_csv('data/personal_expenditure_no_food_energy.csv')
    spending_df['DATE'] = pd.to_datetime(spending_df['DATE'])
    spending_df = spending_df.set_index('DATE')
    spending = spending_df['DPCCRC1M027SBEA']

    plt.figure(figsize=(20, 10))
    plt.plot(spending, label='')
    # plt.title('consumer_spending_no_food_no_energy')
    plot_recessions()
    plt.xlabel('Year')
    plt.ylabel('US Consumer Spending')
    plt.legend()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(spending, label='')
    plt.yscale('log')
    # plt.title('consumer_spending_no_food_no_energy log')
    plot_recessions()
    plt.xlabel('Year')
    plt.ylabel('log(US Consumer Spending)')
    plt.legend()
    pdf.savefig()
    plt.close()

    # inflation
    df = pd.read_csv('data/inflation.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.set_index('DATE')
    inflation = df['FPCPITOTLZGUSA']

    plt.figure(figsize=(20, 10))
    plt.plot(inflation, label='')
    # plt.title('inflation')
    plot_recessions()
    plt.xlabel('Year')
    plt.ylabel('US Inflation Rate')
    plt.legend()
    pdf.savefig()
    plt.close()

    # unemployment
    unemployment_df = pd.read_csv('data/unemployment_rate.csv')
    unemployment_df['DATE'] = pd.to_datetime(unemployment_df['DATE'])
    unemployment_df = unemployment_df.set_index('DATE')
    unemployment_df = unemployment_df.loc['1950-1-1':]
    unemployment = unemployment_df['UNRATE']

    plt.figure(figsize=(20, 10))
    plt.plot(unemployment, label='')
    # plt.title('unemployment_rate')
    plot_recessions()
    plt.xlabel('Year')
    plt.ylabel('US Unemployment Rate')
    plt.legend()
    pdf.savefig()
    plt.close()

    unemployment_pct_change_mean6 = unemployment.pct_change().rolling(
        closed='right', window=6).mean()
    start_unemployment_pct_change_mean6 = []
    end_unemployment_pct_change_mean6 = []
    max_unemployment_pct_change_mean6 = []
    for r in recessions[:7]:
        recession_rows = unemployment_pct_change_mean6.loc[r[0]:r[1]]
        max_unemployment_pct_change_mean6.append(recession_rows.max())
        start_unemployment_pct_change_mean6.append(
            recession_rows.head(1).values[0])
        end_unemployment_pct_change_mean6.append(
            recession_rows.tail(1).values[0])

    begin_thresh = 0  #np.mean(start_unemployment_pct_change_mean6)
    end_thresh = np.min(max_unemployment_pct_change_mean6)

    plt.figure(figsize=(20, 10))
    plt.plot(unemployment_pct_change_mean6, label='')
    # plt.title('unemployment_rate pct_change mean6months (2a)')
    plot_recessions()
    plot_low_high_cole_thresh_preds(begin_thresh, end_thresh,
                                    unemployment_pct_change_mean6)
    plt.xlabel('Year')
    plt.ylabel('Mean % Change of US Unemployment Rate')
    plt.legend()
    pdf.savefig()
    plt.close()

    # US government debt
    df = pd.read_csv('data/government_debt.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.set_index('DATE')
    federal_debt = df['FYGFD']

    plt.figure(figsize=(20, 10))
    plt.plot(federal_debt, label='')
    # plt.title('federal_debt')
    plot_recessions()
    plt.xlabel('Year')
    plt.ylabel('US Federal Debt')
    plt.legend()
    pdf.savefig()
    plt.close()

    # GDP
    gdp_df = pd.read_csv('data/GDP.csv')
    gdp_df['DATE'] = pd.to_datetime(gdp_df['DATE'])
    gdp_df = gdp_df.set_index('DATE')
    gdp_df = gdp_df.loc['1950-1-1':]

    plt.figure(figsize=(20, 10))
    plt.plot(gdp_df['GDP'], label='')
    # plt.title('US GDP')
    plot_recessions()
    plt.xlabel('Year')
    plt.ylabel('US GDP')
    plt.legend()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(gdp_df['GDP'].pct_change(), label='')
    # plt.title('GDP pct_change')
    plot_recessions()
    plt.xlabel('Year')
    plt.ylabel('% Change of US GDP')
    plt.legend()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(gdp_df['GDP'].pct_change().pct_change(), label='')
    # plt.title('GDP pct_change pct_change')
    plot_recessions()
    plt.xlabel('Year')
    plt.ylabel('% Change of % Change of US GDP')
    plt.legend()
    pdf.savefig()
    plt.close()

    # plt.figure(figsize=(20, 10))
    # plt.plot(gdp_df['GDP'].pct_change().rolling(closed='right',
    #                                             window=2).mean())
    # plt.title('GDP pct_change mean6months')
    # plot_recessions()
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    counts, bin_edges = np.histogram(
        sandp_df['Close'].pct_change().abs().dropna().to_list(), bins=5000)
    plt.figure(figsize=(20, 10))
    plt.loglog(bin_edges[:-1], counts, linestyle='None', marker='o')
    # plt.title('Close Pct Change Frequency')
    plt.ylabel('Frequency of Close Pct Change')
    plt.xlabel('Magnitude of Close Pct Change')
    plt.loglog(
        bin_edges[:-1],
        0.002 * np.power(bin_edges[:-1], -2),
        color='red',
        label='x^-2')
    plt.ylim(0.8, 2e2)
    plt.legend()
    pdf.savefig()
    plt.close()

    # counts, bin_edges = np.histogram(
    #     sandp_df['Volume'].pct_change().abs().dropna().to_list(), bins=5000)
    # plt.figure(figsize=(20, 10))
    # plt.loglog(bin_edges[:-1], counts, linestyle='None', marker='o')
    # plt.title('Volume Pct Change Frequency')
    # plt.ylabel('Frequency of Volume Pct Change')
    # plt.xlabel('Magnitude of Volume Pct Change')
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    counts, bin_edges = np.histogram(
        gdp_df['GDP'].pct_change().abs().dropna().to_list(), bins=5000)
    plt.figure(figsize=(20, 10))
    plt.loglog(bin_edges[:-1], counts, linestyle='None', marker='o')
    # plt.title('GDP Pct Change Frequency')
    plt.ylabel('Frequency of GDP % Change')
    plt.xlabel('Magnitude of GDP % Change')
    plt.legend()
    pdf.savefig()
    plt.close()

    counts, bin_edges = np.histogram(
        unemployment.pct_change().abs().dropna().to_list(), bins=5000)
    plt.figure(figsize=(20, 10))
    plt.loglog(bin_edges[:-1], counts, linestyle='None', marker='o')
    # plt.title('unemployment Pct Change Frequency')
    plt.ylabel('Frequency of US Unemployment % Change')
    plt.xlabel('Magnitude of US Unemployment % Change')
    plt.legend()
    pdf.savefig()
    plt.close()

    counts, bin_edges = np.histogram(
        interest_rates.diff().abs().dropna().to_list(), bins=5000)
    plt.figure(figsize=(20, 10))
    plt.loglog(bin_edges[:-1], counts, linestyle='None', marker='o')
    # plt.title('interest_rates Pct Change Frequency')
    plt.ylabel('Frequency of US Interest Rate % Change')
    plt.xlabel('Magnitude of US Interest Rate % Change')
    plt.legend()
    pdf.savefig()
    plt.close()
