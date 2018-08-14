import pandas as pd
import numpy as np
import datetime
from scipy.interpolate import interp1d
from itertools import groupby
from operator import itemgetter
import sys
sys.path.insert(0,'/Users/isabella/github')
from useful_funcs import group_ranges, compute_num_days
from CCM import CCM
import matplotlib.pyplot as plt
import seaborn as sns


start_year = 1995
end_year = 1997
years=range(start_year,end_year+1)

def build_goldstein_ts(years):

    '''
    Builds Goldstein time series from rae data.

    Returns:
    --------
    all_dyads [dict]: nested dictionary
                        1st level keys : country pairs ('Country A', 'Country B')
                        2nd level keys : dates ('YYYY-MM-DD')
    '''

    all_dyads = {}
    for year in years:
        data = pd.read_table('~/Documents/Masters/research/int-cooperation/data/events.%d.tab' % year)
        a = data.groupby(['Source Country', 'Target Country'], axis=0)

        for name, group in a:
            # print('GROUP',len([i for i,r in group.iterrows()]))
            # Remove self loops
            if name[0] != name[1]:

                country_key=name
                dyad = {r['Event Date']: [] for i,r in group.iterrows()}
                dates = group.groupby('Event Date')

                # print (name)

                # print(len(dyad))
                # print (len(dates))

                for name,group in dates:
                    dyad[name]=np.mean(group['Intensity'])
                    # print(group['Intensity'])

                for key in dyad.keys():
                    if country_key in all_dyads.keys():
                        all_dyads[country_key][key]=dyad[key]
                    else:
                        all_dyads[country_key]= dyad
                        all_dyads[country_key][key]=dyad[key]


    # print(len(all_dyads[('United States', 'China')]))
    # print('all dyads', len(all_dyads.keys()))
    return all_dyads


def filter_by_gap(data, max_gap, start_year, end_year, inter_kind='linear'):

    '''
    Args:
    -----

    data [dict]: nested dicts keyed by country pair ('countryA', 'countryB')
    gap [int] : number of consecutive days with missing data

    Returns:
    --------

    ts [dict]: nested dict :
            1st level keys: [dates, pairs]
            2nd level keys (in pairs): ('Country A', 'Country B')
            2nd level vals: list with interpolated ts of goldsetin scores for every day between start_year and end_year
    '''

    ts = {}
    # 1. Check if the pair has more inteaction days than 0.5 * time slice
    total_days = compute_num_days(start_year, end_year)
    half_life = 0.20*total_days
    all_keys = list(data.keys())

    for key in all_keys:
        if len(data[key]) <= half_life:
            del data[key]

    # Adding dates that don't have interactions
    all_days = []
    date1 = '%d-01-01' % start_year
    date2 = '%d-12-31' % (end_year - 1)
    start = datetime.datetime.strptime(date1, '%Y-%m-%d')
    end = datetime.datetime.strptime(date2, '%Y-%m-%d')
    step = datetime.timedelta(days=1)
    while start <= end:
        all_days.append(start.strftime('%Y-%m-%d'))
        start += step

    ts['dates']=all_days
    ts['pairs']={}

    # 2. Remove if there is a interaction gap >= 'gap'
    sorted_keys = sorted([key for key in data.keys()])
    for key in data.keys():
        for date in all_days:
            dates_with_data = data[key].keys()
            if date not in dates_with_data:
                data[key][str(date)]=np.nan

        a = sorted([(k,data[key][k]) for k in data[key].keys()], key=lambda x: x[0])
        a = {'Dates': [i[0] for i in a], 'Intensity': [i[1] for i in a]}
        df = pd.DataFrame(data=a)

        # for the dates where there is no data, add nan
        aux_df = df[df['Intensity'] == np.nan]

        # get the largest gap with nan
        indexes = list(aux_df.index.values)
        indexes = group_ranges(indexes)
        indexes = [list(i) for i in indexes]
        gap = filter(lambda x: x >= max_gap, [i[1]-i[0] for i in indexes if len(i) > 1])
        gap = len(list(gap)) >= 1

        if gap:
            pass

        else:
            # print(key, 'there is no gap, we keep this key')
            intensity=list(df['Intensity'].interpolate(method='linear'))
            ts['pairs'][key] = intensity

    return ts


def align_ts(ts1,ts2, dual=True):

    '''
    Given that interpd1 does not interpolate the strating or trailing NaNs in timeseries,
    this function removes strating and trailing NaNs in time series making them the same shape.

    Args:
    -----
    ts1 [list]: Goldstein timeseries
    ts2 [list]: Goldstein timeseries

    Returns:
    --------
    ts_fixes [dict]: nested dict
                    1st level keys: 'ts1' & 'ts2'
                    2nd level keys: 'head_fix' & 'tail_fix'
                    'head_fix' [tuple] : (index of starting NaN, las index with NaN)
                    'head_tail' [tuple] : (index of starting NaN, las index with NaN)

                    * head_fix and tail_fix can be empty if there is no starting or trailing NaNs.

    '''

    counter=1
    if dual:
        ts_fixes={'ts1':{},'ts2':{}}
        my_list = [ts1,ts2]
    else:
        ts_fixes = {'ts1':{}}
        my_list=[ts1]

    for ts in my_list:
        ts_bool=pd.isna(pd.DataFrame(ts))

        # for head align
        tsh = ts_bool[ts_bool[0]== True][:100]
        if tsh.empty:
            ts_fixes['ts%d' % counter]['head_fix']=[]
        else:
            head_ts = (min(tsh.index.values), max(tsh.index.values))
            ts_fixes['ts%d' % counter]['head_fix']=head_ts

        # for tail fixes
        tst = ts_bool[ts_bool[0]== True][int(len(ts_bool)*0.5):]
        if tst.empty:
            ts_fixes['ts%d' % counter]['tail_fix']=[]
        else:
            tail_ts = (min(tst.index.values), max(tst.index.values))
            ts_fixes['ts%d' % counter]['tail_fix']=tail_ts

        counter+=1
    return ts_fixes


def compute_CCM(timeseries, e,t):
    '''
    Computes CCM on aligened time series.

    Args:
    -----
    timseries [dict]: keyed by country pair
    e : Embedding dimensions
    t : tau

    Returns:
    --------
    results [dict]: nested dicts:
                    1st level keys: ('Country A', 'Country B')
                    2nd level keys: 'corrs' & 'ps'
    '''

    pairs = list(timeseries['pairs'].keys())
    results = {}

    # pairs = [('United States', 'China')]

    for pair in pairs:
        A = timeseries['pairs'][pair]

        if (pair[1],pair[0]) in pairs:
            B = timeseries['pairs'][(pair[1], pair[0])]

            fixes = align_ts(A,B)
            if (fixes['ts1']['head_fix'] != []) and (fixes['ts2']['head_fix'] != []):
                head = max(fixes['ts1']['head_fix'][1], fixes['ts2']['head_fix'][1])
            elif (fixes['ts1']['head_fix'] == []) and (fixes['ts2']['head_fix'] == []):
                head=0
            else:
                head = [i for i in [fixes['ts1']['head_fix'],fixes['ts2']['head_fix']] if i != []]
                head = max(head[0],head[1])

            A = A[head:]
            B = B[head:]

            if (fixes['ts1']['tail_fix'] != []) and (fixes['ts2']['tail_fix'] != []):
                tail = min(fixes['ts1']['tail_fix'][0], fixes['ts2']['tail_fix'][0])

            elif (fixes['ts1']['tail_fix'] == []) and (fixes['ts2']['tail_fix'] == []):
                tail=-1
            else:
                tail = [i for i in [fixes['ts1']['tail_fix'],fixes['ts2']['tail_fix']] if i != []]
                print ('Tail in the making', tail)
                tail = min(tail[0],tail[1])

            A = A[:tail]
            B = B[:tail]


            CCM_coeff, pval =CCM (A,B,E=e,tau=t)
            results[pair]={'corrs': list(CCM_coeff), 'ps': list(pval)}

            pairs.remove(pair)
            pairs.remove((pair[1], pair[0]))

        else:
            pairs.remove(pair)

    return results



def average_gs_net(ts, weighted=True, granularity='quarterly', years=None):

    '''
    Args:
    ------
    ts [dict]: timeseries keyed country pairs
    weighted [bool]: if TRUE, returns edges (src, tgt, weight) otherwise (src,tgt)
    granuralrity [str] : 'Aggregate' -> average network of all years
                          'Yearly' ->  average network by year
                          'quarterly' -> average network by quarter per year
    years (optional): range of years for yearly or quarterly analysis

    Returns:
    ---------
    edges[list]: When 'Aggregate' ('A', 'B', weight)
    averaged [dicr]: otherwise -> {year: '('A', 'B'): weight}

    '''

    if granularity !='aggregate' and years == None:
        raise TypeError ('You have to pass a range or list of years when running %s networks!' % granularity)

    else:

        if granularity=='aggregate':

            edges = []
            averaged = {}
            for key in ts.keys():

                na_fixes = align_ts(ts[key],'no second ts', dual=False)
                if na_fixes['ts1']['head_fix'] == []:
                    head = 0
                else:
                    head = na_fixes['ts1']['head_fix'][1]
                if na_fixes['ts1']['tail_fix'] == []:
                    tail=-1
                else:
                    tail = na_fixes['ts1']['tail_fix'][0]

                nan_corrected = ts[key][head:tail]

                if weighted:
                    edges.append((key[0], key[1], np.mean(nan_corrected)))
                else:
                    edges.append((key[0], key[1]))

            return edges

        elif granularity=='yearly':

            counter = 0
            aux=[]
            averaged = {year:{} for year in years}

            for year in years:
                days = compute_num_days(year,year)

                for key in ts.keys():

                    # get year data
                    year_data = ts[key][counter:counter+days+1]
                    counter+=days

                    # get fixes
                    na_fixes = align_ts(year_data,'no second ts', dual=False)
                    if na_fixes['ts1']['head_fix'] == []:
                        head = 0
                    else:
                        head = na_fixes['ts1']['head_fix'][1]
                    if na_fixes['ts1']['tail_fix'] == []:
                        tail=-1
                    else:
                        tail = na_fixes['ts1']['tail_fix'][0]

                    nan_corrected = year_data[head+1:tail]
                    aux.append((key[0],key[1],np.mean(nan_corrected)))

                averaged[year]=aux
                aux=[]

            return averaged

        else:

            counter = 0
            averaged = {year:{'Q1':{}, 'Q2':{}, 'Q3':{}, 'Q4':{}} for year in years}
            aux=[[],[],[],[]]
            for year in years:

                days=compute_num_days(year,year)
                if days == 365:
                    quarters = [90,91,92,92]
                else:
                    quarters = [91,91,92,92]

                for key in ts.keys():

                    # get year data
                    year_data = ts[key][counter:counter+days+1]

                    q_counter=0
                    for i in range(4):

                        # I can only remove the head and tail with nan after slicing quarters,
                        # otherwise days won't match
                        from math import isnan
                        quarter_data=year_data[q_counter:q_counter+quarters[i]+1]
                        quarter=[i for i in quarter_data if not isnan(i)]

                        if quarter != []:
                            aux[i].append((key[0],key[1] , np.mean(quarter)))
                        q_counter+=quarters[i]+1

                for a in range(len(aux)):
                    label = 'Q%d' % (a+1)
                    averaged[year][label]= {(k[0],k[1]): k[2] for k in aux[a]}

                aux=[[],[],[],[]]

            return averaged


def polarity_networks(data,granularity='quarterly'):

    if granularity=='aggregate':
        nets = {'positive':{}, 'negative':{}, 'balanced': {}}

        for key in data.keys():
            if data[key] > 0:
                nets['positive'][key]=data[key]
            elif data[key]< 0:
                nets['negative'][key]=data[key]
            else:
                nets['balanced'][key]=data[key]

    elif granularity=='yearly':

        nets = {
        'positive': { year:{} for year in data},
        'negative': { year:{} for year in data},
        'balanced': { year:{} for year in data},
        }

        for year in data.keys():
            for key in data[year].keys():
                if data[year][key] > 0:
                    nets['positive'][year][key]=data[year][key]
                elif data[year][key]< 0:
                    nets['negative'][year][key]=data[year][key]
                else:
                    nets['balanced'][year][key]=data[year][key]

    else:
        nets = {
        'positive': { year:{'Q1':{},'Q2':{}, 'Q3':{}, 'Q4':{}} for year in data},
        'negative': { year:{'Q1':{},'Q2':{}, 'Q3':{}, 'Q4':{}} for year in data},
        'balanced': { year:{'Q1':{},'Q2':{}, 'Q3':{}, 'Q4':{}} for year in data},
        }

        for year in data.keys():
            for quarter in data[year].keys():
                for key in data[year][quarter].keys():
                    if data[year][quarter][key] > 0:
                        print(year, quarter,key)
                        nets['positive'][year][quarter][key]=data[year][quarter][key]
                    elif data[year][quarter][key]< 0:
                        nets['negative'][year][quarter][key]=data[year][quarter][key]
                    else:
                        nets['balanced'][year][quarter][key]=data[year][quarter][key]

    return nets

# ###  -------  Run the Code --------  ###
# Get the CCM coeffcient between country paris


# Build Average Goldsetein netowrks
ts=build_goldstein_ts(years)
filtered_ts=filter_by_gap(ts, 100, start_year, end_year)
networks=average_gs_net(filtered_ts,granularity='quarterly', years=range(start_year,end_year))


# build poliarity networks
ts=build_goldstein_ts(years)
filtered_ts=filter_by_gap(ts, 120, start_year, end_year)
polarity_networks(filtered_ts,granularity='quarterly')
