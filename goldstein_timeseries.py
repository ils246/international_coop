import pandas as pd
import numpy as np
import datetime
from itertools import groupby
import sys
sys.path.insert(0,'/Users/isabella/github')
from useful_funcs import group_ranges, compute_num_days
from CCM import CCM


def filter_raw_data(years):

    '''
    filters raw data and returns a (very) large dataframe with
    all the aggregate data

    Args:
    -----
    years (iter): iterator with all the years to be considered

    Returns:
    --------
    final_df [dataframe] : dataframe with columns: Source,
    Target and Date
    '''

    final_df = pd.DataFrame()
    for year in years:

        print(year)
        data=pd.read_table('~/Documents/Masters/research/int-cooperation/data/events.%d.tab' % year)

        data = data.drop(columns = ['Source Name','Event ID', 'Source Sectors', 'Event Text', 'Target Name', 'Publisher', 'Sentence Number', 'Target Sectors', 'Story ID', 'City', 'Province', 'District', 'Latitude','Longitude'], axis=1)

        data = data.rename(columns={'Source Country': 'Source', 'Target Country': 'Target', 'Event Date': 'Date'})

        final_df = pd.concat([final_df, data])

    return final_df


def filter_by_dyad(dataframe, src, tgt, year,year2, m1, m2, d1, d2):

    """
    Args:
    -----------
    src [str] : source country name
    tgt [str] : target country name
    convers [dict] : dictionary of cameo:intensity values
    year [int] : year to analyze


    Returns:
    ----------
    ts_gs [pandas df] : timeseries of mean goldsetein score
    for country dyad

    """

    timeseries = []

    # filter by rows - only the dyad of interest
    filt = dataframe[dataframe.Source == src]
    filt = filt[filt.Target == tgt]

    # dates to analyze
    d1 = datetime.date(year,m1,d1)
    d2 = datetime.date(year2,m2,d2)

    # list containing all of the dates
    dates = [d1 + datetime.timedelta(days=x) for x in range((d2-d1).days + 1)]

    for date in dates:
        aux = filt[filt.Date == date.strftime("%Y-%m-%d")]

        if aux.empty:
            timeseries.append(np.nan)

        else:

            #averaging occurrences
            average = np.average(aux['Intensity'])
            timeseries.append(average)

    return timeseries


def filter_by_gap(ts, max_gap, start_year, end_year, interpolated, inter_kind='linear'):

    '''
    Args:
    -----
    doesn't need to take nested dict, just normal dict --> {(Country A, Country B): [list of Goldstein scores]}, actually needs to be read from csv created by filter_by_dyad
    data [dict]: nested dicts keyed by country pair ('countryA', 'countryB') --> dyads
    gap [int] : number of consecutive days with missing data

    Returns:
    --------

    ts [dict]: nested dict :
            1st level keys: [dates, pairs]
            2nd level keys (in pairs): ('Country A', 'Country B')
            2nd level vals: list with interpolated ts of goldsetin scores for every day between start_year and end_year
    '''

    # Removing paris with nans for more than 50% of days in the sample
    total_days = compute_num_days(start_year, end_year)
    thresh = 0.80*total_days

    from math import isnan
    num_nans = sum([1 for i in ts if isnan(i)])
    if num_nans >= thresh:
        return None

    else:
        # Remove pairs with gaps larger than maxGap
        ts_ = pd.DataFrame({'Timeseries': ts})
        ts_nan = pd.isna(ts_)
        ts_['nans']=ts_nan
        ts_nan = ts_[ts_['nans'] == True]
        indexes = list(ts_nan.index.values)
        indexes = group_ranges(indexes)
        indexes = [list(i) for i in indexes]
        gap = filter(lambda x: x >= max_gap, [i[1]-i[0] for i in indexes if len(i) > 1])
        gap = len(list(gap)) >= 1

        if gap:
            return None
        else:
            if interpolated == 'True':
                ts_interpolated = list(ts_['Timeseries'].interpolate(method=inter_kind))
                return ts_interpolated
            else:
                ts = list(ts_['Timeseries'])
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
        if ts !=[]:
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
        else:
            ts_fixes['ts%d' % counter]['head_fix']=[]
            ts_fixes['ts%d' % counter]['tail_fix']=[]

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

                if ts[key] is not None:

                    na_fixes = align_ts(ts[key],'no second ts', dual=False)
                    if na_fixes['ts1']['head_fix'] == []:
                        head = 0
                    else:
                        head = na_fixes['ts1']['head_fix'][1]
                    if na_fixes['ts1']['tail_fix'] == []:
                        tail=-1
                    else:
                        tail = na_fixes['ts1']['tail_fix'][0]

                    nan_corrected = ts[key][head+1:tail-1]

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
                    if ts[key] is not None:

                        if year != 2016:
                            year_data = ts[key][counter:counter+days+1]
                            # print(counter, counter+days+1)

                        else:
                            year_data = ts[key][counter:]

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

                        nan_corrected = year_data[head+1:tail-1]
                        aux.append((key[0],key[1],np.mean(nan_corrected)))

                averaged[year]=aux
                aux=[]
                counter+=days

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
                    if ts[key] is not None:

                        if year != 2016:
                            year_data = ts[key][counter:counter+days+1]
                            # print(counter, counter+days+1)

                        else:
                            year_data = ts[key][counter:]

                        # if counter <= 275:
                        #     # get year data
                        #     year_data = ts[key][counter:counter+days+1]
                        # else:
                        #     year_data = ts[key][counter:]

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


def polarity_networks(data, granularity='quarterly'):

    '''
    Returns network edges based on their polarity [postive, negative, balanced]. * Balanced are those nets with weights = 0, in our data 0 does not mean that the edge doesn't exist, just the average of interactions is 0

    Args:
    -----
    data [dict]: aggregate shape -> {('A','B'): weights}
                 yearly shape -> {year:[('A','B', weights]}
                 quarterly shape -> {year:{Q: {('A','B'): weights}}}
    Returns:
    --------
    nets [nested dict]: network edges divided by polarity,
                        and similar structure to input.
    '''

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
            for edge in data[year]:
                if edge[2] > 0:
                    nets['positive'][year][edge]= edge[2]
                elif edge[2]< 0:
                    nets['negative'][year][edge]=edge[2]
                else:
                    nets['balanced'][year][edge]=edge[2]

    else:
        nets = {
        'positive': { year:{'Q1':{},'Q2':{}, 'Q3':{}, 'Q4':{}} for year in data},
        'negative': { year:{'Q1':{},'Q2':{}, 'Q3':{}, 'Q4':{}} for year in data},
        'balanced': { year:{'Q1':{},'Q2':{}, 'Q3':{}, 'Q4':{}} for year in data},
        }

        for year in data.keys():
            for quarter in data[year].keys():

                if data[year][quarter] != []:

                    for key in data[year][quarter].keys():
                        if data[year][quarter][key] > 0:
                            # print(year, quarter,key)
                            nets['positive'][year][quarter][key]=data[year][quarter][key]
                        elif data[year][quarter][key]< 0:
                            nets['negative'][year][quarter][key]=data[year][quarter][key]
                        else:
                            nets['balanced'][year][quarter][key]=data[year][quarter][key]

    return nets


def wrapper_filter(df,dyad, max_gap, interpolated):
    '''
    Wrapper function to parallelize the creation of
    timeseries.

    Args:
    -----
    df[dataframe]: dataframe
    dyad [tuple]: tuple with shape ('Source', 'Target')

    Returns:
    --------
    tuple of shape ('Source', 'Target', timeseries)

    '''

    b = filter_by_dyad(df, dyad[0], dyad[1], 1995, 2016, 1, 12, 1, 31)
    b1 = filter_by_gap(b, max_gap, 1995, 2016, interpolated)
    return (dyad[0], dyad[1], b1)
