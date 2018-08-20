from goldstein_timeseries import *
import sys
from joblib import Parallel, delayed
import pickle
import argparse
import pandas as pd
import time
import networkx   as nx

if __name__ == '__main__':

    start_year = 1995
    end_year = 2016
    years=range(start_year,end_year+1)

    parser = argparse.ArgumentParser(description='Make goldstein timeseries')
    parser.add_argument('-c', '--cores', type=int, help='Number of cores to use', required=False, default=2)
    parser.add_argument('-i', '--input', type=str, help='Path to dataframe with raw data', default='../output/data.csv')
    parser.add_argument('-o', '--output', type=str, help='Path to write dataframe with raw data', default='../output/data.csv')
    parser.add_argument('-gap', '--gap', type=int, help='Max gap of missing data in each pair', default=120)
    parser.add_argument('-df', '--df', type=bool, help='If create df or read existing file', default='False')
    args=parser.parse_args()


'''
------------------------------------------------
  Build interpolated timeseries from raw data
------------------------------------------------
'''

create_df_ = args.df
read_df_ = args.input

def build_gs_ts(years, create_df=create_df_, read_df=read_df_):

    if create_df == 'True':
        data = filter_raw_data(years)
        data.to_csv(str(args.output), index =False)

    else:
        data = pd.read_csv(read_df)

    # get all the possible pairs
    groups = data.groupby(['Source', 'Target'], axis=0)
    names_from_groups=[name for name,group in groups if (name[0]!=name[1]) and
    (len(group)>=1000)]

    # # Build Average Goldsetein netowrks
    result = Parallel(n_jobs=args.cores)(delayed(wrapper_filter)
    (df=data, dyad=i, max_gap=args.gap) for i in names_from_groups)
    filter_dict = {(i[0], i[1]): i[2] for i in result}

    # pickle timeseries
    file_ = open('timeseries_data_g%d.pickle' % args.gap, 'wb')
    pickle.dump(filter_dict, file_, protocol=pickle.HIGHEST_PROTOCOL)
    file_.close()

    print('Timeseries are done!')

    return filter_dict


'''
------------------------------------------------
  Build networks in different timescales
------------------------------------------------
'''

def build_networks(filter_dict):

    print(' ...Working on Aggregate Network... ')

    # get the filter dict so I don't need to keep running this forever
    file_ = open('timeseries.pickle', 'wb')
    pickle.dump(filter_dict,file_)
    file_.close()

    # Aggregate network
    # build aggregate network
    networks = average_gs_net(filter_dict,granularity='aggregate', years=None)
    sources = [i[0] for i in networks]
    targets = [i[1] for i in networks]
    weights = [i[2] for i in networks]
    aggregate_df = pd.DataFrame({'src': sources, 'tgt': targets, 'weights': weights})
    aggregate_df.to_csv('aggregate_gs_network_g%d.csv' % args.gap, index=None)

    # polarity yerarly polarity nets
    data_for_polarity = {(sources[i],targets[i]):weights[i] for i in range(len(sources))}
    polar_nets = polarity_networks(data_for_polarity,granularity='aggregate')
    print(polar_nets)
    # polar_file = open('polar_networks_g%d.pickle' % args.gap, 'wb')
    # pickle.dump(polar_nets,polar_file,protocol=pickle.HIGHEST_PROTOCOL)
    # polar_file.close()
    #
    # print(' ...Working on Yearly Networks... ')
    #
    # # build yearly networks
    # yearly = average_gs_net(filter_dict,granularity='yearly', years=years)
    # yearly_file = open('yearly_gs_network_g%d.pickle' % args.gap, 'wb')
    # pickle.dump(yearly,yearly_file,protocol=pickle.HIGHEST_PROTOCOL)
    # yearly_file.close()
    #
    # # polarity yearly nets
    # polar_yearly=polarity_networks(yearly,granularity='yearly')
    # yearly_polar_file = open('yearly_polar_network_g%d.pickle' % args.gap, 'wb')
    # pickle.dump(polar_yearly,yearly_polar_file,protocol=pickle.HIGHEST_PROTOCOL)
    # yearly_polar_file.close()
    #
    # print(' ...Working on Quarterly Networks... ')
    #
    # # # build quarterly timeseries
    # quarterly = average_gs_net(filter_dict,granularity='quarterly', years=years)
    # quarterly_file = open('quarterly_gs_network_g%d.pickle' % args.gap, 'wb')
    # pickle.dump(quarterly,quarterly_file,protocol=pickle.HIGHEST_PROTOCOL)
    # quarterly_file.close()
    #
    # # polarity quarterly nets
    # polar_quarterly=polarity_networks(quarterly,granularity='quarterly')
    # quarterly_polar_file = open('quarterly_polar_network_g%d.pickle' % args.gap, 'wb')
    # pickle.dump(polar_quarterly,quarterly_polar_file,protocol=pickle.HIGHEST_PROTOCOL)
    # quarterly_polar_file.close()

'''
------------------------------------------------
                Run the code
------------------------------------------------
'''
start_time = time.time()

timeseries=build_gs_ts(years, create_df=args.df, read_df=args.input)
build_networks(timeseries)

print("------  %0.2f hours  ------" % ((time.time() - start_time)/3600))


'''
------------------------------------------------
    Analyze networks at different timescales
------------------------------------------------
'''
# Aggregate timeseries

'''
------------------------------------------------

------------------------------------------------
'''
