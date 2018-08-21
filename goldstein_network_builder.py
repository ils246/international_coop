from goldstein_timeseries import *
import sys
from joblib import Parallel, delayed
import pickle
import argparse
import pandas as pd
import time
import networkx as nx
from network_tools import basic_net_stats
import matplotlib.pyplot as plt

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
    parser.add_argument('-interpol', '--interpol', type=bool, help='If return interpolated or raw', default='False')

    args=parser.parse_args()


'''
------------------------------------------------
  Build interpolated timeseries from raw data
------------------------------------------------
'''

create_df_ = args.df
read_df_ = args.input
interpol = args.interpol

def build_gs_ts(years, create_df=create_df_, read_df=read_df_, interpolated=interpol):

    print('Arguments: years: %d, create_df: %s, interpol: %s ' % (len(years), str(create_df), interpol ))

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
    if interpol:
        file_name = 'timeseries_data_g%d.pickle' % args.gap
    else:
        file_name = 'non_interpol_timeseries_data_g%d.pickle' % args.gap

    file_ = open(filename, 'wb')
    pickle.dump(filter_dict, file_, protocol=pickle.HIGHEST_PROTOCOL)
    file_.close()

    print('Timeseries are done!')

    return filter_dict


'''
------------------------------------------------
  Build networks in different timescales
------------------------------------------------
'''

def build_interpol_networks(filter_dict):

    # Aggregate network
    # build aggregate network

    print(' ...Working on Aggregate Network... ')

    networks = average_gs_net(filter_dict,granularity='aggregate', years=None)
    sources = [i[0] for i in networks]
    targets = [i[1] for i in networks]
    weights = [i[2] for i in networks]
    aggregate_df = pd.DataFrame({'src': sources, 'tgt': targets, 'weights': weights})
    aggregate_df.to_csv('aggregate_gs_network_g%d.csv' % args.gap, index=None)

    # polarity yerarly polarity nets
    data_for_polarity = {(sources[i],targets[i]):weights[i] for i in range(len(sources))}
    polar_nets = polarity_networks(data_for_polarity,granularity='aggregate')
    polar_file = open('polar_networks_g%d.pickle' % args.gap, 'wb')
    pickle.dump(polar_nets,polar_file,protocol=pickle.HIGHEST_PROTOCOL)
    polar_file.close()

    print(' ...Working on Yearly Networks... ')

    # build yearly networks
    yearly = average_gs_net(filter_dict,granularity='yearly', years=years)
    yearly_file = open('yearly_gs_networks_g%d.pickle' % args.gap, 'wb')
    pickle.dump(yearly,yearly_file,protocol=pickle.HIGHEST_PROTOCOL)
    yearly_file.close()

    # # polarity yearly nets
    polar_yearly=polarity_networks(yearly,granularity='yearly')
    yearly_polar_file = open('yearly_polar_networks_g%d.pickle' % args.gap, 'wb')
    pickle.dump(polar_yearly,yearly_polar_file,protocol=pickle.HIGHEST_PROTOCOL)
    yearly_polar_file.close()

    print(' ...Working on Quarterly Networks... ')

    # # # build quarterly timeseries
    quarterly = average_gs_net(filter_dict,granularity='quarterly', years=years)
    quarterly_file = open('quarterly_gs_networks_g%d.pickle' % args.gap, 'wb')
    pickle.dump(quarterly,quarterly_file,protocol=pickle.HIGHEST_PROTOCOL)
    quarterly_file.close()

    # # polarity quarterly nets
    polar_quarterly=polarity_networks(quarterly,granularity='quarterly')
    quarterly_polar_file = open('quarterly_polar_networks_g%d.pickle' % args.gap, 'wb')
    pickle.dump(polar_quarterly,quarterly_polar_file,protocol=pickle.HIGHEST_PROTOCOL)
    quarterly_polar_file.close()

'''
------------------------------------------------
                Run the code
------------------------------------------------
'''
start_time = time.time()

# Make timeseries
build_gs_ts(years)

# Make all the networks
# data = pickle.load(open('../output/timeseries_data_g120.pickle', 'rb'))
# build_networks(data)

print("------  %0.2f hours  ------" % ((time.time() - start_time)/3600))

'''
------------------------------------------------
    Analyze networks at different timescales
------------------------------------------------
'''
# get basic data
# for normal networks
data=pickle.load(open('yearly_gs_networks_g120.pickle', 'rb'))
basic_info={}
for i in data:
    stats=basic_net_stats(data[i],verbose=False)
    basic_info[i]=stats

# visualize the network in time
font = {'Fontname': 'Arial Narrow'}
attributes = ['No_Nodes','No_Edges', 'Density', 'No_Triangles', 'Transitivity', 'Average_link_strength']
for i in attributes:
    d = [basic_info[j][i] for j in basic_info]
    plt.plot(d)
    plt.title(i, size=14, **font)
    plt.xticks(range(21), range(1995,2017), rotation=90, **font)
    plt.xlabel('Years', **font)
    plt.show()


# visualize the


# # for quarterly networks
# data=pickle.load(open('quarterly_gs_network_g120.pickle', 'rb'))
# basic_info={y:{'Q1':[], 'Q2':[], 'Q3':[], 'Q4':[]} for y in data}
# for i in data:
#     for j in ['Q1', 'Q2', 'Q3', 'Q4']:
#         my_list=[(k[0],k[1],data[i][j][k]) for k in data[i][j]]
#         stats=basic_net_stats(my_list,verbose=False)
#         basic_info[i][j]=stats
#
# # for polar yearly Networks
# data=pickle.load(open('yearly_polar_networks_g120.pickle', 'rb'))
# basic_info={'positive':{}, 'negative':{}, 'balanced':{}}
# for i in ['positive','negative']:
#     for j in data[i].keys():
#         my_list = [ii for ii in data[i][j].keys()]
#         stats=basic_net_stats(my_list,verbose=False)
#         basic_info[i][j]=stats
#
# # for polary quarterly networks
# basic_info={
#             'positive':{y:{'Q1':[], 'Q2':[], 'Q3':[], 'Q4':[]} for y in range(1995,2017)},
#             'negative':{y:{'Q1':[], 'Q2':[], 'Q3':[], 'Q4':[]} for y in range(1995,2017)},
#             }
#
# data=pickle.load(open('quarterly_polar_networks_g120.pickle', 'rb'))
# for i in ['positive','negative']:
#     for j in data[i].keys():
#         for k in ['Q1', 'Q2', 'Q3', 'Q4']:
#             my_list=[(l[0],l[1],data[i][j][k][l]) for l in data[i][j][k]]
#             stats=basic_net_stats(my_list,verbose=False)
#             basic_info[i][j][k]=stats





'''
------------------------------------------------

------------------------------------------------
'''
