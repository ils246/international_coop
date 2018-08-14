import numpy as np
import pandas as pd
from collections import Counter
import json
import ast

'''
This code applies PCA analysis to the raw data from the ICEWS dataset
per year.

    Returns:
    --------

    - total-interactions.csv : csv with the total num of out and in-going
                               interactions per country between 1995-2016.

    - __raw_counts_'year'.csv :  csv with a matrix that counts the number of
                               in-coming interactions per country per year
                               by month.

    - countries-'year'.csv :   ordered list of countries included in the
                               analysis that year.

    - eigen-'year'.csv :       eigevalues, eigenvectors, and variance explained
                               resulting from PCA.

'''

def interaction_counts(countries, years, thresh_outs, thresh_ins, out_path,filename, global_=True, source_country=None, monthly=True):

    '''
    Args:
    -----
    years [list]        : years
    thresh_outs [float] : threshold for number of out-going interaction
    thresh_ins [float]  : threshold for number of ins-going interaction
    out_path [str]      : path to output folder
    global_ [bool]       : True if we want to see how all countries behave
                          towards eachother, False if we want to see how ONE
                          country behaves
    source_country [str]: source country for interactions towards the rest of the world

    Returns:
    --------

    __raw_counts: csv with a matrix that counts the number of
                  in-coming interactions per country per year
                  by month.
    '''

    for year in years:

        print('--- Starting %d ---' % year)

        # raw = pd.read_table("~/Documents/Masters/research/int-cooperation/ICEWS-data/events.%d.tab" % year)
        # raw = pd.read_table("../ICEWS_data/events.%d.tab" % year)
        raw=pd.read_table('~/Documents/Masters/research/int-cooperation/ICEWS-data/events.%d.tab' % year)


        # Make table of number of interactions:
        # interactions=[]
        # outs = Counter(raw['Source Country'])
        # ins = Counter(raw['Target Country'])
        #
        # countries = list(raw['Target Country'])
        # countries.extend(list(raw['Source Country']))
        # countries = list(set(countries))

        # print(len(countries))

        # for country in list(set(countries)):
        #     a = (country, outs[country], ins[country])
        #     interactions.append(a)

        # total_interactions = pd.DataFrame()
        # total_interactions['Country'] = [i[0] for i in interactions]
        # total_interactions['Outs'] = [i[1] for i in interactions]
        # total_interactions['Ins'] = [i[2] for i in interactions]

        # mean_outs = np.median(total_interactions['Outs'])
        # mean_ins = np.median(total_interactions['Ins'])
        #
        # countries_to_include = total_interactions[(total_interactions.Outs >=
        # (thresh_outs * mean_outs)) & (total_interactions.Ins>= (thresh_ins * mean_ins)) ]

        # Make a file with out-going and incoming interactions

        if global_:

            print('%d Countries included in year %d' % (len(countries), year))


        # ### -------  Make the matrix of interactions -------- ###

        # drop countries that are not in the 'countries_to_include'
        data = raw[raw['Target Country'].isin(countries)]

        # make dates month only
        months=data['Event Date'].apply(lambda x : x[5:7])
        data=data.drop('Event Date', axis=1)
        data['Date'] = months

        # Counts of actions towards countries
        coop_=[]
        non_coop_=[]
        results = pd.DataFrame()

        if monthly:
            if global_:
                for c in countries:
                    for month in ["%02d" % i for i in range(1,13)]:

                        subset = data[(data['Target Country'] == c) &
                                      (data['Date'] == month) &
                                      (data['Source Country'] != c)]

                        coop = len(subset[subset['Intensity'] >= 0])
                        non_coop = len(subset[subset['Intensity'] < 0])

                        coop_.append(coop)
                        non_coop_.append(non_coop)

                    sub_results=pd.DataFrame()
                    sub_results['c_%s' % c] = coop_
                    sub_results['nc_%s' % c] = non_coop_

                    results=pd.concat([results,sub_results], axis=1)
                    coop_ = []
                    non_coop_=[]
                    sub_results=[]

                results.to_csv(out_path+'%s_raw_counts_%d.csv' % (filename,year), index=False)

            else:

                for c in countries:

                    for month in ["%02d" % i for i in range(1,13)]:

                        subset = data[ (data['Source Country'] == source_country) &
                                      (data['Target Country'] == c) &
                                      (data['Date'] == month) &
                                      (data['Source Country'] != c)]

                        coop = len(subset[subset['Intensity'] >= 0])
                        non_coop = len(subset[subset['Intensity'] < 0])

                        coop_.append(coop)
                        non_coop_.append(non_coop)

                    sub_results=pd.DataFrame()
                    sub_results['c_%s' % c] = coop_
                    sub_results['nc_%s' % c] = non_coop_

                    results=pd.concat([results,sub_results], axis=1)
                    coop_ = []
                    non_coop_=[]
                    sub_results=[]

                results.to_csv(out_path+'%s_raw_counts_%d.csv' % (source_country,year), index=False)

        else:
            for c in countries:
                subset = data[ (data['Source Country'] == source_country) &
                              (data['Target Country'] == c) &
                              (data['Source Country'] != c)]

                coop = len(subset[subset['Intensity'] >= 0])
                non_coop = len(subset[subset['Intensity'] < 0])

                coop_.append(coop)
                non_coop_.append(non_coop)

                sub_results=pd.DataFrame()
                sub_results['c_%s' % c] = coop_
                sub_results['nc_%s' % c] = non_coop_

                results=pd.concat([results,sub_results], axis=1)
                coop_ = []
                non_coop_=[]
                sub_results=[]

        results.to_csv(out_path+'%s_yearly_raw_counts_%d.csv' % (source_country,year), index=False)


def community_behaviors(countries, community_, years,outpath, community_name):

    community=np.zeros([len(community_),60])
    c=0
    for country in countries:
        results=np.zeros([len(years),60])
        a=0
        for year in years:
            data = pd.read_csv('%s%s_yearly_raw_counts_%d.csv' %(outpath,country,year))
            results[a]=list(data.values[0])
            a+=1

        community[c] = np.mean(results,axis=0)
        c+=1

    # community matrix - M number of rows per actors in the community
    # by H (coop,nocoop per country)

    # ------ Get the primary eigenbehaviors of the community -------

    # Get average behavior for community
    community_average = np.mean(community,axis=0)
    print(community_average)

    # Get the PCA for the community
    # Should I normalize the columns by their mean??
    centralized  = community-np.mean(community,axis=0)
    scat = np.matmul(centralized.T,centralized)
    cov =  scat/((2*np.shape(community)[1]))

    ### Eigenvector decomposition
    from numpy import linalg
    lambdas,vecs = linalg.eigh(cov)

    ### Amount of variance explained by the eigenvectors:
    total = sum(lambdas)
    var_explained=[(var/total)*100 for var in lambdas]

    # Make results csvs
    results = pd.DataFrame()
    results['lambda'] =  [i for i in lambdas]
    results['vectors'] = [list(vecs[:,i]) for i in range(len(lambdas))]
    results['var_explained']=var_explained

    #distance from community
    df=distance_from_community(community, community_, community_average, countries, vecs)
    # df.to_csv('../output/'+community_name+'-community-distances.csv', index=False)

def distance_from_community(community, community_, community_average, countries, vecs):

    # get vector omega with weights - optimal weighting configuration
    # to get an individual's behavior as close as possible to the community
    # behavior space

    omegas = np.zeros([len(community_),60])
    for i in range(len(community_)):
        omegas[i] = vecs[i]*(community[i]-community_average)

    diffs_frame=pd.DataFrame()
    counter=0
    for j in range(len(community_)):
        la=[]
        for i in range(len(community_)):
            dist = np.linalg.norm(omegas[j]-omegas[i])
            la.append(dist)
            counter+=1
        diffs_frame[counter]=la

    return diffs_frame



### -------  Make the matrix of interactions -------- ###

def PCA(out_path,years,data_f, source_country, read_file=True, all_years =False, all_years_df=None):

    '''
    Args:
    -----

    years [list] : years for analysis
    data [str]  : string of file name

    Returns:
    --------

    results [df] : dataframe with eigenvalues, eigenvectors
                   and variance explained
    '''
    if all_years:

        data = all_years_df
        cols = len(data.columns)
        data=data.values
        centralized  = data-np.mean(data,axis=0)

        scat = np.matmul(centralized.T,centralized)
        cov =  scat/((2*cols))

        ### Eigenvector decomposition
        from numpy import linalg
        lambdas,vecs = linalg.eigh(cov)

        ### Variance explained by eigenvectors
        total = sum(lambdas)
        var_explained=[(var/total)*100 for var in lambdas]

        # Make results csvs
        results = pd.DataFrame()
        results['lambda'] =  [i for i in lambdas]
        results['vectors'] = [list(vecs[:,i]) for i in range(len(lambdas))]
        print(sum(results['vectors'][0]))
        results['var_explained']=var_explained
        results.to_csv(out_path+'after-crisis-years-eigen.csv', index=False)

    else:

        for year in years:

            data_file = '%s%s_%d.csv' %(out_path,data_f,year)
            data=pd.read_csv(data_file)
            countries_csv = list(data.columns)

            cols = len(data.columns)
            data=data.values

            centralized  = data-np.mean(data,axis=0)

            # # multiply the matrices
            # # The transpose is first, then the normalize - this way I get a 218 by 218 matrix
            scat = np.matmul(centralized.T,centralized) # this is the scatter matrix

            # #  multiply by 2 and then divide
            cov =  scat/((2*cols)) # by dividing over

            # # # Eigenvector decomposition
            from numpy import linalg
            lambdas,vecs = linalg.eigh(cov)
            # eigens = [(lambdas[i],vecs[i]) for i in range(len(lambdas))]
            # eigens = sorted(eigens, key=lambda x: x[0], reverse=True)

            # # # Amount of variance explained by the eigenvectors:
            total = sum(lambdas)
            var_explained=[(var/total)*100 for var in lambdas]

            # Make results csvs
            results = pd.DataFrame()
            results['lambda'] =  [i for i in lambdas]
            results['vectors'] = [list(vecs[:,i]) for i in range(len(lambdas))]
            results['var_explained']=var_explained
            results.to_csv(out_path+source_country+'-eigen-%d.csv' % year, index=False)


### -------  All year PCA -------- ###

def aggregate_years(years, out_path):

    all_years = []

    for year in years:
        data_file = '%sglobal_raw_counts_%d.csv' %(out_path,year)
        data = pd.read_csv(data_file)
        all_years.append(data)

    result = pd.concat(all_years)

    return result

### -------  Analyzing PCA results -------- ###

def country_ranking(file_name,countries, years,out_path):

    countries_=np.zeros([len(years),len(countries)])

    for year in range(len(years)):

        data_file = '%s%s-eigen-%d.csv' % (out_path,file_name,years[year])
        data = pd.read_csv(data_file)
        data = data.sort_values(by='lambda', ascending=False)
        c = ast.literal_eval(data['vectors'][0])[0::2]
        c = np.array([abs(i) for i in c])
        nc = ast.literal_eval(data['vectors'][0])[1::2]
        nc = np.array([abs(i) for i in nc])
        t = np.mean(np.array([ c, nc ]), axis=0 )
        countries_[year] = np.array(t)


    final = {k:[] for k in countries}
    for i in range(len(countries)):
        final[countries[i]]=list(countries_[:,i])

    return final

# print(countries)
# print(len(countries['Israel']))

















### ------- Run code, run -------- ###


# years = list(range(1995,2016))
# thresh_outs=0
# thresh_ins=0
# out_path = '~/github/international-cooperation/output/'
#
# # countries = ['Afghanistan','United States','Russian Federation','North Korea','China','Iraq', 'Egypt','Israel','Japan','United Kingdom','Iran','India','Germany','Vietnam','Australia','Syria','Ukraine','Spain','Taiwan','Georgia','Lebanon','Indonesia',
# # 'Thailand', 'South Korea','Pakistan','Serbia','Italy','Occupied Palestinian Territory','France','Turkey']
#
# rc = ['United States', 'United Kingdom', 'Russian Federation', 'China', 'Japan', 'Germany', 'France', 'Italy', 'Turkey']
# comm_names = ['USA','UK','RUS','CHI','JAP','GER','FRA','ITA','TUR']

# This gets the yearly average for a particular country
# for ct in ['Turkey']:
#     interaction_counts(years, thresh_outs, thresh_ins, out_path, global_=False, source_country=ct, monthly=False)
#     print(ct)

# community_behaviors(rc,years,out_path, 'rc')
# community_behaviors(rc,years,out_path, 'rc')
