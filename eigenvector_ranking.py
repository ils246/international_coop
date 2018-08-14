import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

csfont = {'fontname':'Arial Narrow'}

# --------------------------------------
#   Rank countries by var. explained
# --------------------------------------
def rank_countries(year,countries):

    data = pd.read_csv('~/github/international-cooperation/output/europe_eigen_%d.csv' % year)
    size=len(countries)*2

    # make the array
    d=[]
    for i,r in data.iterrows():
        v=ast.literal_eval(r['vectors'])
        d.append(v)
    arr = np.array(d)
    arr = np.square(arr)
    columns_pairs = [(i,i+1) for i in range(0,size,2)]

    # get the average for cooperative and non-cooperative
    # averages = np.zeros([60,30])
    averages = np.zeros([size, int(size/2)])
    counter=0
    for pair in columns_pairs:
        averages[:,counter] = np.sum(arr[:,[pair[0],pair[1]]],axis=1)
        counter+=1

    # multiply by the lambda
    lambdas=np.array(data['lambda'])
    total = sum(lambdas)
    lambdas = [(i/total)*100 for i in lambdas]


    # corrected = np.zeros([60,30])
    corrected = np.zeros([size,int(size/2)])

    for i in range(int(size/2)):
        corrected[:,i] = lambdas * averages[:,i]

    # get the average of all Eigenvectors
    corrected=np.mean(corrected,axis=0)

    # # get the ranking
    country_ranking = [ (countries[country], corrected[country]) for country in range(int(size/2)) ]
    country_ranking = sorted(country_ranking, key=lambda x : x[1], reverse=True)

    ranking = {}
    for i in range(len(country_ranking)):
        ranking[country_ranking[i][0]]=country_ranking[i][1]
        # ranking[country_ranking[i][0]]=i+1

    # for i in range(30):
    #     print((country_ranking[i][0],country_ranking[i][1]))

    return  ranking
    # return country_ranking


def plot_eigen_rankings(data,countries,name):

    # Colors
    colors = ['#0C4F9E', '#0c989e', '#0C96E8', '#00CAFF', '#0CDFE8',
              '#14EDD0', '#a0afc8', '#99e9ff','#5c5c5c']

    other_color='#E6E6E6'
    subtitle_color='#5c5c5c'

    fig,ax=plt.subplots(figsize=(14,9))

    # I want the rich club countries to be top layer
    # so they are ploted last
    order = [i for i in data.keys() if i not in countries]
    order.extend(countries)

    c=0
    for key in order:

        y = data[key]

        if key not in countries:
            if (key == 'Iraq'):

                cs=['#E6E6E6' if i < 0.40 else '#FF8140' for i in data[key]]
                plt.plot(data[key], markersize=0, lw=2, color=other_color, alpha=0.4, label=key)

                for i in range(20):
                    plt.plot(i,data[key][i], 's', markersize=6, lw=2, color=cs[i], alpha=0.4, label=key)


                plt.annotate('Iraq',(8+0.25,data[key][8]+0.007), size=11, **csfont )
                plt.annotate('(Invasion of Iraq)',(8+0.25,data[key][8]-0.008), style='italic', size=8.5, color=subtitle_color, **csfont)

            elif (key == 'Afghanistan'):

                cs=['#E6E6E6' if i < 0.42 else '#FF8140' for i in data[key]]
                plt.plot(data[key], markersize=0, lw=2, color=other_color, alpha=0.4, label=key)


                for i in range(20):
                    plt.plot(i,data[key][i], 's', markersize=6, lw=2, color=cs[i], alpha=0.4, label=key)

                plt.annotate('Afghanistan',(6+0.25,data[key][6]-0.005), size=11, **csfont)
                plt.annotate('(Invasion of Afgh.)',(6+0.2,data[key][6]-0.02),size=8, style='italic', color=subtitle_color, **csfont)

            elif (key == 'Lebanon'):

                cs=['#E6E6E6' if i < 0.40 else '#FF8140' for i in data[key]]
                plt.plot(data[key], markersize=0, lw=2, color=other_color, alpha=0.4, label=key)


                for i in range(20):
                    plt.plot(i,data[key][i], 's', markersize=6, lw=2, color=cs[i], alpha=0.4, label=key)

                plt.annotate('Lebanon',(11+0.2,data[key][11]), **csfont)
                plt.annotate('(war)',(11+0.25,(data[key][11]-0.015)), style='italic', size=8, color=subtitle_color, **csfont)


            elif (key == 'Palestine'):

                cs=['#E6E6E6' if i < 0.42 else '#FF8140' for i in data[key]]
                plt.plot(data[key], markersize=0, lw=2, color=other_color, alpha=0.4, label=key)

                for i in range(20):
                    plt.plot(i,data[key][i], 's', markersize=6, lw=2, color=cs[i], alpha=0.4, label=key)

                plt.annotate('Palestine',(14+0.2,data[key][14]), **csfont)
                plt.annotate('(Passover masacre)',(14+0.2,(data[key][14]-0.016)), size=8, style='italic', color=subtitle_color, **csfont)

            elif (key == 'Syria'):

                cs=['#E6E6E6' if i < 0.41 else '#FF8140' for i in data[key]]
                plt.plot(data[key], markersize=0, lw=2, color=other_color, alpha=0.4, label=key)

                for i in range(20):
                    plt.plot(i,data[key][i], 's', markersize=6, lw=2, color=cs[i], alpha=0.4, label=key)

                plt.annotate('Syria',(17+0.2,0.47), **csfont)
                plt.annotate('TBD',(17+0.35,0.455), size=8, style='italic', color=subtitle_color, **csfont)

            elif (key == 'Georgia'):

                cs=['#E6E6E6' if i < 0.42 else '#FF8140' for i in data[key]]
                plt.plot(data[key], markersize=0, lw=2, color=other_color, alpha=0.4, label=key)

                for i in range(20):
                    plt.plot(i,data[key][i], 's', markersize=6, lw=2, color=cs[i], alpha=0.4, label=key)

                plt.annotate('Georgia',(13+0.2,data[key][13]), **csfont)
                plt.annotate('(Russian war)',(13+0.26,(data[key][13]-0.017)), style='italic', size=8, color=subtitle_color, **csfont)

            else:
                plt.plot(data[key], 's', markersize=3, color=other_color, alpha=1, label=key)

                plt.plot(data[key], markersize=0, lw=2, color=other_color, alpha=0.4, label=key)
        else:
            plt.plot(data[key], 's', markersize=3, color=colors[c], alpha=1, label=key)

            plt.plot(data[key], markersize=0, lw=2, color=colors[c], alpha=0.4, label=key)
            c+=1

    y_labs=[]
    for i in countries:
        y_labs.append((i,data[i][0]))

    plt.title('Global Eigenbehaviors', size=15, **csfont)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.spines['top'].set_visible(False)
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,         # ticks along the top edge are off
        ) # labels along the bottom edge are off

    y_tick_labs.append(str(0.45))
    y_tick_labs.append(str(0))

    # plt.yticks(y_tick_coors,y_tick_labs, size=11, **csfont)
    # plt.xticks(range(len(data['China'])),range(1995,2016), size=11, rotation=90, **csfont)
    plt.savefig(name, bbox_inches='tight', dpi=800)
    # plt.show()



# ---------------------------------------------------------
#  Run the code
# ---------------------------------------------------------

years = range(1995,2016)
countries = ['Afghanistan', 'United States', 'Russia', 'North Korea', 'China', 'Iraq', 'Egypt', 'Israel', 'Japan', 'United Kingdom', 'Iran', 'India', 'Germany', 'Vietnam', 'Australia', 'Syria', 'Ukraine', 'Spain', 'Taiwan', 'Georgia', 'Lebanon', 'Indonesia', 'Thailand', 'South Korea', 'Pakistan', 'Serbia', 'Italy', 'Palestine', 'France', 'Turkey']

# countries=['United States', 'China']

# Rank countries
ranks = {i:[] for i in countries}
for year in years:
    r = rank_countries(year,countries)

    for c in countries:
        ranks[c].append(r[c])

# ---------------------------------------------------------
# Plot eigenbehaviors coefficients per country per year
# ---------------------------------------------------------

rc = [ 'Russian Federation', 'United Kingdom', 'Germany', 'France']
plot_eigen_rankings(ranks, rc, 'europe_eigen_plot_coeffs.png')

# ---------------------------------------------------------
#  Joy plot of eigenvector coefficients
# ---------------------------------------------------------
import joypy

# get the variances of the dists
order=[]
for key in ranks.keys():
    v=np.var(ranks[key])
    order.append((key,v))

order = sorted(order, key=lambda x: x[1])

# get the dictionary in dataframe shape
df = pd.DataFrame()
for j in order:
    df[j[0]]=ranks[j[0]]

fig,axes=joypy.joyplot(df,colormap=plt.get_cmap('YlGnBu'), bins=20)
plt.show()
