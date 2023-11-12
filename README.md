# Classification-Model-Development

This analysis focuses on identifying the most crucial elements for successful cross-selling new subscription to clients. A model to determine if a consumer will effectively cross-sell is included in this investigation.

1
# importing libraries
2
import pandas            as pd                       # data science essentials
3
import matplotlib.pyplot as plt                      # data visualization
4
import seaborn           as sns                      # enhanced data viz
5
import numpy             as np                       # Linear Algebra 
6
​
7
# Machine Learning Modules
8
from sklearn.model_selection import train_test_split # train-test split
9
from sklearn.linear_model import LogisticRegression  # logistic regression
10
import statsmodels.formula.api as smf                # logistic regression
11
from sklearn.metrics import confusion_matrix         # confusion matrix
12
from sklearn.metrics import roc_auc_score            # auc score
13
from sklearn.neighbors import KNeighborsClassifier   # KNN for classification
14
from sklearn.neighbors import KNeighborsRegressor    # KNN for regression
15
from sklearn.preprocessing import StandardScaler     # standard scaler
16
from sklearn.tree import DecisionTreeClassifier      # classification trees
17
from sklearn.tree import plot_tree                   # tree plots
18
from sklearn.model_selection import RandomizedSearchCV
19
from sklearn.metrics import make_scorer
20
from io import StringIO
21
from sklearn.ensemble import RandomForestClassifier     # random forest
22
from sklearn.ensemble import GradientBoostingClassifier # gbm
23
​
24
​
25
# specifying the path and file name
26
file = './Cross_Sell_Success_Dataset_2023.xlsx'
27
​
28
# reading the file into Python
29
cs = pd.read_excel(io = file)
30
​
31
# setting pandas print options
32
pd.set_option('display.max_rows', 500)
33
pd.set_option('display.max_columns', 500)
34
pd.set_option('display.width', 1000)
35
pd.set_option('display.max_colwidth', 100)
36
​
37
#obtaining the top 10 rows as output to enhance data understanding
38
cs.head(n = 10)
39
​
CROSS_SELL_SUCCESS	EMAIL	REVENUE	TOTAL_MEALS_ORDERED	UNIQUE_MEALS_PURCH	CONTACTS_W_CUSTOMER_SERVICE	PRODUCT_CATEGORIES_VIEWED	AVG_TIME_PER_SITE_VISIT	CANCELLATIONS_AFTER_NOON	PC_LOGINS	MOBILE_LOGINS	WEEKLY_PLAN	LATE_DELIVERIES	AVG_PREP_VID_TIME	LARGEST_ORDER_SIZE	AVG_MEAN_RATING	TOTAL_PHOTOS_VIEWED
0	1	steffon.baratheon@yahoo.com	4920.0	493	9	1	10	265.6	5	5	2	0	0	137.41	6	2.894737	456
1	0	harlon.greyjoy@visa.com	6150.0	361	9	1	6	247.0	2	5	1	0	0	120.20	5	2.631579	680
2	0	monster@protonmail.com	3435.0	278	6	1	4	164.4	0	6	1	5	0	127.00	3	3.684211	145
3	1	damon.lannister.(lord)@yahoo.com	3330.0	269	8	1	2	176.0	5	5	2	0	0	129.78	6	3.157895	418
4	1	raynald.westerling@jnj.com	3427.5	276	7	1	10	164.6	0	6	1	14	0	34.42	3	3.157895	174
5	0	rollam.westerling@jnj.com	2917.5	246	7	1	2	240.3	1	5	1	20	0	240.90	8	4.473684	16
6	0	utt@passport.com	3901.5	316	10	1	5	183.8	2	5	2	46	4	140.01	6	3.157895	584
7	0	bandy@aol.com	5000.0	251	10	1	1	269.6	4	5	2	40	0	143.00	8	2.894737	1095
8	1	rickard.karstark@live.com	3090.0	256	7	1	5	218.4	2	6	1	30	0	120.34	6	2.894737	198
9	1	androw.frey@protonmail.com	5745.0	297	8	1	6	266.8	1	6	1	32	0	140.88	8	2.894737	664
After loading the libraries, I will read the Excel file into Python from the working directory. Set pandas print parameters to make sure I can produce and examine data on all variables, not just the first few and the final few. After loading the file, I will examine the dataset's structure, variable types, and other details to get a first impression.

1
########################################
2
# visual_cm
3
########################################
4
def visual_cm(true_y, pred_y, labels = None):
5
    """
6
Creates a visualization of a confusion matrix.
7
​
8
PARAMETERS
9
----------
10
true_y : true values for the response variable
11
pred_y : predicted values for the response variable
12
labels : , default None
13
    """
14
    # visualizing the confusion matrix
15
​
16
    # setting labels
17
    lbls = labels
18
    
19
​
20
    # declaring a confusion matrix object
21
    cm = confusion_matrix(y_true = true_y,
22
                          y_pred = pred_y)
23
​
24
​
25
    # heatmap
26
    sns.heatmap(cm,
27
                annot       = True,
28
                xticklabels = lbls,
29
                yticklabels = lbls,
30
                cmap        = 'Blues',
31
                fmt         = 'g')
32
​
33
​
34
    plt.xlabel('Predicted')
35
    plt.ylabel('Actual')
36
    plt.title('Confusion Matrix of the Classifier')
37
    plt.show()
1
#Finding the dataset's missing or null values
2
cs.isna().sum()
CROSS_SELL_SUCCESS             0
EMAIL                          0
REVENUE                        0
TOTAL_MEALS_ORDERED            0
UNIQUE_MEALS_PURCH             0
CONTACTS_W_CUSTOMER_SERVICE    0
PRODUCT_CATEGORIES_VIEWED      0
AVG_TIME_PER_SITE_VISIT        0
CANCELLATIONS_AFTER_NOON       0
PC_LOGINS                      0
MOBILE_LOGINS                  0
WEEKLY_PLAN                    0
LATE_DELIVERIES                0
AVG_PREP_VID_TIME              0
LARGEST_ORDER_SIZE             0
AVG_MEAN_RATING                0
TOTAL_PHOTOS_VIEWED            0
dtype: int64
The dataset contains no missing values. Next, I will analyze the descriptive statistics of each numeric variable.

1
#defining and evaluating every numerical column
2
cs.describe().round(2)
CROSS_SELL_SUCCESS	REVENUE	TOTAL_MEALS_ORDERED	UNIQUE_MEALS_PURCH	CONTACTS_W_CUSTOMER_SERVICE	PRODUCT_CATEGORIES_VIEWED	AVG_TIME_PER_SITE_VISIT	CANCELLATIONS_AFTER_NOON	PC_LOGINS	MOBILE_LOGINS	WEEKLY_PLAN	LATE_DELIVERIES	AVG_PREP_VID_TIME	LARGEST_ORDER_SIZE	AVG_MEAN_RATING	TOTAL_PHOTOS_VIEWED
count	1946.00	1946.00	1946.00	1946.00	1946.0	1946.00	1946.00	1946.00	1946.00	1946.00	1946.00	1946.00	1946.00	1946.00	1946.00	1946.00
mean	0.68	2107.29	74.63	6.98	4.9	5.38	150.56	1.57	5.52	1.48	11.33	2.05	99.60	4.44	3.55	113.15
std	0.47	1138.29	55.31	2.28	2.5	3.04	49.45	1.61	0.58	0.53	13.57	3.79	62.34	1.55	0.61	177.15
min	0.00	131.00	11.00	1.00	1.0	1.00	33.40	0.00	4.00	0.00	0.00	0.00	10.33	1.00	1.32	1.00
25%	0.00	1350.00	39.00	5.00	3.0	3.00	114.40	0.00	5.00	1.00	1.00	0.00	72.00	3.00	3.16	10.00
50%	1.00	1740.00	60.00	7.00	5.0	5.00	145.60	1.00	6.00	1.00	7.00	0.00	94.16	4.00	3.42	17.00
75%	1.00	2670.00	95.00	8.00	7.0	8.00	173.78	2.00	6.00	2.00	13.00	3.00	117.29	5.00	3.95	174.00
max	1.00	8793.75	493.00	18.00	19.0	10.00	564.20	13.00	7.00	3.00	52.00	18.00	1645.60	11.00	5.00	1600.00
I noticed some intriguing findings based on the results:

The top 25% of consumers earn significantly more revenue than the other 75%. 75% of Apprentice Chef's clients produce income ranging from 131 to 2670. Between 2670 and 8800 is generated by the top 25% of clients. This is a wide range, and in light of this data, I infer that in addition to individual users, we may also have corporate consumers for whom we fulfill substantial orders. If this were a real-world scenario, I would carefully research our clientele. This is a project, so keeping that in mind is already a noteworthy remark.

Also, there is an exponential gap between the number of meals ordered by the top 25% of clients and the rest. As this is an experiment rather than a real-world job, I lack the tools to examine Apprentice Chef's clientele thoroughly. However, this finding supports my initial hypothesis that the organization has individual and corporate clients based on the distribution of values in the "REVENUE" column. The average number of meals ordered is 75, while clients in the third quartile order about 95. As compared to clients in the 75th percentile, the top 25% of customers order up to 493 meals. In conclusion, we have both individuals and businesses as customers.

1
#Details on each variable
2
cs.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1946 entries, 0 to 1945
Data columns (total 17 columns):
 #   Column                       Non-Null Count  Dtype  
---  ------                       --------------  -----  
 0   CROSS_SELL_SUCCESS           1946 non-null   int64  
 1   EMAIL                        1946 non-null   object 
 2   REVENUE                      1946 non-null   float64
 3   TOTAL_MEALS_ORDERED          1946 non-null   int64  
 4   UNIQUE_MEALS_PURCH           1946 non-null   int64  
 5   CONTACTS_W_CUSTOMER_SERVICE  1946 non-null   int64  
 6   PRODUCT_CATEGORIES_VIEWED    1946 non-null   int64  
 7   AVG_TIME_PER_SITE_VISIT      1946 non-null   float64
 8   CANCELLATIONS_AFTER_NOON     1946 non-null   int64  
 9   PC_LOGINS                    1946 non-null   int64  
 10  MOBILE_LOGINS                1946 non-null   int64  
 11  WEEKLY_PLAN                  1946 non-null   int64  
 12  LATE_DELIVERIES              1946 non-null   int64  
 13  AVG_PREP_VID_TIME            1946 non-null   float64
 14  LARGEST_ORDER_SIZE           1946 non-null   int64  
 15  AVG_MEAN_RATING              1946 non-null   float64
 16  TOTAL_PHOTOS_VIEWED          1946 non-null   int64  
dtypes: float64(4), int64(12), object(1)
memory usage: 258.6+ KB
1
# finding the correlation between the cross-sell success and the rest of the fields
2
df_corr = cs.corr(method = 'pearson').round(decimals=2)
3
​
4
df_corr['CROSS_SELL_SUCCESS'].sort_values(ascending = False)
CROSS_SELL_SUCCESS             1.00
CANCELLATIONS_AFTER_NOON       0.14
MOBILE_LOGINS                  0.06
UNIQUE_MEALS_PURCH             0.04
PC_LOGINS                      0.04
AVG_TIME_PER_SITE_VISIT        0.03
LATE_DELIVERIES                0.02
LARGEST_ORDER_SIZE             0.02
TOTAL_MEALS_ORDERED            0.01
AVG_PREP_VID_TIME              0.01
TOTAL_PHOTOS_VIEWED            0.01
REVENUE                        0.00
CONTACTS_W_CUSTOMER_SERVICE   -0.00
PRODUCT_CATEGORIES_VIEWED      0.00
WEEKLY_PLAN                   -0.01
AVG_MEAN_RATING               -0.04
Name: CROSS_SELL_SUCCESS, dtype: float64
Create histograms to examine the distribution of each continuous variable.

1
# developing a histogram using DISPLOT
2
sns.displot(data   = cs,
3
           x       = 'REVENUE',
4
           height  = 5,
5
           aspect  = 2)
6
​
7
# displaying the histogram
8
plt.show()

1
# log transforming REVENUE and saving it to the dataset
2
cs['log_REVENUE'] = np.log10(cs['REVENUE'])
3
​
4
# developing a histogram using DISPLOT for REVENUE after log transformation
5
sns.displot(data   = cs,
6
            x      = 'log_REVENUE',
7
            height = 5,
8
            aspect = 2)
9
​
10
​
11
# displaying the histogram
12
plt.show()

The first finding from the distribution is that revenue is favorably skewed. The revenue variable tends to become more regularly distributed after log transformation than the raw data.

1
# developing a histogram using DISPLOT
2
# AVG_TIME_PER_SITE_VISIT
3
sns.displot(data   = cs,
4
            x      = 'AVG_TIME_PER_SITE_VISIT',
5
            height = 5,
6
            aspect = 2)
7
​
8
# displaying the histogram
9
plt.show()

1
# log transforming and saving it to the dataset
2
cs['log_AVG_TIME_PER_SITE_VISIT'] = np.log10(cs['AVG_TIME_PER_SITE_VISIT'])
3
​
4
# developing a histogram using DISPLOT
5
sns.displot(data   = cs,
6
            x      = 'log_AVG_TIME_PER_SITE_VISIT',
7
            height = 5,
8
            aspect = 2)
9
​
10
​
11
# displaying the histogram
12
plt.show()

1
# developing a histogram using DISPLOT
2
# AVG_PREP_VID_TIME
3
sns.displot(data   = cs,
4
            x      = 'AVG_PREP_VID_TIME',
5
            height = 5,
6
            aspect = 2)
7
​
8
​
9
​
10
​
11
# displaying the histogram
12
plt.show()

1
# log transforming and saving it to the dataset
2
cs['log_AVG_PREP_VID_TIME'] = np.log10(cs['AVG_PREP_VID_TIME'])
3
​
4
# developing a histogram using DISPLOT
5
sns.displot(data   = cs,
6
            x      = 'log_AVG_PREP_VID_TIME',
7
            height = 5,
8
            aspect = 2)
9
​
10
​
11
# displaying the histogram
12
plt.show()

1
# developing a histogram using DISPLOT
2
# AVG_MEAN_RATING
3
sns.displot(data   = cs,
4
            x      = 'AVG_MEAN_RATING',
5
            height = 5,
6
            aspect = 2)
7
​
8
​
9
​
10
​
11
# displaying the histogram
12
plt.show()

1
# log transforming and saving it to the dataset
2
cs['log_AVG_MEAN_RATING'] = np.log10(cs['AVG_MEAN_RATING'])
3
​
4
# developing a histogram using DISPLOT
5
sns.displot(data   = cs,
6
            x      = 'log_AVG_MEAN_RATING',
7
            height = 5,
8
            aspect = 2)
9
​
10
​
11
# displaying the histogram
12
plt.show()

As can be observed from the distribution above, AVG_TIME_PER_SITE_VISIT is skewed positive with outliers on the right side. After applying log transformation, the AVG_TIME_PER_SITE_VISIT variable tends to become more generally distributed than the original data. Also, AVG_PREP_VID_TIME and AVG_PREP_VID_TIME were skewed positive. After applying log transformation, AVG_PREP_VID_TIME and AVG_PREP_VID_TIME variables become more normally distributed than the original data.

Checking the correlation between CROSS SELL SUCCESS and continuous variables.

1
# building a correlation matrix based on continuous features
2
cont_corr = cs.loc[ : , ['REVENUE',
3
                             'AVG_TIME_PER_SITE_VISIT',
4
                             'AVG_PREP_VID_TIME',
5
                             'AVG_MEAN_RATING',
6
                             'log_AVG_PREP_VID_TIME',
7
                             'log_REVENUE',
8
                             'log_AVG_TIME_PER_SITE_VISIT',
9
                             'log_AVG_MEAN_RATING',
10
                             'CROSS_SELL_SUCCESS']  ].corr(method = 'pearson')\
11
                                                     .round(decimals = 2)
12
​
13
​
14
# calculating the correlations between each continuous column and CROSS_SELL_SUCCESS
15
cont_corr.loc['CROSS_SELL_SUCCESS', : ].sort_values(ascending = False)
CROSS_SELL_SUCCESS             1.00
AVG_TIME_PER_SITE_VISIT        0.03
log_AVG_TIME_PER_SITE_VISIT    0.03
log_AVG_PREP_VID_TIME          0.02
AVG_PREP_VID_TIME              0.01
log_REVENUE                    0.01
REVENUE                        0.00
log_AVG_MEAN_RATING           -0.03
AVG_MEAN_RATING               -0.04
Name: CROSS_SELL_SUCCESS, dtype: float64
Continuous variables have weak correlation with CROSS_SELL_SUCCESS.

Analysising the correlation between the explanatory variables and the response variable (CROSS SELL SUCCESS).

1
# developing a correlation matrix
2
interval_corr = cs.drop(['REVENUE',
3
                             'AVG_TIME_PER_SITE_VISIT',
4
                             'AVG_PREP_VID_TIME',
5
                             'AVG_MEAN_RATING',
6
                             'log_AVG_PREP_VID_TIME',
7
                             'log_REVENUE',
8
                             'log_AVG_TIME_PER_SITE_VISIT',
9
                             'log_AVG_MEAN_RATING',], axis = 1)\
10
                    .corr(method = 'pearson')\
11
                    .round(decimals = 2)
12
​
13
# checking the correlations
14
interval_corr.loc['CROSS_SELL_SUCCESS', : ]\
15
                    .sort_values(ascending = False)
CROSS_SELL_SUCCESS             1.00
CANCELLATIONS_AFTER_NOON       0.14
MOBILE_LOGINS                  0.06
UNIQUE_MEALS_PURCH             0.04
PC_LOGINS                      0.04
LATE_DELIVERIES                0.02
LARGEST_ORDER_SIZE             0.02
TOTAL_MEALS_ORDERED            0.01
TOTAL_PHOTOS_VIEWED            0.01
CONTACTS_W_CUSTOMER_SERVICE   -0.00
PRODUCT_CATEGORIES_VIEWED      0.00
WEEKLY_PLAN                   -0.01
Name: CROSS_SELL_SUCCESS, dtype: float64
CANCELLATIONS_BEFORE_NOON

MOBILE_LOGINS

UNIQUE_MEALS_PURCH

PC_LOGINS

These variables have a Positive correlation with the success of cross-selling promotion, will use some of them in feature engineering.

1  Feature Engineering
1
# total number of logins
2
cs['Total_Logins'] = cs['MOBILE_LOGINS'] + cs['PC_LOGINS']
3
​
4
# printing value counts
5
print(f"""
6
--------------------------------
7
Total number of logins
8
--------------------------------
9
{cs['Total_Logins'].value_counts(normalize = False).sort_index()}
10
​
11
""")

--------------------------------
Total number of logins
--------------------------------
5     30
6    487
7    927
8    472
9     30
Name: Total_Logins, dtype: int64


1
# developing a correlation matrix
2
logins_corr = cs.loc[ : , ['PC_LOGINS',
3
                             'MOBILE_LOGINS',
4
                             'Total_Logins',
5
                             'CROSS_SELL_SUCCESS']  ].corr(method = 'pearson')\
6
                                              .round(decimals = 2)
7
​
8
​
9
# printing value counts and correlations
10
print(f"""
11
Total Logins
12
=================
13
​
14
-------------------------------
15
Total_Logins Correlations
16
-------------------------------
17
{logins_corr['CROSS_SELL_SUCCESS']}
18
​
19
""")

Total Logins
=================

-------------------------------
Total_Logins Correlations
-------------------------------
PC_LOGINS             0.04
MOBILE_LOGINS         0.06
Total_Logins          0.07
CROSS_SELL_SUCCESS    1.00
Name: CROSS_SELL_SUCCESS, dtype: float64


Total_Logins have a Positive correlation with CROSS_SELL_SUCCESS.

1
# counting the number of zeroes
2
​
3
rev_zeroes         = len(cs['REVENUE'][cs["REVENUE"] == 0]) # REVENUE
4
tmo_zeroes         = len(cs['TOTAL_MEALS_ORDERED'][cs["TOTAL_MEALS_ORDERED"] == 0]) # TOTAL_MEALS_ORDERED 
5
purch_zeroes       = len(cs['UNIQUE_MEALS_PURCH'][cs["UNIQUE_MEALS_PURCH"] == 0]) # UNIQUE_MEALS_PURCH 
6
customer_zeroes    = len(cs['CONTACTS_W_CUSTOMER_SERVICE'][cs["CONTACTS_W_CUSTOMER_SERVICE"] == 0]) # CONTACTS_W_CUSTOMER_SERVICE 
7
pd_cat_zeroes      = len(cs['PRODUCT_CATEGORIES_VIEWED'][cs["PRODUCT_CATEGORIES_VIEWED"] == 0]) # PRODUCT_CATEGORIES_VIEWED
8
site_zeroes        = len(cs['AVG_TIME_PER_SITE_VISIT'][cs["AVG_TIME_PER_SITE_VISIT"] == 0]) # AVG_TIME_PER_SITE_VISIT
9
cancel_zeroes      = len(cs['CANCELLATIONS_AFTER_NOON'][cs["CANCELLATIONS_AFTER_NOON"] == 0]) # CANCELLATIONS_AFTER_NOON
10
pc_zeroes          = len(cs['PC_LOGINS'][cs["PC_LOGINS"] == 0]) # PC_LOGINS 
11
mobile_zeroes      = len(cs['MOBILE_LOGINS'][cs["MOBILE_LOGINS"] == 0]) # MOBILE_LOGINS
12
plan_zeroes        = len(cs['WEEKLY_PLAN'][cs["WEEKLY_PLAN"] == 0]) # WEEKLY_PLAN
13
late_dv_zeroes     = len(cs['LATE_DELIVERIES '][cs["LATE_DELIVERIES "] == 0]) # LATE_DELIVERIES
14
vid_zeroes         = len(cs['AVG_PREP_VID_TIME'][cs["AVG_PREP_VID_TIME"] == 0]) # AVG_PREP_VID_TIME
15
order_zeroes       = len(cs['LARGEST_ORDER_SIZE'][cs["LARGEST_ORDER_SIZE"] == 0]) # LARGEST_ORDER_SIZE 
16
rating_zeroes      = len(cs['AVG_MEAN_RATING'][cs["AVG_MEAN_RATING"] == 0]) # AVG_MEAN_RATING
17
photo_zeroes       = len(cs['TOTAL_PHOTOS_VIEWED'][cs["TOTAL_PHOTOS_VIEWED"] == 0]) # TOTAL_PHOTOS_VIEWED
18
​
19
# printing a table of the results
20
print(f"""
21
                 No\t\tYes
22
               ---------------------
23
Revenue              | {rev_zeroes}\t\t{len(cs) - rev_zeroes}
24
Meals Ordered        | {tmo_zeroes}\t\t{len(cs) - tmo_zeroes}
25
Unique Meals         | {purch_zeroes}\t\t{len(cs) - purch_zeroes}
26
Customer Service     | {customer_zeroes}\t\t{len(cs) - customer_zeroes}
27
Product Categories   | {pd_cat_zeroes}\t\t{len(cs) - pd_cat_zeroes}
28
Site Visit           | {site_zeroes}\t\t{len(cs) - site_zeroes}
29
Cancellations        | {cancel_zeroes}\t\t{len(cs) - cancel_zeroes}
30
PC Logins            | {pc_zeroes}\t\t{len(cs) - pc_zeroes}
31
Mobile Logins        | {mobile_zeroes}\t\t{len(cs) - mobile_zeroes}
32
Weekly Plans         | {plan_zeroes}\t\t{len(cs) - plan_zeroes}
33
Late Delivery        | {late_dv_zeroes}\t\t{len(cs) - late_dv_zeroes}
34
Video Time           | {vid_zeroes}\t\t{len(cs) - vid_zeroes}
35
Largest Order        | {order_zeroes}\t\t{len(cs) - order_zeroes}
36
Ratings              | {rating_zeroes}\t\t{len(cs) - rating_zeroes}
37
Photos               | {photo_zeroes}\t\t{len(cs) - photo_zeroes} """)

                 No		Yes
               ---------------------
Revenue              | 0		1946
Meals Ordered        | 0		1946
Unique Meals         | 0		1946
Customer Service     | 0		1946
Product Categories   | 0		1946
Site Visit           | 0		1946
Cancellations        | 572		1374
PC Logins            | 0		1946
Mobile Logins        | 16		1930
Weekly Plans         | 468		1478
Late Delivery        | 1433		513
Video Time           | 0		1946
Largest Order        | 0		1946
Ratings              | 0		1946
Photos               | 0		1946 
Splitting Emails. It's crucial to divide the emails so that we may use them effectively in various groups. This will help us track down our consumers more effectively and gain more valuable information.

1
# placeholder list
2
email_lst = []
3
​
4
# looping over each email address
5
for index, col in cs.iterrows():
6
    
7
    # splitting email domain at '@'
8
    email_info = cs.loc[index,'EMAIL'].split(sep = '@')
9
​
10
    
11
    # appending placeholder_lst with the results
12
    email_lst.append(email_info)
13
    
14
# converting placeholder_lst into a DataFrame 
15
df_temp = pd.DataFrame(email_lst)
1
# checking results 
2
df_temp
0	1
0	steffon.baratheon	yahoo.com
1	harlon.greyjoy	visa.com
2	monster	protonmail.com
3	damon.lannister.(lord)	yahoo.com
4	raynald.westerling	jnj.com
...	...	...
1941	wat.(barleycorn)	live.com
1942	hendry.bracken	exxon.com
1943	aenys.i.targaryen	passport.com
1944	leo.blackbar	pg.com
1945	anguy	boeing.com
1946 rows × 2 columns

1
#Setting Personal, Professional and Junk Emails
2
personal      = ['@gmail.com',
3
                               '@yahoo.com',
4
                               '@protonmail.com']
5
​
6
professional                 = ['@mmm.com',
7
                               '@amex.com',
8
                               '@apple.com',
9
                               '@boeing.com',
10
                               '@caterpillar.com',
11
                               '@chevron.com',
12
                               '@cisco.com',
13
                               '@cocacola.com',
14
                               '@disney.com',
15
                               '@dupont.com',
16
                               '@exxon.com',
17
                               '@ge.org',
18
                               '@goldmansacs.com',
19
                               '@homedepot.com',
20
                               '@ibm.com',
21
                               '@intel.com',
22
                               '@jnj.com',
23
                               '@jpmorgan.com',
24
                               '@mcdonalds.com',
25
                               '@merck.com',
26
                               '@microsoft.com',
27
                               '@nike.com',
28
                               '@pfizer.com',
29
                               '@pg.com',
30
                               '@travelers.com',
31
                               '@unitedtech.com',
32
                               '@unitedhealth.com',
33
                               '@verizon.com',
34
                               '@visa.com',
35
                               '@walmart.com']
36
​
37
junk                        = ['@me.com',
38
                            '@aol.com',
39
                            '@hotmail.com',
40
                            '@live.com',
41
                            '@msn.com',
42
                            '@passport.com']
1
# placeholder list
2
email_lst2 = []
3
​
4
​
5
# looping to group observations by domain type
6
for domain in df_temp[1]:
7
        if '@' + domain in personal:
8
            email_lst2.append('personal')
9
            
10
        elif '@' + domain in professional:
11
            email_lst2.append('professional')
12
           
13
        elif '@' + domain in junk:
14
            email_lst2.append('junk')
15
            
16
        else:
17
            print('Unknown')
18
​
19
​
20
# concatenating with original DataFrame
21
cs['DOMAIN'] = pd.Series(email_lst2)
22
​
23
# checking results
24
cs['DOMAIN'].value_counts()
personal        861
professional    696
junk            389
Name: DOMAIN, dtype: int64
1
# checking results
2
cs.head()
CROSS_SELL_SUCCESS	EMAIL	REVENUE	TOTAL_MEALS_ORDERED	UNIQUE_MEALS_PURCH	CONTACTS_W_CUSTOMER_SERVICE	PRODUCT_CATEGORIES_VIEWED	AVG_TIME_PER_SITE_VISIT	CANCELLATIONS_AFTER_NOON	PC_LOGINS	MOBILE_LOGINS	WEEKLY_PLAN	LATE_DELIVERIES	AVG_PREP_VID_TIME	LARGEST_ORDER_SIZE	AVG_MEAN_RATING	TOTAL_PHOTOS_VIEWED	log_REVENUE	log_AVG_TIME_PER_SITE_VISIT	log_AVG_PREP_VID_TIME	log_AVG_MEAN_RATING	Total_Logins	DOMAIN
0	1	steffon.baratheon@yahoo.com	4920.0	493	9	1	10	265.6	5	5	2	0	0	137.41	6	2.894737	456	3.691965	2.424228	2.138018	0.461609	7	personal
1	0	harlon.greyjoy@visa.com	6150.0	361	9	1	6	247.0	2	5	1	0	0	120.20	5	2.631579	680	3.788875	2.392697	2.079904	0.420216	6	professional
2	0	monster@protonmail.com	3435.0	278	6	1	4	164.4	0	6	1	5	0	127.00	3	3.684211	145	3.535927	2.215902	2.103804	0.566344	7	personal
3	1	damon.lannister.(lord)@yahoo.com	3330.0	269	8	1	2	176.0	5	5	2	0	0	129.78	6	3.157895	418	3.522444	2.245513	2.113208	0.499398	7	personal
4	1	raynald.westerling@jnj.com	3427.5	276	7	1	10	164.6	0	6	1	14	0	34.42	3	3.157895	174	3.534977	2.216430	1.536811	0.499398	7	professional
1
#Creating dummies for Emails
2
Dummy      = pd.get_dummies(cs['DOMAIN'])
1
cs        = cs.drop('EMAIL', axis = 1)
2
cs        = cs.drop('DOMAIN', axis = 1)
1
#joining column
2
cs        = cs.join([Dummy])
1
# checking results
2
cs
CROSS_SELL_SUCCESS	REVENUE	TOTAL_MEALS_ORDERED	UNIQUE_MEALS_PURCH	CONTACTS_W_CUSTOMER_SERVICE	PRODUCT_CATEGORIES_VIEWED	AVG_TIME_PER_SITE_VISIT	CANCELLATIONS_AFTER_NOON	PC_LOGINS	MOBILE_LOGINS	WEEKLY_PLAN	LATE_DELIVERIES	AVG_PREP_VID_TIME	LARGEST_ORDER_SIZE	AVG_MEAN_RATING	TOTAL_PHOTOS_VIEWED	log_REVENUE	log_AVG_TIME_PER_SITE_VISIT	log_AVG_PREP_VID_TIME	log_AVG_MEAN_RATING	Total_Logins	junk	personal	professional
0	1	4920.000	493	9	1	10	265.6	5	5	2	0	0	137.41	6	2.894737	456	3.691965	2.424228	2.138018	0.461609	7	0	1	0
1	0	6150.000	361	9	1	6	247.0	2	5	1	0	0	120.20	5	2.631579	680	3.788875	2.392697	2.079904	0.420216	6	0	0	1
2	0	3435.000	278	6	1	4	164.4	0	6	1	5	0	127.00	3	3.684211	145	3.535927	2.215902	2.103804	0.566344	7	0	1	0
3	1	3330.000	269	8	1	2	176.0	5	5	2	0	0	129.78	6	3.157895	418	3.522444	2.245513	2.113208	0.499398	7	0	1	0
4	1	3427.500	276	7	1	10	164.6	0	6	1	14	0	34.42	3	3.157895	174	3.534977	2.216430	1.536811	0.499398	7	0	0	1
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
1941	0	2940.000	89	6	13	3	143.0	1	5	2	8	0	53.00	5	2.894737	16	3.468347	2.155336	1.724276	0.461609	7	1	0	0
1942	1	7087.500	155	7	13	9	169.4	0	6	2	0	8	39.00	5	2.631579	302	3.850493	2.228913	1.591065	0.420216	8	0	0	1
1943	0	6562.500	65	9	13	6	187.4	1	5	1	0	9	48.00	6	2.631579	310	3.817069	2.272770	1.681241	0.420216	6	1	0	0
1944	1	5355.000	58	9	18	10	238.5	3	6	1	0	0	31.00	6	2.631579	275	3.728759	2.377488	1.491362	0.420216	7	0	0	1
1945	0	5194.875	96	7	19	10	168.9	2	6	2	0	0	47.00	5	2.368421	5	3.715575	2.227630	1.672098	0.374459	8	0	0	1
1946 rows × 24 columns

Prepare train-test split for statsmodels.

1
# declaring explanatory variables
2
cs_data = cs.drop('CROSS_SELL_SUCCESS', axis = 1)
3
​
4
​
5
# declaring response variable
6
cs_target = cs['CROSS_SELL_SUCCESS']
1
# train-test split with stratification
2
x_train, x_test, y_train, y_test = train_test_split(
3
            cs_data,
4
            cs_target,
5
            test_size    = 0.25,
6
            random_state = 219,
7
            stratify     = cs_target) # preserving balance
8
​
9
​
10
# merging training data for statsmodels
11
cs_train = pd.concat([x_train, y_train], axis = 1)
12
​
13
print(f"""
14
​
15
Response Variable Proportions (Training Set)
16
--------------------------------------------
17
{y_train.value_counts(normalize = True).round(decimals = 2)}
18
​
19
​
20
​
21
Response Variable Proportions (Testing Set)
22
--------------------------------------------
23
{y_test.value_counts(normalize = True).round(decimals = 2)}
24
""")


Response Variable Proportions (Training Set)
--------------------------------------------
1    0.68
0    0.32
Name: CROSS_SELL_SUCCESS, dtype: float64



Response Variable Proportions (Testing Set)
--------------------------------------------
1    0.68
0    0.32
Name: CROSS_SELL_SUCCESS, dtype: float64

1
# instantiating a logistic regression model object
2
logistic_small = smf.logit(formula = """CROSS_SELL_SUCCESS ~ CANCELLATIONS_AFTER_NOON + 
3
                                        PC_LOGINS +
4
                                        MOBILE_LOGINS +
5
                                        Total_Logins +
6
                                        UNIQUE_MEALS_PURCH +
7
                                        professional""",
8
                           data    = cs_train)
9
​
10
​
11
# fitting the model object
12
results_logistic = logistic_small.fit()
13
​
14
​
15
# checking the results SUMMARY
16
results_logistic.summary2() # summary2() has AIC and BIC
Optimization terminated successfully.
         Current function value: 0.594776
         Iterations 9
Model:	Logit	Pseudo R-squared:	0.053
Dependent Variable:	CROSS_SELL_SUCCESS	AIC:	1747.5552
Date:	2023-03-03 23:42	BIC:	1779.2682
No. Observations:	1459	Log-Likelihood:	-867.78
Df Model:	5	LL-Null:	-916.19
Df Residuals:	1453	LLR p-value:	2.4575e-19
Converged:	1.0000	Scale:	1.0000
No. Iterations:	9.0000		
Coef.	Std.Err.	z	P>|z|	[0.025	0.975]
Intercept	-1.9699	0.6145	-3.2059	0.0013	-3.1742	-0.7656
CANCELLATIONS_AFTER_NOON	0.2031	0.0397	5.1121	0.0000	0.1252	0.2810
PC_LOGINS	0.0338	395651.2579	0.0000	1.0000	-775462.1822	775462.2497
MOBILE_LOGINS	0.1633	395651.2579	0.0000	1.0000	-775462.0527	775462.3792
Total_Logins	0.1970	395651.2579	0.0000	1.0000	-775462.0190	775462.4130
UNIQUE_MEALS_PURCH	0.0482	0.0253	1.9074	0.0565	-0.0013	0.0978
professional	0.8856	0.1288	6.8760	0.0000	0.6331	1.1380
1
for val in cs_data:
2
    print(f" {val} + ")
 REVENUE + 
 TOTAL_MEALS_ORDERED + 
 UNIQUE_MEALS_PURCH + 
 CONTACTS_W_CUSTOMER_SERVICE + 
 PRODUCT_CATEGORIES_VIEWED + 
 AVG_TIME_PER_SITE_VISIT + 
 CANCELLATIONS_AFTER_NOON + 
 PC_LOGINS + 
 MOBILE_LOGINS + 
 WEEKLY_PLAN + 
 LATE_DELIVERIES  + 
 AVG_PREP_VID_TIME + 
 LARGEST_ORDER_SIZE + 
 AVG_MEAN_RATING + 
 TOTAL_PHOTOS_VIEWED + 
 log_REVENUE + 
 log_AVG_TIME_PER_SITE_VISIT + 
 log_AVG_PREP_VID_TIME + 
 log_AVG_MEAN_RATING + 
 Total_Logins + 
 junk + 
 personal + 
 professional + 
1
# instantiating a logistic regression model object
2
logistic_full = smf.logit(formula = """  CROSS_SELL_SUCCESS ~
3
                                          REVENUE + 
4
                                          TOTAL_MEALS_ORDERED + 
5
                                          UNIQUE_MEALS_PURCH + 
6
                                          CONTACTS_W_CUSTOMER_SERVICE + 
7
                                          PRODUCT_CATEGORIES_VIEWED + 
8
                                          AVG_TIME_PER_SITE_VISIT + 
9
                                          CANCELLATIONS_AFTER_NOON + 
10
                                          PC_LOGINS + 
11
                                          MOBILE_LOGINS + 
12
                                          WEEKLY_PLAN +
13
                                          AVG_PREP_VID_TIME + 
14
                                          LARGEST_ORDER_SIZE + 
15
                                          AVG_MEAN_RATING + 
16
                                          TOTAL_PHOTOS_VIEWED + 
17
                                          log_REVENUE + 
18
                                          log_AVG_TIME_PER_SITE_VISIT + 
19
                                          log_AVG_PREP_VID_TIME + 
20
                                          log_AVG_MEAN_RATING + 
21
                                          Total_Logins + 
22
                                          junk + 
23
                                          personal + 
24
                                          professional """,
25
                                         data    = cs_train)
26
​
27
# fitting the model object
28
results_full = logistic_full.fit()
29
​
30
​
31
# checking the results SUMMARY
32
results_full.summary2()
Optimization terminated successfully.
         Current function value: 0.566752
         Iterations 7
Model:	Logit	Pseudo R-squared:	0.097
Dependent Variable:	CROSS_SELL_SUCCESS	AIC:	1695.7830
Date:	2023-03-03 23:42	BIC:	1806.7786
No. Observations:	1459	Log-Likelihood:	-826.89
Df Model:	20	LL-Null:	-916.19
Df Residuals:	1438	LLR p-value:	1.8204e-27
Converged:	1.0000	Scale:	1.0000
No. Iterations:	7.0000		
Coef.	Std.Err.	z	P>|z|	[0.025	0.975]
Intercept	-2.1156	nan	nan	nan	nan	nan
REVENUE	-0.0002	0.0002	-1.0888	0.2762	-0.0006	0.0002
TOTAL_MEALS_ORDERED	0.0007	0.0015	0.4476	0.6544	-0.0022	0.0035
UNIQUE_MEALS_PURCH	0.0650	0.0294	2.2096	0.0271	0.0073	0.1226
CONTACTS_W_CUSTOMER_SERVICE	0.0255	0.0264	0.9658	0.3342	-0.0262	0.0772
PRODUCT_CATEGORIES_VIEWED	-0.0040	0.0200	-0.2027	0.8393	-0.0432	0.0351
AVG_TIME_PER_SITE_VISIT	0.0073	0.0063	1.1502	0.2501	-0.0051	0.0196
CANCELLATIONS_AFTER_NOON	0.2127	0.0413	5.1483	0.0000	0.1317	0.2937
PC_LOGINS	0.0170	164217.8085	0.0000	1.0000	-321860.9732	321861.0072
MOBILE_LOGINS	0.1720	164217.8085	0.0000	1.0000	-321860.8182	321861.1622
WEEKLY_PLAN	0.0000	0.0044	0.0080	0.9936	-0.0086	0.0086
AVG_PREP_VID_TIME	-0.0017	0.0024	-0.7025	0.4823	-0.0064	0.0030
LARGEST_ORDER_SIZE	-0.0714	0.0698	-1.0228	0.3064	-0.2081	0.0654
AVG_MEAN_RATING	-0.7857	0.7652	-1.0267	0.3045	-2.2855	0.7141
TOTAL_PHOTOS_VIEWED	0.0000	0.0004	0.0630	0.9498	-0.0008	0.0009
log_REVENUE	0.4623	1.1805	0.3916	0.6954	-1.8514	2.7760
log_AVG_TIME_PER_SITE_VISIT	-0.6290	2.1607	-0.2911	0.7710	-4.8639	3.6058
log_AVG_PREP_VID_TIME	0.2172	0.5961	0.3643	0.7156	-0.9512	1.3855
log_AVG_MEAN_RATING	5.4235	6.0361	0.8985	0.3689	-6.4070	17.2540
Total_Logins	0.1890	164217.8085	0.0000	1.0000	-321860.8012	321861.1792
junk	-1.7114	nan	nan	nan	nan	nan
personal	-0.4324	nan	nan	nan	nan	nan
professional	0.0281	nan	nan	nan	nan	nan
1
#####################
2
# Sample Solution 1 #
3
#####################
4
​
5
# instantiating a logistic regression model object
6
logit_sig = smf.logit(formula = """ CROSS_SELL_SUCCESS ~ 
7
                                          CANCELLATIONS_AFTER_NOON + 
8
                                          log_AVG_TIME_PER_SITE_VISIT + 
9
                                          log_AVG_PREP_VID_TIME + 
10
                                          log_AVG_MEAN_RATING + 
11
                                          Total_Logins +  
12
                                          professional """,
13
                                    data = cs_train)
14
​
15
# fitting the model object
16
logit_sig = logit_sig.fit()
17
​
18
​
19
# checking the results SUMMARY
20
logit_sig.summary2()
Optimization terminated successfully.
         Current function value: 0.595250
         Iterations 5
Model:	Logit	Pseudo R-squared:	0.052
Dependent Variable:	CROSS_SELL_SUCCESS	AIC:	1750.9404
Date:	2023-03-03 23:42	BIC:	1787.9390
No. Observations:	1459	Log-Likelihood:	-868.47
Df Model:	6	LL-Null:	-916.19
Df Residuals:	1452	LLR p-value:	2.2319e-18
Converged:	1.0000	Scale:	1.0000
No. Iterations:	5.0000		
Coef.	Std.Err.	z	P>|z|	[0.025	0.975]
Intercept	-2.5502	1.4140	-1.8034	0.0713	-5.3216	0.2213
CANCELLATIONS_AFTER_NOON	0.2064	0.0397	5.2054	0.0000	0.1287	0.2841
log_AVG_TIME_PER_SITE_VISIT	0.6102	0.4854	1.2573	0.2086	-0.3411	1.5615
log_AVG_PREP_VID_TIME	-0.1359	0.2768	-0.4910	0.6234	-0.6783	0.4066
log_AVG_MEAN_RATING	-0.4452	0.8511	-0.5230	0.6009	-2.1132	1.2229
Total_Logins	0.2731	0.0750	3.6418	0.0003	0.1261	0.4201
professional	0.8855	0.1288	6.8757	0.0000	0.6331	1.1379
Proportion of CROSS_SELL_SUCCESS in different campaigns.

1
print(f"""
2
Proportion of CROSS_SELL_SUCCESS
3
============================
4
{cs['CROSS_SELL_SUCCESS'].value_counts(normalize = False).sort_index()}
5
​
6
""")

Proportion of CROSS_SELL_SUCCESS
============================
0     625
1    1321
Name: CROSS_SELL_SUCCESS, dtype: int64


1
# create explanatory variable sets
2
candidate_dict = {
3
​
4
 # original x-variables
5
 'logit_full'   :['REVENUE', 'TOTAL_MEALS_ORDERED', 
6
                 'UNIQUE_MEALS_PURCH', 
7
                 'CONTACTS_W_CUSTOMER_SERVICE',
8
                 'PRODUCT_CATEGORIES_VIEWED',
9
                 'AVG_TIME_PER_SITE_VISIT', 
10
                 'CANCELLATIONS_AFTER_NOON',
11
                 'PC_LOGINS',
12
                 'MOBILE_LOGINS',
13
                 'WEEKLY_PLAN',
14
                 'AVG_PREP_VID_TIME',
15
                 'LARGEST_ORDER_SIZE',
16
                 'AVG_MEAN_RATING',
17
                 'TOTAL_PHOTOS_VIEWED'],
18
    
19
 # significant variables only (set 1)
20
 'logit_sig1'    : ['MOBILE_LOGINS',
21
                    'AVG_MEAN_RATING','PC_LOGINS' ,'CANCELLATIONS_AFTER_NOON'],
22
    
23
 # significant variables only (set 2)
24
 'logit_sig2'    : ['CANCELLATIONS_AFTER_NOON','MOBILE_LOGINS',
25
                    'personal', 'professional'],}
26
​
27
# defining explanatory variable set to run in the models
28
key_to_run = 'logit_sig2'
1
# train/test split with the full model
2
cs_data   =  cs.loc[ : , candidate_dict['logit_full']]
3
cs_target =  cs.loc[ : , 'CROSS_SELL_SUCCESS']
4
​
5
​
6
# this is the exact code we were using before
7
x_train, x_test, y_train, y_test = train_test_split(
8
            cs_data,
9
            cs_target,
10
            random_state = 219,
11
            test_size    = 0.25,
12
            stratify     = cs_target)
13
​
14
​
15
# INSTANTIATING a logistic regression model
16
logreg = LogisticRegression(solver = 'lbfgs',
17
                            C = 1.4,
18
                            warm_start= True,
19
                            random_state = 219)
20
​
21
​
22
# FITTING the training data
23
logreg_fit = logreg.fit(x_train, y_train)
24
​
25
​
26
# PREDICTING based on the testing set
27
logreg_pred = logreg_fit.predict(x_test)
28
​
29
​
30
# SCORING the results
31
print('LogReg Training ACCURACY:', logreg_fit.score(x_train, y_train).round(4))
32
print('LogReg Testing  ACCURACY:', logreg_fit.score(x_test, y_test).round(4))
33
​
34
# saving scoring data for future use
35
logreg_train_score = logreg_fit.score(x_train, y_train).round(4) # accuracy
36
logreg_test_score  = logreg_fit.score(x_test, y_test).round(4)   # accuracy
37
​
38
​
39
# displaying and saving the gap between training and testing
40
print('LogReg Train-Test Gap   :', abs(logreg_train_score - logreg_test_score).round(4))
41
logreg_test_gap = abs(logreg_train_score - logreg_test_score).round(4)
LogReg Training ACCURACY: 0.6785
LogReg Testing  ACCURACY: 0.6776
LogReg Train-Test Gap   : 0.0009
/Users/tanginaparvez/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
1
# creating a confusion matrix
2
print(confusion_matrix(y_true = y_test,
3
                       y_pred = logreg_pred))
[[  0 156]
 [  1 330]]
1
# unpacking the confusion matrix
2
logreg_tn, \
3
logreg_fp, \
4
logreg_fn, \
5
logreg_tp = confusion_matrix(y_true = y_test, y_pred = logreg_pred).ravel()
6
​
7
​
8
# printing each result one-by-one
9
print(f"""
10
True Negatives : {logreg_tn}
11
False Positives: {logreg_fp}
12
False Negatives: {logreg_fn}
13
True Positives : {logreg_tp}
14
""")

True Negatives : 0
False Positives: 156
False Negatives: 1
True Positives : 330

1
# calling the visual_cm function
2
visual_cm(true_y = y_test,
3
          pred_y = logreg_pred,
4
          labels = ['CANCELLATIONS_AFTER_NOON', 'professional'])

1
# area under the roc curve (auc)
2
print(roc_auc_score(y_true  = y_test,
3
                    y_score = logreg_pred).round(decimals = 4))
4
​
5
​
6
# saving AUC score for future use
7
logreg_auc_score = roc_auc_score(y_true  = y_test,
8
                                 y_score = logreg_pred).round(decimals = 4)
0.4985
1
# INSTANTIATING a classification tree object
2
full_tree = DecisionTreeClassifier()
3
​
4
​
5
# FITTING the training data
6
full_tree_fit = full_tree.fit(x_train, y_train)
7
​
8
​
9
# PREDICTING on new data
10
full_tree_pred = full_tree_fit.predict(x_test)
11
​
12
​
13
# SCORING the model
14
print('Full Tree Training ACCURACY:', full_tree_fit.score(x_train,
15
                                                     y_train).round(4))
16
​
17
print('Full Tree Testing ACCURACY :', full_tree_fit.score(x_test,
18
                                                     y_test).round(4))
19
​
20
print('Full Tree AUC Score:', roc_auc_score(y_true  = y_test,
21
                                            y_score = full_tree_pred).round(4))
22
​
23
​
24
# saving scoring data for future use
25
full_tree_train_score = full_tree_fit.score(x_train, y_train).round(4) # accuracy
26
full_tree_test_score  = full_tree_fit.score(x_test, y_test).round(4)   # accuracy
27
​
28
​
29
# saving AUC
30
full_tree_auc_score   = roc_auc_score(y_true  = y_test,
31
                                      y_score = full_tree_pred).round(4) # auc
Full Tree Training ACCURACY: 1.0
Full Tree Testing ACCURACY : 0.538
Full Tree AUC Score: 0.4754
2  Classification Modeling
1
########################################
2
# optimal_neighbors
3
########################################
4
def optimal_neighbors(X_data,
5
                      y_data,
6
                      standardize = True,
7
                      pct_test=0.25,
8
                      seed=219,
9
                      response_type='reg',
10
                      max_neighbors=20,
11
                      show_viz=True):
12
    """
13
Exhaustively compute training and testing results for KNN across
14
[1, max_neighbors]. Outputs the maximum test score and (by default) a
15
visualization of the results.
16
PARAMETERS
17
----------
18
X_data        : explanatory variable data
19
y_data        : response variable
20
standardize   : whether or not to standardize the X data, default True
21
pct_test      : test size for training and validation from (0,1), default 0.25
22
seed          : random seed to be used in algorithm, default 219
23
response_type : type of neighbors algorithm to use, default 'reg'
24
    Use 'reg' for regression (KNeighborsRegressor)
25
    Use 'class' for classification (KNeighborsClassifier)
26
max_neighbors : maximum number of neighbors in exhaustive search, default 20
27
show_viz      : display or surpress k-neigbors visualization, default True
28
"""    
29
    
30
    
31
    if standardize == True:
32
        # optionally standardizing X_data
33
        scaler             = StandardScaler()
34
        scaler.fit(X_data)
35
        X_scaled           = scaler.transform(X_data)
36
        X_scaled_df        = pd.DataFrame(X_scaled)
37
        X_data             = X_scaled_df
38
​
39
​
40
​
41
    # train-test split
42
    X_train, X_test, y_train, y_test = train_test_split(X_data,
43
                                                        y_data,
44
                                                        test_size = pct_test,
45
                                                        random_state = seed)
46
​
47
​
48
    # creating lists for training set accuracy and test set accuracy
49
    training_accuracy = []
50
    test_accuracy = []
51
    
52
    
53
    # setting neighbor range
54
    neighbors_settings = range(1, max_neighbors + 1)
55
​
56
​
57
    for n_neighbors in neighbors_settings:
58
        # building the model based on response variable type
59
        if response_type == 'reg':
60
            clf = KNeighborsRegressor(n_neighbors = n_neighbors)
61
            clf.fit(X_train, y_train)
62
            
63
        elif response_type == 'class':
64
            clf = KNeighborsClassifier(n_neighbors = n_neighbors)
65
            clf.fit(X_train, y_train)            
66
            
67
        else:
68
            print("Error: response_type must be 'reg' or 'class'")
69
        
70
        
71
        # recording the training set accuracy
72
        training_accuracy.append(clf.score(X_train, y_train))
73
    
74
        # recording the generalization accuracy
75
        test_accuracy.append(clf.score(X_test, y_test))
76
​
77
​
78
    # optionally displaying visualization
79
    if show_viz == True:
80
        # plotting the visualization
81
        fig, ax = plt.subplots(figsize=(12,8))
82
        plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
83
        plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
84
        plt.ylabel("Accuracy")
85
        plt.xlabel("n_neighbors")
86
        plt.legend()
87
        plt.show()
88
    
89
    
90
    # returning optimal number of neighbors
91
    print(f"The optimal number of neighbors is: {test_accuracy.index(max(test_accuracy))+1}")
92
    return test_accuracy.index(max(test_accuracy))+1
1
# dropping categorical variables after they've been encoded and irrelevant
2
cs = cs.drop(['CONTACTS_W_CUSTOMER_SERVICE', 'AVG_TIME_PER_SITE_VISIT',], axis = 1)
3
​
1
# checking the results
2
cs.columns
Index(['CROSS_SELL_SUCCESS', 'REVENUE', 'TOTAL_MEALS_ORDERED', 'UNIQUE_MEALS_PURCH', 'PRODUCT_CATEGORIES_VIEWED', 'CANCELLATIONS_AFTER_NOON', 'PC_LOGINS', 'MOBILE_LOGINS', 'WEEKLY_PLAN', 'LATE_DELIVERIES ', 'AVG_PREP_VID_TIME', 'LARGEST_ORDER_SIZE', 'AVG_MEAN_RATING', 'TOTAL_PHOTOS_VIEWED', 'log_REVENUE', 'log_AVG_TIME_PER_SITE_VISIT', 'log_AVG_PREP_VID_TIME', 'log_AVG_MEAN_RATING', 'Total_Logins', 'junk', 'personal', 'professional'], dtype='object')
Applying value counts to Response The proportion of successful vs unsuccessful campaigns is a variable that may be understood.

1
print(f"""
2
Proportion of CROSS_SELL_SUCCESS
3
============================
4
{cs['CROSS_SELL_SUCCESS'].value_counts(normalize = False).sort_index()}
5
​
6
""")

Proportion of CROSS_SELL_SUCCESS
============================
0     625
1    1321
Name: CROSS_SELL_SUCCESS, dtype: int64


1
#Correlation analysis between the explanatory variables and the response variable.
2
df_corr = cs.corr(method = 'pearson').round(decimals = 2)
3
​
4
df_corr['CROSS_SELL_SUCCESS'].sort_values(ascending = False)
CROSS_SELL_SUCCESS             1.00
professional                   0.19
CANCELLATIONS_AFTER_NOON       0.14
Total_Logins                   0.07
MOBILE_LOGINS                  0.06
personal                       0.04
PC_LOGINS                      0.04
UNIQUE_MEALS_PURCH             0.04
log_AVG_TIME_PER_SITE_VISIT    0.03
LARGEST_ORDER_SIZE             0.02
LATE_DELIVERIES                0.02
log_AVG_PREP_VID_TIME          0.02
log_REVENUE                    0.01
TOTAL_MEALS_ORDERED            0.01
TOTAL_PHOTOS_VIEWED            0.01
AVG_PREP_VID_TIME              0.01
REVENUE                        0.00
PRODUCT_CATEGORIES_VIEWED      0.00
WEEKLY_PLAN                   -0.01
log_AVG_MEAN_RATING           -0.03
AVG_MEAN_RATING               -0.04
junk                          -0.28
Name: CROSS_SELL_SUCCESS, dtype: float64
Creating a dictionary to store candidate models.

1
# printing candidate variable sets
2
print(f"""
3
/--------------------------\\
4
|Explanatory Variable Sets |
5
\\--------------------------/
6
​
7
Full Model:
8
-----------
9
{candidate_dict['logit_full']}
10
​
11
​
12
First Significant p-value Model:
13
--------------------------------
14
{candidate_dict['logit_sig1']}
15
​
16
​
17
Second Significant p-value Model:
18
---------------------------------
19
{candidate_dict['logit_sig2']}
20
""")

/--------------------------\
|Explanatory Variable Sets |
\--------------------------/

Full Model:
-----------
['REVENUE', 'TOTAL_MEALS_ORDERED', 'UNIQUE_MEALS_PURCH', 'CONTACTS_W_CUSTOMER_SERVICE', 'PRODUCT_CATEGORIES_VIEWED', 'AVG_TIME_PER_SITE_VISIT', 'CANCELLATIONS_AFTER_NOON', 'PC_LOGINS', 'MOBILE_LOGINS', 'WEEKLY_PLAN', 'AVG_PREP_VID_TIME', 'LARGEST_ORDER_SIZE', 'AVG_MEAN_RATING', 'TOTAL_PHOTOS_VIEWED']


First Significant p-value Model:
--------------------------------
['MOBILE_LOGINS', 'AVG_MEAN_RATING', 'PC_LOGINS', 'CANCELLATIONS_AFTER_NOON']


Second Significant p-value Model:
---------------------------------
['CANCELLATIONS_AFTER_NOON', 'MOBILE_LOGINS', 'personal', 'professional']

1
# declaring explanatory variables
2
cs_data   = cs.loc[ : , candidate_dict[key_to_run] ]
3
​
4
# declaring response variable
5
cs_target = cs.loc[ : , 'CROSS_SELL_SUCCESS']
Prepare train-test split for statsmodels.

1
# train-test split with stratification
2
x_train, x_test, y_train, y_test = train_test_split(
3
            cs_data,
4
            cs_target,
5
            test_size    = 0.25,
6
            random_state = 219,
7
            stratify     = cs_target)
8
​
9
​
10
# merging training data for statsmodels
11
cs_train = pd.concat([x_train, y_train], axis = 1)
1
#Print explanatory variables to copy to the models
2
for val in cs_data:
3
    print(f" {val} + ")
 CANCELLATIONS_AFTER_NOON + 
 MOBILE_LOGINS + 
 personal + 
 professional + 
1
 # instantiating a logistic regression model object
2
logistic_sig2 = smf.logit(formula = """CROSS_SELL_SUCCESS ~
3
                                         CANCELLATIONS_AFTER_NOON + 
4
                                         MOBILE_LOGINS + 
5
                                         personal + 
6
                                         professional""", 
7
                           data    = cs_train)
8
​
9
​
10
# fitting the model object
11
results_sig2 = logistic_sig2.fit()
12
​
13
​
14
# checking the results SUMMARY
15
results_sig2.summary()
Optimization terminated successfully.
         Current function value: 0.573267
         Iterations 5
Logit Regression Results
Dep. Variable:	CROSS_SELL_SUCCESS	No. Observations:	1459
Model:	Logit	Df Residuals:	1454
Method:	MLE	Df Model:	4
Date:	Fri, 03 Mar 2023	Pseudo R-squ.:	0.08710
Time:	23:42:22	Log-Likelihood:	-836.40
converged:	True	LL-Null:	-916.19
Covariance Type:	nonrobust	LLR p-value:	1.787e-33
coef	std err	z	P>|z|	[0.025	0.975]
Intercept	-1.2049	0.217	-5.554	0.000	-1.630	-0.780
CANCELLATIONS_AFTER_NOON	0.2155	0.041	5.278	0.000	0.135	0.296
MOBILE_LOGINS	0.3513	0.113	3.103	0.002	0.129	0.573
personal	1.2514	0.150	8.319	0.000	0.957	1.546
professional	1.7100	0.164	10.454	0.000	1.389	2.031
Logistic Regression with Hyperparameter Tuning.

1
########################################
2
# RandomizedSearchCV
3
########################################
4
​
5
# declaring a hyperparameter space
6
C_space          = np.arange(0.1, 5.0, 0.1)   # similar to alpha in lasso regression but inverse => lower C will try to help us select sig variables
7
warm_start_space = [True, False]
8
solver_space     = ['newton-cg', 'sag', 'lbfgs']
9
​
10
# creating a hyperparameter grid
11
param_grid = {'C'          : C_space,
12
              'warm_start' : warm_start_space,
13
              'solver'     : solver_space}
14
​
15
# INSTANTIATING the model object without hyperparameters
16
lr_tuned = LogisticRegression(random_state = 219,
17
                              max_iter     = 1200)
18
​
19
# GridSearchCV object
20
lr_tuned_cv = RandomizedSearchCV(estimator           = lr_tuned,   # the model object
21
                                 param_distributions = param_grid, # parameters to tune
22
                                 cv                  = 3,          # how many folds in cross-validation
23
                                 n_iter              = 500,        # number of combinations of hyperparameters to try
24
                                 random_state        = 219,        # starting point for random sequence
25
                                 scoring = make_scorer(
26
                                           roc_auc_score,
27
                                           needs_threshold = False))# scoring criteria (AUC)
28
​
29
# FITTING to the FULL DATASET (due to cross-validation)
30
lr_tuned_cv.fit(cs_data, cs_target)
31
​
32
# PREDICT step is not needed
33
​
34
# printing the optimal parameters and best score
35
print("Tuned Parameters: ", lr_tuned_cv.best_params_)
36
print("Tuned CV AUC: ", lr_tuned_cv.best_score_.round(4))
/Users/tanginaparvez/opt/anaconda3/lib/python3.9/site-packages/sklearn/model_selection/_search.py:292: UserWarning: The total space of parameters 294 is smaller than n_iter=500. Running 294 iterations. For exhaustive searches, use GridSearchCV.
  warnings.warn(
Tuned Parameters:  {'warm_start': True, 'solver': 'newton-cg', 'C': 0.4}
Tuned CV AUC:  0.6202
Checking the best estimator for the model.

1
lr_tuned_cv.best_estimator_
LogisticRegression(C=0.4, max_iter=1200, random_state=219, solver='newton-cg',
                   warm_start=True)
1
# building a model based on hyperparameter tuning results
2
​
3
# INSTANTIATING a logistic regression model with tuned values
4
lr_tuned = lr_tuned_cv.best_estimator_
5
​
6
​
7
# FIT step is not needed
8
​
9
​
10
# PREDICTING based on the testing set
11
lr_tuned_pred = lr_tuned.predict(x_test)
12
​
13
​
14
# SCORING the results
15
print('LR Tuned Training ACCURACY:', lr_tuned.score(x_train, y_train).round(4))
16
print('LR Tuned Testing  ACCURACY:', lr_tuned.score(x_test, y_test).round(4))
17
print('LR Tuned AUC Score        :', roc_auc_score(y_true  = y_test,
18
                                          y_score = lr_tuned_pred).round(4))
19
​
20
​
21
# saving scoring data for future use
22
lr_tuned_train_score = lr_tuned.score(x_train, y_train).round(4) # accuracy
23
lr_tuned_test_score  = lr_tuned.score(x_test, y_test).round(4)   # accuracy
24
​
25
​
26
# saving the AUC score
27
lr_tuned_auc         = roc_auc_score(y_true  = y_test,
28
                                     y_score = lr_tuned_pred).round(4) # auc
LR Tuned Training ACCURACY: 0.7231
LR Tuned Testing  ACCURACY: 0.7228
LR Tuned AUC Score        : 0.6215
Unpack the confusion matrix.

1
# unpacking the confusion matrix
2
lr_tuned_tn, \
3
lr_tuned_fp, \
4
lr_tuned_fn, \
5
lr_tuned_tp = confusion_matrix(y_true = y_test, y_pred = lr_tuned_pred).ravel()
6
​
7
​
8
# printing each result one-by-one
9
print(f"""
10
True Negatives : {lr_tuned_tn}
11
False Positives: {lr_tuned_fp}
12
False Negatives: {lr_tuned_fn}
13
True Positives : {lr_tuned_tp}
14
""")

True Negatives : 53
False Positives: 103
False Negatives: 32
True Positives : 299

Store the model results.

1
# declaring model performance objects
2
lr_train_acc = lr_tuned_train_score
3
lr_test_acc  = lr_tuned_test_score
4
lr_auc       = lr_tuned_auc
3  Classification Trees (CART Models)
1
########################################
2
# display_tree
3
########################################
4
def display_tree(tree, feature_df, height = 500, width = 800):
5
    """
6
    PARAMETERS
7
    ----------
8
    tree       : fitted tree model object
9
        fitted CART model to visualized
10
    feature_df : DataFrame
11
        DataFrame of explanatory features (used to generate labels)
12
    height     : int, default 500
13
        height in pixels to which to constrain image in html
14
    width      : int, default 800
15
        width in pixels to which to constrain image in html
16
    """
17
​
18
    # visualizing the tree
19
    dot_data = StringIO()
20
​
21
    
22
    # exporting tree to graphviz
23
    export_graphviz(decision_tree      = tree,
24
                    out_file           = dot_data,
25
                    filled             = True,
26
                    rounded            = True,
27
                    special_characters = True,
28
                    feature_names      = feature_df.columns)
29
​
30
​
31
    # declaring a graph object
32
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
33
​
34
​
35
    # creating image
36
    img = Image(graph.create_png(),
37
                height = height,
38
                width  = width)
39
    
40
    return img
41
​
42
########################################
43
# plot_feature_importances for CART model
44
########################################
45
def plot_feature_importances(model, train, export = False):
46
    """
47
    Plots the importance of features from a CART model.
48
    
49
    PARAMETERS
50
    ----------
51
    model  : CART model
52
    train  : explanatory variable training data
53
    export : whether or not to export as a .png image, default False
54
    """
55
    
56
    # declaring the number
57
    n_features = x_train.shape[1]
58
    
59
    # setting plot window
60
    fig, ax = plt.subplots(figsize=(12,9))
61
    
62
    plt.barh(range(n_features), model.feature_importances_, align='center')
63
    plt.yticks(pd.np.arange(n_features), train.columns)
64
    plt.xlabel("Feature importance")
65
    plt.ylabel("Feature")
66
    
67
    if export == True:
68
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')
Creating dictionary of explanatory variable sets for CART models.

1
# create explanatory variable sets
2
​
3
candidate_dict_CART = {
4
​
5
    # original x-variables
6
    'original_CART': [
7
        'REVENUE',
8
        'TOTAL_MEALS_ORDERED',
9
        'UNIQUE_MEALS_PURCH',
10
        'CANCELLATIONS_AFTER_NOON',
11
        'PC_LOGINS',
12
        'MOBILE_LOGINS',
13
        'WEEKLY_PLAN',
14
        'AVG_PREP_VID_TIME',
15
        'LARGEST_ORDER_SIZE',
16
        'AVG_MEAN_RATING'
17
    ],
18
​
19
    # significant variables only (set 1)
20
    'logit_sig1': [
21
        'PRODUCT_CATEGORIES_VIEWED']}
22
​
23
# defining explanatory variable set to run in the models
24
key_to_run_CART = 'original_CART'
1
# declaring explanatory variables
2
cs_data_CART = cs.loc[:, candidate_dict_CART[key_to_run_CART]]
Prepare train-test split.

1
# train-test split with stratification
2
x_train_CART, x_test_CART, y_train_CART, y_test_CART = train_test_split(
3
            cs_data_CART,
4
            cs_target,
5
            test_size    = 0.25,
6
            random_state = 219,
7
            stratify     = cs_target)
8
​
9
​
10
# merging training data for statsmodels
11
cs_train_CART = pd.concat([x_train_CART, y_train_CART], axis = 1)
Hyperparameter Tuning on Classification Trees.

1
# declaring a hyperparameter space
2
criterion_space = ['gini', 'entropy']
3
splitter_space  = ['best', 'random']
4
depth_space     = pd.np.arange(1, 8, 1)
5
leaf_space      = pd.np.arange(1, 100, 1)
6
​
7
​
8
# creating a hyperparameter grid
9
param_grid = {'criterion'        : criterion_space,
10
              'splitter'         : splitter_space,
11
              'max_depth'        : depth_space,
12
              'min_samples_leaf' : leaf_space}
13
​
14
​
15
# INSTANTIATING the model object without hyperparameters
16
tuned_tree_CART = DecisionTreeClassifier(random_state = 219)
17
​
18
​
19
# RandomizedSearchCV object
20
tuned_tree_CART_cv = RandomizedSearchCV(estimator        = tuned_tree_CART,
21
                                   param_distributions   = param_grid,
22
                                   cv                    = 3,   
23
                                   n_iter                = 1000,
24
                                   random_state          = 219,
25
                                   scoring = make_scorer(roc_auc_score,
26
                                             needs_threshold = False))
27
​
28
​
29
# FITTING to the FULL DATASET (due to cross-validation)
30
tuned_tree_CART_cv.fit(cs_data_CART, cs_target)
31
​
32
​
33
# PREDICT step is not needed
34
​
35
​
36
# printing the optimal parameters and best score
37
print("Tuned Parameters  :", tuned_tree_CART_cv.best_params_)
38
print("Tuned Training AUC:", tuned_tree_CART_cv.best_score_.round(4))
/var/folders/qd/k600f85s3mlcxldc5c0t12g80000gn/T/ipykernel_69647/1142073231.py:4: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead.
  depth_space     = pd.np.arange(1, 8, 1)
/var/folders/qd/k600f85s3mlcxldc5c0t12g80000gn/T/ipykernel_69647/1142073231.py:5: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead.
  leaf_space      = pd.np.arange(1, 100, 1)
Tuned Parameters  : {'splitter': 'best', 'min_samples_leaf': 35, 'max_depth': 6, 'criterion': 'entropy'}
Tuned Training AUC: 0.5376
With the results of the hyperparameter tweaking, create a classification tree model.

1
# building a model based on hyperparameter tuning results
2
​
3
# INSTANTIATING a logistic regression model with tuned values
4
tree_tuned_CART = tuned_tree_CART_cv.best_estimator_
5
​
6
​
7
# FIT step is not needed
8
​
9
​
10
# PREDICTING based on the testing set
11
tree_tuned_CART_pred = tree_tuned_CART.predict(x_test_CART)
12
​
13
​
14
# SCORING the results
15
print('Training ACCURACY:', tree_tuned_CART.score(x_train_CART, y_train_CART).round(4))
16
print('Testing  ACCURACY:', tree_tuned_CART.score(x_test_CART, y_test_CART).round(4))
17
print('AUC Score        :', roc_auc_score(y_true  = y_test_CART,
18
                                          y_score = tree_tuned_CART_pred).round(4))
19
​
20
​
21
# saving scoring data for future use
22
tree_tuned_CART_train_score = tree_tuned_CART.score(x_train_CART, y_train_CART).round(4) # accuracy
23
tree_tuned_CART_test_score  = tree_tuned_CART.score(x_test_CART, y_test_CART).round(4)   # accuracy
24
​
25
​
26
# saving the AUC score
27
tree_tuned_CART_auc         = roc_auc_score(y_true  = y_test_CART,
28
                                     y_score = tree_tuned_CART_pred).round(4) # auc
29
​
Training ACCURACY: 0.6991
Testing  ACCURACY: 0.6797
AUC Score        : 0.539
1
# unpacking the confusion matrix
2
tuned_tree_CART_tn, \
3
tuned_tree_CART_fp, \
4
tuned_tree_CART_fn, \
5
tuned_tree_CART_tp = confusion_matrix(y_true = y_test, y_pred = tree_tuned_CART_pred).ravel()
6
​
7
​
8
# printing each result one-by-one
9
print(f"""
10
True Negatives : {tuned_tree_CART_tn}
11
False Positives: {tuned_tree_CART_fp}
12
False Negatives: {tuned_tree_CART_fn}
13
True Positives : {tuned_tree_CART_tp}
14
""")

True Negatives : 23
False Positives: 133
False Negatives: 23
True Positives : 308

Using the dictionary's candidate models, construct a classification tree model.

1
# INSTANTIATING a classification tree object
2
full_tree = DecisionTreeClassifier(max_depth=8, random_state=219)
3
​
4
# FITTING the training data
5
full_tree_fit = full_tree.fit(x_train, y_train)
6
​
7
# PREDICTING on new data
8
full_tree_pred = full_tree_fit.predict(x_test)
9
​
10
# SCORING the model
11
print('Full Tree Training ACCURACY:', full_tree_fit.score(x_train, y_train).round(4))
12
print('Full Tree Testing ACCURACY:', full_tree_fit.score(x_test, y_test).round(4))
13
print('Full Tree AUC Score:', roc_auc_score(y_true=y_test, y_score=full_tree_pred).round(4))
14
​
15
# saving scoring data for future use
16
full_tree_train_score = full_tree_fit.score(x_train, y_train).round(4) # accuracy
17
full_tree_test_score = full_tree_fit.score(x_test, y_test).round(4) # accuracy
18
​
19
# saving AUC
20
full_tree_auc_score = roc_auc_score(y_true=y_test, y_score=full_tree_pred).round(4) # auc
Full Tree Training ACCURACY: 0.7341
Full Tree Testing ACCURACY: 0.7248
Full Tree AUC Score: 0.6146
1
# unpacking the confusion matrix
2
full_tree_tn, \
3
full_tree_fp, \
4
full_tree_fn, \
5
full_tree_tp = confusion_matrix(y_true = y_test, y_pred = full_tree_pred ).ravel()
6
​
7
​
8
# printing each result one-by-one
9
print(f"""
10
True Negatives : {full_tree_tn}
11
False Positives: {full_tree_fp}
12
False Negatives: {full_tree_fn}
13
True Positives : {full_tree_tp}
14
""")

True Negatives : 48
False Positives: 108
False Negatives: 26
True Positives : 305

1
# plotting feature importance
2
plot_feature_importances(full_tree_fit,
3
                         train  = x_train,
4
                         export = False)
/var/folders/qd/k600f85s3mlcxldc5c0t12g80000gn/T/ipykernel_69647/3629681362.py:63: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead.
  plt.yticks(pd.np.arange(n_features), train.columns)
Create a new classification tree model to prune the tree.

1
# INSTANTIATING a classification tree object
2
pruned_tree = DecisionTreeClassifier(max_depth = 8,
3
                                     min_samples_leaf = 50,
4
                                     random_state = 219)
5
​
6
​
7
# FITTING the training data
8
pruned_tree_fit  = pruned_tree.fit(x_train, y_train)
9
​
10
​
11
# PREDICTING on new data
12
pruned_tree_pred = pruned_tree_fit.predict(x_test)
13
​
14
​
15
# SCORING the model
16
print('Training ACCURACY:', pruned_tree_fit.score(x_train, y_train).round(4))
17
print('Testing  ACCURACY:', pruned_tree_fit.score(x_test, y_test).round(4))
18
print('AUC Score        :', roc_auc_score(y_true  = y_test,
19
                                          y_score = pruned_tree_pred).round(4))
20
​
21
​
22
# saving scoring data for future use
23
pruned_tree_train_score = pruned_tree_fit.score(x_train, y_train).round(4) # accuracy
24
pruned_tree_test_score  = pruned_tree_fit.score(x_test, y_test).round(4)   # accuracy
25
​
26
​
27
# saving auc score
28
pruned_tree_auc_score   = roc_auc_score(y_true  = y_test,
29
                                        y_score = pruned_tree_pred).round(4) # auc
Training ACCURACY: 0.7245
Testing  ACCURACY: 0.7228
AUC Score        : 0.6198
1
# unpacking the confusion matrix
2
pruned_tree_tn, \
3
pruned_tree_fp, \
4
pruned_tree_fn, \
5
pruned_tree_tp = confusion_matrix(y_true = y_test, y_pred = pruned_tree_pred).ravel()
6
​
7
​
8
# printing each result one-by-one
9
print(f"""
10
True Negatives : {pruned_tree_tn}
11
False Positives: {pruned_tree_fp}
12
False Negatives: {pruned_tree_fn}
13
True Positives : {pruned_tree_tp}
14
""")

True Negatives : 52
False Positives: 104
False Negatives: 31
True Positives : 300

Hyperparameter Tuning on Classification Trees.

1
# declaring a hyperparameter space
2
criterion_space = ['gini', 'entropy']
3
splitter_space  = ['best', 'random']
4
depth_space     = pd.np.arange(1, 25, 1)
5
leaf_space      = pd.np.arange(1, 100, 1)
6
​
7
​
8
# creating a hyperparameter grid
9
param_grid = {'criterion'        : criterion_space,
10
              'splitter'         : splitter_space,
11
              'max_depth'        : depth_space,
12
              'min_samples_leaf' : leaf_space}
13
​
14
​
15
# INSTANTIATING the model object without hyperparameters
16
tuned_tree = DecisionTreeClassifier(random_state = 219)
17
​
18
​
19
# RandomizedSearchCV object
20
tuned_tree_cv = RandomizedSearchCV(estimator             = tuned_tree,
21
                                   param_distributions   = param_grid,
22
                                   cv                    = 3,    
23
                                   n_iter                = 1500,
24
                                   random_state          = 219,
25
                                   scoring = make_scorer(roc_auc_score,
26
                                             needs_threshold = False))
27
​
28
​
29
# FITTING to the FULL DATASET (due to cross-validation)
30
tuned_tree_cv.fit(cs_data, cs_target)
31
​
32
​
33
# PREDICT step is not needed
34
​
35
​
36
# printing the optimal parameters and best score
37
print("Tuned Parameters  :", tuned_tree_cv.best_params_)
38
print("Tuned Training AUC:", tuned_tree_cv.best_score_.round(4))
/var/folders/qd/k600f85s3mlcxldc5c0t12g80000gn/T/ipykernel_69647/2611757762.py:4: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead.
  depth_space     = pd.np.arange(1, 25, 1)
/var/folders/qd/k600f85s3mlcxldc5c0t12g80000gn/T/ipykernel_69647/2611757762.py:5: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead.
  leaf_space      = pd.np.arange(1, 100, 1)
Tuned Parameters  : {'splitter': 'best', 'min_samples_leaf': 54, 'max_depth': 2, 'criterion': 'entropy'}
Tuned Training AUC: 0.6202
1
#Check the best estimator for the model
2
​
3
tuned_tree_cv.best_params_
{'splitter': 'best',
 'min_samples_leaf': 54,
 'max_depth': 2,
 'criterion': 'entropy'}
1
# Build a classification tree model based on the hyperparameter tuning results.
2
​
3
# INSTANTIATING a classification tree object with tuned values
4
tree_tuned = tuned_tree_cv.best_estimator_
5
​
6
​
7
# FIT step is not needed
8
​
9
​
10
# PREDICTING based on the testing set
11
tree_tuned_pred = tree_tuned.predict(x_test)
12
​
13
​
14
# SCORING the results
15
print('Training ACCURACY:', tree_tuned.score(x_train, y_train).round(4))
16
print('Testing  ACCURACY:', tree_tuned.score(x_test, y_test).round(4))
17
print('AUC Score        :', roc_auc_score(y_true  = y_test,
18
                                          y_score = tree_tuned_pred).round(4))
19
​
20
​
21
# saving scoring data for future use
22
tree_tuned_train_score = tree_tuned.score(x_train, y_train).round(4) # accuracy
23
tree_tuned_test_score  = tree_tuned.score(x_test, y_test).round(4)   # accuracy
24
​
25
​
26
# saving the AUC score
27
tree_tuned_auc         = roc_auc_score(y_true  = y_test,
28
                                     y_score = tree_tuned_pred).round(4) # auc
Training ACCURACY: 0.7121
Testing  ACCURACY: 0.7125
AUC Score        : 0.6225
1
# unpacking the confusion matrix
2
tuned_tree_tn, \
3
tuned_tree_fp, \
4
tuned_tree_fn, \
5
tuned_tree_tp = confusion_matrix(y_true = y_test, y_pred = tree_tuned_pred).ravel()
6
​
7
​
8
# printing each result one-by-one
9
print(f"""
10
True Negatives : {tuned_tree_tn}
11
False Positives: {tuned_tree_fp}
12
False Negatives: {tuned_tree_fn}
13
True Positives : {tuned_tree_tp}
14
""")

True Negatives : 58
False Positives: 98
False Negatives: 42
True Positives : 289

1
# declaring model performance objects
2
tree_train_acc = tree_tuned.score(x_train, y_train).round(4)
3
tree_test_acc  = tree_tuned.score(x_test, y_test).round(4)
4
tree_auc       = roc_auc_score(y_true  = y_test,
5
                              y_score = tree_tuned_pred).round(4)
Scale the explanatory data and create a KNN classification model with optimal neighbors.

1
# INSTANTIATING StandardScaler()
2
scaler = StandardScaler()
3
​
4
​
5
# FITTING the data
6
scaler.fit(cs_data)
7
​
8
​
9
# TRANSFORMING the data
10
x_scaled     = scaler.transform(cs_data)
11
​
12
​
13
# converting to a DataFrame
14
x_scaled_df  = pd.DataFrame(x_scaled) 
15
​
16
​
17
# train-test split with the scaled data
18
x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
19
            x_scaled_df,
20
            cs_target,
21
            random_state = 219,
22
            test_size = 0.25,
23
            stratify = cs_target)
24
​
25
​
26
# INSTANTIATING a KNN classification model with optimal neighbors
27
knn_opt = KNeighborsClassifier(n_neighbors = 19)
28
​
29
​
30
# FITTING the training data
31
knn_fit = knn_opt.fit(x_train_scaled, y_train_scaled)
32
​
33
​
34
# PREDICTING based on the testing set
35
knn_pred = knn_fit.predict(x_test_scaled)
36
​
37
​
38
# SCORING the results
39
print('Training ACCURACY:', knn_fit.score(x_train_scaled, y_train_scaled).round(4))
40
print('Testing  ACCURACY:', knn_fit.score(x_test_scaled, y_test_scaled).round(4))
41
print('AUC Score        :', roc_auc_score(y_true  = y_test,
42
                                          y_score = knn_pred).round(4))
43
​
44
​
45
# saving scoring data
46
knn_train_score = knn_fit.score(x_train_scaled, y_train_scaled).round(4)
47
knn_test_score  = knn_fit.score(x_test_scaled, y_test_scaled).round(4)
48
​
49
​
50
# saving AUC score
51
knn_auc_score   = roc_auc_score(y_true  = y_test,
52
                                          y_score = knn_pred).round(4)
Training ACCURACY: 0.6861
Testing  ACCURACY: 0.7023
AUC Score        : 0.6403
/Users/tanginaparvez/opt/anaconda3/lib/python3.9/site-packages/sklearn/neighbors/_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
  mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
/Users/tanginaparvez/opt/anaconda3/lib/python3.9/site-packages/sklearn/neighbors/_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
  mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
/Users/tanginaparvez/opt/anaconda3/lib/python3.9/site-packages/sklearn/neighbors/_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
  mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
/Users/tanginaparvez/opt/anaconda3/lib/python3.9/site-packages/sklearn/neighbors/_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
  mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
/Users/tanginaparvez/opt/anaconda3/lib/python3.9/site-packages/sklearn/neighbors/_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
  mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
1
# unpacking the confusion matrix
2
knn_tree_tn, \
3
knn_tree_fp, \
4
knn_tree_fn, \
5
knn_tree_tp = confusion_matrix(y_true = y_test, y_pred = knn_pred).ravel()
6
​
7
​
8
# printing each result one-by-one
9
print(f"""
10
True Negatives : {knn_tree_tn}
11
False Positives: {knn_tree_fp}
12
False Negatives: {knn_tree_fn}
13
True Positives : {knn_tree_tp}
14
""")

True Negatives : 73
False Positives: 83
False Negatives: 62
True Positives : 269

Using RandomizedSearchCV, optimize the random forest's hyperparameters.

1
# declaring a hyperparameter space
2
estimator_space  = pd.np.arange(100, 1100, 250)
3
leaf_space       = pd.np.arange(1, 31, 10)
4
criterion_space  = ['gini', 'entropy']
5
bootstrap_space  = [True, False]
6
warm_start_space = [True, False]
7
​
8
​
9
# creating a hyperparameter grid
10
param_grid = {'n_estimators'     : estimator_space,
11
              'min_samples_leaf' : leaf_space,
12
              'criterion'        : criterion_space,
13
              'bootstrap'        : bootstrap_space,
14
              'warm_start'       : warm_start_space}
15
​
16
​
17
# INSTANTIATING the model object without hyperparameters
18
forest_grid = RandomForestClassifier(max_depth    = 8,
19
                                     random_state = 219)
20
​
21
​
22
# GridSearchCV object
23
forest_cv = RandomizedSearchCV(estimator           = forest_grid,
24
                               param_distributions = param_grid,
25
                                        cv         = 3,
26
                                        n_iter     = 1000,
27
                                        scoring    = make_scorer(roc_auc_score,
28
                                   needs_threshold = False))
29
​
30
​
31
# FITTING to the FULL DATASET (due to cross-validation)
32
forest_cv.fit(cs_data, cs_target)
33
​
34
​
35
# PREDICT step is not needed
36
​
37
​
38
# printing the optimal parameters and best score
39
print("Tuned Parameters  :", forest_cv.best_params_)
40
print("Tuned Training AUC:", forest_cv.best_score_.round(4))
/var/folders/qd/k600f85s3mlcxldc5c0t12g80000gn/T/ipykernel_69647/3317191593.py:2: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead.
  estimator_space  = pd.np.arange(100, 1100, 250)
/var/folders/qd/k600f85s3mlcxldc5c0t12g80000gn/T/ipykernel_69647/3317191593.py:3: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead.
  leaf_space       = pd.np.arange(1, 31, 10)
/Users/tanginaparvez/opt/anaconda3/lib/python3.9/site-packages/sklearn/model_selection/_search.py:292: UserWarning: The total space of parameters 96 is smaller than n_iter=1000. Running 96 iterations. For exhaustive searches, use GridSearchCV.
  warnings.warn(
Tuned Parameters  : {'warm_start': True, 'n_estimators': 100, 'min_samples_leaf': 1, 'criterion': 'entropy', 'bootstrap': True}
Tuned Training AUC: 0.6158
1
# best estimators based on RandomizedSearchCV
2
forest_cv.best_estimator_
RandomForestClassifier(criterion='entropy', max_depth=8, random_state=219,
                       warm_start=True)
Creating tuned random forest model based on hyperparameter tuning results.

1
# to avoid running another RandomizedSearch
2
forest_tuned = RandomForestClassifier(bootstrap      =False , 
3
                                      ccp_alpha      =0.0 , 
4
                                      class_weight   =None ,
5
                                      criterion      ='entropy',
6
                                      max_depth      =8 , 
7
                                      max_features   ='auto',
8
                                      max_leaf_nodes =None , 
9
                                      max_samples    =None ,
10
                                      min_impurity_decrease  =0.0 , 
11
                                      min_samples_leaf=11 , 
12
                                      min_samples_split=2 ,
13
                                      min_weight_fraction_leaf=0.0, 
14
                                      n_estimators=350,
15
                                      n_jobs=None, 
16
                                      oob_score=False, 
17
                                      random_state=219,
18
                                      verbose=0, 
19
                                      warm_start=True)
20
​
21
​
22
# FITTING the model object
23
forest_tuned_fit = forest_tuned.fit(x_train, y_train)
24
​
25
​
26
# PREDICTING based on the testing set
27
forest_tuned_pred = forest_tuned_fit.predict(x_test)
28
​
29
​
30
# SCORING the results
31
print('Forest Tuned Training ACCURACY:', forest_tuned.score(x_train, y_train).round(4))
32
print('Forest Tuned Testing  ACCURACY:', forest_tuned.score(x_test, y_test).round(4))
33
print('Forest Tuned AUC Score        :', roc_auc_score(y_true  = y_test,
34
                                                   y_score = forest_tuned_pred).round(4))
35
​
36
​
37
# saving scoring data for future use
38
forest_tuned_train_score = forest_tuned.score(x_train, y_train).round(4) # accuracy
39
forest_tuned_test_score  = forest_tuned.score(x_test, y_test).round(4)   # accuracy
40
​
41
​
42
# saving the AUC score
43
forest_tuned_auc = roc_auc_score(y_true  = y_test,
44
                                 y_score = forest_tuned_pred).round(4) # auc
Forest Tuned Training ACCURACY: 0.7286
Forest Tuned Testing  ACCURACY: 0.7269
Forest Tuned AUC Score        : 0.6127
1
# plotting feature importances
2
plot_feature_importances(forest_tuned_fit,
3
                         train = x_train,
4
                         export = False)
/var/folders/qd/k600f85s3mlcxldc5c0t12g80000gn/T/ipykernel_69647/3629681362.py:63: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead.
  plt.yticks(pd.np.arange(n_features), train.columns)
1
# unpacking the confusion matrix
2
tuned_rf_tn, \
3
tuned_rf_fp, \
4
tuned_rf_fn, \
5
tuned_rf_tp = confusion_matrix(y_true = y_test, y_pred = forest_tuned_pred).ravel()
6
​
7
​
8
# printing each result one-by-one
9
print(f"""
10
True Negatives : {tuned_rf_tn}
11
False Positives: {tuned_rf_fp}
12
False Negatives: {tuned_rf_fn}
13
True Positives : {tuned_rf_tp}
14
""")

True Negatives : 46
False Positives: 110
False Negatives: 23
True Positives : 308

Save the tweaked random forest model's output in model performance for further use.

1
# declaring model performance objects
2
tuned_rf_train_acc = forest_tuned_fit.score(x_train, y_train).round(4)
3
tuned_rf_test_acc  = forest_tuned_fit.score(x_test, y_test).round(4)
4
tuned_rf_auc       = roc_auc_score(y_true  = y_test,
5
                                   y_score = forest_tuned_pred).round(4)
Build a gradient boosting classifier model with the hyperparameters tuning.

1
# declaring a hyperparameter space
2
learn_space        = pd.np.arange(0.1, 4.0, 0.2)
3
estimator_space    = pd.np.arange(100, 200, 5)
4
depth_space        = pd.np.arange(1, 8, 1)
5
warm_start_space   = [True, False]
6
​
7
# creating a hyperparameter grid
8
param_grid = {'learning_rate' : learn_space,
9
              'max_depth'     : depth_space,
10
              'n_estimators'  : estimator_space,
11
              'warm_start'     : warm_start_space}
12
​
13
​
14
# INSTANTIATING the model object without hyperparameters
15
full_gbm_grid = GradientBoostingClassifier(random_state = 219)
16
​
17
​
18
# GridSearchCV object
19
full_gbm_cv = RandomizedSearchCV(estimator     = full_gbm_grid,
20
                           param_distributions = param_grid,
21
                           cv                  = 3,
22
                           n_iter              = 900,
23
                           random_state        = 219,
24
                           scoring             = make_scorer(roc_auc_score,
25
                                                 needs_threshold = False))
26
​
27
​
28
# FITTING to the FULL DATASET (due to cross-validation)
29
full_gbm_cv.fit(cs_data, cs_target)
30
​
31
​
32
# PREDICT step is not needed
33
​
34
​
35
# printing the optimal parameters and best score
36
print("Tuned Parameters  :", full_gbm_cv.best_params_)
37
print("Tuned Training AUC:", full_gbm_cv.best_score_.round(4))
/var/folders/qd/k600f85s3mlcxldc5c0t12g80000gn/T/ipykernel_69647/258157216.py:2: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead.
  learn_space        = pd.np.arange(0.1, 4.0, 0.2)
/var/folders/qd/k600f85s3mlcxldc5c0t12g80000gn/T/ipykernel_69647/258157216.py:3: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead.
  estimator_space    = pd.np.arange(100, 200, 5)
/var/folders/qd/k600f85s3mlcxldc5c0t12g80000gn/T/ipykernel_69647/258157216.py:4: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead.
  depth_space        = pd.np.arange(1, 8, 1)
Tuned Parameters  : {'warm_start': False, 'n_estimators': 175, 'max_depth': 1, 'learning_rate': 1.3000000000000003}
Tuned Training AUC: 0.6193
1
# checking the best estimator for the model
2
full_gbm_cv.best_estimator_
GradientBoostingClassifier(learning_rate=1.3000000000000003, max_depth=1,
                           n_estimators=175, random_state=219)
1
# INSTANTIATING the model object without hyperparameters
2
​
3
# hyperparameter tuning
4
gbm_tuned = GradientBoostingClassifier(ccp_alpha=0.0, 
5
                                       criterion='friedman_mse', 
6
                                       init=None,
7
                                       learning_rate=0.5000000000000001, 
8
                                       loss='deviance',
9
                                       max_depth=1, 
10
                                       max_features=None, 
11
                                       max_leaf_nodes=None,
12
                                       min_impurity_decrease=0.0,
13
                                       min_samples_leaf=1, 
14
                                       min_samples_split=2,
15
                                       min_weight_fraction_leaf=0.0, 
16
                                       n_estimators=180,
17
                                       n_iter_no_change=None, 
18
                                       random_state=219, 
19
                                       subsample=1.0, tol=0.0001,
20
                                       validation_fraction=0.1, 
21
                                       verbose=0,
22
                                       warm_start=True)
23
​
24
# FIT step is needed as we are not using .best_estimator
25
gbm_tuned_fit = gbm_tuned.fit(x_train, y_train)
26
​
27
​
28
# PREDICTING based on the testing set
29
gbm_tuned_pred = gbm_tuned_fit.predict(x_test)
30
​
31
​
32
# SCORING the results
33
print('Training ACCURACY:', gbm_tuned_fit.score(x_train, y_train).round(4))
34
print('Testing  ACCURACY:', gbm_tuned_fit.score(x_test, y_test).round(4))
35
print('AUC Score        :', roc_auc_score(y_true  = y_test,
36
                                          y_score = gbm_tuned_pred).round(4))
Training ACCURACY: 0.73
Testing  ACCURACY: 0.7269
AUC Score        : 0.6229
1
# unpacking the confusion matrix
2
gbm_tuned_tn, \
3
gbm_tuned_fp, \
4
gbm_tuned_fn, \
5
gbm_tuned_tp = confusion_matrix(y_true = y_test, y_pred = gbm_tuned_pred).ravel()
6
​
7
​
8
# printing each result one-by-one
9
print(f"""
10
True Negatives : {gbm_tuned_tn}
11
False Positives: {gbm_tuned_fp}
12
False Negatives: {gbm_tuned_fn}
13
True Positives : {gbm_tuned_tp}
14
""")

True Negatives : 52
False Positives: 104
False Negatives: 29
True Positives : 302

1
# declaring model performance objects
2
gbm_train_acc = gbm_tuned_fit.score(x_train, y_train).round(4)
3
gbm_test_acc  = gbm_tuned_fit.score(x_test, y_test).round(4)
4
gbm_auc       = roc_auc_score(y_true  = y_test,
5
                              y_score = gbm_tuned_pred).round(4)
4  Comparing Results
1
model_performance.iloc[ 3 , :  ]


The "Tuned Tree" model has a training accuracy of 0.7121 and a testing accuracy of 0.7125. The AUC score for this model is 0.6225, which indicates that the model has a fair discrimination performance. The confusion matrix shows that the model correctly classified 289 out of 331 observations in the testing set, with 58 false negatives and 98 false positives.
