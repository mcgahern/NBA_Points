import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(suppress=True, precision = 2)

nba = pd.read_csv('./nba_games.csv')

# Subset Data to 2010 Season, 2014 Season
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

print(nba_2010.head())
print(nba_2014.head())

# Get knicks and Nets points for 2010
knicks_pts = nba_2010.pts[nba.fran_id == 'Knicks']
nets_pts = nba_2010.pts[nba.fran_id == 'Nets']

# Difference of mean points for 2010
diff_means_2010 = np.mean(knicks_pts) - np.mean(nets_pts)
print(diff_means_2010)

# PLot overlapping histograms to compare points
plt.hist(knicks_pts, alpha = 0.5, label = 'Knicks 2010')
plt.hist(nets_pts, alpha = 0.5, label = 'Nets 2010')
plt.title('Knicks vs Nets Points 2010 Season')
plt.ylabel('Points')
plt.legend()
plt.show()

# Get team points for 2014
knicks_pts_14= nba_2014.pts[nba.fran_id == 'Knicks']
nets_pts_14 = nba_2014.pts[nba.fran_id == 'Nets']

# Difference of mean points for 2014
diff_means_2014 = np.mean(knicks_pts_14) - np.mean(nets_pts_14)
print(diff_means_2014)

plt.clf()
# PLot overlapping histograms to compare points
plt.hist(knicks_pts_14, alpha = 0.5, label = 'Knicks 2014')
plt.hist(nets_pts_14, alpha = 0.5,label = 'Nets 2014')
plt.title('Knicks vs Nets Points 2014 Season')
plt.ylabel('Points')
plt.legend()
plt.show()

plt.clf()
sns.boxplot(data = nba_2010, x = nba_2010.pts, y = nba_2010.fran_id)
plt.title('Varience of Points')
plt.show()


#Create a contingency table to compare game result and game location
location_result_freq = pd.crosstab(nba_2010.game_location, nba_2010.game_result)
print(location_result_freq)

# Convert contingency table to a frequency table
location_result_proportions = location_result_freq/len(location_result_freq)
print(location_result_proportions)

# Find the chi-square contingency
chi2, pval, dof, expected = chi2_contingency(location_result_proportions)
print(expected)
print(chi2)

# Find covariance between forcast in winning and points diff in the end
forcast_points_diff = np.cov(nba_2010.forecast, nba_2010.point_diff)
print(forcast_points_diff)

# Find correlation between the two above
forcast_points_corr, p = pearsonr(nba_2010.forecast, nba_2010.point_diff)
print(forcast_points_corr)

# Scatter plot to verify correlation
plt.clf()
plt.scatter( x = nba_2010.forecast, y = nba_2010.point_diff)
plt.xlabel('Forecasted Win Probability')
plt.ylabel('Point Differential')
plt.title('Win Probability vs Points Differential')
plt.show()
