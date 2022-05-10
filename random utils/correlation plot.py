import seaborn as sns
import matplotlib.pyplot as plt

pdf_temp = some_df

BIG = 12
MED = 10
SMALL = 8
plt.rcParams.update({'font.size': BIG})
plt.rcParams.update({'axes.labelsize': BIG})
plt.rcParams.update({'xtick.labelsize': MED})
plt.rcParams.update({'ytick.labelsize': MED})
plt.rcParams.update({'figure.figsize':[10,10]}) #this is important as it makes the visualizations much easier to read and clip/export!(w,h)

pdf_data_clusters_cor = pdf_temp[["OneWay","InvCountsAvg","Workload","expected_rentals","expected_days","ranking_value"]]

Var_Corr = pdf_data_clusters_cor.corr()
# plot the heatmap and annotation on it
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
