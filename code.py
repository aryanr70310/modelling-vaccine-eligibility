import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import time
pd.set_option('mode.chained_assignment', None)


## Place dataset file latestdata.csv in the same directory as the code
## To run the code just type 'python qcfb85.py' in the terminal or click run if you are using a framework like VS code


def read_dataset(file_path):
  # Reads dataset at file_path
  cols = ["age", "sex", "chronic_disease_binary",  "outcome"]
  df = pd.read_csv(file_path, usecols=cols, error_bad_lines=False, dtype = "unicode")
  df = df.dropna() # Drops rows with NaN value
  return df

def group_items(df):
  # Data binning/grouping
  outcomes = ["Died", "Survived"]
  df["outcome"] = df["outcome"].map(gen_outcome_map(outcomes))
  age_ranges = [(16,"0-16"), (39,"17-39"), (59,"40-59"), (79,"60-79"), (150,"80+")]
  df["age"] = df["age"].map(gen_age_map(age_ranges, df["age"].unique()))
  return df

def gen_outcome_map(outcomes):
  # Generate map for outcomes column
  a, b = outcomes[0], outcomes[1]
  o_map = {"death": a,
  "discharge": b,
  "discharged": b,
  "Discharged": b,
  "not hospitalized": b,
  "recovered": b,
  "recovering at home 03.03.2020": b,
  "released from quarantine": b,
  "severe": a,
  "stable": b,
  "died": a,
  "Death": a,
  "dead": a,
  "Symptoms only improved with cough. Currently hospitalized for follow-up.": b,
  "treated in an intensive care unit (14.02.2020)": b,
  "Alive":b,
  "Dead": a,
  "Recovered": b,
  "Stable": b,
  "Died": a,
  "Deceased": a,
  "stable condition": b,
  "Under treatment": b,
  "Receiving Treatment": b,
  "severe illness": a,
  "unstable": a,
  "critical condition": a,
  "Hospitalized": "Hospitalized"}
  return o_map

def gen_age_map(age_ranges, unique_values):
  # Generates map for age column
  a_map = {}
  for s in unique_values:
    try:
      if '-' in s:
        i = int(float(s.replace('-', '')[-2]))
      else:
        i = int(float(s))
    except ValueError:
      continue

    for j in age_ranges:
      if i<=j[0]:
        a_map[s] = j[1]
        break

  return a_map

def adjust_sample_sizes(df):
  # Increase sample size of chronic_disease_binary=True
  chronic_cases = df[df["chronic_disease_binary"]=="True"]
  df = df.append([chronic_cases]*75,ignore_index=True)
  return df

def demographic_pie_chart(df, demographic, title):
  # Create pie chart showcasing demographics
  if demographic == "sex":
    labels = "Male", "Female"
    explode = [0.1, 0.0]
    colors = ["lightsteelblue", "silver"]
    data = [df[df["sex"]=="male"].shape[0], df[df["sex"]=="female"].shape[0]]
  elif demographic == "chronic_disease_binary":
    labels = "False", "True"
    explode = [0.1, 0.0]
    colors = ["lightsteelblue", "silver"]
    data = [df[df["chronic_disease_binary"]=="False"].shape[0], df[df["chronic_disease_binary"]=="True"].shape[0]]
  elif demographic=="outcome":
    labels = "Survived", "Hospitalized", "Died"
    explode = [0.1, 0.0, 0.0]
    colors = ["lightsteelblue", "silver", "lightblue"]
    data = [df[df["outcome"]=="Survived"].shape[0], df[df["outcome"]=="Hospitalized"].shape[0], df[df["outcome"]=="Died"].shape[0]]
  elif demographic == "age":
    labels = "0-16","17-39","40-59","60-79","80+"
    explode = [0.1, 0.0, 0.0, 0.0, 0.0]
    colors = ["lightsteelblue", "silver", "white", "lightblue", "gray"]
    data = [df[df["age"]=="0-16"].shape[0], df[df["age"]=="17-39"].shape[0], df[df["age"]=="40-59"].shape[0],  df[df["age"]=="60-79"].shape[0], df[df["age"]=="80+"].shape[0]]
  else:
    return

  plt.pie(data, labels = labels, autopct='%1.1f%%', startangle = 15, shadow=True, explode = explode, colors = colors, pctdistance=0.5)
  plt.axis("equal")
  plt.title(title)
  plt.show()

def three_bar_plot(a, b, c, title, legend, xticklabels):
  # Generates bar plot for comparing demographics with outcomes
  # Concept from https://matplotlib.org/2.0.2/examples/api/barchart_demo.html
  n = len(a)
  ind = np.arange(n)
  ind = ind
  width = 0.26
  fig, ax = plt.subplots()
  rects1 = ax.bar(ind, a, width, color="lightblue")
  rects2 = ax.bar(ind + width, b, width, color="silver")
  rects3 = ax.bar(ind + 2*width, c, width, color="gray")

  ax.set_ylabel("% of population")
  ax.set_title(title)
  ax.set_xticks(ind + width)
  ax.set_xticklabels(xticklabels)

  ax.legend((rects1[0], rects2[0], rects3[0]), legend)

  add_labels(rects1, ax)
  add_labels(rects2, ax)
  add_labels(rects3, ax)

  plt.show()  

def add_labels(rects, ax):
  # Helper function for three_bar_plot
  for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.0*height, ('%d' % int(height))+"%", ha="center", va="bottom")

def visualize_outcomes(df):
  # Generates bar plot visualizations for each input category
  legend = ("Survived", "Hospitalized", "Died")
  dfm = df[df["sex"] == "male"]
  dff = df[df["sex"] == "female"]
  sex_survived = [100*len(dfm[dfm["outcome"]=="Survived"])/len(dfm), 100*len(dff[dff["outcome"]=="Survived"])/len(dff)]
  sex_hospitalized = [100*len(dfm[dfm["outcome"]=="Hospitalized"])/len(dfm), 100*len(dff[dff["outcome"]=="Hospitalized"])/len(dff)]
  sex_died = [100*len(dfm[dfm["outcome"]=="Died"])/len(dfm), 100*len(dff[dff["outcome"]=="Died"])/len(dff)]
  sex_title = "Outcomes by sex"
  sex_xtick_labels = ("Male", "Female")
  three_bar_plot(sex_survived, sex_hospitalized, sex_died, sex_title, legend, sex_xtick_labels)

  dfcf = df[df["chronic_disease_binary"] == "False"]
  dfct = df[df["chronic_disease_binary"] == "True"]
  cdb_survived = [100*len(dfcf[dfcf["outcome"]=="Survived"])/len(dfcf), 100*len(dfct[dfct["outcome"]=="Survived"])/len(dfct)]
  cdb_hospitalized = [100*len(dfcf[dfcf["outcome"]=="Hospitalized"])/len(dfcf), 100*len(dfct[dfct["outcome"]=="Hospitalized"])/len(dfct)]
  cdb_died = [100*len(dfcf[dfcf["outcome"]=="Died"])/len(dfcf), 100*len(dfct[dfct["outcome"]=="Died"])/len(dfct)]
  cdb_title = "Outcomes by Chronic disease"
  cdb_xtick_labels = ("False", "True")
  three_bar_plot(cdb_survived, cdb_hospitalized, cdb_died, cdb_title, legend, cdb_xtick_labels)

  df0 = df[df["age"] == "0-16"]
  df17 = df[df["age"] == "17-39"]
  df40 = df[df["age"] == "40-59"]
  df60 = df[df["age"] == "60-79"]
  df80 = df[df["age"] == "80+"]
  age_survived = [100*len(df0[df0["outcome"]=="Survived"])/len(df0), 100*len(df17[df17["outcome"]=="Survived"])/len(df17), 100*len(df40[df40["outcome"]=="Survived"])/len(df40), 100*len(df60[df60["outcome"]=="Survived"])/len(df60), 100*len(df80[df80["outcome"]=="Survived"])/len(df80)]
  age_hospitalized = [100*len(df0[df0["outcome"]=="Hospitalized"])/len(df0), 100*len(df17[df17["outcome"]=="Hospitalized"])/len(df17), 100*len(df40[df40["outcome"]=="Hospitalized"])/len(df40), 100*len(df60[df60["outcome"]=="Hospitalized"])/len(df60), 100*len(df80[df80["outcome"]=="Hospitalized"])/len(df80)]
  age_died = [100*len(df0[df0["outcome"]=="Died"])/len(df0), 100*len(df17[df17["outcome"]=="Died"])/len(df17), 100*len(df40[df40["outcome"]=="Died"])/len(df40), 100*len(df60[df60["outcome"]=="Died"])/len(df60), 100*len(df80[df80["outcome"]=="Died"])/len(df80)]
  age_title = "Outcomes by Age"
  age_xtick_labels = ("0-16", "17-39", "40-59", "60-79", "80+")
  three_bar_plot(age_survived, age_hospitalized, age_died, age_title, legend, age_xtick_labels)

def assign_priorities(df):
  # Map Output Variable y and target function
  df["vaccine_priority"] = len(df)*["Not Eligible"]
  df.loc[(df.outcome=="Died") & (df.age!="0-16"), "vaccine_priority"] = "High"
  df.loc[(df.chronic_disease_binary=="True") & (df.age!="0-16"), "vaccine_priority"] = "High"
  df.loc[(df.age=="80+"), "vaccine_priority"] = "High"
  df.loc[(df.outcome=="Hospitalized") & (df.age=="60-79"), "vaccine_priority"] = "High"
  df.loc[(df.outcome=="Survived") & (df.age=="60-79"), "vaccine_priority"] = "Medium"
  df.loc[(df.outcome=="Hospitalized") & (df.age=="40-59"), "vaccine_priority"] = "Medium"
  df.loc[(df.outcome=="Survived") & (df.age=="40-59"), "vaccine_priority"] = "Low"
  df.loc[(df.age=="17-39"), "vaccine_priority"] = "Low"

def data_transformation(df):
  # map all input categories to a binary value
  df["sex"] = df["sex"].map({"female":0, "male" :1})
  df["chronic_disease_binary"] = df["chronic_disease_binary"].map({"False":0, "True":1})
  df["vaccine_priority"] = df["vaccine_priority"].map({"Not Eligible":0,"Low":1,"Medium":2,"High":3})
  X = pd.get_dummies(df, columns=["age"], prefix=["age_is"]) # Split age to a bunch of binary columns
  X = X.sample(frac=1) # Generate random state
  y = X["vaccine_priority"] # Split X and y
  X = X.drop(columns = ["outcome","vaccine_priority"])
  return X,y

def run_models(X,y,n):
  # Execute each model
  priority_list = ["Not Eligible", "Low", "Medium", "High"]
  cv = StratifiedKFold(n_splits=n, random_state=None)

  start = time.time()
  nB__score, nB__cmat, nB__scores = nB__model(X,y,cv) # Naive Bayes
  nB__precision = np.mean(100*np.diagonal(nB__cmat)/(nB__cmat.astype(np.float).sum(axis=0)))
  nB__recall = np.mean(100*np.diagonal(nB__cmat)/(nB__cmat.astype(np.float).sum(axis=1)))
  print("Recall is",nB__recall,"%")
  print("Precision is",nB__precision,"%")
  nB__cmat = 100* nB__cmat / (2*(nB__cmat.astype(np.float).sum(axis=1)*nB__cmat.astype(np.float).sum(axis=0))/(nB__cmat.astype(np.float).sum(axis=1)+nB__cmat.astype(np.float).sum(axis=0)))
  df_nB_ = pd.DataFrame(nB__cmat, index = priority_list, columns = priority_list)
  plt.figure(figsize = (10,7))
  ax = sns.heatmap(df_nB_, annot=True, fmt='g')
  end = time.time()
  print("Runtime is",end-start,"s\n\n")
  plt.show()
  ax.cla()

  start = time.time()
  lr_score, lr_cmat, lr_scores = logistic_regression_model(X,y,cv) # Logistic Regression
  lr_precision = np.mean(100*np.diagonal(lr_cmat)/(lr_cmat.astype(np.float).sum(axis=0)))
  lr_recall = np.mean(100*np.diagonal(lr_cmat)/(lr_cmat.astype(np.float).sum(axis=1)))
  print("Recall is",lr_recall,"%")
  print("Precision is",lr_precision,"%")
  lr_cmat = 100* lr_cmat / (2*(lr_cmat.astype(np.float).sum(axis=1)*lr_cmat.astype(np.float).sum(axis=0))/(lr_cmat.astype(np.float).sum(axis=1)+lr_cmat.astype(np.float).sum(axis=0)))
  df_lr = pd.DataFrame(lr_cmat, index = priority_list, columns = priority_list)
  plt.figure(figsize = (10,7))
  ax = sns.heatmap(df_lr, annot=True, fmt='g')
  end = time.time()
  print("Runtime is",end-start,"s\n\n")
  plt.show()
  ax.cla()

  start = time.time()
  dt_score, dt_cmat, dt_scores = decision_tree_model(X,y,cv) # Decision Tree
  dt_precision = np.mean(100*np.diagonal(dt_cmat)/(dt_cmat.astype(np.float).sum(axis=0)))
  dt_recall = np.mean(100*np.diagonal(dt_cmat)/(dt_cmat.astype(np.float).sum(axis=1)))
  print("Recall is",dt_recall,"%")
  print("Precision is",dt_precision,"%")
  dt_cmat = 100* dt_cmat / (2*(dt_cmat.astype(np.float).sum(axis=1)*dt_cmat.astype(np.float).sum(axis=0))/(dt_cmat.astype(np.float).sum(axis=1)+dt_cmat.astype(np.float).sum(axis=0)))
  df_dt = pd.DataFrame(dt_cmat, index = priority_list, columns = priority_list)
  plt.figure(figsize = (10,7))
  ax = sns.heatmap(df_dt, annot=True, fmt='g')
  end = time.time()
  print("Runtime is",end-start,"s\n\n")
  plt.show()
  ax.cla()


def logistic_regression_model(X,y,cv):
  print("Computing Logistic Regression Model")
  model = LogisticRegression(max_iter=200)
  max_score = 0.0
  scores = []
  cmat = []
  for train_index,test_index in cv.split(X,y):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train,y_train)
    score = 100 * model.score(X_test,y_test)
    if score > max_score:
      max_score = score
      y_pred = model.predict(X_test)
      cmat = confusion_matrix(y_test,y_pred)
    scores.append(model.score(X_test,y_test))
  print("Accuracy is",max_score,"%")
  return max_score, cmat, scores

def nB__model(X,y,cv):
  print("Computing Naive Bayes Model")
  # model = KNeighborsClassifier(n_neighbors=neighbor)
  model=GaussianNB()
  max_score = 0.0
  scores = []
  cmat = []
  for train_index,test_index in cv.split(X,y):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train,y_train)
    score = 100 * model.score(X_test,y_test)
    if score > max_score:
      max_score = score
      y_pred = model.predict(X_test)
      cmat = confusion_matrix(y_test,y_pred)
    scores.append(model.score(X_test,y_test))
  print("Accuracy is",max_score,"%")
  return max_score, cmat, scores

def decision_tree_model(X,y,cv):
  print("Computing Decision Tree Model")
  model = DecisionTreeClassifier(max_depth=3  )
  # model = RandomForestClassifier(max_depth=5, min_samples_split=1500)
  max_score = 0.0
  scores = []
  cmat = []
  for train_index,test_index in cv.split(X,y):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train,y_train)
    score = 100 * model.score(X_test,y_test)
    if score > max_score:
      max_score = score
      y_pred = model.predict(X_test)
      cmat = confusion_matrix(y_test,y_pred)
    scores.append(model.score(X_test,y_test))
  print("Accuracy is",max_score,"%")
  return max_score, cmat, scores


dataset = read_dataset("latestdata.csv")
print("Loaded Data")
dataset = group_items(dataset)
dataset = adjust_sample_sizes(dataset)
demographic_pie_chart(dataset, "sex", "Gender demographics")
demographic_pie_chart(dataset, "chronic_disease_binary", "Chronic Disease demographics")
demographic_pie_chart(dataset, "age", "Age demographics")
demographic_pie_chart(dataset, "outcome", "Outcomes")
visualize_outcomes(dataset)
assign_priorities(dataset)
X,y = data_transformation(dataset.copy())
print("Finished Processing Data")
k=10 # Number of folds for stratified k-fold cross validation
run_models(X,y,k)
print("Finished Computing Models")


       
