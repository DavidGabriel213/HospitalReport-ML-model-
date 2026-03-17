import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(9,9))
gs=plt.GridSpec(4,4)
df=pd.read_csv("/storage/emulated/0/Download/nigerian_health_messy.csv")
df.drop_duplicates(inplace=True)
df["Age"]=np.abs(df["Age"])
q1=df["Age"].quantile(0.25)
q3=df["Age"].quantile(0.75)
iqr=q3-q1
min1=q3-1.5*iqr
max1=q3+1.5*iqr
df["Age"]=df["Age"].clip(min1,max1)
df["Age"]=(df["Age"].fillna(df["Age"].median())).astype(int)
df["Gender"]=df["Gender"].astype(str).str.strip()
gender_correction={"Male":"Male","Female":"Female","male":"Male","female":"Female","M":"Male","F":"Female"}
df["Gender"]=df["Gender"].map(gender_correction)
df["Gender"]=df["Gender"].fillna(df["Gender"].mode()[0])
df["State"]=df["State"].astype(str).str.strip()
df["Occupation"]=df["Occupation"].astype(str).str.strip()
df["Diastolic_BP"]=(df["Diastolic_BP"].fillna(df["Systolic_BP"])).astype(str).str.strip()
df["Systolic_BP"]=df["Systolic_BP"].astype(str).str.strip()
def systolic_correction(c):
    if "/" in c:
        k=c.index("/")
        return c[:k]
    else:
        return c
df["Systolic_BP"]=df["Systolic_BP"].apply(lambda x:systolic_correction(x))
df["Systolic_BP"]=pd.to_numeric(df["Systolic_BP"],errors="coerce")
def diastolic_correction(c):
    if "/" in c:
        k=c.index("/")
        return c[k+1:]
    else:
        return c
df["Diastolic_BP"]=df["Diastolic_BP"].apply(lambda x: diastolic_correction(x))
df["Diastolic_BP"]=pd.to_numeric(df["Diastolic_BP"],errors="coerce")
max2=df["Diastolic_BP"].quantile(0.75)+1.5*(df["Diastolic_BP"].quantile(0.75)-df["Diastolic_BP"].quantile(0.25))
min2=df["Diastolic_BP"].quantile(0.75)-1.5*(df["Diastolic_BP"].quantile(0.75)-df["Diastolic_BP"].quantile(0.25))
df["Diastolic_BP"]=df["Diastolic_BP"].clip(min2,max2)
df["BMI"]=df["BMI"].astype(str).str.replace("kg/m2","").str.strip()
df["BMI"]=np.abs(pd.to_numeric(df["BMI"],errors="coerce"))
df["BMI"]=df["BMI"].fillna(df["BMI"].median())

max3=df["Cholesterol"].quantile(0.75)+1.5*(df["Cholesterol"].quantile(0.75)-df["Cholesterol"].quantile(0.25))
min3=df["Cholesterol"].quantile(0.75)-1.5*(df["Cholesterol"].quantile(0.75)-df["Cholesterol"].quantile(0.25))
df["Cholesterol"]=df["Cholesterol"].clip(min3,max3)
df["Cholesterol"]=df["Cholesterol"].fillna(df["Cholesterol"].mean()).round(1)
df["BloodSugar"]=df["BloodSugar"].astype(str).str.replace("mg/dL","").str.strip()
df["BloodSugar"]=pd.to_numeric(df["BloodSugar"])
df["HeartRate"]=df["HeartRate"].fillna(df["HeartRate"].mean())
smoking_correction={"1":"Yes","0":"No"}
df["Smoking"]=df["Smoking"].astype(str).str.capitalize().str.strip()
df["Smoking"]=df["Smoking"].replace(smoking_correction)
df["AlcoholConsumption"]=df["AlcoholConsumption"].astype(str).str.capitalize().str.strip()
df["ExerciseFrequency"]=df["ExerciseFrequency"].astype(str).str.capitalize().str.strip()
df["FamilyHistory"]=df["FamilyHistory"].astype(str).str.capitalize().str.strip()
FHistory_correction={"1":"Yes","0":"No"}
df["FamilyHistory"]=df["FamilyHistory"].replace(FHistory_correction)
df["DiseaseRisk"]=df["DiseaseRisk"].astype(str).str.strip()
df["PulsePressure"]=df["Systolic_BP"]-df["Diastolic_BP"]
df["MAP"]=(df["Diastolic_BP"]+df["PulsePressure"]/3).round(2)
bins=[0,18.5,24.9,34.9,39.9,100]
labels=["Underweight","Normal","Overweight","Obesity Class1","Severe Obesity"]
df["BMI classification"]=pd.cut(df['BMI'],bins=bins,labels=labels)
bins1=[0,35,60,160]
labels1=["Young","Middle","Senior"]
df["Age Group"]=pd.cut(df["Age"],bins=bins1,labels=labels1)
ax_eda1 = fig.add_subplot(gs[3, :2])
df_risk = df["DiseaseRisk"].value_counts()
ax_eda1.bar(["Low Risk", "High Risk"], df_risk.values, color=["green","red"])
ax_eda1.set_title("Disease Risk Distribution")
ax_eda2 = fig.add_subplot(gs[3, 2:])
avg_age = df.groupby("DiseaseRisk")["Age"].mean()
ax_eda2.bar(["Low Risk", "High Risk"], avg_age.values, color=["blue","orange"])
ax_eda2.set_title("Average Age by Disease Risk")
df.to_csv("Hosital_clearn+engineering.csv")
#ML
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
le=LabelEncoder()
df["Smoking"]=le.fit_transform(df["Smoking"])
df["AlcoholConsumption"]=le.fit_transform(df["AlcoholConsumption"])
df["ExerciseFrequency"]=le.fit_transform(df["ExerciseFrequency"])
df["FamilyHistory"]=le.fit_transform(df["FamilyHistory"])
df["DiseaseRisk"]=le.fit_transform(df["DiseaseRisk"])
df["Occupation"]=le.fit_transform(df["Occupation"])
df["Age Group"]=le.fit_transform(df["Age Group"])
X = df[["Age", "Systolic_BP", "Diastolic_BP", "BMI", "Cholesterol", "BloodSugar", "HeartRate", "PulsePressure", "MAP","Smoking", "AlcoholConsumption", "ExerciseFrequency","FamilyHistory", "Age Group"]]
y=df["DiseaseRisk"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=7)
#logistics
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
classification=classification_report(y_test,y_pred)
print(f"Accuracy(LogisticModel): {accuracy:.4f}")
print(f"ClassificationReport(LogisticModel): {classification}")
print("---------------------------")
cm=confusion_matrix(y_test,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Low","High"])
ax=fig.add_subplot(gs[0,:2])
disp.plot(ax=ax,cmap='Blues')
ax.set_title("LogisticRegression")
#Tree
model1=DecisionTreeClassifier(max_depth=5,random_state=7)
model1.fit(X_train,y_train)
y_pred1=model1.predict(X_test)
accuracy1=accuracy_score(y_test,y_pred1)
classification1=classification_report(y_test,y_pred1)
print(f"Accuracy(DecisionTree): {accuracy1:.4f}")
print(f"ClassificationReport(DecisionTree): {classification1}")
cm1=confusion_matrix(y_test,y_pred1)
disp1=ConfusionMatrixDisplay(confusion_matrix=cm1,display_labels=["Low","High"])
ax1=fig.add_subplot(gs[0,2:])
disp1.plot(ax=ax1,cmap="Blues")
ax1.set_title("DecisionTree")
print("---------------------------")
#Forest
model2=RandomForestClassifier(n_estimators=100,random_state=7)
model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)
accuracy2=accuracy_score(y_test,y_pred2)
classification2=classification_report(y_test,y_pred2)
print(f"Accuracy(RandomForest): {accuracy2:.4f}")
print(f"ClassificationReport: {classification2}")
print("----------------------------")
cm2=confusion_matrix(y_test,y_pred2)
disp2=ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=["Low","High"])
ax2=fig.add_subplot(gs[2,:2])
disp2.plot(ax=ax2,cmap="Blues")
ax2.set_title("RandomForest")
#featureimportance
importance=model2.feature_importances_
feature=X.columns
df_FImportance=pd.DataFrame({"importance":importance,"features":feature})
df_FImportance=df_FImportance.sort_values("importance")
ax3=fig.add_subplot(gs[1,:])
ax3.barh(df_FImportance["features"],df_FImportance["importance"])
ax3.set_xlabel("importance")
ax3.set_title("ImportanceChart")
#finetuningthe best
params={"min_samples_split":[3,7,9,13],"max_depth":[3,5,9,11,None],"n_estimators":[40,70,100,190,220]}
grid=GridSearchCV(RandomForestClassifier(random_state=7),params,cv=5,scoring="accuracy",verbose=1)
grid.fit(X_train,y_train)
print("best finetuned parameters:",grid.best_params_)
print("grid best score: ",grid.best_score_)
y_pred3=grid.best_estimator_.predict(X_test)
accuracy3=accuracy_score(y_test,y_pred3)
classi_report3=classification_report(y_test,y_pred3)
print(f"AccuracyScore(RandomForest): {accuracy3:.4f}")
print(f"ClassificationReport(RandomForest): {classi_report3}")
cm3=confusion_matrix(y_test,y_pred3)
disp3=ConfusionMatrixDisplay(confusion_matrix=cm3,display_labels=["Low","High"])
ax3=fig.add_subplot(gs[2,2:])
disp3.plot(ax=ax3,cmap="Blues")
ax3.set_title("Finetuned(RandomForest)")
plt.tight_layout()
plt.show()