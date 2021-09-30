import pandas as pd

# importing data
train_upsampled = pd.read_csv("preprocessed_data/train.csv")

# Splitting Data
from sklearn.model_selection import train_test_split

Y = train_upsampled['winner']
X = train_upsampled.drop(columns=['winner'],inplace=False)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,random_state=2)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,accuracy_score,f1_score,precision_score,recall_score

def metrics(Y_test,Y_pred,average='macro'):
    return str("Accuracy --- > "+str(accuracy_score(Y_test,Y_pred))+"\n"+
               "F1-Score --- > "+str(f1_score(Y_test,Y_pred,average=average))+"\n"+
                "Precision --- > "+str(precision_score(Y_test,Y_pred,average=average))+"\n"+
                "Recall --- > "+str(recall_score(Y_test,Y_pred,average=average)))


rfc = RandomForestClassifier(n_estimators=110,
                            min_samples_split=20,
                            max_features='auto',
                            max_depth=26)

rfc.fit(X_train,Y_train)


rfc.score(X_train,Y_train)
rfc.score(X_test,Y_test)
pred_of_forest = rfc.predict(X_test)

# data viz
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15,6))
sns.heatmap(confusion_matrix(Y_test,pred_of_forest),annot=True,cmap='cool',linewidths=0.5,fmt='d')
plt.ylabel("ACTUAL")
plt.xlabel("PREDICTED")
plt.savefig("CONFUSION MATRIX FOR RANDOM FOREST.png",dpi=120)
plt.close()

with open("metrics.txt",'w') as outfile:
    outfile.write(metrics(Y_test,pred_of_forest))

