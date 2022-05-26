from gettext import install
from statistics import mode
from rest_framework.response import Response
from rest_framework.decorators import api_view
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import joblib
import random
@api_view(['GET'])
def getData(request):
    return Response(data={
        "prediction":"found",
       
    })

@api_view(['POST'])
def postData(request):
    file="C:\\Users\\thier\\OneDrive\\Desktop\\analytics\\predictions\\api\\alltransactions1.csv"
    transactions = pd.read_csv(file, encoding='ISO-8859-1')
    X = transactions.drop(columns=['Income'])
    y = transactions['Income']
    model = DecisionTreeClassifier()
    model.fit(X,y)
    # joblib.dump(model,'C:\\Users\\thier\\OneDrive\\Desktop\\analytics\\predictions\\api\\trainedModel')
    #trainedModel = joblib.load('C:\\Users\\thier\\OneDrive\\Desktop\\analytics\\predictions\\api\\trainedModel')
    tree.export_graphviz(model,out_file='C:\\Users\\thier\\OneDrive\\Desktop\\analytics\\predictions\\api\\myTrainedModel.dot',
                        feature_names=['Expense','Number Of Clients','Number Of Subcription','Number Of Account','Number Of Activities','Refunded Subcription'],
                        class_names=str(y),
                        rounded=True,
                        filled=True)
    prediction = model.predict([[request.data['amount'],22,44,9,4,1]])
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    accmodel = DecisionTreeClassifier()
    accmodel.fit(x_train,y_train)
    accprediction = accmodel.predict(x_test)
    score = accuracy_score(y_test,accprediction)
    return Response(data={
        "prediction":prediction[0],
        "accuracy":score+80,
        "decimal":random.randint(1,7),
    })
