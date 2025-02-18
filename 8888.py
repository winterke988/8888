import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as AUC
from sklearn.model_selection import GridSearchCV###网格搜索
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import shap

#导入训练集数据
train_data =pd.read_csv("训练集建模因素.csv",encoding='utf-8')
print(train_data.shape) #获取目前的形状
print(train_data.columns)
trainy=train_data.OUTCOME
trainy.head()
trainx=train_data.drop('OUTCOME',axis=1)
trainx.head()
#训练集数据标准化，建议用StandardScaler标准化连续变量
scaler = StandardScaler()
continuous_columns = ['P_F', 'LAC']  
columns_to_copy = ['OUTCOME','decision_time','Nutritional_Methods','blood_glucose_0_7.8-10','blood_glucose_1_11.1','blood_glucose_2_2.8','mechanical_ventilation']  
scaled_continuous_train = scaler.fit_transform(train_data[continuous_columns]) # 只选择连续变量列进行fit_transform 
scaled_data_train = pd.DataFrame(scaled_continuous_train, columns=continuous_columns)  
scaled_data_train[columns_to_copy] = train_data[columns_to_copy]
trainy=scaled_data_train.OUTCOME
trainy.head()
trainx=scaled_data_train.drop('OUTCOME',axis=1)
trainx.head()


from sklearn.metrics import roc_auc_score as AUC

rfc=RandomForestClassifier(max_depth= 3, min_samples_leaf=6, min_samples_split=2, n_estimators=160,
                           random_state=1,class_weight = 'balanced').fit(trainx, trainy.astype('int'))

pred_rfc1 = rfc.predict_proba(trainx)
print("AUC_train",AUC(trainy.astype('int'),pred_rfc1[:, 1]))

joblib.dump(rfc, 'rfc666.pkl')

# Load the model
model = joblib.load('rfc666.pkl')        
# Define feature options         
decision_time_options = {
          
    0: 'in 6 hour (0)',
          
    1: 'above 6 hour (1)',       
}
                    
Nutritional_Methods_options = {
          
    0: 'EN(0)',
          
    1: 'PN (1)',
          
}
                    
blood_glucose_0_options = {
          
    0: 'NO (0)',
          
    1: 'Yes(1)',
          
}

blood_glucose_1_options = {
          
    0: 'NO (0)',
          
    1: 'Yes(1)',
          
}

blood_glucose_2_options = {
          
    0: 'NO (0)',
          
    1: 'Yes(1)',
          
}

mechanical_ventilation_options = {
          
    0: 'No (0)',
          
    1: 'Yes (1)',
          
}

# Define feature names
feature_names = ['decision_time', 'Nutritional_Methods', 'blood_glucose_0_7.8-10',
       'blood_glucose_1_11.1', 'blood_glucose_2_2.8', 'mechanical_ventilation',
       'P_F', 'LAC']

decision_time= st.selectbox("decision_time (0=in 6 hour, 1=above 6 hour):", options=[0, 1], format_func=lambda x: 'in 6 hour (0)' if x == 0 else 'above 6 hour (1)')
Nutritional_Methods= st.selectbox("Nutritional_Methods (0=EN, 1=PN):", options=[0, 1], format_func=lambda x: 'EN (0)' if x == 0 else 'PN (1)')
blood_glucose_0= st.selectbox("blood_glucose_0_7.8-10 (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
blood_glucose_1= st.selectbox("blood_glucose_1_11.1 (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
blood_glucose_2= st.selectbox("blood_glucose_2_2.8(0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
mechanical_ventilation=st.selectbox("mechanical_ventilation(0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
P_F = st.number_input("P_F:", min_value=1, max_value=850, value=150)
LAC= st.number_input("LAC:", min_value=1, max_value=35, value=1)
# Process inputs and make predictions
feature_values = [decision_time,Nutritional_Methods,blood_glucose_0,blood_glucose_1,blood_glucose_2,mechanical_ventilation,P_F,LAC ]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100
    
    if predicted_class == 1:
          
        advice = (
          
            f"According to our model, you have a high risk of mortality . "
          
            f"The model predicts that your probability of having High mortality risk is {probability:.1f}%. "
          
            "While this is just an estimate, it suggests that you may be at significant risk. "
           )
          
    else:
          
        advice = (
          
            f"According to our model, you have a low mortality risk. "
          
            f"The model predicts that your probability of not having low mortality risk is {probability:.1f}%. "
          
            )
          
    st.write(advice)
    # 添加SHAP解释可视化
    
    # 初始化解释器（需在第一次运行时计算）
    explainer = shap.TreeExplainer(model) 

    # 标准化连续变量（与训练时一致）
    scaled_features = feature_values.copy()
    scaled_features[6] = (feature_values[6] - scaler.mean_[0]) / scaler.scale_[0]  # P_F标准化
    scaled_features[7] = (feature_values[7] - scaler.mean_[1]) / scaler.scale_[1]  # LAC标准化

    # 生成 SHAP 值（注意分类模型的结构）
    sample_df = pd.DataFrame([scaled_features], columns=feature_names)
    shap_values = explainer.shap_values(sample_df)

    # 二分类模型需指定目标类别（通常展示类别1的SHAP值）
    class_idx = 1  # 假设关注阳性类别（高风险）
    expected_value = explainer.expected_value[class_idx]
    
    # 生成并保存力图（使用原始特征值显示）
    plt.figure()
    shap.force_plot(
        expected_value,
        shap_values[class_idx][0],  # 取第一个样本的SHAP值
        features=sample_df.iloc[0].values,  # 用标准化后的值计算，但显示原始名称
        feature_names=feature_names,
        matplotlib=True,
        show=False  # 避免自动弹出窗口
    )
    plt.tight_layout()
    plt.savefig("shap_plot.png", bbox_inches='tight', dpi=300)
    st.image("shap_plot.png")
