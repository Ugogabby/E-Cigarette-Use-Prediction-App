# Welcone to E-Cigarette Use Prediction Webserver [`open in streamlit`](https://e-cigarette-use-prediction-app.streamlit.app/)  
E-cigarette use has gained prominence as a public health concern, 
particularly among individuals who have battled cancer. Cancer survivors, 
as a vulnerable population, face unique challenges, and understanding their 
use of e-cigarettes is of utmost importance. Veterans exhibit higher rates of tobacco 
product use, including e-cigarettes, compared to the general population [`Odani et al., 2018`](https://doi.org/10.15585/mmwr.mm6701a2). 
This study investigates the relationship between military veteran status and e-cigarette use among cancer survivors and explores the impact of sociodemographic factors on this association.
***   
To accomplish this, we employed Random Forest Classifier to analyze data obtained from the `National Health 
and Nutrition Examination survey`. Our analysis revealed some notable findings. We observed that several factors 
significantly influenced e-cigarette use among cancer survivors. These included gender, age, country of birth, U.S. 
citizenship status, the highest level of education achieved, and the size of the family unit. It was fascinating to observe the 
substantial influence of these variables in predicting e-cigarette usage within this population during our model building.    
## model implimentation  
***
In this work, we present a `Random Forest Classifier` for predicting e-cigarette use among cancer survivors. First, we built and compared the performance of selected machine learning models for the
prediction of e-cigarette use. In logistic regression, `highest_education_grade_received`, `age(year)`, `race_and_hispanic_origin`,
`country_of_birth`, `marital_status`, `veteran_status`, and `gender` were significant predictors. 
Upon model building, the `AUC` was roughly average because of class imbalance, as the positive clsass was 
was `199` as against `12465`. We applied `ADASYN`, a synthetic oversampling techniques. By incorporating `ADASYN`, and Random forest algorithms, the model's capacity to capture minority class was improved.    
***    
***
