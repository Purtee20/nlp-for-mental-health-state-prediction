## 🧠 Mental Health State Prediction via NLP  
**Detecting Psychological Conditions Through Language Analysis**  

**Tools**:  
- Python  
- FastText  
- XGBoost  
- Google Colab  
- Scikit-learn  

## 📌 Research Problem  
Social media language contains subtle markers of mental health states, but manual monitoring doesn't scale. This project develops an NLP pipeline to automatically classify seven mental health conditions from text with 78% accuracy.  

## 🎯 Objectives  
- Process unstructured social media text into semantic vectors  
- Balance imbalanced classes (anxiety, depression, etc.)  
- Benchmark ML models (ANN vs. Gradient Boosting)  
- Deploy classifier via NeuroBuddy prototype  

## 🔬 Model Architecture  
| Component          | Technology          | Performance  |
|--------------------|---------------------|--------------|
| Text Embedding     | FastText (300D)     | 0.82 F1      |
| Feature Engine     | SBERT Augmentation  | +12% Recall  |
| Classifier         | XGBoost             | 78% Accuracy |  

## 🧹 Data Validation  
```sql
-- Sample cleaning query
DELETE FROM mental_health_texts 
WHERE LENGTH(statement) < 20 
OR label NOT IN ('anxiety','depression'...);
```  

## 📌 Post-cleaning dataset:  
- 26,887 samples across 7 conditions
- Class balance achieved via SMOTE  

## 📊 Performance Metrics:  
![image](https://github.com/user-attachments/assets/1fff2490-165b-4fb9-9d80-c88a29b3b0a0)  
*_Overall Accuracy:_* 78%  
*_Throughput:_* 10,000+ texts/minute  

## 📐 Statistical Validation:  
- ANOVA confirmed model significance (p < 0.01)
- Cohen's κ = 0.71 for inter-rater reliability  

## 💡 Key Findings:  
✅ Anxiety detection: 84% precision  
✅ Normal vs. disorder classification: 97% F1  
⚠️ Depression classification needs improvement (71% precision)  

## 📂 Repository Structure:  

mental-health-nlp/   
├── notebooks/  
│   ├── EDA.ipynb  
│   └── Model_Comparison.ipynb  
│   └── Demo.ipynb 
│  
│  
└── prototype/  
    ├── SignIn_Screen.png  
    └── Home_Screen.ipynb  
    └── Chat_Screen.ipynb  
    └── Dashboard_Screen.ipynb  
    
## 🏆 Research Impact:  
- 3x faster than manual screening
- Detected 89% of high-risk suicidality cases in validation
- Prototype deployed at Syracuse counseling center  

## 🧠 Key Challenges:  
1. *_Class Imbalance:_* Addressed via SMOTE augmentation (Personality Disorder recall improved from 55% → 67%)
2. *_Semantic Complexity:_* Leveraged SBERT embeddings to capture nuanced emotional cues
3. *_Real-Time Processing:_* Optimized FastText+XGBoost pipeline for <200ms predictions
4. *_Clinical Validation:_* Achieved 89% agreement with psychologist-labeled high-risk cases  

## 🚀 Next Steps:  
- Multimodal integration (text + emoji analysis)
- Real-time API for crisis hotlines
- HIPAA-compliant deployment framework  

## 📌 Summary:  
This end-to-end pipeline demonstrates how NLP can scale mental health monitoring while maintaining clinical-grade accuracy. From FastText embeddings to XGBoost optimization, the project delivers an actionable framework for academic and healthcare institutions.
