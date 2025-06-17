## 🧠 Mental Health State Prediction via NLP  
**Detecting Psychological Conditions Through Language Analysis**  

**Tools**: Python • FastText • XGBoost • Google Colab • Scikit-learn  

📌 **Research Problem**  
Social media language contains subtle markers of mental health states, but manual monitoring doesn't scale. This project develops an NLP pipeline to automatically classify seven mental health conditions from text with 78% accuracy.

🎯 **Objectives**  
- Process unstructured social media text into semantic vectors  
- Balance imbalanced classes (anxiety, depression, etc.)  
- Benchmark ML models (ANN vs. Gradient Boosting)  
- Deploy classifier via NeuroBuddy prototype  

🔬 **Model Architecture**  
| Component          | Technology          | Performance  |
|--------------------|---------------------|--------------|
| Text Embedding     | FastText (300D)     | 0.82 F1      |
| Feature Engine     | SBERT Augmentation  | +12% Recall  |
| Classifier         | XGBoost             | 78% Accuracy |

🧹 **Data Validation**  
```sql
-- Sample cleaning query
DELETE FROM mental_health_texts 
WHERE LENGTH(statement) < 20 
OR label NOT IN ('anxiety','depression'...);

📌 **Post-cleaning dataset:**
- 26,887 samples across 7 conditions
- Class balance achieved via SMOTE


