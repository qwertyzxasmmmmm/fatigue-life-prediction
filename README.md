# Introduction
We use the machine learning models to prediction small sample fatigue life, and use data augmentation to improve the accuracy


### machine learning models for early-life and half life prediction
We choose feature selection to improve the accuracy
<img width="917" height="464" alt="image" src="https://github.com/user-attachments/assets/665b2758-76b1-4770-90d7-6cc4b7cc2a04" />


### sample expansion for MBNN
The new data for ML model is formulated as:X=[X_(n_1 )  ,X_(n_2 ),E,n_2-n_1] ,Y=N-n_2 (1≤ n_1≤n_2≤N/2). Here, X_(n_1 ) and X_(n_2 ) denote the seven features extracted from the hysteresis loops at fatigue cycles n_1 and n_2, respectively. 

### MBNN models for fatigue life prediction
<img width="865" height="224" alt="image" src="https://github.com/user-attachments/assets/e8a08b1d-e0c7-4df7-a02d-20136aee1646" />

### Conclusion
(1) By extracting and pairing hysteresis loop features from different fatigue stages, the PIML-ANN model and PIML-MBNN model was proposed for fatigue life prediction, which could effectively capture the evolving damage mechanisms of IN718 superalloy across both the crack initiation and propagation phases. 
(2) Compared with prediction performance of the PIML-ANN model, the PIML-MBNN model demonstrates significantly improved prediction accuracy, stability, and generalization ability—achieving a MSE that is only 30.0823.35% of that obtained by conventional models. 
(3) The proposed MBNN architecture significantly enhances the predictive performance of the PIML framework by facilitating stage-aware feature learning and seamless integration. When combined with the data augmentation strategy that preserves physical authenticity, MBNN acts as a robust tool for multi-feature analysis, effectively improving the robustness and interpretability of physics-informed fatigue life prediction models.



