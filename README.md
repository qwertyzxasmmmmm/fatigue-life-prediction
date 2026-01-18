# Introduction
To overcome the challenge of insufficient samples, a facile physics-informed machine learning-based approach is proposed that enables accurate fatigue life prediction using small sample sets. This study mainly solves the key problem of limited dataset size in fatigue life prediction of IN718 supperalloy. Firstly, feature dimensionality reduction techniques were applied to optimize parameter extraction from early-life and half-life fatigue hysteresis loops physics information, followed by physics-informed machine learning based on artificial neural network (PIML-ANN) modeling for life prediction. Innovatively, a data augmentation strategy was developed by pairing the fatigue life n_i with the remaining life N-n_i, expanding the original dataset of 47 valid specimens tested at 400℃ by a factor of 46 (from 47 to 2,177 samples). Building upon this, a physics-informed machine learning based on multi-branch neural network(PIML-MBNN) architecture was designed, leveraging branched ANNs to fully exploit hysteresis loop features for  fatigue life prediction. Experimental results demonstrate that the PIML-ANN model reduced the mean squared error (MSE) by 67% compared to conventional models, while the PIML-MBNN achieved a further 75% reduction in MSE relative to the PIML-ANN, attaining a state-of-the-art MSE of 0.002.


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



