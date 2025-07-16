# VIGIL: Video-based Inference for Ground-level Activity Monitoring and Recognition Using LRCN-Enabled Unmanned Aerial Systems

## Description

This project implements an advanced computer vision system for detecting suspicious human activities in crowds using AI-driven Unmanned Aerial Systems (UAS). The system employs multiple deep learning architectures including VGG16-LSTM, Motion Influence Map (MIM), and Long-term Recurrent Convolutional Networks (LRCN) to analyze video sequences and classify human activities.

The project addresses the critical need for automated surveillance systems that can identify potentially dangerous or suspicious behaviors in crowded environments, providing real-time monitoring capabilities for security applications.

## Dataset Information

### Movies Fight Detection Dataset
- **Source**: Kaggle (uploaded by naveenk903)
- **Dataset URL**: [Movies Fight Detection Dataset](https://www.kaggle.com/datasets/naveenk903/movies-fight-detection-dataset)
- **Description**: This dataset contains video clips extracted from various movies, labeled for fight and non-fight scenes. It is designed for the development and benchmarking of action recognition and violence/fight detection algorithms in video surveillance and entertainment analytics.
- **Classes**: Fight, Non-Fight
- **Format**: Video files (various formats)
- **Usage**: Used as the primary dataset for training and testing suspicious activity (fight) detection models.

### KTH Actions Dataset
- **Source**: Royal Institute of Technology (KTH), Sweden
- **Dataset URL**: [KTH Actions Dataset](https://www.csc.kth.se/cvap/actions/)
- **Description**: The KTH Action Dataset contains videos of six human action categories performed several times by 25 subjects in four different scenarios (outdoors, outdoors with scale variation, outdoors with different clothes, indoors). The videos are grayscale with controlled backgrounds and minimal camera motion.
- **Classes**: 6 human action categories (walking, jogging, running, boxing, hand waving, hand clapping)
- **Format**: Video files (.avi format)
- **Usage**: Used as a benchmark for human activity recognition algorithms

### Third-Party Data Citation
This research utilizes two third-party datasets:
1. **Movies Fight Detection Dataset**: Curated and uploaded by naveenk903 on Kaggle, publicly available for research purposes and widely used for fight/violence detection in video analytics.
2. **KTH Actions Dataset**: Provided by KTH Royal Institute of Technology, commonly used for human activity recognition research and validation.

## Code Information

### Algorithms and Implementation

The project implements three main deep learning architectures:

1. **VGG16-LSTM Model**
   - Architecture: VGG16 as feature extractor + LSTM for temporal modeling
   - Purpose: Extracts spatial features using VGG16 and processes temporal sequences with LSTM
   - Performance: 82.67% accuracy, 82.99% precision, 94.25% ROC AUC (on benchmark dataset)

2. **Motion Influence Map (MIM) Model**
   - Architecture: Custom CNN-based model for motion analysis
   - Purpose: Focuses on motion patterns and temporal dynamics
   - Performance: 60.0% accuracy (on benchmark dataset)

3. **Long-term Recurrent Convolutional Network (LRCN)**
   - Architecture: CNN + LSTM hybrid for video sequence analysis
   - Purpose: Combines spatial and temporal feature learning
   - Performance: Comparable to VGG16-LSTM model

### Key Features
- Multi-model ensemble approach
- Real-time video processing capabilities
- Comprehensive evaluation metrics
- Pre-trained models available for immediate use

## Usage Instructions

### Prerequisites
1. Ensure you have Python 3.7+ installed
2. Download the Movies Fight Detection Dataset from Kaggle and place it in a `Dataset` folder
3. (Optional) Download the KTH Actions Dataset for benchmarking
4. Install required dependencies (see Requirements section)

### Implementation Steps

1. **Dataset Setup**
   ```bash
   # Create dataset directory
   mkdir Dataset
   # Download and extract Movies Fight Detection Dataset to Dataset folder
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook "Suspicious Action Detection in Crowds Using AI Driven UAS (1).ipynb"
   ```

4. **Model Training**
   - Execute cells in sequence to train all three models
   - Models will be saved as `.h5` files

5. **Model Evaluation**
   - Run evaluation cells to assess model performance
   - View confusion matrices and classification reports

6. **Real-time Detection**
   - Use the provided video processing functions for real-time analysis
   - Load pre-trained models for immediate inference

### Pre-trained Models
- `Suspicious_Human_Activity_Detection_VGG16_LSTM_Model.h5` (68MB)
- `Suspicious_Human_Activity_Detection_MIM_Model.h5` (18MB)

## Requirements

### Python Libraries
```
tensorflow>=2.4.0
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
pandas>=1.2.0
seaborn>=0.11.0
PIL>=8.0.0
moviepy>=1.0.3
pafy>=0.5.5
```

### System Requirements
- **Operating System**: Windows 10/11, Linux, macOS
- **Hardware**: 
  - Minimum: 8GB RAM, 4GB GPU memory
  - Recommended: 16GB RAM, 8GB+ GPU memory
- **Storage**: 10GB free space for dataset and models
- **GPU**: NVIDIA GPU with CUDA support (recommended)

### Computing Infrastructure
- **Development Environment**: Jupyter Notebook
- **Deep Learning Framework**: TensorFlow 2.x
- **GPU Support**: CUDA-compatible NVIDIA GPUs
- **Memory Management**: Automatic memory allocation with TensorFlow

## Methodology

### Data Processing Pipeline
1. **Video Preprocessing**
   - Frame extraction at 30 FPS
   - Resize frames to 64x64 pixels
   - Normalize pixel values to [0,1] range
   - Handle both Movies Fight Detection and KTH datasets with consistent preprocessing

2. **Feature Extraction**
   - VGG16: Extract spatial features from individual frames
   - LSTM: Process temporal sequences of features
   - MIM: Analyze motion patterns and dynamics

3. **Model Training**
   - Train/Test split: 75%/25%
   - Batch size: 4-8 samples
   - Epochs: 50 (with early stopping)
   - Optimizer: Adam with learning rate 0.001
   - Cross-dataset validation using KTH dataset (optional)

4. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-Score
   - ROC AUC, Matthews Correlation Coefficient
   - Confusion Matrix Analysis
   - Cross-dataset performance evaluation (if using KTH)

### Model Architecture Details

#### VGG16-LSTM Model
```python
# VGG16 base for feature extraction
cnn_base = VGG16(input_shape=(64, 64, 3), include_top=False)
# LSTM layer for temporal modeling
encoded_sequence = LSTM(256)(encoded_frames)
```

#### MIM Model
```python
# Custom CNN architecture for motion analysis
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    # Additional layers...
])
```

## Results and Performance

### Model Performance Comparison

| Model | Accuracy | Precision | F1-Score | ROC AUC |
|-------|----------|-----------|----------|---------|
| VGG16-LSTM | 82.67% | 82.99% | 82.80% | 94.25% |
| MIM | 60.0% | N/A | N/A | N/A |
| LRCN | Comparable | Comparable | Comparable | Comparable |

### Key Findings
- VGG16-LSTM model achieves the best performance
- Motion Influence Map shows potential but needs optimization
- Ensemble approaches could improve overall accuracy

## Limitations

### Technical Limitations
1. **Computational Complexity**: High GPU memory requirements for real-time processing
2. **Dataset Bias**: Limited to fight/non-fight action categories
3. **Environmental Factors**: Performance may degrade in poor lighting conditions
4. **Scale Sensitivity**: Model performance varies with object scale and distance

### Research Limitations
1. **Limited Dataset**: Only fight/non-fight categories from Movies Fight Detection and 6 from KTH
2. **Dataset Diversity**: Limited cross-dataset validation between Movies Fight Detection and KTH
3. **Real-time Constraints**: Processing speed limitations for live video
4. **Generalization**: May not generalize well to unseen action categories

### Future Improvements
1. **Multi-dataset Training**: Incorporate additional datasets for better generalization
2. **Real-time Optimization**: Implement model compression and optimization
3. **Ensemble Methods**: Combine multiple models for improved accuracy
4. **Transfer Learning**: Explore pre-trained models on larger datasets

## Citations

### Dataset Citations
```
Naveen Kumar (2022). Movies Fight Detection Dataset. Kaggle. https://www.kaggle.com/datasets/naveenk903/movies-fight-detection-dataset

Schuldt, C., Laptev, I., & Caputo, B. (2004). Recognizing human actions: A local SVM approach. In Proceedings of the 17th International Conference on Pattern Recognition (ICPR'04) (Vol. 3, pp. 32-36). IEEE.
```

### Related Work
```
Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

Donahue, J., Anne Hendricks, L., Guadarrama, S., Rohrbach, M., Venugopalan, S., Saenko, K., & Darrell, T. (2015). Long-term recurrent convolutional networks for visual recognition and description. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2625-2634).
```

## License & Contribution Guidelines

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code of Conduct
- Be respectful and inclusive
- Follow Python PEP 8 style guidelines
- Add comprehensive documentation for new features
- Include tests for new functionality

---

**Note**: This project is for research and educational purposes. Please ensure compliance with local regulations when deploying surveillance systems in real-world applications.
