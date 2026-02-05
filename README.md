# 📘 Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow  
<img width="2752" height="1536" alt="unnamed" src="https://github.com/user-attachments/assets/00e57f80-155b-4a28-943f-5023953d4a7f" />

![Hands-On Machine Learning](https://images-na.ssl-images-amazon.com/images/I/81O8BkMAc-L.jpg)

> **Personal learning journey through "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd Edition)" by Aurélien Géron**

This repository documents my hands-on practice, notes, experiments, and implementations while working through one of the most comprehensive machine learning books available. It serves as both a personal study log and a reference resource for anyone learning practical machine learning with Python.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/jupyter-notebooks-orange.svg)](https://jupyter.org/)

---

## 📖 About This Repository

This repository contains:

- **Jupyter Notebooks**: Step-by-step implementations of algorithms and concepts
- **Personal Notes**: Concise explanations and key takeaways from each chapter
- **Code Examples**: Practical implementations using Scikit-Learn, Keras, and TensorFlow
- **Experiments**: Additional explorations and variations beyond the book's examples
- **Exercises**: Solutions to end-of-chapter exercises

The content is organized chapter-by-chapter, following the structure of the book's 3rd edition.

---

## 📂 Repository Structure

```
Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow/
│
├── Chapter 1 - The Machine Learning Landscape/
│   ├── notebooks/
│   ├── notes/
│   └── exercises/
│
├── Chapter 2 - End-to-End Machine Learning Project/
│   ├── notebooks/
│   ├── notes/
│   └── exercises/
│
├── Chapter 3 - Classification/
│   └── [Coming soon]
│
├── Chapter 4 - Training Models/
│   └── [Coming soon]
│
└── ... [Additional chapters]
```

---

## 📚 Book Coverage

### Part I: The Fundamentals of Machine Learning

1. **The Machine Learning Landscape** - Introduction to ML concepts, types of learning systems, and main challenges
2. **End-to-End Machine Learning Project** - Complete workflow from data acquisition to model deployment
3. **Classification** - Binary and multiclass classification, performance metrics
4. **Training Models** - Linear regression, gradient descent, regularization
5. **Support Vector Machines** - SVMs for classification and regression
6. **Decision Trees** - Tree-based models and the CART algorithm
7. **Ensemble Learning and Random Forests** - Voting classifiers, bagging, boosting, stacking
8. **Dimensionality Reduction** - PCA, manifold learning, and other techniques
9. **Unsupervised Learning Techniques** - Clustering, anomaly detection

### Part II: Neural Networks and Deep Learning

10. **Introduction to Artificial Neural Networks with Keras** - Building and training neural networks
11. **Training Deep Neural Networks** - Advanced optimization techniques and regularization
12. **Custom Models and Training with TensorFlow** - Building custom models and training loops
13. **Loading and Preprocessing Data with TensorFlow** - Data pipelines and preprocessing
14. **Deep Computer Vision Using Convolutional Neural Networks** - CNNs and image processing
15. **Processing Sequences Using RNNs and CNNs** - Sequence models for time series and NLP
16. **Natural Language Processing with RNNs and Attention** - Advanced NLP techniques
17. **Autoencoders, GANs, and Diffusion Models** - Generative models
18. **Reinforcement Learning** - Q-learning, policy gradients, and deep RL
19. **Training and Deploying TensorFlow Models at Scale** - Production deployment strategies

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Basic understanding of Python programming
- Familiarity with NumPy and Pandas (helpful but not required)
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/M-F-Tushar/Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow.git
   cd Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

### Required Libraries

The main libraries used throughout this repository:

- **Scikit-Learn**: Machine learning algorithms and tools
- **TensorFlow**: Deep learning framework
- **Keras**: High-level neural networks API
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization

---

## 💡 Key Learning Objectives

By working through this repository, you'll learn:

- **Fundamentals**: Core ML concepts, supervised/unsupervised learning, overfitting/underfitting
- **Practical Skills**: Data preprocessing, feature engineering, model evaluation
- **Classical ML**: Linear models, SVMs, decision trees, ensemble methods
- **Deep Learning**: Neural networks, CNNs, RNNs, transformers
- **Advanced Topics**: Transfer learning, GANs, reinforcement learning
- **Production**: Model deployment, scaling, and best practices

---

## 🎯 How to Use This Repository

### For Self-Study
1. Start with Chapter 1 and progress sequentially
2. Read the book chapter first
3. Review the notes in this repository
4. Run the Jupyter notebooks
5. Attempt the exercises independently
6. Compare your solutions with those provided

### For Quick Reference
- Use the chapter folders to find specific topics
- Review notes for quick conceptual refreshers
- Refer to code examples for implementation details

### For Practice
- Modify the provided notebooks with different datasets
- Experiment with hyperparameters
- Try implementing variations of the algorithms
- Apply techniques to your own projects

---

## 📊 Datasets Used

Throughout this repository, we work with various datasets including:

- **California Housing**: Regression problem for predicting house prices
- **MNIST**: Handwritten digit classification
- **Fashion MNIST**: Clothing image classification
- **CIFAR-10**: Object recognition in images
- **IMDB Reviews**: Sentiment analysis
- **Custom datasets**: Various real-world examples

Most datasets are automatically downloaded by the notebooks when needed.

---

## 🔗 Additional Resources

### Official Resources
- [Book's Official GitHub Repository](https://github.com/ageron/handson-ml3)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)

### Recommended Learning
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)

### Communities
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [Kaggle](https://www.kaggle.com/)
- [Papers With Code](https://paperswithcode.com/)

---

## 🤝 Contributing

While this is primarily a personal learning repository, contributions are welcome! If you find errors or have suggestions:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📝 Notes and Disclaimer

- This repository is for **educational purposes only**
- The code and notes are based on the 3rd edition of the book
- Some implementations may differ from the book's examples as they reflect my personal understanding and experiments
- This repository is **not affiliated** with O'Reilly Media or the book's author
- Please purchase the book to support the author: [Hands-On Machine Learning on O'Reilly](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 About

**Author**: M-F-Tushar  
**Purpose**: Personal learning and knowledge sharing  
**Status**: In Progress (Continuously updated as I progress through the book)

---

## 🌟 Acknowledgments

- **Aurélien Géron** - For writing this exceptional book
- **O'Reilly Media** - For publishing quality technical content
- The **open-source community** - For maintaining Scikit-Learn, TensorFlow, and Keras
- Fellow learners who share their knowledge and experiences

---

## 📧 Contact & Feedback

If you find this repository helpful or have suggestions:

- ⭐ **Star** this repository
- 🐛 **Report issues** via GitHub Issues
- 💬 **Start discussions** about ML concepts
- 🔀 **Fork** and create your own learning journey

---

<div align="center">

**Happy Learning! 🚀**

*"The only way to learn machine learning is by doing machine learning"*

</div>
