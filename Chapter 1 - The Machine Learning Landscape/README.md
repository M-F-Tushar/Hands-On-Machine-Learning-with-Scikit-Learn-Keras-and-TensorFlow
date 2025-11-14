# Machine Learning: Key Concepts & Definitions

## Core Definitions

### What is Machine Learning?
**Simple Definition**: Programming computers so they can learn from data.

**Arthur Samuel (1959)**: Field of study that gives computers the ability to learn without being explicitly programmed.

**Tom Mitchell (1997)**: A program learns from experience E with respect to task T and performance measure P, if its performance on T improves with experience E.

---

## Essential Terms

### Training & Data Terms

| Term | Definition |
|------|------------|
| **Training Set** | Examples the system uses to learn |
| **Training Instance/Sample** | Each individual training example |
| **Model** | The part that learns and makes predictions (e.g., neural networks, random forests) |
| **Label** | The desired solution/answer in supervised learning |
| **Features** | Input variables/attributes used for predictions (also called predictors or attributes) |
| **Target** | The value being predicted (used in regression) |

### Performance Terms

| Term | Definition |
|------|------------|
| **Accuracy** | Ratio of correctly classified items |
| **Generalization Error** | Error rate on new, unseen cases (out-of-sample error) |
| **Training Error** | Error rate on training data |
| **Performance Measure** | How you measure if the model is good |

---

## Types of Machine Learning Systems

### 1. By Training Supervision

#### **Supervised Learning**
- **What**: Training with labeled data (answers provided)
- **Tasks**:
  - **Classification**: Categorizing items (spam vs. ham emails)
  - **Regression**: Predicting numbers (house prices, GDP)
- **Example**: Spam filter trained with examples of spam and non-spam emails

#### **Unsupervised Learning**
- **What**: Training with unlabeled data (no answers provided)
- **Tasks**:
  - **Clustering**: Grouping similar items together
  - **Dimensionality Reduction**: Simplifying data while keeping important info
  - **Anomaly Detection**: Finding unusual patterns
  - **Association Rule Learning**: Discovering relationships in data
- **Example**: Grouping blog visitors by behavior without telling the system what groups exist

#### **Semi-Supervised Learning**
- **What**: Mix of labeled and unlabeled data
- **Example**: Google Photos recognizing faces (clustering) then you label who they are

#### **Self-Supervised Learning**
- **What**: System creates its own labels from unlabeled data
- **Example**: Masking parts of images and training model to restore them
- **Key Point**: Uses transfer learning - knowledge from one task helps another

#### **Reinforcement Learning**
- **What**: Agent learns by trial and error, getting rewards/penalties
- **Components**:
  - **Agent**: The learning system
  - **Policy**: Strategy defining what actions to take
  - **Rewards**: Points for good/bad actions
- **Example**: AlphaGo learning to play Go

---

### 2. By Learning Style

#### **Batch Learning (Offline Learning)**
- **What**: Trained on all data at once, then deployed
- **Process**: Train → Launch → Use (no more learning)
- **Problem**: Model rot/data drift - performance decays over time
- **When to Use**: When you can retrain periodically with all data

#### **Online Learning (Incremental Learning)**
- **What**: Learns continuously from new data
- **Key Parameter**: Learning rate (how fast it adapts)
  - High rate: Adapts quickly but forgets old data
  - Low rate: More stable but slower to learn
- **Uses**:
  - Rapidly changing data (stock markets)
  - Limited computing resources
  - **Out-of-core learning**: Huge datasets that don't fit in memory

---

### 3. By How They Generalize

#### **Instance-Based Learning**
- **What**: Learns examples by heart, compares new cases using similarity
- **Example**: K-nearest neighbors - classifying based on most similar training examples

#### **Model-Based Learning**
- **What**: Builds a mathematical model from examples
- **Process**:
  1. Study data
  2. Select model type
  3. Train model (find best parameters)
  4. Make predictions
- **Example**: Linear regression predicting life satisfaction from GDP

---

## Important Machine Learning Concepts

### Feature Engineering
**What**: Creating good features for training

**Steps**:
1. **Feature Selection**: Pick most useful existing features
2. **Feature Extraction**: Combine features to create better ones
3. **Create New Features**: Gather new data

### Model Parameters vs. Hyperparameters

| Model Parameters | Hyperparameters |
|-----------------|-----------------|
| Learned during training | Set before training |
| Define the model's predictions | Control the learning process |
| Example: θ₀, θ₁ in linear regression | Example: learning rate, regularization strength |

### Regularization
**What**: Constraining a model to make it simpler and prevent overfitting

**How**: Limits model complexity (e.g., keeping parameter values small)

**Purpose**: Balance between fitting training data and generalizing to new data

---

## Main Challenges in Machine Learning

### 1. Bad Data Problems

#### **Insufficient Training Data**
- **Problem**: Most ML needs thousands/millions of examples
- **Solution**: Get more data or use transfer learning

#### **Nonrepresentative Training Data**
- **Problem**: Training data doesn't match real-world cases
- **Causes**:
  - **Sampling Noise**: Small sample not representative by chance
  - **Sampling Bias**: Flawed sampling method
- **Example**: 1936 election poll failed because sample was biased toward wealthy voters

#### **Poor-Quality Data**
- **Problems**: Errors, outliers, noise, missing values
- **Solutions**:
  - Remove outliers
  - Fix errors manually
  - Fill in missing values
  - Ignore problematic attributes

#### **Irrelevant Features**
- **Problem**: "Garbage in, garbage out"
- **Solution**: Feature engineering to get relevant features

---

### 2. Bad Model Problems

#### **Overfitting**
- **What**: Model performs well on training data but poorly on new data
- **Cause**: Model too complex for the amount of data
- **Signs**: Low training error, high generalization error
- **Solutions**:
  - Simplify model (fewer parameters)
  - Get more training data
  - Reduce noise in data
  - Apply regularization

#### **Underfitting**
- **What**: Model too simple to learn data patterns
- **Cause**: Model not powerful enough
- **Solutions**:
  - Use more powerful model
  - Better feature engineering
  - Reduce model constraints (less regularization)

---

## Testing & Validation

### Data Splits

#### **Training Set** (typically 80%)
- Data used to train the model

#### **Test Set** (typically 20%)
- Data used to estimate generalization error
- Never used during training
- Simulates real-world performance

#### **Validation Set (Dev Set)**
- Held-out portion of training data
- Used to select best model and tune hyperparameters
- Prevents "cheating" by repeatedly testing on test set

#### **Train-Dev Set**
- Used when training data differs from production data
- Helps identify if problems are from overfitting or data mismatch

### Cross-Validation
**What**: Using multiple small validation sets

**How**: Average performance across all validation sets for more accurate measure

**Tradeoff**: More accurate but takes longer to train

---

## Key Strategies & Workflows

### Typical ML Project Workflow
1. **Study the data**
2. **Select a model**
3. **Train the model** (learning algorithm finds best parameters)
4. **Make predictions** (inference on new cases)

### When to Use Machine Learning

✅ **Good for**:
- Problems needing lots of fine-tuning or long rule lists
- Complex problems with no traditional solution
- Fluctuating environments (can retrain on new data)
- Getting insights from large data (data mining)

❌ **Not needed for**:
- Simple rule-based problems
- Problems with clear algorithmic solutions
- When you lack sufficient data

---

## Important Principles

### No Free Lunch Theorem
**What**: No model is guaranteed to work best for all problems

**Meaning**: You must test different models for your specific data

**Implication**: Make reasonable assumptions and evaluate multiple reasonable models

### Data vs. Algorithms
- For complex problems, having more data often matters more than having better algorithms
- Small/medium datasets still common - don't abandon good algorithms

---

## Common ML Applications

| Application | Technique |
|-------------|-----------|
| Image classification | CNNs, Transformers |
| Tumor detection | Semantic segmentation (CNNs) |
| Text classification | RNNs, CNNs, Transformers |
| Text summarization | NLP tools |
| Revenue forecasting | Regression models |
| Speech recognition | RNNs, CNNs, Transformers |
| Fraud detection | Anomaly detection |
| Customer segmentation | Clustering (k-means, DBSCAN) |
| Data visualization | Dimensionality reduction |
| Recommender systems | Neural networks |
| Game bots | Reinforcement learning |

---

## Quick Reference: Key Warnings

⚠️ **Critical Points to Remember**:
1. Test set must be representative of production data
2. Never tune hyperparameters on test set (use validation set)
3. Clean your data - garbage in, garbage out
4. Balance model complexity with data amount
5. Monitor for model rot/data drift in production
6. Overfitting = good training performance, poor test performance
7. Underfitting = poor performance everywhere

---

## Glossary of Terms

- **Accuracy**: Percentage of correct predictions
- **Agent**: Learning system in reinforcement learning
- **Cost Function**: Measures how bad model predictions are
- **Data Mining**: Discovering hidden patterns in large data
- **Fitness Function**: Measures how good model is
- **Inference**: Using trained model to make predictions
- **Model Rot/Data Drift**: Performance decay over time
- **Novelty Detection**: Finding instances different from training set
- **Policy**: Strategy for choosing actions (reinforcement learning)
- **Transfer Learning**: Using knowledge from one task for another
- **Utility Function**: Same as fitness function
