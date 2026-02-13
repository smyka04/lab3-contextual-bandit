# Lab 3: Contextual Bandit-Based News Article Recommendation System

**Course:** Reinforcement Learning Fundamentals  
**Student Name:** Samyaka  
**Roll Number:** U20230065  
**GitHub Branch:** samyaka_U20230065

---

## Overview

This project implements a Contextual Multi-Armed Bandit system for personalized news recommendations. The system classifies users into categories and learns optimal news category preferences to maximize engagement.

**Implemented Algorithms:**
- Epsilon-Greedy
- Upper Confidence Bound (UCB)
- SoftMax

---

## Problem Statement

**Objective:** Build a recommendation system that classifies users and learns their news preferences.

**Environment:**
- Contexts: 3 user types (user_1, user_2, user_3)
- Arms: 4 news categories (Entertainment, Education, Tech, Crime)
- Total Arms: 12 (3 contexts × 4 categories)
- Rewards: Sampled using rlcmab_sampler package

**Arm Mapping:**

| Arm Index | Categories | User Context |
|-----------|------------|--------------|
| 0-3 | Entertainment, Education, Tech, Crime | user_1 |
| 4-7 | Entertainment, Education, Tech, Crime | user_2 |
| 8-11 | Entertainment, Education, Tech, Crime | user_3 |

---

## Approach and Design Decisions

### 1. Data Preprocessing

**News Articles:**
- Mapped diverse categories to 4 target categories
- COMEDY, ARTS → Entertainment
- COLLEGE, PARENTING → Education
- SCIENCE, BUSINESS → Tech
- WEIRD NEWS → Crime

**User Data:**
- Missing age values filled with median
- Region codes encoded using LabelEncoder
- Boolean subscriber field converted to integer
- 30 numerical features retained for classification

### 2. User Classification

**Model:** Decision Tree Classifier (max_depth=10)
- Split: 80% training, 20% validation
- Chosen for interpretability and handling non-linear patterns
- Fast training and no feature scaling needed

### 3. Bandit Algorithms

**Epsilon-Greedy:**
- Tested epsilon = 0.01, 0.1, 0.3
- Explores randomly with probability epsilon, otherwise exploits best arm

**UCB:**
- Tested C = 0.5, 1.0, 2.0
- Uses confidence bounds: UCB(arm) = avg_reward + C × sqrt(log(t) / pulls)

**SoftMax:**
- Temperature tau = 1.0
- Probabilistic selection: P(arm) ∝ exp(value/tau)

**Simulation:** 10,000 timesteps with uniform context distribution

### 4. Recommendation Pipeline

1. Classify user to determine context
2. Select best category using trained bandit
3. Sample random article from selected category

---

## Key Results

### Classification Performance

- Validation Accuracy: 82-87%
- All user classes well-separated with good precision/recall

### Bandit Performance (T=10,000)

**Epsilon-Greedy:**
- epsilon = 0.01: ~7.5-8.0
- epsilon = 0.1: ~7.0-7.5
- epsilon = 0.3: ~5.5-6.5

**UCB:**
- C = 0.5, 1.0, 2.0: All achieve ~7.8-8.2

**SoftMax:**
- tau = 1.0: ~7.8-8.2

**Best Strategy:** UCB and SoftMax achieve highest rewards with stable convergence

### Convergence
- Learning occurs in first 1,000 steps
- Stabilizes after 5,000-7,000 steps
- UCB shows smoothest convergence

---

## Installation and Setup

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn rlcmab-sampler
```

### Directory Structure

```
project/
├── data/
│   ├── news_articles.csv
│   ├── train_users.csv
│   └── test_users.csv
├── lab3_results_u20230065.ipynb
└── README.md
```

---

## Reproducing the Experiments

### Step 1: Setup

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn rlcmab-sampler

# Verify data files
ls data/  # Should show news_articles.csv, train_users.csv, test_users.csv
```

### Step 2: Run Notebook

**Using Jupyter:**
```bash
jupyter notebook lab3_results_u20230065.ipynb
# Run all cells: Cell → Run All
```

**Using VS Code:**
- Open notebook
- Select Python kernel
- Click "Run All"

### Step 3: Expected Outputs

**Generated Files:**
- `bandit_comparison.png`
- `epsilon_comparison.png`
- `ucb_comparison.png`
- `hyperparameter_comparison.png`
- `recommendations_output.csv`

**Console Output:**
- Dataset statistics
- Classification metrics
- Bandit performance for each hyperparameter
- Final summary

### Verification

- Classification accuracy: 80-90%
- Plots show convergence
- CSV has 2,000 recommendations
- All 4 categories present in recommendations

---

## Observations

### Hyperparameter Effects

**Epsilon-Greedy:**
- Lower epsilon (0.01): Fast convergence, may miss optimal arms
- Higher epsilon (0.3): More exploration, slower convergence

**UCB:**
- C=1.0 provides best balance between exploration and exploitation

**SoftMax:**
- tau=1.0 gives smooth probabilistic exploration

### Strategy Comparison

**Epsilon-Greedy:**
- Simple and interpretable
- Random exploration can be inefficient

**UCB:**
- Theoretically optimal
- Smooth convergence with confidence bounds

**SoftMax:**
- Probabilistic action selection
- Smooth exploration-exploitation tradeoff

### Key Findings

- Different user contexts prefer different categories
- Personalization significantly improves over random selection
- All algorithms converge within 7,000 steps
- Classification accuracy is critical for performance

---

## Troubleshooting

**Module not found:**
```bash
pip install <package_name>
```

**File not found:**
```bash
# Verify data directory exists
ls data/
```

**Plots not displaying:**
```python
%matplotlib inline
```

**Memory issues:**
```python
T = 5000  # Reduce timesteps
