# AI From Scratch

Implementing core ML algorithms from scratch using only NumPy.
No sklearn for the actual implementation — only used at the end 
to validate that my results match.

## Algorithms

| Algorithm | Dataset | Visualizations |
|-----------|---------|----------------|
| Linear Regression | Custom (self-generated) | Yes — regression plots |
| Logistic Regression | Toy dataset | No |
| Decision Tree | Iris dataset | No |
| Random Forest | Iris dataset | No |
| K-means Clustering | Blob dataset | Yes — cluster plots |

## Highlights
- Random Forest built on top of my own Decision Tree implementation
- Binary cross-entropy math implemented manually for Logistic Regression
- K-means and Linear Regression include visual output

## Why I built this
Using sklearn is easy. Understanding what it's actually doing 
under the hood is the point. Building these forced me to actually 
understand the math before touching a library.

## Stack
- Python
- NumPy (implementations)
- scikit-learn (validation only — to verify my results match)
- Matplotlib (visualizations)
