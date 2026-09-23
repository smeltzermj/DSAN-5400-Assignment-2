# Problem 1

### Part B -

**If you only had access to the prior probabilities, would you be more likely to guess that Kennedy or Johnson authored an unlabeled paper? Why?**

The prior probability is how likely an event or parameter is before collecting any other data. In this case, the prior probabilities would be $P(K)$, i.e., the probability that a paper in the training set is Kennedy authored, and $P(J)$, the probability that a paper in the training set is Johnson authored. Given that there are 36 Kennedy-authored papers and 66-Johnson authored papers, totaling 102 papers, the priors are:

$$
P(\text{Kennedy}) = \frac{36}{36+66} = \frac{36}{102} \approx 0.353
$$

$$
P(\text{Johnson}) = \frac{66}{36+66} = \frac{66}{102} \approx 0.647
$$

Thus, you'd be more likely to guess Johnson authored an unlabeled paper.

### Part C

**Does one author's documents tend to be shorter?**

Kennedy's speeches tend to be shorter in this collection. Kennedy's speeches have mean and median lengths of 2879 and 2291, respectively, while Johnson's are 3620 and 3848. Both mean and median suggest Johnson was a fair bit more long-winded than Kennedy. 

**What are the most common words used by each author?**

The 20 most common words for both authors are dominated by stop words such as “the,” “of,” “and,” and “to.” One noticeable difference is the relative frequency of first-person language. Johnson uses “I” at roughly twice the rate of Kennedy (1.72% versus 0.85%), while Kennedy uses the first-person plural possessive “our” somewhat more frequently than Johnson (1.48% versus 1.02%).

**Do either authors tend to start or end their works in a consistent way?**

I completed a simple examination of a sample of the speeches and did not find a strong or consistent pattern in how either Kennedy or Johnson began or ended their speeches. For both men, speeches often began with formal references to the audience, while the endings varied considerably, with some concluding in expressions of thanks. This surface-level analysis did not reveal any especially distinctive patterns, though more systematic methods might identify patterns that are not obvious from manual inspection.

### Part E

**What are your two prior probability estimates?**\

The estimated prior probabilities were approximately 0.353 for Kennedy and 0.647 for Johnson.

**What is the shape of the matrix storing your likelihoods?**

he likelihood matrix had shape \((2, 22961)\), corresponding to two authors and 22,961 vocabulary terms.

**What happens when you vary the smoothing hyperparameter** $\alpha$ (alpha)?

Increasing the Lidstone smoothing parameter (alpha) reduced accuracy in this test set. Accuracy remained at 0.80 for \(\alpha=0.01\) and \(\alpha=0.1\), declined to 0.70 at \(\alpha=1.0\), and fell to 0.50 at \(\alpha=10.0\), suggesting that excessive smoothing weakened the differences in word likelihoods between the two authors. 

### Part F

**What are the predicted authors for each of the unlabeled works?**



# Problem 2

### Part B

**How do your predictions from your hand-built Naive Bayes classifier compare with the scikit-learn implementation?**

# Problem 3

### Part A

| ---                     | Accuracy      | F1-Score      |
|-------------------------|---------------|---------------|
| Naive Bayes Classifier  | 0.80 | .75 |
| Scikit-learn Classifier | .90 | .8889 |

### Part B

**For each classifier, what do you notice from the confusion matrix?**