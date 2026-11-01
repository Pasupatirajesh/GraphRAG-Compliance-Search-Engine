# Mini Evaluation: Self-Correction ON vs OFF

Same query set, same corpus (attention.pdf), two graph configurations.

| Query | Expected | Loop | Verdict | Rewrite rounds | Faithfulness | Relevance | Time (s) |
|---|---|---|---|---|---|---|---|
| How does scaled dot-product attention work? | in-scope | ON  | pass | 1 | 1.0 | 1.0 | 4.9 |
| How does scaled dot-product attention work? | in-scope | OFF | pass | 1 | 1.0 | 1.0 | 4.4 |
| What problem does multi-head attention solve over a single attention head? | in-scope | ON  | pass | 1 | 1.0 | 1.0 | 3.3 |
| What problem does multi-head attention solve over a single attention head? | in-scope | OFF | pass | 1 | 1.0 | 1.0 | 3.3 |
| What optimizer and learning rate schedule were used to train the model? | in-scope | ON  | pass | 1 | 1.0 | 1.0 | 4.0 |
| What optimizer and learning rate schedule were used to train the model? | in-scope | OFF | pass | 1 | 1.0 | 1.0 | 3.6 |
| attention is all you need paper summary | in-scope (terse phrasing) | ON  | pass | 1 | 1.0 | 1.0 | 4.0 |
| attention is all you need paper summary | in-scope (terse phrasing) | OFF | pass | 1 | 1.0 | 1.0 | 4.0 |
| What is today's date? | out-of-scope | ON  | unanswerable | 1 | 1.0 | 0.0 | 3.3 |
| What is today's date? | out-of-scope | OFF | unanswerable | 1 | 1.0 | 0.0 | 3.0 |
| Can you recommend a good pizza restaurant nearby? | out-of-scope | ON  | unanswerable | 1 | 1.0 | 1.0 | 3.1 |
| Can you recommend a good pizza restaurant nearby? | out-of-scope | OFF | unanswerable | 1 | 1.0 | 1.0 | 2.9 |
| Write me a recursive quicksort in Python. | out-of-scope (code-generation request) | ON  | unanswerable | 1 | 1.0 | 1.0 | 3.8 |
| Write me a recursive quicksort in Python. | out-of-scope (code-generation request) | OFF | unanswerable | 1 | 1.0 | 0.0 | 3.2 |
