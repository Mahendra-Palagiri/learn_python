# Week 8 Mini Capstone — Model Defense

## Evaluation Rules (Locked)
- We keep a **final holdout test set** and evaluate on it **once**.
- We perform all preprocessing **inside a sklearn Pipeline**.
- We compare models using the **same CV strategy** and the **same primary metric**.
- We select models based on **CV mean + variance** (stability matters).
- We fix and record all random seeds.