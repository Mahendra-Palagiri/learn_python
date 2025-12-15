''' Week 6 · Day 1 — Regression From Scratch (No Libraries)

🎯 Learning Goal

By the end of today, we will understand:
    • What linear regression is actually doing (predicting with a line: ŷ = wx + b)
    • What “loss” means in regression and why MSE is commonly used
    • How gradient descent updates w and b (what dw and db represent intuitively)
    • How learning rate affects training behavior (smooth → oscillation → explosion)
    • How to verify training worked using evidence:
        - Loss curve decreases over iterations
        - Learned line visually fits the data
        - Learned (w, b) approach the true (w_true, b_true) on synthetic data
    • Why feature scaling impacts optimization (what happens if X is multiplied by 100)

This is where we stop treating regression like a black box and start understanding the engine.

'''