# What is this folder?

This folder is where the best models are saved during training.

Models will **only** be overwritten if they are better than the previous attempt.

# How do I update a model here?

Files in this directory are ignored by default in the `.gitignore` file.
If your model exceeds the current model on main, please commit your model forcibly, and a good message, e.g.:
 - `git add -f best_models/pitch.h5`
 - `git commit -m "Loss: 8502"`