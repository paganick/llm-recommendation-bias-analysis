# Git Repository Setup

The code has been committed to a local git repository. To push to GitHub:

## Option 1: Push to Existing GitHub Repository

If you already have a GitHub repository:

```bash
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Or if using SSH:
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin master
```

## Option 2: Create New GitHub Repository

1. Go to https://github.com/new
2. Create a new repository (e.g., "llm-recommendation-bias-analysis")
3. DO NOT initialize with README, .gitignore, or license (we already have these)
4. Copy the repository URL
5. Run:

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/llm-recommendation-bias-analysis.git

# Push code
git push -u origin master
```

## Check Current Repository Status

```bash
# See what's committed
git log --oneline

# See current remotes
git remote -v

# See current status
git status
```

## Recent Commits

```
b4d937e Add multi-trial analysis capability and improve bias reporting
2b2491c Initial commit: LLM recommendation bias analysis framework
```

## What's Included in the Repository

- ✅ All source code (data loaders, inference, recommenders, analysis)
- ✅ Analysis scripts (simple_test.py, run_complete_analysis.py, run_multi_trial_analysis.py)
- ✅ Batch experiment scripts (run_experiment.sh, run_batch_experiments.sh)
- ✅ Documentation (README.md, METHODOLOGY.md, IMPROVEMENTS_SUMMARY.md)
- ✅ Configuration examples (config.yaml.example)
- ✅ .gitignore (excludes outputs, data files, API keys, etc.)

## What's Excluded (via .gitignore)

- ❌ Data files (TwitterAAE-full-v1.zip, etc.) - too large
- ❌ Output directories (outputs/, logs/)
- ❌ API keys and credentials
- ❌ Python cache files (__pycache__, *.pyc)
- ❌ Virtual environments

## Collaborating

If you want to collaborate with others:

1. Push to GitHub (as above)
2. Share the repository URL
3. Others can clone with:
   ```bash
   git clone https://github.com/YOUR_USERNAME/llm-recommendation-bias-analysis.git
   cd llm-recommendation-bias-analysis
   ```
4. They'll need to:
   - Set up their own API keys (OPENAI_API_KEY or ANTHROPIC_API_KEY)
   - Download the data files (TwitterAAE-full-v1.zip)
   - Install dependencies: `pip install -r requirements.txt` (if you create one)

## Making Future Changes

```bash
# After making changes
git status                           # See what changed
git add -A                          # Stage all changes
git commit -m "Description of changes"
git push                            # Push to GitHub
```
