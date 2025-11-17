# GitHub Push Guide

## Repository Information

**Target Repository:** https://github.com/takakhoo/AI_Neural_AudioCodec_Remastering

**Current Remote:** https://github.com/takakhoo/ENGS_Honors_Thesis (needs to be updated)

## Completed Tasks

### 1. ✅ Image Organization
All images have been organized into the following structure:
- `images/loss_curves/` - All loss curve visualizations
- `images/spectrograms/` - Spectrogram comparisons
- `images/token_visualizations/` - Token coverage and heatmaps
- `images/training_results/` - Training overview and hyperparameter tables
- `images/poster_assets/` - Poster-related images (if any)

### 2. ✅ Comprehensive README.md
Created a detailed README.md with:
- Complete mathematical foundations
- Architecture walkthrough
- Training pipeline documentation
- Evaluation results
- Usage guide
- Code structure documentation

### 3. ✅ Python Scripts Identified

#### Scripts to Push (12 total):

**CBAMFiLMUNet + InvLSTM src/** (4 scripts):
- train3.py
- models3.py
- evaluate3.py
- dataset3.py

**DeepUnet & LSTM src/** (4 scripts):
- train2.py
- models2.py
- evaluate2.py
- dataset2.py

**VocoderUNet & LSTM src/** (4 scripts):
- train.py
- models.py
- evaluate.py
- dataset.py

#### Folders with No Python Scripts:
- Baseline Test/ (only .wav files)
- curriculum_loss_curves/ (only images, already moved)
- CurriculumInference/ (no .py files)
- CurriculumTraining/ (training outputs only)
- GriffinLimNetEval/ (no .py files)
- GriffinLimNetTraining/ (no .py files)
- MasterNetEval/ (no .py files)
- MasterNetTraining/ (no .py files)
- TokenEval/ (no .py files)
- TokenTraining/ (training outputs only)
- poster_assets/ (no .py files)
- poster_assets_L3_FINAL/ (no .py files)

## Next Steps to Push to GitHub

### Step 1: Update Git Remote
```bash
cd /scratch2/f004h1v/thesis_project
git remote set-url origin https://github.com/takakhoo/AI_Neural_AudioCodec_Remastering.git
git remote -v  # Verify
```

### Step 2: Add Python Scripts from Specified Folders
```bash
# Add baseline experimental scripts
git add "CBAMFiLMUNet + InvLSTM src/*.py"
git add "DeepUnet & LSTM src/*.py"
git add "VocoderUNet & LSTM src/*.py"

# Add main source code (if not already tracked)
git add src/*.py

# Add root-level Python scripts
git add *.py

# Add README and documentation
git add README.md
git add PUSH_SUMMARY.md
git add GITHUB_PUSH_GUIDE.md
```

### Step 3: Add Organized Image Folders
```bash
git add images/
```

### Step 4: Handle Git LFS Issues (if needed)
If you encounter LFS errors with large files:
```bash
# Option 1: Skip LFS for now
git config lfs.fetchexclude "*"
git config lfs.pushskip "*"

# Option 2: Add large files to .gitignore
echo "*.wav" >> .gitignore
echo "*.pt" >> .gitignore
echo "checkpoints/*.pt" >> .gitignore
```

### Step 5: Commit Changes
```bash
git commit -m "Add comprehensive README, organize images, and include baseline experimental scripts

- Added detailed README.md with mathematical foundations and architecture walkthrough
- Organized all images into structured folders (loss_curves, spectrograms, token_visualizations, training_results)
- Included baseline experimental scripts from CBAMFiLMUNet, DeepUnet, and VocoderUNet folders
- Added documentation for GitHub push process"
```

### Step 6: Push to GitHub
```bash
# Check current branch
git branch

# Push to main branch (or create new branch if needed)
git push origin main

# If pushing to new branch:
# git checkout -b main
# git push -u origin main
```

## Important Notes

1. **Git LFS:** The repository appears to use Git LFS for large files. You may need to configure LFS properly or exclude large files from tracking.

2. **Large Files:** Consider adding to .gitignore:
   - `*.wav` (audio files)
   - `*.pt` (PyTorch checkpoints)
   - `checkpoints/` (checkpoint directories)
   - `CurriculumTraining/` (training outputs)
   - `TokenTraining/` (training outputs)

3. **Repository Structure:** The main source code is in `src/` directory, which should already be tracked. The baseline experimental scripts are in separate folders as documented above.

4. **Documentation:** All documentation is now in README.md. Additional guides are in:
   - `PUSH_SUMMARY.md` - Summary of Python scripts
   - `GITHUB_PUSH_GUIDE.md` - This file

## Verification Checklist

Before pushing, verify:
- [ ] README.md is complete and accurate
- [ ] All images are organized in `images/` folder
- [ ] Python scripts from specified folders are added
- [ ] Git remote is set to correct repository
- [ ] Large files are handled appropriately (LFS or .gitignore)
- [ ] No sensitive information in code or configs
- [ ] All paths in README are correct

## Troubleshooting

### Git LFS Permission Errors
If you see LFS permission errors:
```bash
# Temporarily disable LFS
git config lfs.fetchexclude "*"
git config lfs.pushskip "*"
```

### Large File Errors
If GitHub rejects large files:
```bash
# Add to .gitignore and remove from tracking
echo "large_file.wav" >> .gitignore
git rm --cached large_file.wav
```

### Remote URL Issues
If remote URL is incorrect:
```bash
git remote set-url origin https://github.com/takakhoo/AI_Neural_AudioCodec_Remastering.git
git remote -v  # Verify
```

