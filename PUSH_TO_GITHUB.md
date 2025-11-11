# How to Push DualGuide-SDXL to GitHub

Your project is ready to push to GitHub! Follow these steps:

## Option 1: Create New GitHub Repository (Recommended)

### Step 1: Create Repository on GitHub
1. Go to https://github.com/new
2. Fill in the repository details:
   - **Repository name**: `DualGuide-SDXL`
   - **Description**: `Dual Guidance for Stable Diffusion XL: Combining Token Perturbation Guidance (TPG) and Normalized Attention Guidance (NAG) for enhanced image generation`
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
3. Click "Create repository"

### Step 2: Push Your Code
After creating the repository on GitHub, run these commands:

```bash
# Add the remote repository (replace 'yourusername' with your GitHub username)
git remote add origin https://github.com/yourusername/DualGuide-SDXL.git

# Push the code
git branch -M main
git push -u origin main
```

## Option 2: Use GitHub CLI (if installed)

If you have GitHub CLI installed:

```bash
# Create repository and push in one go
gh repo create DualGuide-SDXL --public --source=. --remote=origin --push

# Or for private repository
gh repo create DualGuide-SDXL --private --source=. --remote=origin --push
```

## What's Been Committed

Your initial commit includes:
- ✅ Complete source code with TPG and NAG implementation
- ✅ Comprehensive README with usage examples
- ✅ Requirements and setup files
- ✅ Basic and advanced usage examples
- ✅ Proper Python package structure
- ✅ Apache 2.0 License
- ✅ .gitignore for Python projects

## After Pushing

1. Update the repository URL in `README.md` and `setup.py` if you use a different username
2. Consider adding:
   - Repository topics/tags: `stable-diffusion`, `sdxl`, `diffusion-models`, `pytorch`, `guidance`
   - Repository description on GitHub
   - Sample images in the README

## Repository Description for GitHub

Copy this for your GitHub repository description:

```
🎨 Training-free enhancement for Stable Diffusion XL combining Token Perturbation Guidance (TPG) and Normalized Attention Guidance (NAG) for superior image quality and prompt adherence.
```

## Suggested Topics/Tags

Add these topics to your GitHub repository:
- `stable-diffusion`
- `sdxl`
- `stable-diffusion-xl`
- `diffusion-models`
- `pytorch`
- `image-generation`
- `ai-art`
- `guidance`
- `token-perturbation`
- `attention-guidance`
- `diffusers`
- `huggingface`

## Need Help?

If you encounter any issues:
- Make sure you're logged into GitHub
- Check that you have push permissions
- Verify your GitHub username is correct in the remote URL

Run `git remote -v` to check your remote configuration.
