#!/bin/bash
set -e

# Usage: ./publish.sh "Commit message for public repo"

if [ -z "$1" ]; then
  echo "❌ Please provide a commit message."
  echo "Usage: $0 \"Your commit message\""
  exit 1
fi

COMMIT_MSG="$1"
PUBLIC_REMOTE="public"
PUBLIC_BRANCH="main"
PUBLIC_REPO="git@github.com:USERNAME/PUBLIC-REPO.git"

# Ensure public remote exists
if ! git remote | grep -q "$PUBLIC_REMOTE"; then
  git remote add $PUBLIC_REMOTE $PUBLIC_REPO
fi

# Create a temporary directory for public repo operations
TMPDIR=$(mktemp -d)
git clone --depth 1 $PUBLIC_REPO $TMPDIR
cd $TMPDIR
git checkout $PUBLIC_BRANCH || git checkout --orphan $PUBLIC_BRANCH

# Copy current code from private repo (without .git)
rsync -a --exclude='.git' "$OLDPWD/" ./

# Add all files and create a new commit
git add -A
git commit -m "$COMMIT_MSG"

# Push new commit
git push $PUBLIC_REMOTE $PUBLIC_BRANCH

# Clean up
cd "$OLDPWD"
rm -rf $TMPDIR

echo "✅ Published snapshot to $PUBLIC_REPO with message:"
echo "   \"$COMMIT_MSG\""
