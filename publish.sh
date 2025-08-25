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
PUBLIC_REPO="git@github.com:usc-mrel/PylotToneMRI.git"
EXCLUDE_FILE="$OLDPWD/publish-exclude.txt"  # Optional file listing tracked files to exclude

# Ensure public remote exists
if ! git remote | grep -q "$PUBLIC_REMOTE"; then
  git remote add $PUBLIC_REMOTE $PUBLIC_REPO
fi

# Create a temporary directory for public repo operations
TMPDIR=$(mktemp -d)
git clone --depth 1 $PUBLIC_REPO $TMPDIR
cd $TMPDIR
git checkout $PUBLIC_BRANCH || git checkout --orphan $PUBLIC_BRANCH

# Get list of tracked files from private repo
git -C "$OLDPWD" ls-files > /tmp/tracked.txt

# If an exclude file exists, tell rsync to skip those tracked files
EXCLUDE_ARG=""
if [ -f "$EXCLUDE_FILE" ]; then
  EXCLUDE_ARG="--exclude-from=$EXCLUDE_FILE"
fi

echo Copying tracked files to temporary repo...
# Copy tracked files to public repo, excluding .git and any exclude list
rsync -a --files-from=/tmp/tracked.txt $EXCLUDE_ARG --exclude='.git' "$OLDPWD/" ./

# Add all files and create a new commit
git add -A
git commit -m "$COMMIT_MSG"

# Push new commit to public repo (normal push, accumulates snapshots)
git push $PUBLIC_REMOTE $PUBLIC_BRANCH

# Clean up
cd "$OLDPWD"
rm -rf $TMPDIR /tmp/tracked.txt

echo "✅ Published snapshot to $PUBLIC_REPO with message:"
echo "   \"$COMMIT_MSG\""
