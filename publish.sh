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

# Ensure we’re on the right branch
git checkout main

# Make sure the public remote exists
if ! git remote | grep -q "$PUBLIC_REMOTE"; then
  git remote add $PUBLIC_REMOTE $PUBLIC_REPO
fi

# Fetch the latest commit from public repo (if it exists)
git fetch $PUBLIC_REMOTE $PUBLIC_BRANCH || true

# Parent commit = last commit in public repo (if any)
PARENT=$(git rev-parse $PUBLIC_REMOTE/$PUBLIC_BRANCH 2>/dev/null || echo "")

if [ -n "$PARENT" ]; then
  echo "ℹ️ Found parent commit $PARENT from public repo"
  NEW_COMMIT=$(git commit-tree HEAD^{tree} -p $PARENT -m "$COMMIT_MSG")
else
  echo "ℹ️ No parent found — creating first snapshot commit"
  NEW_COMMIT=$(git commit-tree HEAD^{tree} -m "$COMMIT_MSG")
fi

# Move branch pointer to new commit
git update-ref refs/heads/$PUBLIC_BRANCH $NEW_COMMIT

# Push normally (no force needed)
git push $PUBLIC_REMOTE $PUBLIC_BRANCH

echo "✅ Published snapshot to $PUBLIC_REPO with message:"
echo "   \"$COMMIT_MSG\""