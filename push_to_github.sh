#!/bin/bash

# Instructions:
# 1. First, create a new repository on GitHub named "SP500-Options-Scanner"
#    - Go to https://github.com/crajarshi
#    - Click "New" repository
#    - Name: SP500-Options-Scanner
#    - Description: Advanced intraday technical analysis scanner for S&P 500 stocks to identify options trading opportunities
#    - Make it Public
#    - DO NOT initialize with README, .gitignore, or license
#    - Click "Create repository"
#
# 2. Then run this script to push your code:

echo "Please enter your GitHub username (likely 'crajarshi'):"
read GITHUB_USERNAME

# Add remote origin
git remote add origin https://github.com/$GITHUB_USERNAME/SP500-Options-Scanner.git

# Push to GitHub
git push -u origin main

echo "Done! Your code is now on GitHub at: https://github.com/$GITHUB_USERNAME/SP500-Options-Scanner"