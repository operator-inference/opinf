#!/bin/bash

# remove the opinf directory if it already exists
if [ -d "opinf" ]; then
    echo "Removing existing opinf/ directory..."
    rm -rf opinf/
fi

# Clone into the repository and download only the data branch
echo "Cloning into the data branch of https://github.com/operator-inference/opinf.git..."
git clone -b data --single-branch --depth=1 https://github.com/operator-inference/opinf.git

# move docs files into api/ directory and all other data files into the data/ directory
echo "Moving basis_example.npy, pre_example.npy, and lstsq_example.npz into docs/source/api/ directory..."
mv opinf/basis_example.npy opinf/lstsq_example.npz opinf/pre_example.npy source/api/

echo "Moving all other data files into data/ directory..."
mv opinf/* data/ 

echo "Removing empty opinf/ directory..."
rm -rf opinf/

echo "Done!"