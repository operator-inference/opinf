#!/bin/bash

# NOTE: this script should be run from the docs/ directory

# remove the temp_data directory if it already exists
if [ -d "temp_data" ]; then
    echo "Removing existing temp_data/ directory..."
    rm -rf temp_data/
fi

# Clone into the repository and download only the data branch
echo "Cloning into the data branch of https://github.com/operator-inference/opinf.git..."
git clone -b data --single-branch --depth=1 https://github.com/operator-inference/opinf.git temp_data/

# move docs files into api/ directory and all other data files into the data/ directory
echo "Moving basis_example.npy, pre_example.npy, and lstsq_example.npz into docs/source/api/ directory..."
mv temp_data/basis_example.npy source/api/
mv temp_data/lstsq_example.npz source/api/
mv temp_data/pre_example.npy source/api/

echo "Moving basics_data.h5, inputs_data.h5, and parametric_data.h5 into docs/source/tutorials/ directory..."
mv temp_data/basics_data.h5 source/tutorials/
mv temp_data/inputs_data.h5 source/tutorials/
mv temp_data/parametric_data.h5 source/tutorials/

echo "Removing empty temp_data/ directory..."
rm -rf temp_data/

echo "Done!"