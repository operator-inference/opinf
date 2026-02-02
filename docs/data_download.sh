#!/bin/bash

# NOTE: this script should be run from the root of the opinf directory

# remove the temp_data directory if it already exists
if [ -d "temp_data" ]; then
    echo "Removing existing temp_data/ directory..."
    rm -rf temp_data/
fi

# Clone into the repository and download only the data branch
echo "Cloning into the data branch of https://github.com/XanderBys/opinf.git..."
git clone -b data --single-branch --depth=1 https://github.com/XanderBys/opinf.git temp_data/

# move docs files into docs/source/api/ directory
echo "Moving basis_example.npy, pre_example.npy, and lstsq_example.npz into docs/source/api/ directory..."
mv temp_data/basis_example.npy docs/source/api/
mv temp_data/lstsq_example.npz docs/source/api/
mv temp_data/pre_example.npy docs/source/api/

# move tutorial data files into docs/source/tutorials/ directory
echo "Moving basics_data.h5, inputs_data.h5, and parametric_data.h5 into docs/source/tutorials/ directory..."
mv temp_data/basics_data.h5 docs/source/tutorials/
mv temp_data/inputs_data.h5 docs/source/tutorials/
mv temp_data/parametric_data.h5 docs/source/tutorials/

echo "Removing empty temp_data/ directory..."
rm -rf temp_data/

echo "Done!"