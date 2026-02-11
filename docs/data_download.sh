#!/bin/bash

# Many of the tutorials and examples in this function rely on pre-generated data.
# This script is meant to be run automatically as a GitHub Action to download
# necessary data, but it can also be run manually if you need to regenerate the data.

# NOTE: this script should be run from the root of the opinf directory
# Example usage: bash docs/data_download.sh

# remove the temp_data directory if it already exists
if [ -d "temp_data" ]; then
    echo "Removing existing temp_data/ directory..."
    rm -rf temp_data/
fi

# Clone into the repository and download only the data branch
DATA_URL="https://github.com/operator-inference/opinf.git"
echo "Cloning into the data branch of ${DATA_URL}..."
git clone -b data --single-branch --depth=1 ${DATA_URL} temp_data/

# move docs files into docs/source/api/ directory
DOC_DIR="docs/source/"
API_DIR="${DOC_DIR}api/"
echo "Moving basis_example.npy, pre_example.npy, and lstsq_example.npz into ${API_DIR} directory..."
mv temp_data/basis_example.npy ${API_DIR}
mv temp_data/lstsq_example.npz ${API_DIR}
mv temp_data/pre_example.npy ${API_DIR}

# move tutorial data files into docs/source/tutorials/ directory
TUTORIALS_DIR="${DOC_DIR}tutorials/"
echo "Moving basics_data.h5, inputs_data.h5, and parametric_data.h5 into ${TUTORIALS_DIR} directory..."
mv temp_data/basics_data.h5 ${TUTORIALS_DIR}
mv temp_data/inputs_data.h5 ${TUTORIALS_DIR}
mv temp_data/parametric_data.h5 ${TUTORIALS_DIR}

echo "Removing empty temp_data/ directory..."
rm -rf temp_data/

echo "Done!"