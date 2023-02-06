#!/bin/bash
# Download and unzip SC2

MAPFILE=`pwd`'/smac/smac/env/starcraft2/maps/SMAC_Maps'

mkdir 3rdparty
cd 3rdparty

export SC2PATH=`pwd`'/StarCraftII'
echo 'SC2PATH is set to '$SC2PATH

if [ ! -d $SC2PATH ]; then
        echo 'StarCraftII is not installed. Installing now ...';
        wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.6.2.69232.zip
        unzip -P iagreetotheeula SC2.4.6.2.69232.zip
        rm -rf SC2.4.6.2.69232.zip
else
        echo 'StarCraftII is already installed.'
fi

echo 'Adding SMAC maps.'
MAP_DIR="$SC2PATH/Maps/"
echo 'MAP_DIR is set to '$MAP_DIR

if [ ! -d $MAP_DIR ]; then
        mkdir -p $MAP_DIR
fi

cp -r $MAPFILE $MAP_DIR

echo 'StarCraft II and SMAC are installed.'
