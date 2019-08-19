#!/bin/bash
# installing python dependencies tool
sudo apt-get install python3-pip
# upgrading pip to the most recent version
pip3 install --upgrade pip
pip3 --version
# installing virtualenv, used to manage python environments
sudo pip3 install virtualenv
virtualenv LEWIS-FYP
source LEWIS-FYP/bin/activate
# creating a new virtual environment for the appropriate packages
pip3 install -r requirements-3.txt
echo 'Modules installed! See encoder/word-model-command.txt for details on running the models.'
