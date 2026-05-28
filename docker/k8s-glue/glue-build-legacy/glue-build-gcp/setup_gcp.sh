#!/bin/bash

curl -LO https://dl.k8s.io/release/v1.29.3/bin/linux/amd64/kubectl

install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

sudo apt-get install -y apt-transport-https ca-certificates gnupg

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

apt-get update -y
apt-get install -y google-cloud-sdk
