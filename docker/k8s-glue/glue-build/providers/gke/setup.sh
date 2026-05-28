#!/bin/bash
set -e

curl -LO https://dl.k8s.io/release/v1.24.0/bin/linux/amd64/kubectl
install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl

apt-get update -qqy
apt-get install -qqy apt-transport-https ca-certificates gnupg

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

apt-get update -qqy
apt-get install -qqy google-cloud-sdk
rm -rf /var/lib/apt/lists/*
