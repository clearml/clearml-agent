#!/bin/bash
set -e

curl -LO https://dl.k8s.io/release/v1.24.0/bin/linux/amd64/kubectl
install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl
