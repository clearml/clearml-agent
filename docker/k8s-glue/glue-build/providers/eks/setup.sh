#!/bin/bash
set -e

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
./aws/install
rm -rf awscliv2.zip aws/

curl -LO https://dl.k8s.io/release/v1.24.0/bin/linux/amd64/kubectl
install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl

curl -o aws-iam-authenticator https://amazon-eks.s3.us-west-2.amazonaws.com/1.21.2/2021-07-05/bin/linux/amd64/aws-iam-authenticator
install -o root -g root -m 0755 aws-iam-authenticator /usr/local/bin/aws-iam-authenticator
rm aws-iam-authenticator
