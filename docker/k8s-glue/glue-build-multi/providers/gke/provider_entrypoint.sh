#!/bin/bash -x

gcloud auth activate-service-account ${CLEARML_SERVICE_ACC} --key-file=/root/keys/${SERVICE_ACC_KEY_JSON}
gcloud container clusters get-credentials ${CLUSTER_CRED}
