FROM ubuntu:18.04

ARG PROVIDER=base

USER root
WORKDIR /root

ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV PYTHONIOENCODING=UTF-8

COPY ./build-resources/setup.sh /root/setup.sh
COPY ./build-resources/entrypoint.sh /root/entrypoint.sh
RUN /root/setup.sh

COPY ./glue-build-multi/providers/${PROVIDER}/setup.sh /root/setup_provider.sh
RUN /root/setup_provider.sh

COPY ./glue-build-multi/providers/${PROVIDER}/provider_entrypoint.sh /root/provider_entrypoint.sh
COPY ./build-resources/k8s_glue_example.py /root/k8s_glue_example.py
COPY ./build-resources/clearml.conf /root/clearml.conf

ENTRYPOINT ["/root/entrypoint.sh"]
