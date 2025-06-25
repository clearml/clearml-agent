FROM ubuntu:22.04

USER root
WORKDIR /root

ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV PYTHONIOENCODING=UTF-8

COPY ./setup.sh /root/setup.sh
RUN /root/setup.sh

COPY ./setup_aws.sh /root/setup_aws.sh
RUN chmod +x /root/setup_aws.sh && /root/setup_aws.sh

COPY ./entrypoint.sh /root/entrypoint.sh
COPY ./provider_entrypoint.sh /root/provider_entrypoint.sh
RUN chmod +x /root/provider_entrypoint.sh
COPY ./k8s_glue_example.py /root/k8s_glue_example.py
COPY ./clearml.conf /root/clearml.conf

ENTRYPOINT ["/root/entrypoint.sh"]