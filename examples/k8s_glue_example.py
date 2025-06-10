"""
This example assumes you have preconfigured services with selectors in the form of
 "ai.clearml.agent.serial=pod-<number>" and a targetPort of 10022.
The K8sIntegration component will label each pod accordingly.
"""
from argparse import ArgumentParser

from clearml_agent.glue.k8s import K8sIntegration


def parse_args():
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group()

    parser.add_argument(
        "--queue",
        type=str,
        help="Queues to pull tasks from. If multiple queues, use comma separated list, e.g. 'queue1,queue2'",
    )
    group.add_argument(
        "--ports-mode",
        action="store_true",
        default=False,
        help="Ports-Mode will add a label to the pod which can be used as service, in order to expose ports"
        "Should not be used with max-pods",
    )
    parser.add_argument(
        "--num-of-services",
        type=int,
        default=20,
        help="Specify the number of k8s services to be used. Use only with ports-mode.",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        help="Used in conjunction with ports-mode, specifies the base port exposed by the services. "
        "For pod #X, the port will be <base-port>+X. Note that pod number is calculated based on base-pod-num"
        "e.g. if base-port=20000 and base-pod-num=3, the port for the first pod will be 20003",
    )
    parser.add_argument(
        "--base-pod-num",
        type=int,
        default=1,
        help="Used in conjunction with ports-mode and base-port, specifies the base pod number to be used by the "
        "service (default: %(default)s)",
    )
    parser.add_argument(
        "--gateway-address",
        type=str,
        default=None,
        help="Used in conjunction with ports-mode, specify the external address of the k8s ingress / ELB",
    )
    parser.add_argument(
        "--pod-clearml-conf",
        type=str,
        help="Configuration file to be used by the pod itself (if not provided, current configuration is used)",
    )
    parser.add_argument(
        "--overrides-yaml", type=str, help="YAML file containing pod overrides to be used when launching a new pod"
    )
    parser.add_argument(
        "--template-yaml",
        type=str,
        help="YAML file containing pod template. If provided pod will be scheduled with kubectl apply "
        "and overrides are ignored, otherwise it will be scheduled with kubectl run",
    )
    parser.add_argument(
        "--ssh-server-port",
        type=int,
        default=0,
        help="If non-zero, every pod will also start an SSH server on the selected port (default: zero, not active)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        help="Specify the namespace in which pods will be created (default: %(default)s)",
        default="clearml",
    )
    group.add_argument(
        "--max-pods",
        type=int,
        help="Limit the maximum number of pods that this service can run at the same time."
        "Should not be used with ports-mode",
    )
    parser.add_argument(
        "--use-owner-token",
        action="store_true",
        default=False,
        help="Generate and use task owner token for the execution of each task",
    )
    parser.add_argument(
        "--create-queue",
        action="store_true",
        default=False,
        help="Create the queue if it does not exist (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    user_props_cb = None
    if args.ports_mode and args.base_port:

        def k8s_user_props_cb(pod_number=0):
            user_prop = {"k8s-pod-port": args.base_port + pod_number}
            if args.gateway_address:
                user_prop["k8s-gateway-address"] = args.gateway_address
            return user_prop

        user_props_cb = k8s_user_props_cb

    k8s = K8sIntegration(
        ports_mode=args.ports_mode,
        num_of_services=args.num_of_services,
        base_pod_num=args.base_pod_num,
        user_props_cb=user_props_cb,
        overrides_yaml=args.overrides_yaml,
        clearml_conf_file=args.pod_clearml_conf,
        template_yaml=args.template_yaml,
        extra_bash_init_script=K8sIntegration.get_ssh_server_bash(ssh_port_number=args.ssh_server_port)
        if args.ssh_server_port
        else None,
        namespace=args.namespace,
        max_pods_limit=args.max_pods or None,
    )
    queue = [q.strip() for q in args.queue.split(",") if q.strip()] if args.queue else None

    k8s.k8s_daemon(queue, use_owner_token=args.use_owner_token, create_queue=args.create_queue)


if __name__ == "__main__":
    main()
