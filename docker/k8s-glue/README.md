# ClearML k8s-glue

Example Docker image and Kubernetes templates for running the ClearML k8s glue
as a pod inside a Kubernetes cluster.

ClearML credentials and server addresses should either be filled into the
`clearml.conf` file before building the glue Docker image, or provided in the
`k8s-glue.yml` template.

## Building the Docker image

A single Dockerfile at `glue-build-multi/Dockerfile` produces the image for
all supported providers. The provider is selected at build time via the
`PROVIDER` build argument.

### Providers

| `PROVIDER` | Installs                                                  |
|------------|-----------------------------------------------------------|
| `base`     | `kubectl`                                                 |
| `eks`      | `kubectl`, AWS CLI v2, `aws-iam-authenticator`            |
| `gke`      | `kubectl`, Google Cloud SDK (`gcloud`)                    |

### Prerequisites

- Docker
- Run the commands below from the repository root

### Build commands

Base image:

```bash
docker build \
  --build-arg PROVIDER=base \
  -t clearml-agent-k8s-base:dev \
  -f docker/k8s-glue/glue-build-multi/Dockerfile \
  docker/k8s-glue/
```

EKS image:

```bash
docker build \
  --build-arg PROVIDER=eks \
  -t clearml-agent-k8s-eks:dev \
  -f docker/k8s-glue/glue-build-multi/Dockerfile \
  docker/k8s-glue/
```

GKE image:

```bash
docker build \
  --build-arg PROVIDER=gke \
  -t clearml-agent-k8s-gke:dev \
  -f docker/k8s-glue/glue-build-multi/Dockerfile \
  docker/k8s-glue/
```

## Deploying to Kubernetes

1. Create a secret from `pod_template.yml`:

   ```bash
   kubectl -n clearml create secret generic k8s-glue-pod-template --from-file=pod_template.yml
   ```

2. Apply the k8s glue template:

   ```bash
   kubectl -n clearml apply -f k8s-glue.yml
   ```
