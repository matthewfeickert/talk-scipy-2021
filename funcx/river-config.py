from funcx_endpoint.endpoint.utils.config import Config
from funcx_endpoint.executors import HighThroughputExecutor
from funcx_endpoint.providers.kubernetes.kube import KubernetesProvider
from funcx_endpoint.strategies import KubeSimpleStrategy
from parsl.addresses import address_by_route

config = Config(
    executors=[
        HighThroughputExecutor(
            max_workers_per_node=1,
            address=address_by_route(),
            strategy=KubeSimpleStrategy(max_idletime=3600),
            container_type="docker",
            scheduler_mode="hard",
            provider=KubernetesProvider(
                init_blocks=0,
                min_blocks=1,
                max_blocks=100,
                init_cpu=2,
                max_cpu=3,
                init_mem="2000Mi",
                max_mem="4600Mi",
                image="bengal1/pyhf-funcx:3.8.0.3.0-1",
                worker_init="pip freeze",
                namespace="servicex",
                incluster_config=True,
            ),
        )
    ],
    heartbeat_period=15,
    heartbeat_threshold=200,
    log_dir="/tmp/worker_logs",
    funcx_service_address="https://api2.funcx.org/v2",
    detach_endpoint=True,
)

# For now, visible_to must be a list of URNs for globus auth users or groups, e.g.:
# urn:globus:auth:identity:{user_uuid}
# urn:globus:groups:id:{group_uuid}
meta = {
    "name": "default",
    "description": "",
    "organization": "",
    "department": "",
    "public": True,
    "visible_to": [],
}
