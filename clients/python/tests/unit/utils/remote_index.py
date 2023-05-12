import time
from loguru import logger
import os
from enum import Enum
import pinecone as pinecone

from urllib3.exceptions import MaxRetryError

QUOTA = 2

class PodType(Enum):
    """
    Enum for pod types
    """
    P1 = 'p1'
    S1 = 's1'
    P2 = 'p2'
    P1_X2 = 'p1.x2'
    S1_X2 = 's1.x2'
    P2_X2 = 'p2.x2'
    S1H = 's1h'

    def __str__(self):
        return self.value

    def as_name(self):
        return self.value.replace('.', '-')

    def is_implicitly_x1(self):
        return '.' not in self.value

class RemoteIndex:
    index = None

    def __init__(self, pods=1, index_name=None, dimension=512, pod_type="p1", metadata_config=None,
                 _openapi_client_config=None, source_collection='',metric='cosine'):
        self.env = os.getenv('PINECONE_ENVIRONMENT')
        self.key = os.getenv('PINECONE_API_KEY')
        self.pod_type = pod_type
        self.pods = pods
        self.index_name = index_name if index_name else 'sdk-citest-{0}'.format(pod_type)
        self.dimension = dimension
        self.metadata_config = metadata_config
        self.source_collection = source_collection
        self.metric = metric
        self.client = pinecone.Client(self.key, self.env)

    def __enter__(self):
        if self.index_name not in self.client.list_indexes():
            index_creation_args = {'name': self.index_name,
                                   'dimension': self.dimension,
                                   'pod_type': str(self.pod_type),
                                   'pods': self.pods,
                                   'metadata_config': self.metadata_config,
                                   'source_collection': self.source_collection,
                                   'metric': self.metric}
            self.client.create_index(**index_creation_args)

        self.index = self.client.get_index(self.index_name)
        return self.index

    @staticmethod
    def wait_for_ready(index):
        logger.info('waiting until index gets ready...')
        max_attempts = 30
        for i in range(max_attempts):
            try:
                time.sleep(1)
                index.describe_index_stats()
                break
            except (Exception, MaxRetryError):
                if i + 1 == max_attempts:
                    logger.info("Index didn't get ready in time. Raising error...")
                    raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('deleting index')
        self.client.delete_index(self.index_name)
