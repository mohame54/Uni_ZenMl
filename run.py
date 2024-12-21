# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from pipelines import serve_model_mlflow
from zenml.logger import get_logger

logger = get_logger(__name__)


def main():
    pipeline_args = {}
    pipeline_args["config_path"] = "config.yaml"
    pipeline_args["enable_cache"] = False
    pipe_configured = serve_model_mlflow.with_options(**pipeline_args)
    pipe_configured()
    logger.info("Inference pipeline finished successfully!")


if __name__ == "__main__":
    main()