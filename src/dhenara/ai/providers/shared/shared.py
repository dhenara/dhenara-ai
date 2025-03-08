import logging

from google.oauth2 import service_account

from dhenara.ai.types.external_api import AIModelAPIProviderEnum
from dhenara.ai.types.genai.ai_model import AIModelAPI

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
class APIProviderSharedFns:
    # -------------------------------------------------------------------------
    # client params for vertext ai is diffente
    @staticmethod
    def get_vertex_ai_credentials(
        api: AIModelAPI,
    ):
        if api.provider == AIModelAPIProviderEnum.GOOGLE_VERTEX_AI:
            client_params = api.get_provider_credentials()

            service_account_json = client_params["service_account_json"]
            project_id = client_params["project_id"]
            location = client_params["location"]

            scopes = [
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/cloud-platform.read-only",
            ]

            sa_credentials = service_account.Credentials.from_service_account_info(service_account_json, scopes=scopes)

            #  Vertex AI API
            final_client_params = {
                "credentials": sa_credentials,
                "project_id": project_id,
                "location": location,
            }
            return final_client_params
        else:
            logger.error(
                f"get_vertex_ai_client_params should only be called for api with provider vertext ai. "
                f"provider={api.provider}"
            )
            return {}
