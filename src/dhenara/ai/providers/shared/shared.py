import logging
from typing import TypedDict

from google.oauth2 import service_account

from dhenara.ai.types.genai.ai_model import AIModelAPI, AIModelAPIProviderEnum

logger = logging.getLogger(__name__)


class VertexAICredentials(TypedDict):
    credentials: service_account.Credentials
    project_id: str
    location: str


# -----------------------------------------------------------------------------
class APIProviderSharedFns:
    # -------------------------------------------------------------------------
    # client params for vertext ai is diffente
    @staticmethod
    def get_vertex_ai_credentials(
        api: AIModelAPI,
    ) -> VertexAICredentials:
        if api.provider != AIModelAPIProviderEnum.GOOGLE_VERTEX_AI:
            error_msg = (
                "get_vertex_ai_client_params should only be called for api with provider vertext ai. "
                f"provider={api.provider}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        client_params = api.get_provider_credentials()

        service_account_json = client_params.get("service_account_json")
        if not isinstance(service_account_json, dict):
            raise ValueError("Vertex AI credentials require service_account_json as a mapping")

        project_id = client_params.get("project_id")
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError("Vertex AI credentials require a non-empty project_id")

        location = client_params.get("location")
        if not isinstance(location, str) or not location.strip():
            raise ValueError("Vertex AI credentials require a non-empty location")

        scopes = [
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/cloud-platform.read-only",
        ]

        sa_credentials = service_account.Credentials.from_service_account_info(service_account_json, scopes=scopes)

        return {
            "credentials": sa_credentials,
            "project_id": project_id,
            "location": location,
        }
