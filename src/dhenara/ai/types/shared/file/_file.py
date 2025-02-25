import logging

from dhenara.ai.types.shared.base import BaseModel
from dhenara.ai.types.shared.file import FileContentData, FileContentFormatEnum, FileFormatEnum, FileMetadata

logger = logging.getLogger(__name__)


# TODO: Need cleanup
# -----------------------------------------------------------------------------
class GenericFile(BaseModel):
    name: str
    metadata: FileMetadata | None = None


# -----------------------------------------------------------------------------
class LocalFile(GenericFile):
    url: str | None


# -----------------------------------------------------------------------------
class StoredFile(GenericFile):
    url: str | None


# -----------------------------------------------------------------------------
class ProcessedFile(GenericFile):
    processed_content: FileContentData

    def get_source_file_name(self) -> str:
        """Get the original source file name"""
        return self.processed_content.name

    def get_file_format(self) -> FileFormatEnum:
        """Get the file format enum"""
        return self.processed_content.file_format

    def get_content_format(self) -> FileContentFormatEnum:
        """Get the content format enum"""
        return self.processed_content.content_format

    def get_metadata(self) -> FileMetadata | None:
        """Get file metadata"""
        # TODO_FUTURE : for processed files, metatdata shalll be duplicated in the content and in the file metadata
        return self.processed_content.metadata

    def get_mime_type(self) -> str | None:
        """Get the mime type of the file"""
        mime_type = self.get_metadata()
        if mime_type:
            return mime_type.mime_type.lower()
        return None

    def get_processed_file_data(self, max_words: int | None = None) -> str:
        """Get processed file data with optional word limit"""
        # content = str(self.processed_content.content or "")
        # content = self.model_dump_json()
        _content = self.model_dump()
        content = str(_content)

        if max_words:
            words = content.split()
            content = " ".join(words[:max_words])
        return content

    def get_processed_file_data_content_only(self):
        if self.processed_content:
            # NOTE: Its impossilbe to return `only` the content in case of zip files.
            # Here we will return the full content.
            # This should be OK as long as zip files won't need to be send as image/byte in context. TODO_FUTURE
            if self.processed_content.contents:
                return self.get_processed_file_data()

            content_str = self.processed_content.content
            return content_str
        else:
            return ""
