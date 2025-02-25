from dhenara.ai.types.shared.base import BaseEnum, BaseModel


# -----------------------------------------------------------------------------
class FileFormatEnum(BaseEnum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    COMPRESSED = "compressed"
    UNDEFINED = "undefined"


# -----------------------------------------------------------------------------
class FileContentFormatEnum(BaseEnum):
    TEXT = "text"
    BINARY = "binary"
    RAW = "raw"
    UNDEFINED = "undefined"


# -----------------------------------------------------------------------------
class FileProcessingStatusEnum(BaseEnum):
    SUCCESS = "success"
    FAIL = "fail"
    UNDEFINED = "undefined"


# -----------------------------------------------------------------------------
class FileMetadata(BaseModel):
    file_name: str
    file_size: int | float
    mime_type: str | None  # TODO: Enforce after fixing current mime gussess
    isinstance_of_dj_fileclass: bool | None = None  # TODO:Remove this internal flag


# -----------------------------------------------------------------------------
class FileContentData(BaseModel):
    name: str
    metadata: FileMetadata | None
    file_format: FileFormatEnum
    content_format: FileContentFormatEnum
    content: str | None
    contents: list["FileContentData"] | None
    folder_structure: str | None  # Represent the folder structure in case of a compressed file
    processing_status: FileProcessingStatusEnum
    processing_message: str

    def get_content(self):
        if self.file_format != FileFormatEnum.COMPRESSED:
            return self.content or ""
        elif self.contents:
            all_contents = "".join([content.get_content() for content in self.contents])
            return all_contents
        else:
            return ""

    def get_dict_without_content(self):
        if self.file_format != FileFormatEnum.COMPRESSED:
            return {
                "name": self.name,
                "metadata": self.metadata,
                "file_format": self.file_format.value,
                "content_format": self.content_format.value,
                "folder_structure": self.folder_structure,
                "processing_status": self.processing_status.value,
                "processing_message": self.processing_message,
            }
        elif self.contents:
            all_contents = "".join([content.get_dict_without_content() for content in self.contents])
            return {
                "name": self.name,
                "metadata": self.metadata,
                "file_format": self.file_format.value,
                "content_format": self.content_format.value,
                "folder_structure": self.folder_structure,
                "processing_status": self.processing_status.value,
                "processing_message": self.processing_message,
                "file_contents": all_contents,
            }
        else:
            return ""
