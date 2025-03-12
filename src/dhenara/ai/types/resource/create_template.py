from _resource_config import ResourceConfig

try:
    ResourceConfig.create_credentials_template()
except Exception as e:
    print(f"Error while creating file: {e}")
