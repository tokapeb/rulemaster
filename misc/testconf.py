import configparser
import json

# Init configuration
config = configparser.ConfigParser()
config.read('example.ini')
sources = json.loads(config.get("books","sources"))

sources_path = config.get("books","source_path")

print(sources_path)

print(sources)

for source in sources:
    print(f"Book: {sources_path}\{source['path']}")
