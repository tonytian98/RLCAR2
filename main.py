import sys
from map_processor import MapProcessor

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <map_file_path>")
        sys.exit(1)

    map_file_path = sys.argv[1]

    map_processor = MapProcessor(map_path=map_file_path)
    map_processor.process_map()
