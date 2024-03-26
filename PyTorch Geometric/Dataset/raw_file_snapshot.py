import yaml
from pathlib import Path

def save_directory_structure(path, output_file):
    def traverse(path):
        structure = {}
        files = []
        path = Path(path)
        what_return = None
        for item in path.iterdir():
            if item.is_dir():
                structure[item.name] = traverse(item)
                what_return = structure
            else:
                files.append(item.name)
                what_return = files

        return what_return

    structure = traverse(path)
    with open(output_file, 'w') as file:
        yaml.dump(structure, file)

# Example usage
path = Path(__file__).parent / 'raw'
output_file = Path(__file__).parent / 'file_list.yaml'
save_directory_structure(path, output_file)
