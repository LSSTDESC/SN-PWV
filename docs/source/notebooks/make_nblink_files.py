from pathlib import Path

file_dir = Path(__file__).resolve().parent
nb_dir = file_dir.parent.parent.parent / 'notebooks'

if __name__ == '__main__':
    # Remove old files
    for file in file_dir.glob('*.nblink'):
        file.unlink()

    # Create new files
    for file in nb_dir.glob('*.ipynb'):
        with (file_dir / file.with_suffix('.nblink').name).open('w') as new_file:
            print(file_dir / file.with_suffix('.nblink').name)
            new_file.write(f'{{"path": "../../../notebooks/{file.name}"}}')
