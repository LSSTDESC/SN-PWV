from pathlib import Path

nblink_dir = Path(__file__).resolve().parent
notebook_dir = nblink_dir.parent.parent.parent / 'notebooks'

# Names of notebook files to create links to
notebook_file_names = [
    'lsst_filters.ipynb',
    'pwv_eff_on_black_body.ipynb',
    'pwv_modeling.ipynb',
    'simulating_lc_for_cadence.ipynb',
    'sne_delta_mag.ipynb'
]

if __name__ == '__main__':
    # Remove old files
    for file in nblink_dir.glob('*.nblink'):
        file.unlink()

    # Create new files
    for file_name in notebook_file_names:
        notebook_path = notebook_dir / file_name
        if not notebook_path.exists():
            raise FileNotFoundError(notebook_path)

        nblink_path = (nblink_dir / file_name).with_suffix('.nblink')
        with nblink_path.open('w') as new_file:
            new_file.write(f'{{"path": "../../../notebooks/{file_name}"}}')
