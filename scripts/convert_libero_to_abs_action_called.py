if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import click
import numpy as np
import os
import pathlib

from convert_libero_to_abs_action import LiberoAbsoluteActionConverter


@click.command()
@click.option('-i', '--input', required=True, help='input hdf5 path')
@click.option('-l', '--path_to_libero_lib', required=True, help='path to libero lib')
@click.option('-o', '--output', required=True, help='output hdf5 path. Parent directory must exist')
@click.option('-d', '--demo_index', default=None, type=int)
@click.option('-g', '--gpu_id', default=None, type=int)
def main(input, path_to_libero_lib, output, demo_index, gpu_id):
    # process inputs
    input = pathlib.Path(input).expanduser()
    assert input.is_file()
    output = pathlib.Path(output).expanduser()
    os.makedirs(output.parent, exist_ok=True)
    assert output.parent.is_dir()
    assert not output.is_dir()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    converter = LiberoAbsoluteActionConverter(input, path_to_libero_lib)
    abs_actions = converter.convert_idx(demo_index)
    
    with open(output, 'wb') as f:
        np.savez(f, abs_actions=abs_actions)


if __name__ == "__main__":
    main()
