sudo apt install -y python3-venv
python3 -m venv pyg_MIDS
source pyg_MIDS/bin/activate
 
python3 -m pip install torch
python3 -m pip install torch_geometric
python3 -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html