FROM python:3.9
COPY data/util /opt/app/data/util
COPY data/get_data.py /opt/app/data/
COPY FL_and_DP /opt/app/FL_and_DP
COPY model /opt/app/model
COPY optimizer /opt/app/optimizer
COPY privacy_analysis /opt/app/privacy_analysis
COPY SGD_and_DP /opt/app/SGD_and_DP
COPY train_and_validation /opt/app/train_and_validation
COPY GNN/*.py /opt/app/GNN/
COPY GNN/example /opt/app/GNN/example
COPY GNN/link_prediction/*.py /opt/app/GNN/link_prediction/
# COPY requirements_bakup.txt /opt/app
WORKDIR /opt/app
RUN pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 opacus==1.3.0 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install pyg_lib==0.1.0+pt113cu116 torch_scatter==2.1.0+pt113cu116 torch_sparse==0.6.16+pt113cu116 torch_cluster==1.6.0+pt113cu116 torch_spline_conv==1.2.1+pt113cu116 torch_geometric==2.2.0 -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install ogb==1.3.5
# RUN pip install -r /opt/app/requirements_bakup.txt
CMD ["bash"]