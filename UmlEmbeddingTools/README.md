# UmlEmbeddingTools
- This tool is used to embed class name, transfer graph data format, map uml, class and method IDs, and integrate 3 datasets (training, valid and test).
- `uml2pyg.py` is to  embed class name and transfer graph data format, `method2class.py` is to map uml, class and method IDs, and `integrate.py` is to integrate datasets.
# Examples:
- Environment configuration
```shell
conda env create -f tf2.yaml
conda activate tf2
```
- Embedding and tranfering
```
# take training set as an example
python uml2pyg.py --data 1k/train/umls.pkl -part train
```
- Mapping ID and Integration
```
bash run_idx_1k.sh
```
